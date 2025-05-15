from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
import asyncio
import base64
import json
import logging
import os
import re
import sys
import time
import uuid
import urllib.parse
from typing import Literal, Optional, List
from pathlib import Path

# Import FastAPI components for static file serving
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Import OpenAI components
from openai import AsyncOpenAI, APIConnectionError, RateLimitError, APIStatusError
from openai.types import ImagesResponse

# Import utility functions
from utils import get_openai_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a dataclass for our application context
@dataclass
class OpenAIImageContext:
    """Context for the OpenAI Image Generation MCP server."""
    openai_client: AsyncOpenAI

@asynccontextmanager
async def openai_image_lifespan(server: FastMCP) -> AsyncIterator[OpenAIImageContext]:
    """
    Manages the OpenAI client lifecycle.
    
    Args:
        server: The FastMCP server instance
        
    Yields:
        OpenAIImageContext: The context containing the OpenAI client
    """
    # Create and return the OpenAI client with the helper function in utils.py
    openai_client = get_openai_client()
    
    try:
        yield OpenAIImageContext(openai_client=openai_client)
    finally:
        # No explicit cleanup needed for the OpenAI client
        pass

# Define constants for images directory and public URL
IMAGES_DIR = Path("ai-images")
IMAGES_DIR.mkdir(exist_ok=True)

# Get server host and port from environment variables
HOST = os.getenv("HOST", "0.0.0.0")
PORT = os.getenv("PORT", "8050")
PUBLIC_URL = os.getenv("PUBLIC_URL", f"http://{HOST}:{PORT}")

# Initialize FastMCP server with the OpenAI client as context
mcp = FastMCP(
    "openai-gpt-image-1",
    description="MCP server for OpenAI GPT Image generation and editing capabilities",
    lifespan=openai_image_lifespan,
    host=HOST,
    port=PORT
)

# Get the FastAPI app from FastMCP to add static file serving
app = mcp.get_app()

# Mount static files directory for serving generated images
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

# Add a download endpoint for downloading images by filename
@app.get("/download/{filename}")
async def download_image(filename: str):
    file_path = IMAGES_DIR / filename
    if not file_path.exists():
        return {"error": "File not found"}
    return FileResponse(path=str(file_path), filename=filename, media_type="image/png")

@mcp.tool()
async def generate_image(
    ctx: Context,
    prompt: str,
    model: str = "gpt-image-1", # Current model as per docs
    n: Optional[int] = 1, # Number of images (default 1)
    size: Optional[Literal["1024x1024", "1536x1024", "1024x1536", "auto"]] = "auto", # Size options
    quality: Optional[Literal["low", "medium", "high", "auto"]] = "auto", # Quality options
    user: Optional[str] = None, # Optional end-user identifier
    save_filename: Optional[str] = None # Optional: specify filename (without extension)
) -> dict:
    """
    Generates an image using OpenAI's gpt-image-1 model based on a text prompt.
    Returns both the base64 data and saves the image locally.

    Args:
        ctx: The MCP context object (automatically passed)
        prompt: The text description of the desired image(s)
        model: The model to use (currently 'gpt-image-1')
        n: The number of images to generate (Default: 1)
        size: Image dimensions ('1024x1024', '1536x1024', '1024x1536', 'auto'). Default: 'auto'
        quality: Rendering quality ('low', 'medium', 'high', 'auto'). Default: 'auto'
        user: An optional unique identifier representing your end-user
        save_filename: Optional filename (without extension). If None, a default name based on the prompt and timestamp is used

    Returns:
        A dictionary containing:
        - status: "success" or "error"
        - saved_path: Local path where image was saved
        - filename: Generated filename
        - image_url: Direct URL to view the image
        - download_url: URL to download the image
        Or an error dictionary if the API call or saving fails.
    """
    logging.info(f"Tool 'generate_image' called with prompt: '{prompt[:50]}...'")

    # Basic validation
    if model != "gpt-image-1":
        logging.warning(f"Model '{model}' specified, but current documentation points to 'gpt-image-1'. Proceeding anyway.")

    try:
        logging.info(f"Requesting image generation from OpenAI with model={model}, size={size}, quality={quality}, n={n}")

        # Get OpenAI client from context
        client = ctx.request_context.lifespan_context.openai_client

        # Prepare arguments, removing None values
        api_args = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
            "quality": quality,
            "user": user
        }
        cleaned_args = {k: v for k, v in api_args.items() if v is not None}

        response: ImagesResponse = await client.images.generate(**cleaned_args)
        logging.info(f"Image generation API call successful.")

        # Process and save the image
        if not response.data or not response.data[0].b64_json:
             logging.error("API response did not contain image data.")
             return {"status": "error", "message": "API call succeeded but no image data received."}

        try:
            image_b64 = response.data[0].b64_json # Extract b64 data
            image_bytes = base64.b64decode(image_b64)

            # Determine filename (generate default if needed)
            final_filename = ""
            if not save_filename:
                 # Generate default filename: first 5 words of prompt + timestamp
                 safe_prompt = re.sub(r'[^\w\s-]', '', prompt).strip().lower()
                 prompt_part = "-".join(safe_prompt.split()[:5])
                 timestamp = time.strftime("%Y%m%d-%H%M%S")
                 final_filename = f"{prompt_part}-{timestamp}.png"
                 logging.info(f"No save_filename provided, generated default: {final_filename}")
            else:
                 # Sanitize provided filename and ensure .png extension
                 safe_filename = re.sub(r'[^\w\s-]', '', save_filename).strip()
                 if not safe_filename.lower().endswith('.png'):
                     final_filename = f"{safe_filename}.png"
                 else:
                     final_filename = safe_filename
                 logging.info(f"Using provided save_filename (sanitized): {final_filename}")

            # Get the current directory
            current_dir = os.getcwd()
            save_dir = os.path.join(current_dir, "ai-images")

            # Ensure directory exists
            os.makedirs(save_dir, exist_ok=True)
            full_save_path = os.path.join(save_dir, final_filename)

            # Save the image locally (optional in Docker environment)
            with open(full_save_path, "wb") as f:
                 f.write(image_bytes)
            logging.info(f"Image successfully saved to: {full_save_path}")
            
            # Generate URLs for direct access and download
            image_url = f"{PUBLIC_URL}/images/{final_filename}"
            download_url = f"{PUBLIC_URL}/download/{final_filename}"
            
            # Return success, paths and URLs for client-side usage
            return {
                "status": "success", 
                "saved_path": full_save_path,
                "filename": final_filename,
                "image_url": image_url,
                "download_url": download_url
            }

        except Exception as save_e:
             logging.error(f"Failed to save image: {save_e}")
             # Return failure message if saving failed
             return {"status": "error", "message": f"Image generated but failed to save: {save_e}"}

    except APIConnectionError as e:
        logging.error(f"OpenAI API request failed to connect: {e}")
        return {"status_code": 503, "status_message": "API Connection Error", "error_details": str(e)}
    except RateLimitError as e:
        logging.error(f"OpenAI API request exceeded rate limit: {e}")
        return {"status_code": 429, "status_message": "Rate Limit Exceeded", "error_details": str(e)}
    except APIStatusError as e:
        logging.error(f"OpenAI API returned an error status: {e.status_code} - {e.response}")
        return {"status_code": e.status_code, "status_message": "API Error", "error_details": e.response.text}
    except Exception as e:
        logging.exception(f"An unexpected error occurred during image generation: {e}")
        return {"status_code": 500, "status_message": "Internal Server Error", "error_details": str(e)}

@mcp.tool()
async def edit_image(
    ctx: Context,
    prompt: str,
    image_paths: List[str], # List of paths to input image(s)
    mask_path: Optional[str] = None, # Optional path to mask image for inpainting
    model: str = "gpt-image-1", # Current model as per docs
    n: Optional[int] = 1, # Number of images (default 1)
    size: Optional[Literal["1024x1024", "1536x1024", "1024x1536", "auto"]] = "auto", # Size options
    quality: Optional[Literal["low", "medium", "high", "auto"]] = "auto", # Quality options
    user: Optional[str] = None, # Optional end-user identifier
    save_filename: Optional[str] = None # Optional: specify filename (without extension)
) -> dict:
    """
    Edits an image or creates variations using OpenAI's gpt-image-1 model.
    Returns both the base64 data and saves the image locally.
    Can use multiple input images as reference or perform inpainting with a mask.

    Args:
        ctx: The MCP context object (automatically passed)
        prompt: The text description of the desired final image or edit
        image_paths: A list of file paths to the input image(s). Must be PNG. < 25MB
        mask_path: Optional file path to the mask image (PNG with alpha channel) for inpainting. Must be same size as input image(s). < 25MB
        model: The model to use (currently 'gpt-image-1')
        n: The number of images to generate (Default: 1)
        size: Image dimensions ('1024x1024', '1536x1024', '1024x1536', 'auto'). Default: 'auto'
        quality: Rendering quality ('low', 'medium', 'high', 'auto'). Default: 'auto'
        user: An optional unique identifier representing your end-user
        save_filename: Optional filename (without extension). If None, a default name based on the prompt and timestamp is used

    Returns:
        A dictionary containing:
        - status: "success" or "error"
        - saved_path: Local path where image was saved
        - filename: Generated filename
        - image_url: Direct URL to view the image
        - download_url: URL to download the image
        Or an error dictionary if the API call or saving fails.
    """
    logging.info(f"Tool 'edit_image' called with prompt: '{prompt[:50]}...'")
    logging.info(f"Input image paths: {image_paths}")
    if mask_path:
        logging.info(f"Mask path: {mask_path}")

    # Basic validation
    if model != "gpt-image-1":
        logging.warning(f"Model '{model}' specified, but current documentation points to 'gpt-image-1'. Proceeding anyway.")
    if not image_paths:
        return {"status_code": 400, "status_message": "Missing required parameter: image_paths cannot be empty."}

    image_files = []
    mask_file = None
    try:
        # Get OpenAI client from context
        client = ctx.request_context.lifespan_context.openai_client
        
        # Open image files
        for path in image_paths:
            if not os.path.exists(path):
                 return {"status_code": 400, "status_message": f"Input image file not found: {path}"}
            image_files.append(open(path, "rb")) # Keep file handles open until API call

        # Open mask file if provided
        if mask_path:
            if not os.path.exists(mask_path):
                 return {"status_code": 400, "status_message": f"Mask file not found: {mask_path}"}
            mask_file = open(mask_path, "rb")

        logging.info(f"Requesting image edit from OpenAI with model={model}, size={size}, quality={quality}, n={n}")

        # Prepare arguments, removing None values
        api_args = {
            "model": model,
            "prompt": prompt,
            "image": image_files, # Pass the list of file objects
            "mask": mask_file, # Pass the mask file object or None
            "n": n,
            "size": size,
            "quality": quality,
            "user": user
        }
        cleaned_args = {k: v for k, v in api_args.items() if v is not None}

        response: ImagesResponse = await client.images.edit(**cleaned_args)
        logging.info(f"Image edit API call successful. Attempting to save.")

        # Process and save the image
        if not response.data or not response.data[0].b64_json:
             logging.error("API response did not contain image data for edit.")
             return {"status": "error", "message": "API call succeeded but no image data received."}

        try:
            image_b64 = response.data[0].b64_json # Extract b64 data
            image_bytes = base64.b64decode(image_b64)

            # Determine filename (generate default if needed)
            final_filename = ""
            if not save_filename:
                 # Generate default filename: first 5 words of prompt + timestamp
                 safe_prompt = re.sub(r'[^\w\s-]', '', prompt).strip().lower()
                 prompt_part = "-".join(safe_prompt.split()[:5])
                 timestamp = time.strftime("%Y%m%d-%H%M%S")
                 final_filename = f"edited-{prompt_part}-{timestamp}.png" # Add 'edited-' prefix
                 logging.info(f"No save_filename provided, generated default: {final_filename}")
            else:
                 # Sanitize provided filename and ensure .png extension
                 safe_filename = re.sub(r'[^\w\s-]', '', save_filename).strip()
                 if not safe_filename.lower().endswith('.png'):
                     final_filename = f"{safe_filename}.png"
                 else:
                     final_filename = safe_filename
                 logging.info(f"Using provided save_filename (sanitized): {final_filename}")

            # Get the current directory
            current_dir = os.getcwd()
            save_dir = os.path.join(current_dir, "ai-images")

            # Ensure directory exists
            os.makedirs(save_dir, exist_ok=True)
            full_save_path = os.path.join(save_dir, final_filename)

            # Save the image locally (optional in Docker environment)
            with open(full_save_path, "wb") as f:
                f.write(image_bytes)
            logging.info(f"Edited image successfully saved to: {full_save_path}")
            
            # Generate URLs for direct access and download
            image_url = f"{PUBLIC_URL}/images/{final_filename}"
            download_url = f"{PUBLIC_URL}/download/{final_filename}"
            
            # Return success, paths and URLs for client-side usage
            return {
                "status": "success", 
                "saved_path": full_save_path,
                "filename": final_filename,
                "image_url": image_url,
                "download_url": download_url
            }

        except Exception as save_e:
            logging.error(f"Failed to save edited image: {save_e}")
            # Return failure message if saving failed
            return {"status": "error", "message": f"Image edited but failed to save: {save_e}"}

    except FileNotFoundError as e:
         logging.error(f"File not found during image edit preparation: {e}")
         return {"status_code": 400, "status_message": "File Not Found", "error_details": str(e)}
    except APIConnectionError as e:
        logging.error(f"OpenAI API request failed to connect: {e}")
        return {"status_code": 503, "status_message": "API Connection Error", "error_details": str(e)}
    except RateLimitError as e:
        logging.error(f"OpenAI API request exceeded rate limit: {e}")
        return {"status_code": 429, "status_message": "Rate Limit Exceeded", "error_details": str(e)}
    except APIStatusError as e:
        logging.error(f"OpenAI API returned an error status: {e.status_code} - {e.response}")
        return {"status_code": e.status_code, "status_message": "API Error", "error_details": e.response.text}
    except Exception as e:
        logging.exception(f"An unexpected error occurred during image edit: {e}")
        return {"status_code": 500, "status_message": "Internal Server Error", "error_details": str(e)}
    finally:
        # Ensure all opened files are closed
        for f in image_files:
            if f:
                f.close()
        if mask_file:
            mask_file.close()

async def main():
    transport = os.getenv("TRANSPORT", "sse")
    if transport == 'sse':
        # Run the MCP server with sse transport
        await mcp.run_sse_async()
    else:
        # Run the MCP server with stdio transport
        await mcp.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())