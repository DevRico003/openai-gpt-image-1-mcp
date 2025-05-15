from openai import AsyncOpenAI
from supabase import create_client, Client
import logging
import os
import sys
import uuid
import time
from typing import Optional, Tuple

def get_openai_client():
    """
    Creates and returns an AsyncOpenAI client with the appropriate API key.
    
    Returns:
        AsyncOpenAI: The initialized OpenAI client
    
    Raises:
        SystemExit: If the required API key is not found
    """
    # Get OpenAI API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        logging.error("FATAL: OPENAI_API_KEY environment variable not set.")
        sys.exit("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    
    # Create and return async client
    try:
        client = AsyncOpenAI(api_key=api_key)
        logging.info("AsyncOpenAI client initialized successfully.")
        return client
    except Exception as e:
        logging.error(f"Failed to initialize AsyncOpenAI client: {e}")
        sys.exit(f"Failed to initialize AsyncOpenAI client: {e}")

def get_supabase_client() -> Client:
    """
    Creates and returns a Supabase client with the appropriate credentials.
    
    Returns:
        Client: The initialized Supabase client
    
    Raises:
        SystemExit: If the required credentials are not found
    """
    # Get Supabase credentials from environment
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        logging.error("FATAL: SUPABASE_URL and SUPABASE_KEY environment variables must be set.")
        sys.exit("Supabase credentials not found. Please set the SUPABASE_URL and SUPABASE_KEY environment variables.")
    
    # Create and return client
    try:
        client = create_client(supabase_url, supabase_key)
        logging.info("Supabase client initialized successfully.")
        return client
    except Exception as e:
        logging.error(f"Failed to initialize Supabase client: {e}")
        sys.exit(f"Failed to initialize Supabase client: {e}")

async def upload_image_to_supabase(image_bytes: bytes, filename: str) -> Tuple[str, str]:
    """
    Uploads an image to Supabase Storage and returns the public URL.
    
    Args:
        image_bytes: The image data as bytes
        filename: Filename to use for the uploaded image
        
    Returns:
        Tuple containing:
        - Public URL of the uploaded image
        - Path of the image in Supabase storage
        
    Raises:
        Exception: If upload fails
    """
    try:
        # Get Supabase client and bucket name
        supabase = get_supabase_client()
        bucket_name = os.getenv("SUPABASE_BUCKET", "image")
        
        # Sanitize filename - remove special characters and replace spaces with underscores
        # Convert to ASCII to remove non-ASCII characters like umlauts
        import re
        import unicodedata
        
        def sanitize_filename(name):
            # Remove accents and convert to ASCII
            name_ascii = unicodedata.normalize('NFKD', name)
            name_ascii = ''.join([c for c in name_ascii if not unicodedata.combining(c)])
            
            # Replace remaining non-alphanumeric chars (except for .png extension) with underscore
            name_safe = re.sub(r'[^\w\.-]', '_', name_ascii)
            return name_safe
        
        safe_filename = sanitize_filename(filename)
        logging.info(f"Sanitized filename for Supabase: {safe_filename}")
        
        # Generate a unique path for the image to prevent conflicts
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        folder_path = f"openai-images/{timestamp}-{unique_id}"
        storage_path = f"{folder_path}/{safe_filename}"
        
        # Prepare file stream for upload
        from io import BytesIO
        file_stream = BytesIO(image_bytes)
        
        # Upload the image with updated API usage
        res = supabase.storage.from_(bucket_name).upload(
            path=storage_path,
            file=file_stream,
            file_options={"content-type": "image/png"}
        )
        
        if not res:
            logging.error(f"Supabase upload failed: {res}")
            raise Exception(f"Failed to upload image to Supabase: {res}")
        
        # Get the public URL for the uploaded image
        public_url = supabase.storage.from_(bucket_name).get_public_url(storage_path)
        
        if not public_url:
            raise Exception("Failed to get public URL from Supabase")
        
        logging.info(f"Image uploaded successfully to Supabase: {public_url}")
        return public_url, storage_path
        
    except Exception as e:
        logging.error(f"Error uploading image to Supabase: {e}")
        raise Exception(f"Failed to upload image to Supabase: {str(e)}")

def get_storage_mode() -> str:
    """
    Gets the current storage mode from environment variables.
    
    Returns:
        str: Either "local" or "supabase"
    """
    storage_mode = os.getenv("STORAGE_MODE", "local").lower()
    if storage_mode not in ["local", "supabase"]:
        logging.warning(f"Invalid STORAGE_MODE '{storage_mode}', falling back to 'local'")
        return "local"
    return storage_mode