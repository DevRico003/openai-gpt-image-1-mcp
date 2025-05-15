<h1 align="center">OpenAI GPT Image Generation MCP Server</h1>

<p align="center">
  <em>AI Image Generation and Editing Capabilities for AI Agents and AI Coding Assistants</em>
</p>

A powerful implementation of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io) integrated with [OpenAI's GPT-image-1 model](https://platform.openai.com/docs/guides/images) for providing AI agents and AI coding assistants with advanced image generation and editing capabilities.

With this MCP server, you can <b>generate images from text descriptions</b> and <b>edit existing images</b> with powerful AI. Images can be stored either locally or in [Supabase Storage](https://supabase.com/storage) for easy access.

## Overview

This MCP server provides tools that enable AI agents to generate and edit images using OpenAI's GPT-image-1 model. It follows the best practices for building MCP servers based on the [Mem0 MCP server template](https://github.com/coleam00/mcp-mem0/). The server handles all aspects of the image generation process including API calls, error handling, and storing the generated images either locally or in Supabase Storage for easy access from anywhere.

## Features

- **Text-to-Image Generation**: Create images from detailed text descriptions
- **Image Editing**: Modify existing images using text instructions
- **Multiple Reference Images**: Use multiple input images as references for generation
- **Inpainting Support**: Edit specific areas of images using masks
- **Image Quality Control**: Configure image size and quality settings
- **Flexible Storage Options**: Store images locally or in Supabase Storage
- **Public Image URLs**: Get direct URLs to access generated images when using Supabase Storage

## Tools

The server provides two essential image generation tools:

1. **`generate_image`**: Create new images from text descriptions
2. **`edit_image`**: Edit existing images based on text prompts, with support for multiple reference images and masks

## Prerequisites

- [Docker/Docker Desktop](https://www.docker.com/products/docker-desktop/) if running the MCP server as a container (recommended)
- [Python 3.10+](https://www.python.org/downloads/) if running the MCP server directly
- [OpenAI API key](https://platform.openai.com/api-keys) with access to GPT-image-1 model
- [Supabase account](https://supabase.com/) (optional, for cloud storage of images)

## Installation

### Using Docker (Recommended)

1. Clone this repository:
   ```bash
   git clone https://github.com/DevRico003/openai-gpt-image-1-mcp.git
   cd openai-gpt-image-1-mcp
   ```

2. Build the Docker image:
   ```bash
   docker build -t openai-gpt-image-1-mcp --build-arg PORT=8050 .
   ```

3. Create a `.env` file based on the `.env.example` file

### Using Python directly (no Docker)

1. Clone this repository:
   ```bash
   git clone https://github.com/DevRico003/openai-gpt-image-1-mcp.git
   cd openai-gpt-image-1-mcp
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

3. Create a `.env` file based on the `.env.example` file

## Configuration

Create a `.env` file in the project root with the following variables:

```
# MCP Server Configuration
HOST=0.0.0.0
PORT=8050
TRANSPORT=sse

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key

# Storage Configuration
# Set to 'local' to store images on the server, or 'supabase' to use Supabase Storage
STORAGE_MODE=supabase

# Supabase Configuration (required if STORAGE_MODE=supabase)
SUPABASE_URL=https://your-project-url.supabase.co
SUPABASE_KEY=your-supabase-service-key
SUPABASE_BUCKET=your-bucket-name

# Optional settings for image generation
# MODEL_NAME=gpt-image-1
# DEFAULT_IMAGE_SIZE=auto
# DEFAULT_IMAGE_QUALITY=auto
```

### Supabase Setup (Optional)

If you want to use Supabase for image storage:

1. Create a Supabase account at [supabase.com](https://supabase.com/)
2. Create a new project and get your project URL and service key
3. Create a storage bucket named `images`
4. Make sure public access is enabled for the bucket if you want the images to be publicly accessible

## Running the Server

### Using Docker

```bash
docker run --env-file .env -p 8050:8050 openai-gpt-image-1-mcp
```

### Using Python

```bash
python src/main.py
```

The server will start and listen on the configured host and port.

## Integration with MCP Clients

### SSE Configuration

Once you have the server running with SSE transport, you can connect to it using this configuration:

```json
{
  "mcpServers": {
    "openai-gpt-image-1": {
      "transport": "sse",
      "url": "http://localhost:8050/sse"
    }
  }
}
```

> **Note for Windsurf users**: Use `serverUrl` instead of `url` in your configuration:
> ```json
> {
>   "mcpServers": {
>     "openai-gpt-image-1": {
>       "transport": "sse",
>       "serverUrl": "http://localhost:8050/sse"
>     }
>   }
> }
> ```
>
> **Note for Docker users**: Use `host.docker.internal` instead of `localhost` if your client is running in a different container.

### Stdio Configuration

Add this server to your MCP configuration for Claude Desktop, Windsurf, or any other MCP client:

```json
{
  "mcpServers": {
    "openai-gpt-image-1": {
      "command": "python",
      "args": ["path/to/openai-gpt-image-1-mcp/src/main.py"],
      "env": {
        "TRANSPORT": "stdio",
        "OPENAI_API_KEY": "your_openai_api_key"
      }
    }
  }
}
```

### Docker with Stdio Configuration

```json
{
  "mcpServers": {
    "openai-gpt-image-1": {
      "command": "docker",
      "args": ["run", "--rm", "-i", 
               "-e", "TRANSPORT", 
               "-e", "OPENAI_API_KEY", 
               "openai-gpt-image-1-mcp"],
      "env": {
        "TRANSPORT": "stdio",
        "OPENAI_API_KEY": "your_openai_api_key"
      }
    }
  }
}
```

## Tool Usage Examples

### Generating an Image with Supabase Storage

```python
# Generate an image of a flying cat with Supabase storage
# Make sure STORAGE_MODE=supabase in your .env file
result = await generate_image(
    prompt="A photorealistic flying cat with wings soaring through clouds",
    size="1024x1024",
    quality="high"
)

# Get the public URL for the image
image_url = result['image_url']        # URL to view the image
download_url = result['download_url']  # URL to download the image
filename = result['filename']          # Filename of the generated image
storage_path = result['storage_path']  # Path in Supabase storage

print(f"Image generated successfully!")
print(f"View the image at: {image_url}")

# Example of displaying the image in a web application
html_img = f'<img src="{image_url}" alt="Flying cat">'

# Example of creating a download link
html_download = f'<a href="{download_url}" download="{filename}">Download Image</a>'
```

### Generating an Image with Local Storage

```python
# Generate an image with local storage
# Make sure STORAGE_MODE=local in your .env file
result = await generate_image(
    prompt="A photorealistic flying cat with wings soaring through clouds",
    size="1024x1024",
    quality="high",
    return_image=True  # Get the image data directly (default is True)
)

# When using local storage, you get file paths and optional base64 data
saved_path = result['saved_path']        # Absolute path on server
relative_path = result['relative_path']  # Relative path from working directory
filename = result['filename']            # Filename of the generated image
directory = result['directory']          # Directory where images are stored

# If return_image=True, you also get image data
image_data = result['image_data']  # Base64 encoded image data
mime_type = result['mime_type']    # "image/png"

# Save image from base64 data
import base64
with open(f"local-{filename}", "wb") as f:
    f.write(base64.b64decode(image_data))

print(f"Image saved locally as: local-{filename}")
```

### Editing an Image

```python
# Edit an existing image to add a hat
result = await edit_image(
    prompt="Add a wizard hat to the cat",
    image_paths=["/path/to/cat_image.png"],
    size="1024x1024",
    quality="high"
)

# Check storage mode used
storage_mode = result['storage_mode']  # "supabase" or "local"

if storage_mode == "supabase":
    # Access the image via public URL
    image_url = result['image_url']
    print(f"View edited image at: {image_url}")
else:
    # When using local storage
    saved_path = result['saved_path']
    
    # If return_image=True was used, you can also access the image data
    if 'image_data' in result:
        image_data = result['image_data']
        # Save locally from base64 data
        import base64
        with open(f"edited-{result['filename']}", "wb") as f:
            f.write(base64.b64decode(image_data))
```

### Storage Options

The MCP server supports two storage modes:

1. **Supabase Storage**: Images are uploaded to Supabase and available via public URLs. This is ideal for:
   - Remote access without filesystem access to the server
   - Sharing images with other systems or users
   - Long-term storage independent of the MCP server

2. **Local Storage**: Images are saved to a local directory on the server. This can be used when:
   - You prefer to manage storage yourself
   - You don't need remote access to the images
   - You're testing or developing locally

### Performance Considerations

If you're using local storage and are concerned about token limits:

```python
# Generate image without returning the base64 data
result = await generate_image(
    prompt="A photorealistic flying cat with wings soaring through clouds",
    return_image=False  # Don't include image data in the response
)

# Now you only have path information
saved_path = result['saved_path']
filename = result['filename']
```

For Docker deployments with shared volumes:

```bash
docker run -v $(pwd)/ai-images:/app/ai-images --env-file .env -p 8050:8050 openai-gpt-image-1-mcp
```

This way, you can access locally stored images in the `ai-images` directory on your host machine.

## License

MIT