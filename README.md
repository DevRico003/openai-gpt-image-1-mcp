<h1 align="center">OpenAI GPT Image Generation MCP Server</h1>

<p align="center">
  <em>AI Image Generation and Editing Capabilities for AI Agents and AI Coding Assistants</em>
</p>

A powerful implementation of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io) integrated with [OpenAI's GPT-image-1 model](https://platform.openai.com/docs/guides/images) for providing AI agents and AI coding assistants with advanced image generation and editing capabilities.

With this MCP server, you can <b>generate images from text descriptions</b> and <b>edit existing images</b> with powerful AI.

## Overview

This MCP server provides tools that enable AI agents to generate and edit images using OpenAI's GPT-image-1 model. It follows the best practices for building MCP servers based on the [Mem0 MCP server template](https://github.com/coleam00/mcp-mem0/). The server handles all aspects of the image generation process including API calls, error handling, and saving the generated images to a local directory.

## Features

- **Text-to-Image Generation**: Create images from detailed text descriptions
- **Image Editing**: Modify existing images using text instructions
- **Multiple Reference Images**: Use multiple input images as references for generation
- **Inpainting Support**: Edit specific areas of images using masks
- **Image Quality Control**: Configure image size and quality settings
- **Automatic Image Saving**: All generated images are stored locally for easy access

## Tools

The server provides two essential image generation tools:

1. **`generate_image`**: Create new images from text descriptions
2. **`edit_image`**: Edit existing images based on text prompts, with support for multiple reference images and masks

## Prerequisites

- [Docker/Docker Desktop](https://www.docker.com/products/docker-desktop/) if running the MCP server as a container (recommended)
- [Python 3.10+](https://www.python.org/downloads/) if running the MCP server directly
- [OpenAI API key](https://platform.openai.com/api-keys) with access to GPT-image-1 model

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

# Optional settings for image generation
# MODEL_NAME=gpt-image-1
# DEFAULT_IMAGE_SIZE=auto
# DEFAULT_IMAGE_QUALITY=auto
```

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

### Generating an Image

```python
# Generate an image of a flying cat
result = await generate_image(
    prompt="A photorealistic flying cat with wings soaring through clouds",
    size="1024x1024",
    quality="high",
    return_image=True  # Get the image data directly (default is True)
)

# By default, the result contains the image data
image_data = result['image_data']  # Base64 encoded image data
mime_type = result['mime_type']    # "image/png"

# The result also contains paths to the saved image on the server
saved_path = result['saved_path']     # Absolute path where the image was saved
relative_path = result['relative_path']  # Relative path from working directory
filename = result['filename']         # The filename of the generated image
directory = result['directory']       # The directory where images are stored

# Save the image locally from the base64 data
import base64
with open(f"local-{filename}", "wb") as f:
    f.write(base64.b64decode(image_data))

print(f"Image saved locally as: local-{filename}")

# Or display it in an HTML context
html_img = f'<img src="data:{mime_type};base64,{image_data}" alt="Flying cat">'
```

### Editing an Image

```python
# Edit an existing image to add a hat
result = await edit_image(
    prompt="Add a wizard hat to the cat",
    image_paths=["/path/to/cat_image.png"],
    size="1024x1024",
    quality="high",
    return_image=True  # Get the image data directly
)

# Use the base64 data to save or display the image
image_data = result['image_data']
filename = result['filename']

# Save locally
import base64
with open(f"edited-{filename}", "wb") as f:
    f.write(base64.b64decode(image_data))
```

### Client-Side Image Handling

The MCP server provides two ways to access generated images:

1. **Direct Image Data**: By default, the image is returned as base64-encoded data in the `image_data` field. This is ideal when:
   - You don't have direct filesystem access to the server (e.g., when running in Docker)
   - You want to display the image directly in a web application
   - You need to save the image locally on the client side

2. **File Paths**: The tools also return file paths where the image is saved on the server. This is useful when:
   - The client has direct filesystem access to the server
   - You're using volume mounts in Docker
   - You need to reference the image location for other server-side processes

### Performance Considerations

If you're concerned about token limits or don't need the image data directly:

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

This way, you can access generated images in the `ai-images` directory on your host machine, even if you set `return_image=False`.

## License

MIT