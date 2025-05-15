# OpenAI GPT Image Generation MCP

This is a Model Control Protocol (MCP) server that provides access to OpenAI's GPT-image-1 model for generating and editing images.

## Features

- Generate images from text descriptions using OpenAI's gpt-image-1 model
- Edit existing images with text instructions
- Support for inpainting with masks
- Automatic image saving to local directory

## Requirements

- Python 3.10 or higher
- OpenAI API key

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `HOST`: Host IP address (default: 0.0.0.0)
- `PORT`: Port number (default: 8050)
- `TRANSPORT`: Transport protocol - 'sse' or 'stdio' (default: 'sse')

## Installation

### Local Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -e .
   ```
3. Run the server:
   ```
   python src/main.py
   ```

### Docker Installation

1. Build the Docker image:
   ```
   docker build -t openai-gpt-image-1-mcp .
   ```
2. Run the container:
   ```
   docker run -p 8050:8050 -e OPENAI_API_KEY=your_api_key_here openai-gpt-image-1-mcp
   ```

## Usage

The MCP server exposes two tools:

### 1. generate_image

Generates an image based on a text description.

Parameters:
- `prompt`: Text description of the desired image
- `model`: Model to use (default: "gpt-image-1")
- `n`: Number of images to generate (default: 1)
- `size`: Image dimensions (default: "auto", options: "1024x1024", "1536x1024", "1024x1536")
- `quality`: Rendering quality (default: "auto", options: "low", "medium", "high")
- `user`: Optional end-user identifier
- `save_filename`: Optional custom filename (without extension)

### 2. edit_image

Edits an existing image based on a text description.

Parameters:
- `prompt`: Text description of the desired edits
- `image_paths`: List of paths to input image files
- `mask_path`: Optional path to a mask image for inpainting
- `model`: Model to use (default: "gpt-image-1")
- `n`: Number of images to generate (default: 1)
- `size`: Image dimensions (default: "auto", options: "1024x1024", "1536x1024", "1024x1536")
- `quality`: Rendering quality (default: "auto", options: "low", "medium", "high")
- `user`: Optional end-user identifier
- `save_filename`: Optional custom filename (without extension)

## Output

Generated images are automatically saved to the `ai-images` directory with filename based on the prompt and timestamp.

## License

MIT