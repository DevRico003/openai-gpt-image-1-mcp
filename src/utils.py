from openai import AsyncOpenAI
import logging
import os
import sys

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