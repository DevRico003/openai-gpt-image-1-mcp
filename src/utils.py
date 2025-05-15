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
        raise Exception("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    
    # Create and return async client
    try:
        client = AsyncOpenAI(api_key=api_key)
        logging.info("AsyncOpenAI client initialized successfully.")
        return client
    except Exception as e:
        logging.error(f"Failed to initialize AsyncOpenAI client: {e}")
        raise Exception(f"Failed to initialize AsyncOpenAI client: {e}")

def get_supabase_client() -> Client:
    """
    Creates and returns a Supabase client with the appropriate credentials.
    Supports both hosted Supabase and self-hosted installations.
    
    Returns:
        Client: The initialized Supabase client
    
    Raises:
        Exception: If the required credentials are not found or client creation fails
    """
    # Get Supabase credentials from environment
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        logging.error("FATAL: SUPABASE_URL and SUPABASE_KEY environment variables must be set.")
        raise Exception("Supabase credentials not found. Please set the SUPABASE_URL and SUPABASE_KEY environment variables.")
    
    # Log detailed info to help with troubleshooting
    logging.info(f"Initializing Supabase client with URL: {supabase_url}")
    
    # Create and return client
    try:
        # Ensure URL doesn't have trailing slash
        supabase_url = supabase_url.rstrip("/")
        
        # Create client with only the basic parameters
        # Explicitly pass just these two parameters to avoid the dict error
        client = create_client(
            supabase_url=supabase_url,
            supabase_key=supabase_key
        )
        
        logging.info("Supabase client initialized successfully.")
        return client
    except Exception as e:
        logging.error(f"Failed to initialize Supabase client: {e}")
        # Werfen einer Exception anstatt sys.exit zu verwenden
        # Dies erlaubt dem Hauptprogramm, auf den Fehler zu reagieren
        raise Exception(f"Failed to initialize Supabase client: {e}")

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
        bucket_name = os.getenv("SUPABASE_BUCKET", "images")
        
        # Ensure the bucket exists
        if not ensure_bucket_exists(supabase, bucket_name):
            logging.warning(f"Could not ensure bucket '{bucket_name}' exists. Proceeding anyway.")
        
        # Sanitize filename - remove special characters and replace spaces with underscores
        # Convert to ASCII to remove non-ASCII characters like umlauts
        import re
        import unicodedata
        import mimetypes
        
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
        
        # Automatische Content-Type-Erkennung
        content_type = mimetypes.guess_type(filename)[0] or "image/png"
        
        # Detaillierte Debug-Informationen
        logging.info(f"Uploading to Supabase bucket: {bucket_name}, path: {storage_path}")
        logging.info(f"Supabase URL: {os.getenv('SUPABASE_URL')}")
        logging.info(f"Detected content type: {content_type}")
        
        # Upload the image mit expliziten Parametern
        try:
            # Die storage.from_ Methode neu aufrufen und mit named arguments arbeiten
            bucket = supabase.storage.from_(bucket_name)
            
            # Explizite Parameter für mehr Stabilität
            res = bucket.upload(
                path=storage_path,
                file=file_stream,
                file_options={"content-type": content_type}
            )
            logging.info(f"Upload response: {res}")
        except Exception as upload_err:
            logging.error(f"Upload error details: {str(upload_err)}")
            # Detaillierte Fehlerinformationen
            if hasattr(upload_err, 'message'):
                logging.error(f"Error message: {upload_err.message}")
            if hasattr(upload_err, 'code'):
                logging.error(f"Error code: {upload_err.code}")
            raise
        
        # Erfolgsvalidierung
        if not res:
            logging.error(f"Supabase upload returned empty response")
            raise Exception("Failed to upload image to Supabase: Empty response")
        
        # Get the public URL for the uploaded image
        try:
            # Verwende die neueste API-Methode für Public URLs
            bucket = supabase.storage.from_(bucket_name)
            public_url = bucket.get_public_url(storage_path)
            
            logging.info(f"Generated public URL via API: {public_url}")
            
            # Wenn die URL leer oder ungültig ist, erstelle sie manuell
            if not public_url or "null" in public_url:
                base_url = os.getenv("SUPABASE_URL").rstrip("/")
                
                # Standard-Pfad für Supabase Storage
                public_url = f"{base_url}/storage/v1/object/public/{bucket_name}/{storage_path}"
                logging.info(f"Generated custom public URL: {public_url}")
        except Exception as url_err:
            logging.error(f"Error getting public URL: {url_err}")
            # Fallback: URL manuell erstellen
            base_url = os.getenv("SUPABASE_URL").rstrip("/")
            public_url = f"{base_url}/storage/v1/object/public/{bucket_name}/{storage_path}"
            logging.info(f"Fallback to manual public URL: {public_url}")
        
        # Final validation
        if not public_url:
            raise Exception("Failed to generate public URL for Supabase object")
        
        logging.info(f"Image uploaded successfully to Supabase: {public_url}")
        return public_url, storage_path
        
    except Exception as e:
        logging.error(f"Error uploading image to Supabase: {e}")
        raise Exception(f"Failed to upload image to Supabase: {str(e)}")

def ensure_bucket_exists(supabase: Client, bucket_name: str) -> bool:
    """
    Ensures that the specified bucket exists, and attempts to create it if it doesn't.
    
    Args:
        supabase: The Supabase client
        bucket_name: Name of the bucket to check/create
        
    Returns:
        bool: True if bucket exists or was created, False otherwise
    """
    try:
        # Get list of all buckets
        buckets = supabase.storage.list_buckets()
        
        # Check if bucket already exists
        bucket_exists = any(bucket.name == bucket_name for bucket in buckets)
        
        if bucket_exists:
            logging.info(f"Bucket '{bucket_name}' already exists")
            return True
            
        # Bucket doesn't exist, try to create it
        logging.info(f"Bucket '{bucket_name}' does not exist, attempting to create it")
        supabase.storage.create_bucket(bucket_name, options={"public": True})
        logging.info(f"Successfully created bucket '{bucket_name}'")
        return True
            
    except Exception as e:
        logging.error(f"Error checking/creating bucket '{bucket_name}': {e}")
        return False

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