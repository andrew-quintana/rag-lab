"""Supabase Storage operations for document persistence"""

import time
import io
from typing import Optional
from datetime import datetime
from supabase import create_client, Client
from PIL import Image
from rag_eval.core.exceptions import DatabaseError
from rag_eval.core.logging import get_logger

logger = get_logger("services.rag.supabase_storage")

# Bucket name for documents
DOCUMENTS_BUCKET = "documents"


def _retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry (callable that takes no arguments)
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds for exponential backoff (default: 1.0)
        
    Returns:
        Result of the function call
        
    Raises:
        DatabaseError: If all retries are exhausted
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(
                    f"Supabase operation failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                time.sleep(delay)
            else:
                logger.error(f"Supabase operation failed after {max_retries + 1} attempts: {e}")
                raise DatabaseError(
                    f"Supabase Storage operation failed after {max_retries + 1} attempts: {str(e)}"
                ) from e
    
    # This should never be reached, but included for type safety
    raise DatabaseError(f"Unexpected error in retry logic: {last_exception}") from last_exception


def _get_supabase_client(config) -> Client:
    """
    Create and return a Supabase client.
    
    Args:
        config: Application configuration with Supabase credentials
        
    Returns:
        Supabase client instance
        
    Raises:
        DatabaseError: If client creation fails
    """
    if not config.supabase_url:
        raise DatabaseError("Supabase URL is not configured")
    
    if not config.supabase_key:
        raise DatabaseError("Supabase key is not configured")
    
    try:
        # Create client without options to avoid ClientOptions compatibility issues
        # The default options should be sufficient for storage operations
        client = create_client(
            config.supabase_url,
            config.supabase_key
        )
        return client
    except Exception as e:
        logger.error(f"Failed to create Supabase client: {e}")
        raise DatabaseError(f"Failed to create Supabase client: {str(e)}") from e


def upload_document_to_storage(
    file_content: bytes,
    document_id: str,
    filename: str,
    config,
    content_type: Optional[str] = None
) -> str:
    """
    Upload a document to Supabase Storage.
    
    This function:
    - Connects to Supabase Storage using credentials from config
    - Uploads the file content with document_id as the file path
    - Implements retry logic with exponential backoff (3 retries max)
    
    Args:
        file_content: Binary content of the file to upload
        document_id: Unique document identifier (used as file path)
        filename: Original filename (for metadata)
        config: Application configuration with Supabase credentials
        content_type: MIME type of the file (optional)
        
    Returns:
        Storage path of the uploaded file
        
    Raises:
        DatabaseError: If upload fails after retries
        ValueError: If file_content is empty or document_id is invalid
    """
    if not file_content:
        raise ValueError("file_content cannot be empty")
    
    if not document_id or not document_id.strip():
        raise ValueError("document_id cannot be empty")
    
    logger.info(
        f"Uploading document '{filename}' (ID: {document_id}, size: {len(file_content)} bytes) "
        f"to bucket '{DOCUMENTS_BUCKET}'"
    )
    
    try:
        client = _get_supabase_client(config)
        storage_path = document_id
        
        def upload_file():
            response = client.storage.from_(DOCUMENTS_BUCKET).upload(
                path=storage_path,
                file=file_content,
                file_options={"content-type": content_type} if content_type else None
            )
            logger.info(f"Successfully uploaded document '{document_id}' to storage")
            return storage_path
        
        _retry_with_backoff(upload_file)
        return storage_path
        
    except DatabaseError:
        raise
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error uploading document to storage: {e}", exc_info=True)
        raise DatabaseError(
            f"Unexpected error uploading document to storage: {str(e)}"
        ) from e


def download_document_from_storage(document_id: str, config) -> bytes:
    """
    Download a document from Supabase Storage.
    
    This function:
    - Connects to Supabase Storage using credentials from config
    - Downloads the file content by document_id
    - Handles missing files gracefully
    - Implements retry logic with exponential backoff (3 retries max)
    
    Args:
        document_id: Document identifier (storage path) to download
        config: Application configuration with Supabase credentials
        
    Returns:
        Binary content of the downloaded file
        
    Raises:
        DatabaseError: If download fails after retries
        ValueError: If document_id is invalid
    """
    if not document_id or not document_id.strip():
        raise ValueError("document_id cannot be empty")
    
    logger.info(f"Downloading document '{document_id}' from bucket '{DOCUMENTS_BUCKET}'")
    
    try:
        client = _get_supabase_client(config)
        
        def download_file():
            response = client.storage.from_(DOCUMENTS_BUCKET).download(document_id)
            logger.info(f"Successfully downloaded document '{document_id}' ({len(response)} bytes)")
            return response
        
        content = _retry_with_backoff(download_file)
        return content
        
    except DatabaseError:
        raise
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error downloading document from storage: {e}", exc_info=True)
        raise DatabaseError(
            f"Unexpected error downloading document from storage: {str(e)}"
        ) from e


def delete_document_from_storage(document_id: str, config) -> None:
    """
    Delete a document from Supabase Storage.
    
    This function:
    - Connects to Supabase Storage using credentials from config
    - Deletes the file by document_id
    - Implements retry logic with exponential backoff (3 retries max)
    
    Args:
        document_id: Document identifier (storage path) to delete
        config: Application configuration with Supabase credentials
        
    Raises:
        DatabaseError: If delete fails after retries
        ValueError: If document_id is invalid
    """
    if not document_id or not document_id.strip():
        raise ValueError("document_id cannot be empty")
    
    logger.info(f"Deleting document '{document_id}' from bucket '{DOCUMENTS_BUCKET}'")
    
    try:
        client = _get_supabase_client(config)
        
        def delete_file():
            client.storage.from_(DOCUMENTS_BUCKET).remove([document_id])
            logger.info(f"Successfully deleted document '{document_id}' from storage")
        
        _retry_with_backoff(delete_file)
        
    except DatabaseError:
        raise
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting document from storage: {e}", exc_info=True)
        raise DatabaseError(
            f"Unexpected error deleting document from storage: {str(e)}"
        ) from e


def get_public_url(document_id: str, config, expires_in: int = 3600) -> str:
    """
    Get a public URL for a document in Supabase Storage.
    
    Args:
        document_id: Document identifier (storage path)
        config: Application configuration with Supabase credentials
        expires_in: URL expiration time in seconds (default: 3600)
        
    Returns:
        Public URL for the document
        
    Raises:
        DatabaseError: If URL generation fails
        ValueError: If document_id is invalid
    """
    if not document_id or not document_id.strip():
        raise ValueError("document_id cannot be empty")
    
    try:
        client = _get_supabase_client(config)
        response = client.storage.from_(DOCUMENTS_BUCKET).create_signed_url(
            document_id,
            expires_in
        )
        return response
    except Exception as e:
        logger.error(f"Failed to generate public URL for document '{document_id}': {e}")
        raise DatabaseError(f"Failed to generate public URL: {str(e)}") from e


def generate_image_preview(
    file_content: bytes,
    document_id: str,
    mime_type: str,
    config,
    max_size: tuple[int, int] = (800, 800),
    quality: int = 85
) -> Optional[str]:
    """
    Generate a preview image for a document and upload it to storage.
    
    Supports:
    - Image files (JPEG, PNG, GIF, etc.) - resizes if needed
    - PDF files - extracts first page as image (requires pdf2image, optional)
    
    Args:
        file_content: Binary content of the original file
        document_id: Document identifier (used for preview path)
        mime_type: MIME type of the original file
        config: Application configuration with Supabase credentials
        max_size: Maximum size for preview image (width, height)
        quality: JPEG quality (1-100, default: 85)
        
    Returns:
        Storage path of the preview image, or None if preview cannot be generated
        
    Raises:
        DatabaseError: If preview generation or upload fails
    """
    preview_path = None
    
    try:
        # Handle image files
        if mime_type and mime_type.startswith('image/'):
            try:
                # Open image
                image = Image.open(io.BytesIO(file_content))
                
                # Convert to RGB if necessary (for JPEG)
                if image.mode in ('RGBA', 'LA', 'P'):
                    # Create white background
                    rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                    if image.mode == 'P':
                        image = image.convert('RGBA')
                    rgb_image.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                    image = rgb_image
                elif image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Resize if needed
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Save to bytes
                preview_bytes = io.BytesIO()
                image.save(preview_bytes, format='JPEG', quality=quality, optimize=True)
                preview_bytes.seek(0)
                preview_content = preview_bytes.read()
                
                # Upload preview
                preview_path = f"{document_id}_preview.jpg"
                upload_document_to_storage(
                    preview_content,
                    preview_path,
                    f"{document_id}_preview.jpg",
                    config,
                    content_type="image/jpeg"
                )
                
                logger.info(f"Generated image preview for '{document_id}': {preview_path}")
                return preview_path
                
            except Exception as e:
                logger.warning(f"Failed to generate preview for image '{document_id}': {e}")
                return None
        
        # Handle PDF files (optional - requires pdf2image and poppler)
        elif mime_type == 'application/pdf':
            try:
                from pdf2image import convert_from_bytes
                import os
                
                # Try to find poppler in common locations
                poppler_path = None
                for path in ['/opt/homebrew/bin', '/usr/local/bin']:
                    if os.path.exists(f'{path}/pdftoppm'):
                        poppler_path = path
                        break
                
                # Convert first page to image
                convert_kwargs = {
                    'first_page': 1,
                    'last_page': 1,
                    'dpi': 150
                }
                if poppler_path:
                    convert_kwargs['poppler_path'] = poppler_path
                
                images = convert_from_bytes(file_content, **convert_kwargs)
                if images:
                    image = images[0]
                    
                    # Resize if needed
                    image.thumbnail(max_size, Image.Resampling.LANCZOS)
                    
                    # Convert to JPEG
                    preview_bytes = io.BytesIO()
                    image.save(preview_bytes, format='JPEG', quality=quality, optimize=True)
                    preview_bytes.seek(0)
                    preview_content = preview_bytes.read()
                    
                    # Upload preview
                    preview_path = f"{document_id}_preview.jpg"
                    upload_document_to_storage(
                        preview_content,
                        preview_path,
                        f"{document_id}_preview.jpg",
                        config,
                        content_type="image/jpeg"
                    )
                    
                    logger.info(f"Generated PDF preview for '{document_id}': {preview_path}")
                    return preview_path
            except ImportError:
                logger.debug("pdf2image not installed, skipping PDF preview generation")
                return None
            except Exception as e:
                error_msg = str(e)
                if "poppler" in error_msg.lower() or "pdftoppm" in error_msg.lower():
                    logger.warning(
                        f"Failed to generate preview for PDF '{document_id}': "
                        f"poppler-utils not found. Install with: brew install poppler"
                    )
                else:
                    logger.warning(f"Failed to generate preview for PDF '{document_id}': {e}")
                return None
        
        # Unsupported file type
        logger.debug(f"Preview generation not supported for MIME type: {mime_type}")
        return None
        
    except Exception as e:
        logger.error(f"Unexpected error generating preview for '{document_id}': {e}", exc_info=True)
        return None

