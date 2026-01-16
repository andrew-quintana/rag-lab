"""Azure Blob Storage operations for document persistence"""

import time
from typing import Optional
from datetime import datetime
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import AzureError, ResourceExistsError, ResourceNotFoundError
from src.core.exceptions import AzureServiceError
from src.core.logging import get_logger

logger = get_logger("services.rag.storage")


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
        AzureServiceError: If all retries are exhausted
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except (AzureError, Exception) as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(
                    f"Azure operation failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                time.sleep(delay)
            else:
                logger.error(f"Azure operation failed after {max_retries + 1} attempts: {e}")
                raise AzureServiceError(
                    f"Azure Blob Storage operation failed after {max_retries + 1} attempts: {str(e)}"
                ) from e
    
    # This should never be reached, but included for type safety
    raise AzureServiceError(f"Unexpected error in retry logic: {last_exception}") from last_exception


def _ensure_container_exists(blob_service_client: BlobServiceClient, container_name: str) -> None:
    """
    Ensure that a blob container exists, creating it if necessary (idempotent).
    
    Args:
        blob_service_client: Azure Blob Service Client
        container_name: Name of the container to ensure exists
        
    Raises:
        AzureServiceError: If container creation fails after retries
    """
    try:
        container_client = blob_service_client.get_container_client(container_name)
        
        # Try to get container properties to check if it exists
        try:
            container_client.get_container_properties()
            logger.debug(f"Container '{container_name}' already exists")
        except ResourceNotFoundError:
            # Container doesn't exist, create it
            logger.info(f"Container '{container_name}' does not exist, creating it...")
            
            def create_container():
                container_client.create_container()
                logger.info(f"Successfully created container '{container_name}'")
            
            _retry_with_backoff(create_container)
    except ResourceExistsError:
        # Container was created by another process, which is fine (idempotent)
        logger.debug(f"Container '{container_name}' already exists (created concurrently)")
    except Exception as e:
        raise AzureServiceError(
            f"Failed to ensure container '{container_name}' exists: {str(e)}"
        ) from e


def upload_document_to_blob(
    file_content: bytes,
    document_id: str,
    filename: str,
    config
) -> str:
    """
    Upload a document to Azure Blob Storage.
    
    This function:
    - Connects to Azure Blob Storage using connection string from config
    - Creates the container if it doesn't exist (idempotent)
    - Uploads the file content with document_id as the blob name
    - Stores metadata (filename, upload timestamp, document_id)
    - Implements retry logic with exponential backoff (3 retries max)
    
    Args:
        file_content: Binary content of the file to upload
        document_id: Unique document identifier (used as blob name)
        filename: Original filename (stored as metadata)
        config: Application configuration with Azure Blob Storage credentials
        
    Returns:
        Blob name (document_id) of the uploaded file
        
    Raises:
        AzureServiceError: If upload fails after retries
        ValueError: If file_content is empty or document_id is invalid
    """
    if not file_content:
        raise ValueError("file_content cannot be empty")
    
    if not document_id or not document_id.strip():
        raise ValueError("document_id cannot be empty")
    
    if not filename:
        filename = document_id
    
    connection_string = config.azure_blob_connection_string
    container_name = config.azure_blob_container_name
    
    if not connection_string:
        raise AzureServiceError("Azure Blob Storage connection string is not configured")
    
    if not container_name:
        raise AzureServiceError("Azure Blob Storage container name is not configured")
    
    logger.info(
        f"Uploading document '{filename}' (ID: {document_id}, size: {len(file_content)} bytes) "
        f"to container '{container_name}'"
    )
    
    try:
        # Create blob service client
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Ensure container exists (idempotent)
        _ensure_container_exists(blob_service_client, container_name)
        
        # Get blob client
        blob_client = blob_service_client.get_blob_client(
            container=container_name,
            blob=document_id
        )
        
        # Prepare metadata
        metadata = {
            "filename": filename,
            "document_id": document_id,
            "upload_timestamp": datetime.utcnow().isoformat(),
            "content_length": str(len(file_content))
        }
        
        # Upload blob with retry logic
        def upload_blob():
            blob_client.upload_blob(
                data=file_content,
                overwrite=True,  # Allow overwriting if blob already exists
                metadata=metadata
            )
            logger.info(f"Successfully uploaded document '{document_id}' to blob storage")
        
        _retry_with_backoff(upload_blob)
        
        return document_id
        
    except AzureServiceError:
        # Re-raise AzureServiceError as-is
        raise
    except ValueError:
        # Re-raise ValueError as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error uploading document to blob storage: {e}", exc_info=True)
        raise AzureServiceError(
            f"Unexpected error uploading document to blob storage: {str(e)}"
        ) from e


def download_document_from_blob(document_id: str, config) -> bytes:
    """
    Download a document from Azure Blob Storage.
    
    This function:
    - Connects to Azure Blob Storage using connection string from config
    - Downloads the blob content by document_id
    - Handles missing blobs gracefully
    - Implements retry logic with exponential backoff (3 retries max)
    
    Args:
        document_id: Document identifier (blob name) to download
        config: Application configuration with Azure Blob Storage credentials
        
    Returns:
        Binary content of the downloaded file
        
    Raises:
        AzureServiceError: If download fails after retries
        ResourceNotFoundError: If blob does not exist (wrapped in AzureServiceError)
        ValueError: If document_id is invalid
    """
    if not document_id or not document_id.strip():
        raise ValueError("document_id cannot be empty")
    
    connection_string = config.azure_blob_connection_string
    container_name = config.azure_blob_container_name
    
    if not connection_string:
        raise AzureServiceError("Azure Blob Storage connection string is not configured")
    
    if not container_name:
        raise AzureServiceError("Azure Blob Storage container name is not configured")
    
    logger.info(f"Downloading document '{document_id}' from container '{container_name}'")
    
    try:
        # Create blob service client
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Get blob client
        blob_client = blob_service_client.get_blob_client(
            container=container_name,
            blob=document_id
        )
        
        # Check if blob exists
        try:
            blob_properties = blob_client.get_blob_properties()
            logger.debug(f"Blob '{document_id}' exists (size: {blob_properties.size} bytes)")
        except ResourceNotFoundError:
            raise AzureServiceError(f"Blob '{document_id}' not found in container '{container_name}'")
        
        # Download blob with retry logic
        def download_blob():
            download_stream = blob_client.download_blob()
            content = download_stream.readall()
            logger.info(f"Successfully downloaded document '{document_id}' ({len(content)} bytes)")
            return content
        
        content = _retry_with_backoff(download_blob)
        return content
        
    except AzureServiceError:
        # Re-raise AzureServiceError as-is
        raise
    except ValueError:
        # Re-raise ValueError as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error downloading document from blob storage: {e}", exc_info=True)
        raise AzureServiceError(
            f"Unexpected error downloading document from blob storage: {str(e)}"
        ) from e

