"""Document ingestion from Azure Blob Storage"""

from typing import Optional
from rag_eval.core.interfaces import Chunk
from rag_eval.core.exceptions import AzureServiceError
from rag_eval.core.logging import get_logger

logger = get_logger("services.rag.ingestion")


def ingest_document(blob_name: str, config) -> str:
    """
    Ingest a document from Azure Blob Storage.
    
    Args:
        blob_name: Name of the blob to ingest
        config: Application configuration
        
    Returns:
        Document text content
        
    Raises:
        AzureServiceError: If ingestion fails
    """
    logger.info(f"Ingesting document: {blob_name}")
    # TODO: Implement Azure Blob Storage ingestion
    raise NotImplementedError("Document ingestion not yet implemented")

