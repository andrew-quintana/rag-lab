"""Document ingestion using Azure Document Intelligence"""

from typing import Optional
import io
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from rag_eval.core.interfaces import Chunk
from rag_eval.core.exceptions import AzureServiceError
from rag_eval.core.logging import get_logger

logger = get_logger("services.rag.ingestion")


def extract_text_from_document(file_content: bytes, config) -> str:
    """
    Extract text from a document using Azure Document Intelligence.
    Handles OCR, table extraction, and text segmentation.
    
    Args:
        file_content: Raw file content (bytes)
        config: Application configuration
        
    Returns:
        Extracted plaintext content
        
    Raises:
        AzureServiceError: If extraction fails
    """
    logger.info("Extracting text using Azure Document Intelligence")
    
    try:
        # Initialize Document Intelligence client
        client = DocumentIntelligenceClient(
            endpoint=config.azure_document_intelligence_endpoint,
            credential=AzureKeyCredential(config.azure_document_intelligence_api_key)
        )
        
        # Analyze document (supports PDF, images, etc.)
        # This performs OCR, table extraction, and text segmentation
        poller = client.begin_analyze_document(
            model_id="prebuilt-read",  # Prebuilt model for text extraction
            analyze_request=file_content,
            content_type="application/octet-stream"
        )
        
        result = poller.result()
        
        # Extract text from all pages
        extracted_text = []
        if result.content:
            extracted_text.append(result.content)
        
        # Also extract text from tables if present
        if result.tables:
            for table in result.tables:
                for cell in table.cells:
                    if cell.content:
                        extracted_text.append(cell.content)
        
        full_text = "\n\n".join(extracted_text)
        logger.info(f"Extracted {len(full_text)} characters from document")
        
        return full_text
        
    except Exception as e:
        logger.error(f"Document Intelligence extraction failed: {e}")
        raise AzureServiceError(f"Failed to extract text from document: {str(e)}")


def ingest_document(file_content: bytes, config) -> str:
    """
    Ingest a document by extracting text using Azure Document Intelligence.
    
    Args:
        file_content: Raw file content (bytes)
        config: Application configuration
        
    Returns:
        Document text content
        
    Raises:
        AzureServiceError: If ingestion fails
    """
    logger.info("Ingesting document")
    return extract_text_from_document(file_content, config)

