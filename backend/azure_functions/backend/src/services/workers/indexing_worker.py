"""Indexing worker for RAG pipeline

This worker handles the final stage of document processing:
1. Loads chunks and embeddings from persistence layer
2. Indexes chunks and embeddings into Azure AI Search
3. Updates final document status
"""

import time
from typing import Any, Dict, Optional
from src.services.workers.queue_client import (
    QueueMessage,
    validate_message,
    ProcessingStage,
)
from src.services.workers.persistence import (
    load_chunks,
    load_embeddings,
    update_document_status,
    should_process_document,
)
from src.services.rag.search import index_chunks
from src.core.exceptions import (
    AzureServiceError,
    DatabaseError,
    ValidationError,
)
from src.core.logging import get_logger

logger = get_logger("services.workers.indexing_worker")

# Maximum retry attempts before failing
MAX_RETRY_ATTEMPTS = 3


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
        Exception: If all retries are exhausted
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
                    f"Operation failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                time.sleep(delay)
            else:
                logger.error(f"Operation failed after {max_retries + 1} attempts: {e}")
                raise
    
    # This should never be reached, but included for type safety
    raise Exception(f"Unexpected error in retry logic: {last_exception}") from last_exception


def indexing_worker(queue_message: dict, context: Optional[Any] = None) -> None:
    """
    Indexing worker: Loads chunks and embeddings, indexes them, and updates final status.
    
    This worker implements the Load → Process → Persist → Enqueue pattern:
    1. **Load**: Retrieve chunks and embeddings for document from persistence layer
    2. **Process**: Index chunks and embeddings into Azure AI Search
    3. **Persist**: Update document status to 'indexed' (no enqueue needed - final stage)
    
    The worker also implements:
    - Idempotency: Checks document status before processing (Azure Search operations are idempotent)
    - Retry logic: Exponential backoff for transient Azure Search errors
    - Partial failure handling: Uses existing logic in index_chunks to handle partial failures
    
    Args:
        queue_message: Dictionary with message fields:
            - document_id: Document identifier (required)
            - source_storage: Storage backend (required)
            - filename: Original filename (required)
            - attempt: Retry attempt counter (required)
            - stage: Current stage (should be "embedded")
            - metadata: Optional metadata dict
        context: Optional context object (for Azure Functions compatibility)
        
    Raises:
        ValidationError: If message is invalid or missing required fields
        AzureServiceError: If indexing fails after retries
        DatabaseError: If load/persist operations fail
    """
    logger.info("Starting indexing worker")
    
    try:
        # Parse and validate queue message
        try:
            message = validate_message(queue_message)
        except ValidationError as e:
            logger.error(f"Invalid queue message: {e}")
            raise
        
        document_id = message.document_id
        source_storage = message.source_storage
        filename = message.filename
        stage = message.stage
        
        logger.info(
            f"Processing document {document_id} (filename: {filename}, "
            f"source: {source_storage}, stage: {stage})"
        )
        
        # Get config from context
        if context and hasattr(context, 'config'):
            config = context.config
        elif context and isinstance(context, dict) and 'config' in context:
            config = context['config']
        else:
            from src.core.config import Config
            try:
                config = Config.from_env()
            except Exception as e:
                logger.error(f"Failed to get config: {e}")
                raise ValidationError("Config not available in context and failed to load from environment")
        
        # Idempotency check: Skip if already indexed
        if not should_process_document(document_id, "indexed", config):
            logger.info(f"Document {document_id} already indexed, skipping indexing")
            return
        
        # Load: Retrieve chunks and embeddings from persistence layer
        logger.info(f"Loading chunks and embeddings for document {document_id}")
        try:
            chunks = load_chunks(document_id, config)
            if not chunks:
                raise ValueError(f"No chunks found for document {document_id}")
            
            embeddings = load_embeddings(document_id, config)
            if not embeddings:
                raise ValueError(f"No embeddings found for document {document_id}")
            
            if len(chunks) != len(embeddings):
                raise ValueError(
                    f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) count mismatch "
                    f"for document {document_id}"
                )
            
            logger.info(
                f"Loaded {len(chunks)} chunks and {len(embeddings)} embeddings "
                f"for document {document_id}"
            )
        except ValueError as e:
            logger.error(f"Failed to load chunks/embeddings for document {document_id}: {e}")
            # Update status to indicate failure
            try:
                update_document_status(document_id, "failed_indexing", config)
            except Exception:
                pass  # Best effort
            raise
        except Exception as e:
            logger.error(f"Failed to load chunks/embeddings for document {document_id}: {e}")
            raise DatabaseError(f"Failed to load chunks/embeddings: {str(e)}") from e
        
        # Process: Index chunks and embeddings into Azure AI Search
        logger.info(f"Indexing {len(chunks)} chunks for document {document_id}")
        try:
            _retry_with_backoff(
                lambda: index_chunks(chunks, embeddings, config),
                max_retries=MAX_RETRY_ATTEMPTS
            )
            logger.info(f"Successfully indexed {len(chunks)} chunks for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to index chunks for document {document_id}: {e}")
            # Update status to indicate failure
            try:
                update_document_status(document_id, "failed_indexing", config)
            except Exception:
                pass  # Best effort
            raise AzureServiceError(f"Failed to index chunks: {str(e)}") from e
        
        # Persist: Update document status to 'indexed' (final stage - no enqueue needed)
        logger.info(f"Updating document status to 'indexed' for document {document_id}")
        try:
            update_document_status(document_id, "indexed", timestamp_field="indexed_at", config=config)
            logger.info(f"Successfully updated document {document_id} status to 'indexed'")
        except Exception as e:
            logger.error(f"Failed to update document status for document {document_id}: {e}")
            # This is a critical failure - chunks are indexed but status isn't updated
            # Log error but don't fail completely (chunks are already indexed)
            raise DatabaseError(f"Failed to update document status: {str(e)}") from e
        
        logger.info(f"Successfully completed indexing for document {document_id}")
        
    except (ValidationError, AzureServiceError, DatabaseError):
        # Re-raise known exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in indexing worker: {e}", exc_info=True)
        raise AzureServiceError(f"Unexpected error in indexing worker: {str(e)}") from e

