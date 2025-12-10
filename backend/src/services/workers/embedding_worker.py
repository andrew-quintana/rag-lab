"""Embedding worker for RAG pipeline

This worker handles the third stage of document processing:
1. Loads chunks from persistence layer
2. Generates embeddings using Azure AI Foundry
3. Persists embeddings to database
4. Enqueues message to indexing queue
"""

import time
from typing import Any, Dict, Optional
from src.services.workers.queue_client import (
    QueueMessage,
    validate_message,
    enqueue_message,
    ProcessingStage,
)
from src.services.workers.persistence import (
    load_chunks,
    persist_embeddings,
    update_document_status,
    should_process_document,
)
from src.services.rag.embeddings import generate_embeddings
from src.core.exceptions import (
    AzureServiceError,
    DatabaseError,
    ValidationError,
)
from src.core.logging import get_logger

logger = get_logger("services.workers.embedding_worker")

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


def embedding_worker(queue_message: dict, context: Optional[Any] = None) -> None:
    """
    Embedding worker: Loads chunks, generates embeddings, persists embeddings, and enqueues to next stage.
    
    This worker implements the Load → Process → Persist → Enqueue pattern:
    1. **Load**: Fetch all chunks for document from persistence layer
    2. **Process**: Generate embeddings using Azure AI Foundry
    3. **Persist**: Store embeddings in database and update document status
    4. **Enqueue**: Send message to indexing queue
    
    The worker also implements:
    - Idempotency: Checks document status before processing (skips if already embedded)
    - Retry logic: Exponential backoff for transient Azure Foundry failures
    
    Args:
        queue_message: Dictionary with message fields:
            - document_id: Document identifier (required)
            - source_storage: Storage backend (required)
            - filename: Original filename (required)
            - attempt: Retry attempt counter (required)
            - stage: Current stage (should be "chunked")
            - metadata: Optional metadata dict
        context: Optional context object (for Azure Functions compatibility)
        
    Raises:
        ValidationError: If message is invalid or missing required fields
        AzureServiceError: If embedding generation fails after retries
        DatabaseError: If load/persist operations fail
    """
    logger.info("Starting embedding worker")
    
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
        
        # Idempotency check: Skip if already embedded
        if not should_process_document(document_id, "embedded", config):
            logger.info(f"Document {document_id} already embedded, skipping embedding")
            return
        
        # Load: Fetch all chunks from persistence layer
        logger.info(f"Loading chunks for document {document_id}")
        try:
            chunks = load_chunks(document_id, config)
            if not chunks:
                raise ValueError(f"No chunks found for document {document_id}")
            logger.info(f"Loaded {len(chunks)} chunks for document {document_id}")
        except ValueError as e:
            logger.error(f"Failed to load chunks for document {document_id}: {e}")
            # Update status to indicate failure
            try:
                update_document_status(document_id, "failed_embedding", config)
            except Exception:
                pass  # Best effort
            raise
        except Exception as e:
            logger.error(f"Failed to load chunks for document {document_id}: {e}")
            raise DatabaseError(f"Failed to load chunks: {str(e)}") from e
        
        # Process: Generate embeddings using Azure AI Foundry
        logger.info(f"Generating embeddings for {len(chunks)} chunks for document {document_id}")
        try:
            embeddings = _retry_with_backoff(
                lambda: generate_embeddings(chunks, config),
                max_retries=MAX_RETRY_ATTEMPTS
            )
            logger.info(f"Generated {len(embeddings)} embeddings for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to generate embeddings for document {document_id}: {e}")
            # Update status to indicate failure
            try:
                update_document_status(document_id, "failed_embedding", config)
            except Exception:
                pass  # Best effort
            raise AzureServiceError(f"Failed to generate embeddings: {str(e)}") from e
        
        # Persist: Store embeddings in database and update status
        logger.info(f"Persisting embeddings for document {document_id}")
        try:
            persist_embeddings(document_id, chunks, embeddings, config)
            update_document_status(document_id, "embedded", timestamp_field="embedded_at", config=config)
            logger.info(f"Successfully persisted {len(embeddings)} embeddings for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to persist embeddings for document {document_id}: {e}")
            # Update status to indicate failure
            try:
                update_document_status(document_id, "failed_embedding", config)
            except Exception:
                pass  # Best effort
            raise DatabaseError(f"Failed to persist embeddings: {str(e)}") from e
        
        # Enqueue: Send message to indexing queue
        logger.info(f"Enqueuing message to indexing queue for document {document_id}")
        try:
            next_message = QueueMessage(
                document_id=document_id,
                source_storage=source_storage,
                filename=filename,
                attempt=1,  # Reset attempt counter for next stage
                stage=ProcessingStage.EMBEDDED,
                metadata=message.metadata
            )
            enqueue_message("ingestion-indexing", next_message, config)
            logger.info(f"Successfully enqueued message to indexing queue for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to enqueue message to indexing queue for document {document_id}: {e}")
            # This is a critical failure - embeddings are persisted but we can't proceed
            # Update status to indicate partial success
            try:
                update_document_status(document_id, "failed_indexing_enqueue", config)
            except Exception:
                pass  # Best effort
            raise DatabaseError(f"Failed to enqueue to indexing queue: {str(e)}") from e
        
        logger.info(f"Successfully completed embedding for document {document_id}")
        
    except (ValidationError, AzureServiceError, DatabaseError):
        # Re-raise known exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in embedding worker: {e}", exc_info=True)
        raise AzureServiceError(f"Unexpected error in embedding worker: {str(e)}") from e

