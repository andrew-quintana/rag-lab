"""Chunking worker for RAG pipeline

This worker handles the second stage of document processing:
1. Loads extracted text from persistence layer
2. Chunks text using fixed-size algorithm
3. Persists chunks to database
4. Enqueues message to embedding queue
"""

from typing import Any, Dict, Optional
from rag_eval.services.workers.queue_client import (
    QueueMessage,
    validate_message,
    enqueue_message,
    ProcessingStage,
)
from rag_eval.services.workers.persistence import (
    load_extracted_text,
    persist_chunks,
    update_document_status,
    should_process_document,
)
from rag_eval.services.rag.chunking import chunk_text
from rag_eval.core.exceptions import (
    DatabaseError,
    ValidationError,
)
from rag_eval.core.logging import get_logger

logger = get_logger("services.workers.chunking_worker")


def chunking_worker(queue_message: dict, context: Optional[Any] = None) -> None:
    """
    Chunking worker: Loads extracted text, chunks it, persists chunks, and enqueues to next stage.
    
    This worker implements the Load → Process → Persist → Enqueue pattern:
    1. **Load**: Retrieve extracted text for document from persistence layer
    2. **Process**: Chunk text using fixed-size algorithm
    3. **Persist**: Store chunks in database and update document status
    4. **Enqueue**: Send message to embedding queue
    
    The worker also implements:
    - Idempotency: Checks document status before processing (handles duplicate messages safely)
    - Error handling: Fail fast for data/validation issues
    
    Args:
        queue_message: Dictionary with message fields:
            - document_id: Document identifier (required)
            - source_storage: Storage backend (required)
            - filename: Original filename (required)
            - attempt: Retry attempt counter (required)
            - stage: Current stage (should be "parsed")
            - metadata: Optional metadata dict
        context: Optional context object (for Azure Functions compatibility)
        
    Raises:
        ValidationError: If message is invalid or missing required fields
        DatabaseError: If load/persist operations fail
        ValueError: If extracted text is missing or invalid
    """
    logger.info("Starting chunking worker")
    
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
            from rag_eval.core.config import Config
            try:
                config = Config.from_env()
            except Exception as e:
                logger.error(f"Failed to get config: {e}")
                raise ValidationError("Config not available in context and failed to load from environment")
        
        # Idempotency check: Skip if already chunked
        if not should_process_document(document_id, "chunked", config):
            logger.info(f"Document {document_id} already chunked, skipping chunking")
            return
        
        # Load: Retrieve extracted text from persistence layer
        logger.info(f"Loading extracted text for document {document_id}")
        try:
            extracted_text = load_extracted_text(document_id, config)
            if not extracted_text or not extracted_text.strip():
                raise ValueError(f"Extracted text is empty for document {document_id}")
            logger.info(f"Loaded {len(extracted_text)} characters of extracted text for document {document_id}")
        except ValueError as e:
            logger.error(f"Failed to load extracted text for document {document_id}: {e}")
            # Update status to indicate failure
            try:
                update_document_status(document_id, "failed_chunking", config)
            except Exception:
                pass  # Best effort
            raise
        except Exception as e:
            logger.error(f"Failed to load extracted text for document {document_id}: {e}")
            raise DatabaseError(f"Failed to load extracted text: {str(e)}") from e
        
        # Process: Chunk text using fixed-size algorithm
        logger.info(f"Chunking text for document {document_id}")
        try:
            # Get chunking parameters from metadata or use defaults
            chunk_size = 1000  # Default chunk size
            overlap = 200  # Default overlap
            
            if message.metadata:
                chunk_size = message.metadata.get("chunk_size", chunk_size)
                overlap = message.metadata.get("chunk_overlap", overlap)
            
            chunks = chunk_text(
                text=extracted_text,
                config=config,
                document_id=document_id,
                chunk_size=chunk_size,
                overlap=overlap
            )
            logger.info(f"Generated {len(chunks)} chunks for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to chunk text for document {document_id}: {e}")
            # Update status to indicate failure
            try:
                update_document_status(document_id, "failed_chunking", config)
            except Exception:
                pass  # Best effort
            raise ValueError(f"Failed to chunk text: {str(e)}") from e
        
        # Persist: Store chunks in database and update status
        logger.info(f"Persisting chunks for document {document_id}")
        try:
            persist_chunks(document_id, chunks, config)
            update_document_status(document_id, "chunked", timestamp_field="chunked_at", config=config)
            logger.info(f"Successfully persisted {len(chunks)} chunks for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to persist chunks for document {document_id}: {e}")
            # Update status to indicate failure
            try:
                update_document_status(document_id, "failed_chunking", config)
            except Exception:
                pass  # Best effort
            raise DatabaseError(f"Failed to persist chunks: {str(e)}") from e
        
        # Enqueue: Send message to embedding queue
        logger.info(f"Enqueuing message to embedding queue for document {document_id}")
        try:
            next_message = QueueMessage(
                document_id=document_id,
                source_storage=source_storage,
                filename=filename,
                attempt=1,  # Reset attempt counter for next stage
                stage=ProcessingStage.CHUNKED,
                metadata=message.metadata
            )
            enqueue_message("ingestion-embeddings", next_message, config)
            logger.info(f"Successfully enqueued message to embedding queue for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to enqueue message to embedding queue for document {document_id}: {e}")
            # This is a critical failure - chunks are persisted but we can't proceed
            # Update status to indicate partial success
            try:
                update_document_status(document_id, "failed_embedding_enqueue", config)
            except Exception:
                pass  # Best effort
            raise DatabaseError(f"Failed to enqueue to embedding queue: {str(e)}") from e
        
        logger.info(f"Successfully completed chunking for document {document_id}")
        
    except (ValidationError, DatabaseError, ValueError):
        # Re-raise known exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chunking worker: {e}", exc_info=True)
        raise DatabaseError(f"Unexpected error in chunking worker: {str(e)}") from e

