"""Ingestion worker for RAG pipeline

This worker handles the first stage of document processing:
1. Downloads file from storage (Supabase or Azure Blob)
2. Validates file size (50MB limit)
3. Detects PDF page count
4. Processes PDF in configurable batches (default: 2 pages)
5. Persists each batch temporarily, then merges and persists final text
6. Enqueues message to chunking queue

The worker supports resumable batch processing with crash recovery.
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Tuple
from src.services.workers.queue_client import (
    QueueMessage,
    validate_message,
    enqueue_message,
    send_to_dead_letter,
    ProcessingStage,
    SourceStorage,
)
from src.services.workers.persistence import (
    persist_extracted_text,
    update_document_status,
    should_process_document,
    persist_batch_result,
    get_completed_batches,
    load_batch_result,
    delete_batch_chunk,
    update_ingestion_metadata,
    get_ingestion_metadata,
)
from src.services.workers.pdf_utils import (
    get_pdf_page_count,
    slice_pdf_to_batch,
    generate_page_batches,
)
from src.services.rag.ingestion import extract_text_from_batch
from src.services.rag.storage import download_document_from_blob
from src.services.rag.supabase_storage import download_document_from_storage
from src.core.exceptions import (
    AzureServiceError,
    DatabaseError,
    ValidationError,
)
from src.core.logging import get_logger

logger = get_logger("services.workers.ingestion_worker")

# Maximum retry attempts before sending to dead-letter queue
MAX_RETRY_ATTEMPTS = 3

# Memory limit: 50MB
MEMORY_LIMIT_BYTES = 50 * 1024 * 1024

# Default batch size
DEFAULT_BATCH_SIZE = 2
MAX_BATCH_SIZE = 10


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


def ingestion_worker(queue_message: dict, context: Optional[Any] = None) -> None:
    """
    Ingestion worker: Downloads file, processes in batches, persists results, and enqueues to next stage.
    
    This worker implements the Load → Process → Persist → Enqueue pattern with batch processing:
    1. **Load**: Download file from storage (Supabase or Azure Blob)
    2. **Validate**: Check file size (fail fast if > 50MB)
    3. **Detect**: Get PDF page count using PyPDF2
    4. **Process Batches**: For each batch (configurable size, default 2 pages):
       - Skip if already completed (idempotency)
       - Slice PDF to batch pages
       - Extract text using Azure Document Intelligence
       - Persist batch result temporarily
       - Merge batch into accumulated text
       - Delete batch chunk immediately
       - Update progress metadata
    5. **Persist**: Store merged extracted text in database
    6. **Enqueue**: Send message to chunking queue
    
    The worker also implements:
    - Idempotency: Checks document status before processing, skips completed batches
    - Retry logic: Exponential backoff for transient failures
    - Dead-letter handling: Sends to dead-letter queue after max retries
    - Resumability: Can resume from last successful batch after crash/retry
    
    Args:
        queue_message: Dictionary with message fields:
            - document_id: Document identifier (required)
            - source_storage: "azure_blob" or "supabase" (required)
            - filename: Original filename (required)
            - attempt: Retry attempt counter (required)
            - stage: Current stage (should be "uploaded")
            - metadata: Optional metadata dict (can contain "batch_size": int)
        context: Optional context object (for Azure Functions compatibility)
        
    Raises:
        ValidationError: If message is invalid or missing required fields
        AzureServiceError: If file download or text extraction fails
        DatabaseError: If persistence fails
    """
    logger.info("Starting ingestion worker")
    
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
        attempt = message.attempt
        stage = message.stage
        
        logger.info(
            f"Processing document {document_id} (filename: {filename}, "
            f"source: {source_storage}, stage: {stage}, attempt: {attempt})"
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
        
        # Idempotency check: Skip if already parsed
        if not should_process_document(document_id, "parsed", config):
            logger.info(f"Document {document_id} already parsed, skipping ingestion")
            return
        
        # Load: Download file from storage
        logger.info(f"Downloading file for document {document_id} from {source_storage}")
        try:
            if source_storage == SourceStorage.AZURE_BLOB:
                file_content = _retry_with_backoff(
                    lambda: download_document_from_blob(document_id, config),
                    max_retries=MAX_RETRY_ATTEMPTS
                )
            elif source_storage == SourceStorage.SUPABASE:
                file_content = _retry_with_backoff(
                    lambda: download_document_from_storage(document_id, config),
                    max_retries=MAX_RETRY_ATTEMPTS
                )
            else:
                raise ValidationError(f"Unsupported source_storage: {source_storage}")
            
            logger.info(f"Downloaded {len(file_content)} bytes for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to download file for document {document_id}: {e}")
            if attempt >= MAX_RETRY_ATTEMPTS:
                logger.error(f"Max retries reached for document {document_id}, sending to dead-letter")
                try:
                    send_to_dead_letter(
                        "ingestion-dead-letter",
                        message,
                        f"Failed to download file after {attempt} attempts: {str(e)}",
                        config
                    )
                    try:
                        update_document_status(document_id, "failed_ingestion", config)
                    except Exception:
                        pass
                except Exception as dl_error:
                    logger.error(f"Failed to send to dead-letter queue: {dl_error}")
            raise AzureServiceError(f"Failed to download file: {str(e)}") from e
        
        # Validate: Check memory limit (50MB)
        if len(file_content) > MEMORY_LIMIT_BYTES:
            error_msg = f"PDF exceeds 50MB memory limit ({len(file_content) / (1024*1024):.2f}MB)"
            logger.error(f"{error_msg} for document {document_id}")
            try:
                send_to_dead_letter(
                    "ingestion-dead-letter",
                    message,
                    error_msg,
                    config
                )
                update_document_status(document_id, "failed_ingestion", config)
                # Update metadata with error
                update_ingestion_metadata(
                    document_id,
                    {
                        "parsing_status": "failed",
                        "errors": [error_msg]
                    },
                    config
                )
            except Exception:
                pass  # Best effort
            raise AzureServiceError(error_msg)
        
        # Get batch size from metadata (default: 2)
        batch_size = DEFAULT_BATCH_SIZE
        if message.metadata and "batch_size" in message.metadata:
            batch_size = int(message.metadata["batch_size"])
            if batch_size <= 0 or batch_size > MAX_BATCH_SIZE:
                raise ValidationError(
                    f"batch_size must be between 1 and {MAX_BATCH_SIZE}, got {batch_size}"
                )
        
        # Detect: Get PDF page count
        logger.info(f"Detecting page count for document {document_id}")
        try:
            num_pages = get_pdf_page_count(file_content)
            logger.info(f"Detected {num_pages} pages for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to detect page count for document {document_id}: {e}")
            if attempt >= MAX_RETRY_ATTEMPTS:
                try:
                    send_to_dead_letter(
                        "ingestion-dead-letter",
                        message,
                        f"Failed to detect page count: {str(e)}",
                        config
                    )
                    update_document_status(document_id, "failed_ingestion", config)
                except Exception:
                    pass
            raise AzureServiceError(f"Failed to detect page count: {str(e)}") from e
        
        # Initialize: Set up batch processing metadata
        ingestion_meta = get_ingestion_metadata(document_id, config)
        if ingestion_meta.get("parsing_status") != "in_progress":
            # First time processing or resuming
            update_ingestion_metadata(
                document_id,
                {
                    "num_pages": num_pages,
                    "num_batches_total": len(generate_page_batches(num_pages, batch_size)),
                    "batch_size": batch_size,
                    "parsing_status": "in_progress",
                    "parsing_started_at": datetime.now(timezone.utc).isoformat(),
                },
                config
            )
        
        # Get completed batches for resumability
        completed_batches = get_completed_batches(document_id, config)
        if completed_batches:
            logger.info(
                f"Found {len(completed_batches)} completed batches for document {document_id}: "
                f"{sorted(completed_batches)}. Will resume from batch {max(completed_batches) + 1}"
            )
        else:
            logger.info(
                f"No completed batches found for document {document_id}. "
                f"Will process all batches from the beginning."
            )
        
        # Generate batch ranges (1-indexed for Azure)
        batch_ranges = generate_page_batches(num_pages, batch_size)
        logger.info(
            f"Generated {len(batch_ranges)} batches for document {document_id} "
            f"({num_pages} pages, batch_size={batch_size})"
        )
        
        # Initialize accumulator for merged text
        merged_text_parts = []
        
        # Time-based execution limiting for Azure Functions timeout management
        # Target: 4 minutes of processing (240 seconds) with 1 minute safety buffer for cleanup
        MAX_EXECUTION_TIME_SECONDS = 240  # 4 minutes
        SAFETY_BUFFER_SECONDS = 60  # Reserve 1 minute for final operations
        MIN_TIME_PER_BATCH = 5  # Minimum estimate for a batch (seconds)
        
        execution_start_time = time.time()
        batches_processed_this_execution = 0
        batch_times = []  # Track actual batch processing times
        
        logger.info(
            f"Starting time-limited batch processing (max {MAX_EXECUTION_TIME_SECONDS}s, "
            f"safety buffer {SAFETY_BUFFER_SECONDS}s)"
        )
        
        # Process: Process each batch
        for batch_index, (start_page, end_page) in enumerate(batch_ranges):
            # Skip if already completed (idempotency)
            if batch_index in completed_batches:
                logger.debug(
                    f"Skipping batch {batch_index} (pages {start_page}-{end_page}) "
                    f"for document {document_id} - already completed"
                )
                # Load the batch result to include in merged text
                try:
                    batch_text = load_batch_result(document_id, batch_index, config)
                    if batch_text:
                        merged_text_parts.append(batch_text)
                except Exception as e:
                    logger.warning(
                        f"Failed to load completed batch {batch_index} for document {document_id}: {e}. "
                        f"Will reprocess."
                    )
                    # Remove from completed set so we reprocess
                    completed_batches.discard(batch_index)
            
            # Check execution time before processing new batch
            if batch_index not in completed_batches:
                elapsed_time = time.time() - execution_start_time
                
                # Estimate time for next batch based on average of previous batches
                if batch_times:
                    avg_batch_time = sum(batch_times) / len(batch_times)
                    estimated_next_batch_time = avg_batch_time * 1.2  # 20% buffer for variance
                else:
                    # First batch - use minimum estimate
                    estimated_next_batch_time = MIN_TIME_PER_BATCH
                
                # Check if we have enough time for another batch
                time_remaining = MAX_EXECUTION_TIME_SECONDS - elapsed_time
                
                if time_remaining < (estimated_next_batch_time + SAFETY_BUFFER_SECONDS):
                    # Not enough time for another batch - re-enqueue for continuation
                    logger.info(
                        f"Time limit approaching: {elapsed_time:.1f}s elapsed, "
                        f"{time_remaining:.1f}s remaining, need {estimated_next_batch_time + SAFETY_BUFFER_SECONDS:.1f}s for next batch. "
                        f"Processed {batches_processed_this_execution} batches this execution. "
                        f"Re-enqueuing for continuation from batch {batch_index}..."
                    )
                    
                    # Re-enqueue message for next batch set
                    continuation_message = QueueMessage(
                        document_id=document_id,
                        source_storage=source_storage,
                        filename=filename,
                        attempt=1,  # Reset attempt - this is continuation, not retry
                        stage=ProcessingStage.UPLOADED,
                        metadata=message.metadata
                    )
                    enqueue_message("ingestion-uploads", continuation_message, config)
                    
                    logger.info(
                        f"Progress saved: {len(completed_batches) + batches_processed_this_execution}/{len(batch_ranges)} "
                        f"batches completed. Continuation message enqueued. "
                        f"Average batch time: {sum(batch_times)/len(batch_times):.1f}s"
                    )
                    
                    # Exit this execution - next function call will continue
                    return
                
                # We have time - process this batch
                batch_start_time = time.time()
                
                logger.info(
                    f"Processing batch {batch_index} (pages {start_page}-{end_page}) "
                    f"for document {document_id} - batch {batches_processed_this_execution + 1} this execution, "
                    f"{elapsed_time:.1f}s elapsed, {time_remaining:.1f}s remaining"
                )
                
                try:
                    # Slice PDF to batch
                    try:
                        batch_pdf = slice_pdf_to_batch(file_content, start_page, end_page)
                    except Exception as e:
                        logger.error(
                            f"Failed to slice PDF for batch {batch_index} "
                            f"(pages {start_page}-{end_page}): {e}"
                        )
                        if attempt >= MAX_RETRY_ATTEMPTS:
                            error_msg = f"Failed to slice PDF for batch {batch_index}: {str(e)}"
                            try:
                                send_to_dead_letter(
                                    "ingestion-dead-letter",
                                    message,
                                    error_msg,
                                    config
                                )
                                update_document_status(document_id, "failed_ingestion", config)
                                # Update metadata with error
                                current_errors = ingestion_meta.get("errors", [])
                                current_errors.append(error_msg)
                                update_ingestion_metadata(
                                    document_id,
                                    {
                                        "parsing_status": "failed",
                                        "errors": current_errors
                                    },
                                    config
                                )
                            except Exception:
                                pass
                        raise AzureServiceError(f"Failed to slice PDF: {str(e)}") from e
                    
                    # Extract text from batch
                    page_range_str = f"{start_page}-{end_page}"
                    try:
                        batch_text = _retry_with_backoff(
                            lambda: extract_text_from_batch(batch_pdf, page_range_str, config),
                            max_retries=MAX_RETRY_ATTEMPTS
                        )
                        logger.info(
                            f"Extracted {len(batch_text)} characters from batch {batch_index} "
                            f"(pages {start_page}-{end_page})"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to extract text from batch {batch_index} "
                            f"(pages {start_page}-{end_page}): {e}"
                        )
                        if attempt >= MAX_RETRY_ATTEMPTS:
                            error_msg = f"Failed to extract text from batch {batch_index}: {str(e)}"
                            try:
                                send_to_dead_letter(
                                    "ingestion-dead-letter",
                                    message,
                                    error_msg,
                                    config
                                )
                                update_document_status(document_id, "failed_ingestion", config)
                                # Update metadata with error
                                current_errors = ingestion_meta.get("errors", [])
                                current_errors.append(error_msg)
                                update_ingestion_metadata(
                                    document_id,
                                    {
                                        "parsing_status": "failed",
                                        "errors": current_errors
                                    },
                                    config
                                )
                            except Exception:
                                pass
                        raise AzureServiceError(f"Failed to extract text from batch: {str(e)}") from e
                    
                    # Persist batch result temporarily
                    try:
                        persist_batch_result(
                            document_id,
                            batch_index,
                            batch_text,
                            start_page,
                            end_page,
                            config
                        )
                    except Exception as e:
                        logger.error(f"Failed to persist batch {batch_index}: {e}")
                        raise DatabaseError(f"Failed to persist batch result: {str(e)}") from e
                    
                    # Merge batch into accumulator
                    merged_text_parts.append(batch_text)
                    
                    # Delete batch chunk immediately (cleanup)
                    try:
                        delete_batch_chunk(document_id, batch_index, config)
                    except Exception as e:
                        logger.warning(f"Failed to delete batch chunk {batch_index}: {e}")
                        # Non-critical, continue
                    
                    # Update progress metadata
                    try:
                        update_ingestion_metadata(
                            document_id,
                            {
                                "last_successful_page": end_page - 1,  # Last page processed (1-indexed)
                                "next_unparsed_batch_index": batch_index + 1,
                                "batches_completed": {**ingestion_meta.get("batches_completed", {}), str(batch_index): True}
                            },
                            config
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update progress metadata: {e}")
                        # Non-critical, continue
                    
                    # Track batch processing time
                    batch_duration = time.time() - batch_start_time
                    batch_times.append(batch_duration)
                    batches_processed_this_execution += 1
                    
                    logger.debug(
                        f"Batch {batch_index} completed in {batch_duration:.1f}s "
                        f"(avg: {sum(batch_times)/len(batch_times):.1f}s)"
                    )
                    
                except (AzureServiceError, DatabaseError):
                    # Re-raise known exceptions
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error processing batch {batch_index}: {e}")
                    raise AzureServiceError(f"Unexpected error processing batch: {str(e)}") from e
        
        # All batches complete!
        total_execution_time = time.time() - execution_start_time
        logger.info(
            f"All batches completed for document {document_id}. "
            f"Processed {batches_processed_this_execution} batches this execution "
            f"in {total_execution_time:.1f}s. "
            f"Average batch time: {sum(batch_times)/len(batch_times):.1f}s"
        )
        
        # Merge all batches
        merged_text = "\n\n".join(merged_text_parts)
        logger.info(
            f"Merged {len(batch_ranges)} batches into {len(merged_text)} characters "
            f"for document {document_id}"
        )
        
        # Persist: Store merged extracted text and update status
        logger.info(f"Persisting merged extracted text for document {document_id}")
        try:
            persist_extracted_text(document_id, merged_text, config)
            update_document_status(document_id, "parsed", timestamp_field="parsed_at", config=config)
            
            # Update metadata to mark as completed
            update_ingestion_metadata(
                document_id,
                {
                    "parsing_status": "completed",
                    "parsing_completed_at": datetime.now(timezone.utc).isoformat()
                },
                config
            )
            
            logger.info(f"Successfully persisted extracted text for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to persist extracted text for document {document_id}: {e}")
            raise DatabaseError(f"Failed to persist extracted text: {str(e)}") from e
        
        # Enqueue: Send message to chunking queue
        logger.info(f"Enqueuing message to chunking queue for document {document_id}")
        try:
            next_message = QueueMessage(
                document_id=document_id,
                source_storage=source_storage,
                filename=filename,
                attempt=1,  # Reset attempt counter for next stage
                stage=ProcessingStage.PARSED,
                metadata=message.metadata
            )
            enqueue_message("ingestion-chunking", next_message, config)
            logger.info(f"Successfully enqueued message to chunking queue for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to enqueue message to chunking queue for document {document_id}: {e}")
            try:
                update_document_status(document_id, "failed_chunking_enqueue", config)
            except Exception:
                pass  # Best effort
            raise AzureServiceError(f"Failed to enqueue to chunking queue: {str(e)}") from e
        
        logger.info(f"Successfully completed ingestion for document {document_id}")
        
    except (ValidationError, AzureServiceError, DatabaseError):
        # Re-raise known exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in ingestion worker: {e}", exc_info=True)
        raise AzureServiceError(f"Unexpected error in ingestion worker: {str(e)}") from e

