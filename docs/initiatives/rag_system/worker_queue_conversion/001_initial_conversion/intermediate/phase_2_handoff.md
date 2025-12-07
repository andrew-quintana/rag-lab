# Phase 2 Handoff — Queue Infrastructure

## Phase 2 Status: ✅ Complete

Phase 2 implementation is complete with all validation requirements met.

## Deliverables

### 1. Core Implementation
- ✅ `rag_eval/services/workers/queue_client.py` - Complete queue client module
  - QueueMessage dataclass with validation
  - Message validation, serialization, deserialization functions
  - All queue operations (enqueue, dequeue, peek, delete, dead-letter, get_queue_length)
  - Convenience function: `dequeue_message_with_receipt()`

### 2. Testing
- ✅ `backend/tests/components/workers/test_queue_client.py` - Comprehensive test suite
  - 51 tests, all passing
  - 93% code coverage (exceeds 80% requirement)
  - Complete error handling and edge case coverage

### 3. Documentation
- ✅ Complete docstrings for all functions
- ✅ `phase_2_decisions.md` - Implementation decisions
- ✅ `phase_2_testing.md` - Testing summary
- ✅ `phase_2_handoff.md` - This document

## Key Components for Phase 3

### QueueMessage Dataclass
```python
from rag_eval.services.workers.queue_client import QueueMessage

message = QueueMessage(
    document_id="doc-123",
    source_storage="supabase",  # or "azure_blob"
    filename="document.pdf",
    attempt=1,
    stage="uploaded",  # or "parsed", "chunked", "embedded", "indexed"
    metadata={"tenant_id": "tenant_abc"}  # optional
)
```

### Queue Operations
```python
from rag_eval.services.workers.queue_client import (
    enqueue_message,
    dequeue_message,
    dequeue_message_with_receipt,
    delete_message,
    send_to_dead_letter,
    get_queue_length
)

# Enqueue a message
enqueue_message("ingestion-uploads", message, config)

# Dequeue a message (returns message or None)
message = dequeue_message("ingestion-uploads", config)

# Dequeue with receipt (for deletion after processing)
result = dequeue_message_with_receipt("ingestion-uploads", config)
if result:
    message, message_id, pop_receipt = result
    # Process message...
    delete_message("ingestion-uploads", message_id, pop_receipt, config)

# Send to dead-letter queue
send_to_dead_letter("ingestion-dead-letter", message, "Max retries exceeded", config)

# Get queue length
length = get_queue_length("ingestion-uploads", config)
```

### Message Validation
```python
from rag_eval.services.workers.queue_client import validate_message, serialize_message, deserialize_message

# Validate dictionary to QueueMessage
message = validate_message({"document_id": "doc-123", ...})

# Serialize to JSON string
json_str = serialize_message(message)

# Deserialize from JSON string
message = deserialize_message(json_str)
```

## Queue Names

Use these exact queue names (as specified in RFC001.md):
- `ingestion-uploads` - Triggers ingestion worker
- `ingestion-chunking` - Triggers chunking worker
- `ingestion-embeddings` - Triggers embedding worker
- `ingestion-indexing` - Triggers indexing worker
- `ingestion-dead-letter` - For poison messages

## Configuration

Queue client uses `config.azure_blob_connection_string` from Config class. This connection string works for both Blob Storage and Queue Storage services.

## Error Handling Patterns

### Invalid Messages (Poison Messages)
- `dequeue_message()` and `peek_message()` return `None` for invalid messages
- Logs warning but doesn't raise exception
- Allows workers to skip corrupted messages

### Queue Not Found
- `dequeue_message()`, `peek_message()`, `get_queue_length()` return `None` or `0` if queue doesn't exist
- Logs warning but doesn't raise exception
- Workers should handle gracefully

### Connection Errors
- All operations raise `AzureServiceError` for connection failures
- Workers should implement retry logic (Phase 3)

## Worker Implementation Pattern

For Phase 3, workers should follow this pattern:

```python
def worker_function(queue_message: dict, context):
    """Worker function triggered by Azure Function queue trigger"""
    from rag_eval.services.workers.queue_client import (
        validate_message,
        dequeue_message_with_receipt,
        delete_message,
        enqueue_message,
        send_to_dead_letter
    )
    
    # Parse and validate message
    try:
        message = validate_message(queue_message)
    except ValidationError as e:
        logger.error(f"Invalid message: {e}")
        return
    
    # Check idempotency (using persistence layer)
    if not should_process_document(message.document_id, target_stage, config):
        logger.info(f"Skipping already processed document: {message.document_id}")
        return
    
    try:
        # Load data from persistence layer
        data = load_data(message.document_id, config)
        
        # Process using existing service module
        result = existing_service_function(data, config)
        
        # Persist results
        persist_data(message.document_id, result, config)
        
        # Update document status
        update_document_status(message.document_id, new_stage, config)
        
        # Enqueue to next queue
        next_message = QueueMessage(
            document_id=message.document_id,
            source_storage=message.source_storage,
            filename=message.filename,
            attempt=1,
            stage=next_stage,
            metadata=message.metadata
        )
        enqueue_message(next_queue_name, next_message, config)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        # Increment attempt and retry or dead-letter
        if message.attempt >= max_retries:
            send_to_dead_letter("ingestion-dead-letter", message, str(e), config)
        else:
            # Retry logic (increment attempt, re-enqueue)
            pass
```

## Dependencies

- `azure-storage-queue>=12.19.0` - Already in requirements.txt
- Uses existing `rag_eval.core.exceptions` (AzureServiceError, ValidationError)
- Uses existing `rag_eval.core.logging` for logging

## Testing Notes

- All queue operations are unit tested with mocked Azure SDK
- Real Azure Storage Account testing deferred to Phase 5 (Integration Testing)
- Workers should use the same mocking patterns for queue operations in their tests

## Known Limitations

1. **Queue Auto-Creation**: Queues are created automatically on first enqueue, but manual creation via Azure Portal is recommended for production
2. **Retry Logic**: Retry logic for transient failures is deferred to Phase 3 (worker implementation)
3. **Message Visibility Timeout**: Default 30 seconds, can be customized in `dequeue_message()` calls

## Next Phase Entry Point

Phase 3 should:
1. Create worker modules in `rag_eval/services/workers/`
2. Implement Load → Process → Persist → Enqueue pattern
3. Use queue_client functions for all queue operations
4. Implement idempotency checks using persistence layer
5. Add retry logic with exponential backoff
6. Handle dead-letter queue for permanent failures

See `prompts/prompt_phase_3_001.md` for detailed Phase 3 requirements.

## Questions or Issues?

If any issues arise during Phase 3 implementation:
1. Check `phase_2_decisions.md` for implementation decisions
2. Review `phase_2_testing.md` for test patterns
3. Refer to RFC001.md for architecture details
4. Document any new decisions in `phase_3_decisions.md`

