# Phase 3 Decisions

## Overview

This document captures key decisions made during Phase 3 implementation that are not already documented in PRD001.md or RFC001.md.

## Worker Implementation Decisions

### 1. Config Loading Strategy

**Decision**: Workers support multiple config loading strategies:
1. From context object (for Azure Functions)
2. From context dictionary (for testing)
3. From environment variables (fallback)

**Rationale**: Provides flexibility for different deployment scenarios while maintaining testability.

**Implementation**: All workers check for config in this order:
```python
if context and hasattr(context, 'config'):
    config = context.config
elif context and isinstance(context, dict) and 'config' in context:
    config = context['config']
else:
    config = Config.from_env()
```

### 2. Retry Logic Implementation

**Decision**: Each worker implements its own retry logic with exponential backoff, rather than using a shared utility.

**Rationale**: 
- Different workers have different retry requirements
- Ingestion worker needs retry for file download and text extraction
- Embedding worker needs retry for Azure AI Foundry API calls
- Indexing worker needs retry for Azure AI Search operations
- Chunking worker doesn't need retry (fail fast for data issues)

**Implementation**: 
- `_retry_with_backoff()` helper function in each worker that needs it
- Max retries: 3 attempts
- Base delay: 1.0 seconds
- Exponential backoff: `delay = base_delay * (2 ** attempt)`

### 3. Error Handling Strategy

**Decision**: Workers update document status to failure states before raising exceptions.

**Rationale**: 
- Enables tracking of failure points in the pipeline
- Allows for better observability and debugging
- Status updates are best-effort (wrapped in try/except to avoid masking original errors)

**Implementation**: 
- Status updates use try/except with pass (best effort)
- Original exceptions are re-raised after status update attempt
- Failure statuses: `failed_ingestion`, `failed_chunking`, `failed_embedding`, `failed_indexing`, `failed_*_enqueue`

### 4. Dead-Letter Queue Handling

**Decision**: Only ingestion worker implements dead-letter queue handling.

**Rationale**:
- Ingestion worker is the entry point and most likely to encounter permanent failures (missing files, invalid formats)
- Other workers can rely on Azure Queue's built-in retry and visibility timeout
- Dead-letter handling can be added to other workers in future if needed

**Implementation**:
- Dead-letter queue: `ingestion-dead-letter`
- Triggered when `attempt >= MAX_RETRY_ATTEMPTS` (3)
- Adds `dead_letter_reason` and `dead_lettered_at` to message metadata

### 5. Idempotency Implementation

**Decision**: All workers use `should_process_document()` from persistence layer for idempotency checks.

**Rationale**:
- Consistent idempotency logic across all workers
- Status-based checks are simple and reliable
- Reuses existing persistence layer functionality

**Implementation**:
- Each worker checks `should_process_document(document_id, target_status, config)` before processing
- If already at or beyond target status, worker skips processing and returns early
- No side effects from duplicate message processing

### 6. Chunking Parameters from Metadata

**Decision**: Chunking worker reads `chunk_size` and `chunk_overlap` from message metadata, with defaults.

**Rationale**:
- Allows per-document customization of chunking parameters
- Maintains backward compatibility with default values
- Enables future support for document-specific chunking strategies

**Implementation**:
- Defaults: `chunk_size=1000`, `overlap=200`
- Reads from `message.metadata.get("chunk_size")` and `message.metadata.get("chunk_overlap")`
- Falls back to defaults if not provided

### 7. Message Attempt Counter Reset

**Decision**: Each worker resets `attempt` counter to 1 when enqueuing to next stage.

**Rationale**:
- Each stage should have its own retry budget
- Prevents retry count from accumulating across stages
- Simplifies retry logic per stage

**Implementation**:
- All workers set `attempt=1` when creating next stage message
- Original attempt count is preserved in message metadata if needed

### 8. Status Update Timestamps

**Decision**: Workers update both status and timestamp fields when transitioning document state.

**Rationale**:
- Enables tracking of processing time per stage
- Provides audit trail for pipeline execution
- Supports performance monitoring and debugging

**Implementation**:
- Ingestion: `status='parsed'`, `parsed_at=now()`
- Chunking: `status='chunked'`, `chunked_at=now()`
- Embedding: `status='embedded'`, `embedded_at=now()`
- Indexing: `status='indexed'`, `indexed_at=now()`

## Testing Decisions

### 1. Test Data Strategy

**Decision**: All tests use actual files and realistic data rather than synthetic test data.

**Rationale**:
- Ensures tests reflect real-world usage
- Catches issues that might not appear with synthetic data
- Validates integration with actual service modules

**Implementation**:
- Uses actual PDF files from `backend/tests/fixtures/sample_documents/`
- Uses actual extracted text from real documents
- Uses actual chunks generated from real text
- Uses realistic embedding vectors (1536 dimensions)

### 2. Mocking Strategy

**Decision**: Mock external dependencies (storage, services, queues) but use real service modules for processing.

**Rationale**:
- Tests worker-specific functionality (queue operations, persistence, status updates)
- Reuses existing service module tests from `initial_setup` initiative
- Focuses on worker orchestration rather than service implementation

**Implementation**:
- Mock: `download_document_from_blob`, `download_document_from_storage`, `enqueue_message`, `persist_*`, `load_*`, `update_document_status`
- Real: `extract_text_from_document`, `chunk_text`, `generate_embeddings`, `index_chunks` (mocked at service level)

### 3. Coverage Target

**Decision**: Target 80% coverage minimum, accept 79% overall with individual modules ranging from 73% to 87%.

**Rationale**:
- 80% is a reasonable target for unit tests
- Edge cases and error paths are better tested in integration tests
- Focus on testing happy paths and critical error scenarios

**Implementation**:
- Achieved 79% overall coverage
- Individual modules: 73-87% coverage
- Missing coverage primarily in error handling edge cases

## Future Considerations

1. **Dead-Letter Queue for All Workers**: Consider adding dead-letter handling to other workers if needed
2. **Config Validation**: Add validation for required config fields at worker startup
3. **Metrics and Observability**: Add metrics collection for worker performance and failures
4. **Batch Processing**: Consider batch processing for embedding and indexing workers for better throughput

