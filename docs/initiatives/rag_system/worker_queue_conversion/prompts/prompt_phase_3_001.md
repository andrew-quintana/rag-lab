# Phase 3 Prompt — Worker Implementation

## Context

This prompt guides the implementation of **Phase 3: Worker Implementation** for the RAG Ingestion Worker–Queue Architecture Conversion. This phase implements all four workers (ingestion, chunking, embedding, indexing) that wrap existing service modules with persistence and queue operations.

**Related Documents:**
- @docs/initiatives/rag_system/worker_queue_conversion/scoping/PRD001.md - Product requirements
- @docs/initiatives/rag_system/worker_queue_conversion/scoping/RFC001.md - Technical design (Worker Architecture section)
- @docs/initiatives/rag_system/worker_queue_conversion/scoping/TODO001.md - Implementation tasks (Phase 3 section - check off tasks as completed)
- @docs/initiatives/rag_system/worker_queue_conversion/scoping/context.md - Project context

## Objectives

1. **Ingestion Worker**: Implement worker that downloads files, extracts text, persists results, and enqueues to next stage
2. **Chunking Worker**: Implement worker that loads extracted text, chunks it, persists chunks, and enqueues to next stage
3. **Embedding Worker**: Implement worker that loads chunks, generates embeddings, persists embeddings, and enqueues to next stage
4. **Indexing Worker**: Implement worker that loads chunks and embeddings, indexes them, and updates final status
5. **Testing**: Create comprehensive unit tests for each worker

## Execution Requirements

### Checklist Modifications
- **REQUIRED**: Check off completed tasks in @docs/initiatives/rag_system/worker_queue_conversion/scoping/TODO001.md Phase 3 section as you complete them
- Update phase status when all validation requirements are met

### Validation
- **REQUIRED**: All unit tests for Phase 3 must pass before proceeding to Phase 4
- **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/workers/ -v`
- **REQUIRED**: Test coverage must meet minimum 80% for all worker modules
- **REQUIRED**: All test assertions must pass (no failures, no errors)

### Documentation
- **REQUIRED**: Create `phase_3_decisions.md` if any decisions are made that aren't in PRD/RFC
- **REQUIRED**: Create `phase_3_testing.md` documenting testing results
- **REQUIRED**: Create `phase_3_handoff.md` summarizing what's needed for Phase 4

## Key References

### Existing Service Modules (Unchanged)
- @backend/rag_eval/services/rag/ingestion.py - `extract_text_from_document(file_content: bytes, config) -> str`
- @backend/rag_eval/services/rag/chunking.py - `chunk_text(text: str, config, document_id: str, chunk_size: int, overlap: int) -> List[Chunk]`
- @backend/rag_eval/services/rag/embeddings.py - `generate_embeddings(chunks: List[Chunk], config) -> List[List[float]]`
- @backend/rag_eval/services/rag/search.py - `index_chunks(chunks: List[Chunk], embeddings: List[List[float]], config) -> None`
- @backend/rag_eval/services/rag/storage.py - Azure Blob Storage functions
- @backend/rag_eval/services/rag/supabase_storage.py - Supabase Storage functions

### Persistence Layer
- @backend/rag_eval/services/workers/persistence.py - Load/persist functions from Phase 1

### Queue Client
- @backend/rag_eval/services/workers/queue_client.py - Queue operations from Phase 2

### Worker Pattern
- **Load → Process → Persist → Enqueue**: Each worker follows this pattern
- **Idempotency**: Workers check document status before processing
- **Error Handling**: Retry logic with exponential backoff for transient failures

## Phase 3 Tasks

### Ingestion Worker
1. Create `rag_eval/services/workers/ingestion_worker.py`
2. Implement `ingestion_worker(queue_message: dict, context) -> None`:
   - Parse queue message and extract `document_id`, `source_storage`
   - **Load**: Resolve file location and download file (Supabase or Azure Blob)
   - **Process**: Call existing `extract_text_from_document(file_content: bytes, config) -> str`
   - **Persist**: Store extracted text in persistence layer
   - Update `documents` table: `status = 'parsed'`, `parsed_at = now()`
   - Enqueue message to `ingestion-chunking` queue with `stage = "parsed"`
   - Implement error handling with retry logic (exponential backoff)
   - Implement idempotency check (skip if already parsed)
   - Handle dead-letter queue for permanent failures

### Chunking Worker
1. Create `rag_eval/services/workers/chunking_worker.py`
2. Implement `chunking_worker(queue_message: dict, context) -> None`:
   - Parse queue message and extract `document_id`
   - **Load**: Retrieve extracted text for `document_id` from persistence layer
   - **Process**: Call existing `chunk_text(text: str, config, document_id: str, chunk_size: int, overlap: int) -> List[Chunk]`
   - **Persist**: Store chunks in persistence layer
   - Update `documents.status = 'chunked'`
   - Enqueue message to `ingestion-embeddings` queue
   - Implement error handling (fail fast for data/validation issues)
   - Implement idempotency check (handle duplicate messages safely)

### Embedding Worker
1. Create `rag_eval/services/workers/embedding_worker.py`
2. Implement `embedding_worker(queue_message: dict, context) -> None`:
   - Parse queue message and extract `document_id`
   - **Load**: Fetch all chunks for `document_id` from persistence layer
   - **Process**: Call existing `generate_embeddings(chunks: List[Chunk], config) -> List[List[float]]`
   - **Persist**: Store embeddings in persistence layer
   - Update `documents.status = 'embedded'`
   - Enqueue message to `ingestion-indexing` queue
   - Implement error handling with retry logic (transient Azure Foundry failures)
   - Implement idempotency check (skip if already embedded)

### Indexing Worker
1. Create `rag_eval/services/workers/indexing_worker.py`
2. Implement `indexing_worker(queue_message: dict, context) -> None`:
   - Parse queue message and extract `document_id`
   - **Load**: Retrieve chunks + embeddings for `document_id` from persistence layer
   - **Process**: Call existing `index_chunks(chunks: List[Chunk], embeddings: List[List[float]], config) -> None`
   - Handle partial failures using existing logic
   - Update `documents.status = 'indexed'`, `indexed_at = now()`
   - Implement error handling with retry logic (transient Azure Search errors)
   - Implement idempotency check (Azure Search operations are idempotent)

### Testing
1. Create test files for each worker:
   - `backend/tests/components/workers/test_ingestion_worker.py`
   - `backend/tests/components/workers/test_chunking_worker.py`
   - `backend/tests/components/workers/test_embedding_worker.py`
   - `backend/tests/components/workers/test_indexing_worker.py`
2. **Note**: Unit tests for underlying service modules already exist from `initial_setup` initiative and can be reused. Focus new tests on worker-specific functionality:
   - Queue message parsing and validation
   - Load/persist operations
   - Status updates
   - Message enqueuing to next queue
   - Retry logic and error handling
   - Idempotency checks

### Documentation
1. Add docstrings to all worker functions
2. Document worker Load → Process → Persist → Enqueue pattern
3. Document error handling and retry strategies for each worker
4. Document idempotency implementation

## Success Criteria

- [ ] All four workers implemented
- [ ] All workers follow Load → Process → Persist → Enqueue pattern
- [ ] All unit tests pass (minimum 80% coverage)
- [ ] All Phase 3 tasks in TODO001.md checked off
- [ ] Phase 3 handoff document created

## Important Notes

- **Service Modules Unchanged**: All existing service modules (`extract_text_from_document`, `chunk_text`, `generate_embeddings`, `index_chunks`) remain unchanged
- **Worker Wrapping**: Workers wrap existing functions with persistence and queue operations
- **Test Reuse**: Reuse existing service module tests from `initial_setup` initiative
- **Idempotency**: Workers check document status before processing to enable safe retries
- **Error Handling**: Implement retry logic with exponential backoff for transient failures

## Blockers

- **BLOCKER**: Phase 4 cannot proceed until Phase 3 validation complete
- **BLOCKER**: Persistence layer (Phase 1) and queue client (Phase 2) must be complete

## Next Phase

After completing Phase 3, proceed to **Phase 4: API Integration** using @docs/initiatives/rag_system/worker_queue_conversion/prompts/prompt_phase_4_001.md

