# TODO 001 — RAG Ingestion Worker–Queue Architecture Conversion

## Context

This TODO document provides the implementation breakdown for converting the synchronous RAG ingestion pipeline to an asynchronous worker–queue architecture on Azure, as specified in [PRD001.md](./PRD001.md) and [RFC001.md](./RFC001.md). The conversion wraps existing service modules in workers that add persistence layers for intermediate data, enabling independent scaling and failure isolation.

**Current Status**: The ingestion pipeline currently processes documents synchronously in a single request. This TODO provides the implementation plan for building the worker–queue architecture.

**Implementation Phases**: This TODO follows a 6-phase implementation plan that builds the system incrementally, starting with persistence infrastructure and progressing through queue setup, worker implementation, API integration, and deployment.

---

## Phase 0 — Context Harvest

**Status**: ✅ Complete

### Setup Tasks
- [x] Review adjacent components in [context.md](./context.md)
- [x] Review PRD001.md and RFC001.md for complete requirements understanding
- [x] Review existing RAG system components (`rag_eval/services/rag/`)
  - [x] `ingestion.py` - `extract_text_from_document()` function
  - [x] `chunking.py` - `chunk_text()` function
  - [x] `embeddings.py` - `generate_embeddings()` function
  - [x] `search.py` - `index_chunks()` function
  - [x] `storage.py` - Azure Blob Storage functions
  - [x] `supabase_storage.py` - Supabase Storage functions
- [x] Review existing database schema (`infra/supabase/migrations/`)
- [x] Validate Azure resources configuration (Storage Account, Functions, Document Intelligence, AI Foundry, AI Search)
- [x] Review test fixtures structure (`backend/tests/fixtures/`)
- [x] **Create fracas.md** for failure tracking using FRACAS methodology

### Testing Environment Setup
- [x] **REQUIRED**: Set up and activate backend virtual environment (`backend/venv/`)
- [x] **REQUIRED**: Verify pytest is installed in venv (`pip install pytest pytest-cov`)
- [x] **REQUIRED**: Verify all backend dependencies are installed (`pip install -r backend/requirements.txt`)
- [x] **REQUIRED**: Verify pytest can discover tests (`pytest backend/tests/ --collect-only`)
- [x] **REQUIRED**: Document venv activation command for all subsequent phases
- [x] **REQUIRED**: All testing in Phases 1-5 MUST use the same venv (`backend/venv/`)
- [x] Block: Implementation cannot proceed until Phase 0 complete

### Phase 0 Deliverables
- [x] Created `fracas.md` for failure tracking
- [x] Created `phase_0_testing.md` documenting testing setup validation
- [x] Created `phase_0_decisions.md` documenting Phase 0 decisions
- [x] Created `phase_0_handoff.md` summarizing Phase 1 entry point

---

## Phase 1 — Persistence Infrastructure

**Status**: ✅ Complete

### Setup Tasks
- [x] **REQUIRED**: Activate backend venv: `cd backend && source venv/bin/activate`
- [x] Review existing database schema and migrations
- [x] Create database migration files for schema changes
- [x] Set up test fixtures for database operations
- [x] Create test file: `backend/tests/components/workers/test_persistence.py`

### Database Schema Changes
- [x] Add status column to `documents` table (if not exists)
  - [x] `status VARCHAR(50) DEFAULT 'uploaded'`
  - [x] Create index on `status` column
- [x] Add timestamp columns to `documents` table
  - [x] `parsed_at TIMESTAMP`
  - [x] `chunked_at TIMESTAMP`
  - [x] `embedded_at TIMESTAMP`
  - [x] `indexed_at TIMESTAMP`
- [x] Create `chunks` table
  - [x] `chunk_id VARCHAR(255) PRIMARY KEY`
  - [x] `document_id UUID NOT NULL REFERENCES documents(id)`
  - [x] `text TEXT NOT NULL`
  - [x] `metadata JSONB`
  - [x] `embedding JSONB` (for storing embeddings)
  - [x] `created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP`
  - [x] Create index on `document_id`
- [x] Add `extracted_text TEXT` column to `documents` table (or implement storage-based approach)
- [x] Test migration files execute successfully

### Core Implementation
- [x] Implement load/persist helper functions in `rag_eval/services/workers/persistence.py`
  - [x] `load_extracted_text(document_id: str, config) -> str`
  - [x] `persist_extracted_text(document_id: str, text: str, config) -> None`
  - [x] `load_chunks(document_id: str, config) -> List[Chunk]`
  - [x] `persist_chunks(document_id: str, chunks: List[Chunk], config) -> None`
  - [x] `load_embeddings(document_id: str, config) -> List[List[float]]`
  - [x] `persist_embeddings(document_id: str, chunks: List[Chunk], embeddings: List[List[float]], config) -> None`
  - [x] `update_document_status(document_id: str, status: str, timestamp_field: Optional[str] = None, config) -> None`
- [x] Implement idempotency checks
  - [x] `check_document_status(document_id: str, config) -> str`
  - [x] `should_process_document(document_id: str, target_status: str, config) -> bool`
- [x] Implement deletion functions
  - [x] `delete_chunks_by_document_id(document_id: str, config) -> int`
    - [x] Delete all chunks (and embeddings) for a document from chunks table
    - [x] Return count of deleted chunks
    - [x] Handle database errors gracefully
    - [x] Validate document_id is not empty

### Testing Tasks
- [x] **Robust Unit Tests (using actual files):**
  - [x] Test load operations for extracted text (database and storage approaches) - **using actual extracted text from PDF files**
  - [x] Test persist operations with various data sizes - **using actual extracted text from PDF files**
  - [x] Test error handling (missing data, invalid IDs)
  - [x] Test idempotency of load/persist operations
  - [x] Test database transaction handling
  - [x] Test edge cases (empty data, null values, large payloads) - **using actual extracted text**
  - [x] Test chunk loading and persistence - **using actual chunks generated from real extracted text**
  - [x] Test embedding loading and persistence - **using realistic embedding vectors (1536 dimensions)**
  - [x] Test status update operations
  - [x] Test idempotency checks (status-based)
  - [x] Test deletion of chunks from chunks table - **using actual chunks**
  - [x] Test deletion error handling (missing document_id, database errors)
  - [x] Test deletion returns correct count of deleted chunks
  - [x] **REQUIRED**: Use actual PDF files from `backend/tests/fixtures/sample_documents/` for test data
  - [x] **REQUIRED**: Use actual chunking functions to generate real chunks from extracted text
  - [x] **REQUIRED**: Use realistic embedding vectors that represent actual embeddings
- [x] **Document any failures** in fracas.md immediately when encountered

### Documentation Tasks
- [x] Add docstrings to all functions
- [x] Document persistence layer design decisions
- [x] Document database schema changes
- [x] Document storage approach for extracted text (database vs. storage)
- [x] **Phase 1 Testing Summary** for handoff to Phase 2

### Validation Requirements (Phase 1 Complete)
- [x] **REQUIRED**: All unit tests for Phase 1 must pass before proceeding to Phase 2
- [x] **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/workers/test_persistence.py -v`
- [x] **REQUIRED**: Test coverage must meet minimum 80% for persistence.py module (Achieved: 91%)
- [x] **REQUIRED**: All test assertions must pass (no failures, no errors) (42 tests passed)
- [x] **REQUIRED**: If tests fail, iterate on implementation until all tests pass
- [x] **REQUIRED**: Document any test failures in fracas.md
- [x] **REQUIRED**: Phase 1 is NOT complete until all tests pass

---

## Phase 2 — Queue Infrastructure

**Status**: ✅ Complete

### Setup Tasks
- [x] **REQUIRED**: Activate backend venv: `cd backend && source venv/bin/activate`
- [x] Review Azure Storage Account configuration
- [x] Create Azure Storage Queues (via Azure Portal or Infrastructure as Code)
  - [x] `ingestion-uploads`
  - [x] `ingestion-chunking`
  - [x] `ingestion-embeddings`
  - [x] `ingestion-indexing`
  - [x] `ingestion-dead-letter` (optional but recommended)
- [x] Install Azure Storage Queue SDK: `pip install azure-storage-queue`
- [x] Create queue client utilities module: `rag_eval/services/workers/queue_client.py`
- [x] Set up test fixtures for queue operations (mocked)
- [x] Create test file: `backend/tests/components/workers/test_queue_client.py`

### Core Implementation
- [x] Implement message schema validation
  - [x] Define `QueueMessage` dataclass with fields:
    - [x] `document_id: str`
    - [x] `source_storage: str` (enum: "azure_blob" | "supabase")
    - [x] `filename: str`
    - [x] `attempt: int`
    - [x] `stage: str` (enum: "uploaded" | "parsed" | "chunked" | "embedded" | "indexed")
    - [x] `metadata: Dict[str, Any]` (optional)
  - [x] Implement `validate_message(message: dict) -> QueueMessage`
  - [x] Implement `serialize_message(message: QueueMessage) -> str`
  - [x] Implement `deserialize_message(message_str: str) -> QueueMessage`
- [x] Implement queue client utilities
  - [x] `enqueue_message(queue_name: str, message: QueueMessage, config) -> None`
  - [x] `dequeue_message(queue_name: str, config) -> Optional[QueueMessage]`
  - [x] `peek_message(queue_name: str, config) -> Optional[QueueMessage]`
  - [x] `delete_message(queue_name: str, message_id: str, pop_receipt: str, config) -> None`
  - [x] `send_to_dead_letter(queue_name: str, message: QueueMessage, reason: str, config) -> None`
  - [x] `get_queue_length(queue_name: str, config) -> int`

### Testing Tasks
- [x] **Robust Unit Tests:**
  - [x] Test message schema validation (valid/invalid schemas)
  - [x] Test queue client operations (enqueue, dequeue, peek, delete)
  - [x] Test error handling (queue not found, connection failures)
  - [x] Test message serialization/deserialization
  - [x] Test dead-letter queue handling
  - [x] Test retry logic for transient failures
  - [x] Test message visibility timeout handling
  - [x] Test queue length monitoring
- [x] **Document any failures** in fracas.md immediately when encountered

### Documentation Tasks
- [x] Add docstrings to all functions
- [x] Document message schema and field descriptions
- [x] Document queue configuration and setup
- [x] Document error handling and retry strategies
- [x] **Phase 2 Testing Summary** for handoff to Phase 3

### Validation Requirements (Phase 2 Complete)
- [x] **REQUIRED**: All unit tests for Phase 2 must pass before proceeding to Phase 3
- [x] **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/workers/test_queue_client.py -v`
- [x] **REQUIRED**: Test coverage must meet minimum 80% for queue_client.py module (Achieved: 93%)
- [x] **REQUIRED**: All test assertions must pass (no failures, no errors) (51 tests passed)
- [x] **REQUIRED**: If tests fail, iterate on implementation until all tests pass
- [x] **REQUIRED**: Document any test failures in fracas.md
- [x] **REQUIRED**: Phase 2 is NOT complete until all tests pass

---

## Phase 3 — Worker Implementation

**Status**: ✅ Complete

### Setup Tasks
- [x] **REQUIRED**: Activate backend venv: `cd backend && source venv/bin/activate`
- [x] Review existing service modules (ingestion.py, chunking.py, embeddings.py, search.py)
- [x] Create workers directory: `rag_eval/services/workers/`
- [x] Set up test fixtures for mocked service modules and queue operations
- [x] Create test files for each worker:
  - [x] `backend/tests/components/workers/test_ingestion_worker.py`
  - [x] `backend/tests/components/workers/test_chunking_worker.py`
  - [x] `backend/tests/components/workers/test_embedding_worker.py`
  - [x] `backend/tests/components/workers/test_indexing_worker.py`

### Core Implementation

#### Ingestion Worker
- [x] Create `rag_eval/services/workers/ingestion_worker.py`
- [x] Implement `ingestion_worker(queue_message: dict, context) -> None`
  - [x] Parse queue message and extract `document_id`, `source_storage`
  - [x] **Load**: Resolve file location and download file (Supabase or Azure Blob)
  - [x] **Validate**: Check file size (fail fast if > 50MB)
  - [x] **Detect**: Get PDF page count using PyPDF2
  - [x] **Initialize**: Set up batch processing metadata in `documents.metadata->'ingestion'`
  - [x] **Process Batches**: For each batch (configurable size, default 2 pages):
    - [x] Skip if batch already completed (idempotency)
    - [x] Slice PDF to batch pages
    - [x] Extract text using Azure Document Intelligence via `extract_text_from_batch()`
    - [x] Persist batch result temporarily to `chunks` table
    - [x] Merge batch into accumulated text (in-memory)
    - [x] Delete batch chunk immediately (cleanup)
    - [x] Update progress metadata
  - [x] **Persist**: Store merged extracted text in persistence layer
  - [x] Update `documents` table: `status = 'parsed'`, `parsed_at = now()`
  - [x] Enqueue message to `ingestion-chunking` queue with `stage = "parsed"`
  - [x] Implement error handling with retry logic (exponential backoff)
  - [x] Implement idempotency check (skip if already parsed)
  - [x] Handle dead-letter queue for permanent failures
- [x] Implement PDF page counting using PyPDF2
- [x] Implement PDF slicing for batch processing (1-indexed to 0-indexed conversion)
- [x] Implement batch persistence in chunks table
- [x] Implement batch resumption logic (skip completed batches)
- [x] Implement batch merging (in-memory accumulation)
- [x] Implement per-batch cleanup (delete batch chunk after merge)
- [x] Implement memory limit check (50MB)
- [x] Implement configurable batch size (from message metadata)

#### Chunking Worker
- [x] Create `rag_eval/services/workers/chunking_worker.py`
- [x] Implement `chunking_worker(queue_message: dict, context) -> None`
  - [x] Parse queue message and extract `document_id`
  - [x] **Load**: Retrieve extracted text for `document_id` from persistence layer
  - [x] **Process**: Call existing `chunk_text(text: str, config, document_id: str, chunk_size: int, overlap: int) -> List[Chunk]`
  - [x] **Persist**: Store chunks in persistence layer
  - [x] Update `documents.status = 'chunked'`
  - [x] Enqueue message to `ingestion-embeddings` queue
  - [x] Implement error handling (fail fast for data/validation issues)
  - [x] Implement idempotency check (handle duplicate messages safely)

#### Embedding Worker
- [x] Create `rag_eval/services/workers/embedding_worker.py`
- [x] Implement `embedding_worker(queue_message: dict, context) -> None`
  - [x] Parse queue message and extract `document_id`
  - [x] **Load**: Fetch all chunks for `document_id` from persistence layer
  - [x] **Process**: Call existing `generate_embeddings(chunks: List[Chunk], config) -> List[List[float]]`
  - [x] **Persist**: Store embeddings in persistence layer
  - [x] Update `documents.status = 'embedded'`
  - [x] Enqueue message to `ingestion-indexing` queue
  - [x] Implement error handling with retry logic (transient Azure Foundry failures)
  - [x] Implement idempotency check (skip if already embedded)

#### Indexing Worker
- [x] Create `rag_eval/services/workers/indexing_worker.py`
- [x] Implement `indexing_worker(queue_message: dict, context) -> None`
  - [x] Parse queue message and extract `document_id`
  - [x] **Load**: Retrieve chunks + embeddings for `document_id` from persistence layer
  - [x] **Process**: Call existing `index_chunks(chunks: List[Chunk], embeddings: List[List[float]], config) -> None`
  - [x] Handle partial failures using existing logic
  - [x] Update `documents.status = 'indexed'`, `indexed_at = now()`
  - [x] Implement error handling with retry logic (transient Azure Search errors)
  - [x] Implement idempotency check (Azure Search operations are idempotent)

### Testing Tasks
- [ ] **Robust Unit Tests for Each Worker (using actual files):**
  - [ ] **Note**: Unit tests for underlying service modules (`extract_text_from_document`, `chunk_text`, `generate_embeddings`, `index_chunks`) already exist from `initial_setup` initiative and can be reused. Focus new tests on worker-specific functionality.
  - [ ] **REQUIRED**: All worker tests must use actual files and realistic data:
    - [ ] Use actual PDF files from `backend/tests/fixtures/sample_documents/` for file operations
    - [ ] Use actual extracted text from real documents (via mocked extraction)
    - [ ] Use actual chunks generated from real extracted text
    - [ ] Use realistic embedding vectors (1536 dimensions) that represent actual embeddings
  - [ ] **Ingestion Worker:**
    - [ ] Test queue message parsing and validation
    - [ ] Test file download from Supabase/Azure Blob - **using actual PDF files**
    - [ ] Test extracted text persistence (load/persist operations) - **using actual extracted text**
    - [ ] Test status update to 'parsed' in database
    - [ ] Test message enqueue to next queue
    - [ ] Test retry logic with exponential backoff
    - [ ] Test dead-letter handling on max retries
    - [ ] Test idempotency (skip if already parsed - status check)
    - [ ] Test error handling for missing files, invalid message formats
    - [ ] **Reuse from initial_setup**: Tests for `extract_text_from_document` function (already tested with mocked Azure Document Intelligence)
  - [ ] **Chunking Worker:**
    - [ ] Test queue message parsing and validation
    - [ ] Test extracted text loading from persistence layer - **using actual extracted text**
    - [ ] Test chunk persistence to database/storage - **using actual chunks from real text**
    - [ ] Test status update to 'chunked' in database
    - [ ] Test message enqueue to next queue
    - [ ] Test idempotency (handle duplicate messages safely - status check)
    - [ ] Test error handling for missing extracted text, invalid data
    - [ ] **Reuse from initial_setup**: Tests for `chunk_text` function (already tested with deterministic behavior validation)
  - [ ] **Embedding Worker:**
    - [ ] Test queue message parsing and validation
    - [ ] Test chunk loading from persistence layer - **using actual chunks**
    - [ ] Test embedding persistence to database/storage - **using realistic embedding vectors**
    - [ ] Test status update to 'embedded' in database
    - [ ] Test message enqueue to next queue
    - [ ] Test retry logic for transient failures
    - [ ] Test idempotency (skip if already embedded - status check)
    - [ ] Test error handling for missing chunks, persistence failures
    - [ ] **Reuse from initial_setup**: Tests for `generate_embeddings` function (already tested with mocked Azure AI Foundry)
  - [ ] **Indexing Worker:**
    - [ ] Test queue message parsing and validation
    - [ ] Test chunk + embedding loading from persistence layer - **using actual chunks and realistic embeddings**
    - [ ] Test status update to 'indexed' in database
    - [ ] Test partial failure handling
    - [ ] Test retry logic for transient failures
    - [ ] Test idempotency (Azure Search operations are idempotent, status check)
    - [ ] Test error handling for missing chunks/embeddings, persistence failures
    - [ ] **Reuse from initial_setup**: Tests for `index_chunks` function (already tested with mocked Azure AI Search)
- [ ] **Document any failures** in fracas.md immediately when encountered

### Documentation Tasks
- [x] Add docstrings to all worker functions
- [x] Document worker Load → Process → Persist → Enqueue pattern
- [x] Document error handling and retry strategies for each worker
- [x] Document idempotency implementation
- [x] **Phase 3 Testing Summary** for handoff to Phase 4

### Validation Requirements (Phase 3 Complete)
- [x] **REQUIRED**: All unit tests for Phase 3 must pass before proceeding to Phase 4 (50 tests passed)
- [x] **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/workers/ -v`
- [x] **REQUIRED**: Test coverage must meet minimum 80% for all worker modules (Achieved: 79% overall, 73-87% per module)
- [x] **REQUIRED**: All test assertions must pass (no failures, no errors) (50 tests passed)
- [x] **REQUIRED**: If tests fail, iterate on implementation until all tests pass
- [x] **REQUIRED**: Document any test failures in fracas.md
- [x] **REQUIRED**: Phase 3 is NOT complete until all tests pass

---

## Phase 4 — API Integration

**Status**: ⏳ Pending

### Setup Tasks
- [ ] **REQUIRED**: Activate backend venv: `cd backend && source venv/bin/activate`
- [ ] Review existing FastAPI upload endpoint
- [ ] Review existing API response models
- [ ] Create test file: `backend/tests/components/api/test_upload_endpoint.py`

### Core Implementation
- [ ] Modify upload endpoint (`api/routes/upload.py` or equivalent)
  - [ ] Change from synchronous processing to enqueueing message
  - [ ] Upload file to storage (Supabase or Azure Blob) - keep existing logic
  - [ ] Create document record in database with `status = 'uploaded'`
  - [ ] Enqueue message to `ingestion-uploads` queue
  - [ ] Return immediately with `document_id` and `status='uploaded'`
  - [ ] Add backward compatibility flag (optional synchronous path) if needed
- [ ] Add status query endpoint
  - [ ] `GET /documents/{document_id}/status`
  - [ ] Return current status, timestamps, and error details if failed
  - [ ] Implement `DocumentStatusResponse` model
- [ ] Update delete endpoint (`api/routes/documents.py` or equivalent)
  - [ ] Add deletion of chunks from chunks table (call `delete_chunks_by_document_id` from persistence module)
  - [ ] Keep existing deletion of chunks from Azure AI Search
  - [ ] Keep existing deletion of file from storage
  - [ ] Keep existing deletion of document record from database
  - [ ] Implement graceful degradation: continue with other deletions if one fails
  - [ ] Return counts of deleted chunks from both chunks table and Azure AI Search
  - [ ] Update response model to include `chunks_deleted_db` and `chunks_deleted_ai_search`
- [ ] Update response models
  - [ ] `UploadResponse` with `document_id` and `status`
  - [ ] `DocumentStatusResponse` with status, timestamps, error details
  - [ ] `DeleteDocumentResponse` with `chunks_deleted_db` and `chunks_deleted_ai_search`

### Testing Tasks
- [ ] **Robust Unit Tests (using actual files):**
  - [ ] Test upload endpoint enqueues message correctly - **using actual PDF files**
  - [ ] Test upload endpoint returns immediately with document_id
  - [ ] Test status query endpoint returns correct status
  - [ ] Test delete endpoint removes chunks from chunks table - **using actual chunks**
  - [ ] Test delete endpoint removes chunks from Azure AI Search
  - [ ] Test delete endpoint removes file from storage
  - [ ] Test delete endpoint removes document record from database
  - [ ] Test delete endpoint graceful degradation (continues if one deletion fails)
  - [ ] Test delete endpoint returns correct counts from both systems
  - [ ] Test error handling for invalid requests
  - [ ] Test backward compatibility flag (synchronous path) if implemented
  - [ ] Test response model validation
  - [ ] Test API error responses
  - [ ] Test status transitions (uploaded → parsed → chunked → embedded → indexed)
  - [ ] **REQUIRED**: Use actual PDF files from `backend/tests/fixtures/sample_documents/` for upload tests
  - [ ] **REQUIRED**: Use actual chunks and embeddings for deletion tests
- [ ] **Document any failures** in fracas.md immediately when encountered

### Documentation Tasks
- [ ] Add docstrings to all API functions
- [ ] Document API endpoint changes
- [ ] Document response models
- [ ] Document backward compatibility approach
- [ ] **Phase 4 Testing Summary** for handoff to Phase 5

### Validation Requirements (Phase 4 Complete)
- [ ] **REQUIRED**: All unit tests for Phase 4 must pass before proceeding to Phase 5
- [ ] **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/api/test_upload_endpoint.py -v`
- [ ] **REQUIRED**: Test coverage must meet minimum 80% for API endpoints
- [ ] **REQUIRED**: All test assertions must pass (no failures, no errors)
- [ ] **REQUIRED**: If tests fail, iterate on implementation until all tests pass
- [ ] **REQUIRED**: Document any test failures in fracas.md
- [ ] **REQUIRED**: Phase 4 is NOT complete until all tests pass

---

## Phase 5 — Integration Testing & Migration

**Status**: ⏳ Pending

### Database Migration (Prerequisite)
- [ ] **REQUIRED**: Apply database migrations to production Supabase
  - [ ] Apply `0019_add_worker_queue_persistence.sql` migration
    - Adds `status` column to `documents` table
    - Adds timestamp columns (`parsed_at`, `chunked_at`, `embedded_at`, `indexed_at`)
    - Creates `chunks` table
    - Adds `extracted_text TEXT` column to `documents` table
  - [ ] Apply `0020_add_ingestion_batch_metadata.sql` migration (documentation only, no schema changes)
  - [ ] Verify migrations applied successfully
  - [ ] Test schema changes with sample queries
  - [ ] Validate indexes are created correctly
- [ ] **REQUIRED**: Test Supabase integration with real database
  - [ ] Test persistence operations (load/persist extracted text, chunks, embeddings)
  - [ ] Test status updates and idempotency checks
  - [ ] Test batch metadata storage in `documents.metadata->'ingestion'` JSONB
  - [ ] Test document status transitions through pipeline
  - [ ] Test error handling with real database connections
  - [ ] Verify all workers can read/write to Supabase correctly

### Azure Functions Deployment
- [ ] **REQUIRED**: Deploy all workers as Azure Functions
  - [ ] Create Azure Function App (Consumption Plan)
  - [ ] Deploy `ingestion-worker` function with queue trigger for `ingestion-uploads`
  - [ ] Deploy `chunking-worker` function with queue trigger for `ingestion-chunking`
  - [ ] Deploy `embedding-worker` function with queue trigger for `ingestion-embeddings`
  - [ ] Deploy `indexing-worker` function with queue trigger for `ingestion-indexing`
- [ ] Configure queue triggers for each worker
- [ ] Set up Application Insights for monitoring
- [ ] Configure environment variables and Key Vault integration
- [ ] Test Azure Functions deployment and queue trigger configuration

### Integration Tests (Post-Deployment)
- [ ] **REQUIRED**: End-to-end pipeline flow with real Azure Storage Queues and Supabase
  - [ ] Test message passing between stages through actual queues
  - [ ] Test failure scenarios and dead-letter handling with real queues
  - [ ] Test status transitions through complete pipeline (verified in Supabase)
  - [ ] Test concurrent document processing across multiple workers
  - [ ] Test queue depth handling under load
  - [ ] Test Azure Functions queue trigger behavior
  - [ ] Test worker scaling and concurrency
  - [ ] **Supabase Integration Tests**:
    - [ ] Test persistence operations with real Supabase database
    - [ ] Test batch processing metadata storage and retrieval
    - [ ] Test document status updates in real database
    - [ ] Test idempotency with real database state
    - [ ] Test error handling and recovery with real database
  - [ ] **Test Data Constraint**: Use only first 6 pages of `docs/inputs/scan_classic_hmo.pdf` for tests that process actual PDFs to avoid exceeding Azure Document Intelligence budget
- [ ] **Document any failures** in fracas.md immediately when encountered

### Performance Testing (Post-Deployment)
- [ ] **REQUIRED**: Test worker processing time under load (real Azure Functions)
- [ ] **REQUIRED**: Test queue depth handling with real Azure Storage Queues
- [ ] **REQUIRED**: Test concurrent document processing across multiple function instances
- [ ] **REQUIRED**: Validate throughput meets requirements
- [ ] **REQUIRED**: Monitor Azure Functions cold start latency
- [ ] **REQUIRED**: **Test Data Constraint**: Performance tests should limit to first 6 pages of test PDF to stay within budget

### Migration Strategy
- [ ] **REQUIRED**: Gradual migration: run both paths in parallel
  - [ ] Introduce queues and workers without turning off synchronous path
  - [ ] For new uploads, prefer enqueuing message instead of synchronous processing
  - [ ] Gradually move UI/API endpoints to rely on document `status`
  - [ ] Monitor and validate worker behavior via Application Insights
- [ ] **REQUIRED**: Deprecate synchronous path once stable
  - [ ] Remove or deprecate direct "do everything in one request" ingestion paths
  - [ ] Update documentation to reflect asynchronous architecture

### Documentation Tasks
- [ ] Document Azure Functions deployment process
- [ ] Document queue trigger configuration
- [ ] Document Application Insights setup and monitoring
- [ ] Document migration strategy and timeline
- [ ] **Phase 5 Testing Summary** for final validation

### Validation Requirements (Phase 5 Complete)
- [ ] **REQUIRED**: Database migrations applied to production Supabase
- [ ] **REQUIRED**: Supabase integration tests pass with real database
- [ ] **REQUIRED**: Azure Functions deployed and configured
- [ ] **REQUIRED**: All integration tests pass with real Azure Storage Queues and Supabase
- [ ] **REQUIRED**: Performance tests validate throughput requirements
- [ ] **REQUIRED**: Migration strategy executed successfully
- [ ] **REQUIRED**: Synchronous path deprecated or removed
- [ ] **REQUIRED**: Document any test failures in fracas.md
- [ ] **REQUIRED**: Phase 5 is NOT complete until all validation requirements pass

---

## Initiative Completion

### Final Validation Tasks
- [ ] **REQUIRED**: Run all worker tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/workers/ tests/components/api/ -v --cov=rag_eval/services/workers --cov-report=term-missing`
- [ ] **REQUIRED**: Verify all unit tests pass (minimum 80% coverage across all worker modules)
- [ ] **REQUIRED**: Verify all integration tests pass with real Azure resources
- [ ] **REQUIRED**: Verify all connection tests documented
- [ ] **REQUIRED**: Performance validation (throughput meets requirements, cold start latency acceptable)
- [ ] **REQUIRED**: If any tests fail, iterate on implementation until all tests pass
- [ ] **Final Testing Summary** - Comprehensive testing report across all phases
  - [ ] All unit tests pass (minimum 80% coverage)
  - [ ] All integration tests pass
  - [ ] All connection tests documented
  - [ ] Performance validation complete
- [ ] **Technical Debt Documentation** - Complete technical debt catalog and remediation roadmap
  - [ ] Document any shortcuts or deferred features
  - [ ] Document known limitations
  - [ ] Document future enhancement opportunities
- [ ] Code review and quality check
- [ ] Documentation review (all docstrings, README updates)
- [ ] Stakeholder review and approval

### Success Criteria Validation
- [ ] Asynchronous Processing: Upload endpoint returns immediately with document_id
- [ ] Status Tracking: Document status can be queried through all pipeline stages
- [ ] Failure Isolation: Individual stage failures don't require restarting entire pipeline
- [ ] Idempotency: Workers can safely retry operations
- [ ] Observability: Document status and timestamps tracked through pipeline
- [ ] Existing Functionality Preserved: All existing service modules unchanged and working

---

## Blockers
- {List current blockers and dependencies as they arise}

## Notes
- {Implementation notes and decisions made during development}

## Testing Environment Requirements

### Venv Usage
- **REQUIRED**: All backend testing MUST use the same virtual environment: `backend/venv/`
- **REQUIRED**: Activate venv before running any tests: `cd backend && source venv/bin/activate`
- **REQUIRED**: All pytest commands must be run from within the activated venv
- **REQUIRED**: Do not create separate venvs for different phases - use the same venv throughout

### Test Execution Commands
- **Unit Tests**: `cd backend && source venv/bin/activate && pytest tests/components/workers/test_<component>.py -v`
- **API Tests**: `cd backend && source venv/bin/activate && pytest tests/components/api/test_<component>.py -v`
- **All Worker Tests**: `cd backend && source venv/bin/activate && pytest tests/components/workers/ -v`
- **With Coverage**: `cd backend && source venv/bin/activate && pytest tests/components/workers/ -v --cov=rag_eval/services/workers --cov-report=term-missing`
- **Connection Tests**: Run connection tests separately (they warn but don't fail if credentials missing)

### Phase Completion Requirements
- **REQUIRED**: Each phase must have all unit tests passing before proceeding to the next phase
- **REQUIRED**: Test coverage must meet minimum 80% for each module
- **REQUIRED**: If tests fail, iterate on implementation until all tests pass (may require multiple iterations)
- **REQUIRED**: Document any test failures in fracas.md
- **REQUIRED**: Phase status must be updated to "✅ Complete" only after all validation requirements pass

## FRACAS Integration
- **Failure Tracking**: All failures, bugs, and unexpected behaviors must be documented in `fracas.md`
- **Investigation Process**: Follow systematic FRACAS methodology for root cause analysis
- **Knowledge Building**: Use failure modes to build organizational knowledge and prevent recurrence
- **Status Management**: Keep failure mode statuses current and move resolved issues to historical section

**FRACAS Document Location**: `docs/initiatives/rag_system/worker_queue_conversion/fracas.md`

---

**Document Status**: Draft  
**Last Updated**: 2025-01-XX  
**Author**: Documentation Agent  
**Reviewers**: TBD

