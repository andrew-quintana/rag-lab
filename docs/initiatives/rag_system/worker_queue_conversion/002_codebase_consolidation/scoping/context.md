# CONTEXT 002 — Codebase Consolidation & Production Readiness

## 1. Purpose

This document defines the context for Initiative 002: Codebase Consolidation and Production Readiness. This initiative focuses on consolidating the codebase structure, eliminating duplication, refining the deployment process, and ensuring the worker-queue architecture is production-ready through comprehensive local and cloud testing.

**Success Criteria:**
- Codebase structure consolidated (eliminate duplication, improve organization)
- All Phase 5 tests pass in both local and cloud environments
- Azure Functions process queue messages correctly in cloud
- Performance requirements validated
- Production deployment verified and stable
- All identified issues from Phase 5 testing resolved
- Deployment process streamlined and documented

---

## 2. Scope

### In Scope
- **Codebase Consolidation**:
  - Eliminate duplicate backend code in Azure Functions directory
  - Consolidate configuration management (environment variables, settings)
  - Streamline deployment process and build scripts
  - Improve code organization and structure
  - Consolidate test infrastructure and utilities
  
- **Phase 5 Local Testing**: Complete test suite execution in local environment (Azurite + local Supabase)
- **Phase 5 Cloud Testing**: Complete test suite execution with deployed Azure Functions
- **Test Infrastructure**: Local development environment setup (Azurite, local Supabase, dotenv configuration)
- **Queue & Worker Validation**: End-to-end pipeline validation with real Azure resources
- **Performance Testing**: Throughput and latency validation with deployed functions
- **Issue Resolution**: Fixing any bugs or issues discovered during Phase 5 testing
- **Production Readiness**: Ensuring system is ready for production use
- **Deployment Optimization**: Streamline Git-based deployment and build process

### Out of Scope
- Changes to core worker logic (already implemented in 001)
- Changes to message schema (already defined in 001)
- Changes to database schema (migrations already applied)
- Query-time RAG pipeline modifications (unchanged per 001)
- New feature development (focus is on consolidation and testing)
- Major architectural changes (consolidation only, not redesign)

---

## 3. Current State (From Initiative 001)

### 3.1 Architecture Implemented

**Worker-Queue Pipeline:**
- ✅ 4 workers implemented: ingestion, chunking, embedding, indexing
- ✅ 5 queues configured: ingestion-uploads, ingestion-chunking, ingestion-embeddings, ingestion-indexing, ingestion-dead-letter
- ✅ Message schema defined and validated
- ✅ Status transitions implemented (uploaded → parsed → chunked → embedded → indexed)
- ✅ Idempotency checks implemented
- ✅ Persistence operations implemented

**Azure Functions Deployment:**
- ✅ Function App created: `func-raglab-uploadworkers`
- ✅ All 4 functions deployed
- ✅ Queue triggers configured
- ✅ Environment variables configured (partially)

**Database:**
- ✅ Migrations 0019 and 0020 applied
- ✅ Status columns and timestamps added
- ✅ Chunks table created
- ✅ Indexes created

### 3.2 Codebase Duplication Issues

**Current State:**
- ⚠️ Duplicate backend code exists in `infra/azure/azure_functions/backend/rag_eval/`
- ⚠️ Build script (`build.sh`) copies `backend/rag_eval/` to deployment package during Git deployment
- ⚠️ Functions currently reference copied code via path manipulation
- ⚠️ Two sources of truth for backend code (source in `backend/rag_eval/` vs. copied in `infra/azure/azure_functions/backend/rag_eval/`)

**Impact:**
- Code duplication increases maintenance burden
- Changes must be synchronized between source and deployment
- Risk of code drift between source and copied code
- Build process unnecessarily copies entire backend directory
- Deployment package size larger than necessary
- Confusion about which code is the source of truth

**Consolidation Goals:**
- Eliminate duplicate code in Azure Functions directory
- Functions should import directly from project root `backend/rag_eval/` using proper path resolution
- Simplify build process to eliminate code copying step
- Single source of truth for all backend code
- Reduce deployment package size
- Improve code maintainability and clarity

### 3.3 Phase 5 Local Testing Results

**Test Summary** (from `001/intermediate/PHASE_5_LOCAL_TEST_RESULTS.md`):

| Test Suite | Total | Passed | Failed | Skipped | Status |
|------------|-------|--------|--------|---------|--------|
| Migration Verification | 2 | 2 | 0 | 0 | ✅ PASSED |
| Supabase Integration | 13 | 12 | 1 | 0 | ⚠️ 92% PASSED |
| E2E Pipeline | 9 | 8 | 0 | 1 | ✅ 89% PASSED |
| Performance | 6 | 0 | 0 | 6 | ⊘ SKIPPED |
| **TOTAL** | **30** | **22** | **1** | **7** | **73% PASSED** |

**Key Validations:**
- ✅ Queue operations work with Azurite
- ✅ Database operations work with local Supabase
- ✅ Status transitions validated
- ✅ Idempotency checks work
- ✅ Message passing validated
- ⚠️ 1 known non-blocking failure: `test_batch_result_persistence` (tuple index error)

**Local Environment Setup:**
- ✅ Azurite configured and running
- ✅ Local Supabase configured
- ✅ `.env.local` setup script created
- ✅ Function entry points load `.env.local` via dotenv
- ✅ `local.settings.json` contains only Azurite connection string
- ✅ All helper scripts created (start/stop Azurite, dev functions, test functions)

### 3.4 Phase 5 Cloud Testing Status

**Deployment Status:**
- ✅ Azure Functions deployed
- ✅ All 4 functions present in Function App
- ⚠️ Some environment variables missing (documented in phase_5_testing.md)
- ⚠️ Queue trigger behavior test failed (FM-005: Functions not processing messages)

**Cloud Test Results:**
- ✅ Message enqueueing works
- ✅ Status transitions work
- ✅ Queue depth handling works
- ❌ Azure Functions queue trigger behavior: FAILED (functions not processing messages)
- ⊘ Performance tests: Not yet executed (require deployed functions)

**Known Issues:**
- **FM-005**: Azure Functions not processing queue messages (Critical)
- **FM-001**: Batch result persistence query error (Non-blocking, local only)

---

## 4. Phase 5 Testing Infrastructure

### 4.1 Codebase Structure Issues

**Duplication:**
- `backend/rag_eval/` - Source code (project root)
- `infra/azure/azure_functions/backend/rag_eval/` - Copied code (duplicate)
- Build script copies source to deployment package during Git deployment

**Consolidation Approach:**
- Functions should import from project root using relative paths
- Eliminate need for code copying in build process
- Use Python path manipulation in function entry points
- Single source of truth for all backend code

### 4.2 Local Development Environment

**Components:**
- **Azurite**: Local Azure Storage emulator (ports 10000, 10001, 10002)
- **Local Supabase**: PostgreSQL database (localhost:54322)
- **Azure Functions Core Tools**: Local function execution
- **Environment Variables**: Loaded from `.env.local` via python-dotenv

**Key Files:**
- `infra/azure/azure_functions/local.settings.json` - Azurite connection strings only
- `.env.local` - All application configuration (project root)
- `scripts/setup_local_env.sh` - Setup script for local environment
- `scripts/dev_functions_local.sh` - Start all local services
- `scripts/test_functions_local.sh` - Run Phase 5 tests locally
- `scripts/start_azurite.sh` / `scripts/stop_azurite.sh` - Azurite management

**Environment Variable Loading Strategy:**
- **Critical Requirement**: All environment variables except Azurite connection string must be loaded from `.env.local` in project root using `python-dotenv`, NOT hardcoded in `local.settings.json`
- Function `__init__.py` files load `.env.local` BEFORE importing backend code
- `local.settings.json` contains ONLY: `AzureWebJobsStorage`, `AZURE_STORAGE_QUEUES_CONNECTION_STRING`, `FUNCTIONS_WORKER_RUNTIME`

### 4.3 Cloud Testing Environment

**Components:**
- **Azure Function App**: `func-raglab-uploadworkers` (Consumption Plan)
- **Azure Storage Account**: Queue storage for worker queues
- **Azure Application Insights**: Monitoring and logging
- **Cloud Supabase**: Production database instance
- **Azure AI Services**: Document Intelligence, AI Foundry, AI Search

**Deployment:**
- Git-based deployment configured
- Build script: `infra/azure/azure_functions/build.sh`
- Environment variables set via Azure Function App configuration
- See `001/intermediate/phase_5_azure_functions_deployment.md` for full deployment guide

### 4.4 Test Suites

**1. Database Migration Verification**
- **Script**: `backend/scripts/verify_phase5_migrations.py`
- **Purpose**: Verify migrations 0019 and 0020 applied correctly
- **Status**: ✅ PASSED (local)

**2. Supabase Integration Tests**
- **File**: `backend/tests/integration/test_supabase_phase5.py`
- **Purpose**: Test persistence operations with real database
- **Status**: ⚠️ 12/13 PASSED (local), 1 known failure
- **Test Classes**: PersistenceOperations, StatusUpdates, BatchMetadata, ErrorHandling, WorkerReadWrite

**3. End-to-End Pipeline Tests**
- **File**: `backend/tests/integration/test_phase5_e2e_pipeline.py`
- **Purpose**: Test complete pipeline flow with real queues and database
- **Status**: ✅ 8/9 PASSED (local), 1 skipped (requires cloud)
- **Test Classes**: EndToEndPipeline, FailureScenarios, SupabaseIntegration
- **Markers**: `@pytest.mark.local`, `@pytest.mark.cloud`, `@pytest.mark.integration`

**4. Performance Tests**
- **File**: `backend/tests/integration/test_phase5_performance.py`
- **Purpose**: Validate throughput and latency requirements
- **Status**: ⊘ All SKIPPED (require deployed Azure Functions)
- **Test Classes**: WorkerProcessingTime, QueueDepthHandling, ConcurrentProcessing, ColdStartLatency, Throughput

---

## 5. Adjacent & Integrating Components

### 5.1 Core RAG Services (Unchanged)
- `rag_eval/services/rag/ingestion.py` - Text extraction (wrapped by ingestion worker)
- `rag_eval/services/rag/chunking.py` - Text chunking (wrapped by chunking worker)
- `rag_eval/services/rag/embeddings.py` - Embedding generation (wrapped by embedding worker)
- `rag_eval/services/rag/search.py` - Vector indexing (wrapped by indexing worker)
- `rag_eval/services/rag/pipeline.py` - Query-time RAG pipeline (unchanged)

### 5.2 Worker Infrastructure (From 001)
- `rag_eval/services/workers/ingestion_worker.py` - Ingestion worker implementation
- `rag_eval/services/workers/chunking_worker.py` - Chunking worker implementation
- `rag_eval/services/workers/embedding_worker.py` - Embedding worker implementation
- `rag_eval/services/workers/indexing_worker.py` - Indexing worker implementation
- `rag_eval/services/workers/queue_client.py` - Queue operations (with local dev support)
- `rag_eval/services/workers/persistence.py` - Database persistence operations

### 5.3 Azure Functions (From 001)
- `infra/azure/azure_functions/ingestion-worker/__init__.py` - Function entry point
- `infra/azure/azure_functions/chunking-worker/__init__.py` - Function entry point
- `infra/azure/azure_functions/embedding-worker/__init__.py` - Function entry point
- `infra/azure/azure_functions/indexing-worker/__init__.py` - Function entry point
- All functions load `.env.local` via dotenv before importing backend code

### 5.4 Database & Storage
- **Supabase**: PostgreSQL database with migrations 0019, 0020
- **Supabase Storage**: Document storage backend
- **Azure Blob Storage**: Alternative storage backend (optional)

### 5.5 Testing Infrastructure
- `backend/tests/integration/test_phase5_e2e_pipeline.py` - E2E pipeline tests
- `backend/tests/integration/test_supabase_phase5.py` - Supabase integration tests
- `backend/tests/integration/test_phase5_performance.py` - Performance tests
- `backend/scripts/verify_phase5_migrations.py` - Migration verification

---

## 6. Interface Contracts to Preserve

### 6.1 Worker Function Signatures
```python
def ingestion_worker(message: dict, context: Optional[Any]) -> None
def chunking_worker(message: dict, context: Optional[Any]) -> None
def embedding_worker(message: dict, context: Optional[Any]) -> None
def indexing_worker(message: dict, context: Optional[Any]) -> None
```

### 6.2 Queue Message Schema
```python
@dataclass
class QueueMessage:
    document_id: str
    source_storage: str  # "azure_blob" | "supabase"
    filename: str
    attempt: int
    stage: str  # "uploaded" | "parsed" | "chunked" | "embedded" | "indexed"
    metadata: Optional[Dict[str, Any]] = None
```

### 6.3 Queue Client Interface
```python
def enqueue_message(queue_name: str, message: QueueMessage, config) -> None
def dequeue_message(queue_name: str, config, visibility_timeout: int = 30) -> Optional[QueueMessage]
def get_queue_length(queue_name: str, config) -> int
def send_to_dead_letter(queue_name: str, message: QueueMessage, reason: str, config) -> None
```

### 6.4 Persistence Interface
```python
def persist_extracted_text(document_id: str, text: str, config) -> None
def load_extracted_text(document_id: str, config) -> Optional[str]
def persist_chunks(document_id: str, chunks: List[Chunk], config) -> None
def load_chunks(document_id: str, config) -> List[Chunk]
def persist_embeddings(document_id: str, chunks: List[Chunk], embeddings: List[List[float]], config) -> None
def update_document_status(document_id: str, status: str, timestamp_column: str, config) -> None
def check_document_status(document_id: str, config) -> str
def should_process_document(document_id: str, target_stage: str, config) -> bool
```

---

## 7. Risks & Constraints from Adjacent Systems

### 7.1 Azure Document Intelligence Constraints
- **Free-tier limit**: 2 pages per request
- **Budget constraint**: Tests limited to first 6 pages of test PDF
- **Impact**: Batch processing required, throughput limited by API constraints

### 7.2 Azure Functions Constraints
- **Consumption Plan**: 10-minute maximum execution time
- **Cold starts**: May cause latency in processing
- **Scaling**: Automatic based on queue depth
- **Queue triggers**: Require `AzureWebJobsStorage` connection string

### 7.3 Supabase Constraints
- **Connection limits**: Database connection pooling required
- **Storage limits**: Document size constraints
- **Schema changes**: Require migrations (already applied in 001)

### 7.4 Local Development Constraints
- **Azurite limitations**: May not perfectly match Azure Storage behavior
- **Local Supabase**: May differ from cloud instance
- **Environment variables**: Must be properly loaded from `.env.local`

---

## 8. Phase 5 Testing Requirements

### 8.1 Local Testing (Completed)

**Prerequisites:**
- ✅ Supabase running locally (`supabase start`)
- ✅ Azurite running (`./scripts/start_azurite.sh`)
- ✅ `.env.local` configured with local Supabase credentials
- ✅ `local.settings.json` configured with Azurite connection string

**Test Execution:**
```bash
# Run all local tests
./scripts/test_functions_local.sh

# Or run individually
cd backend
source venv/bin/activate
export AZURE_STORAGE_QUEUES_CONNECTION_STRING="UseDevelopmentStorage=true"
pytest tests/integration/test_phase5_e2e_pipeline.py tests/integration/test_supabase_phase5.py -v -m integration
```

**Results:**
- ✅ 22/23 runnable tests PASSED (96% pass rate)
- ⚠️ 1 known non-blocking failure
- ⊘ 7 tests skipped (require cloud deployment)

### 8.2 Cloud Testing (In Progress)

**Prerequisites:**
- ✅ Azure Functions deployed
- ✅ Environment variables configured in Function App
- ✅ Queues created in Azure Storage
- ✅ Database migrations applied to cloud Supabase

**Test Execution:**
```bash
cd backend
source venv/bin/activate
# Use cloud connection string from Azure
pytest tests/integration/test_phase5_e2e_pipeline.py -v -m integration -m cloud
pytest tests/integration/test_phase5_performance.py -v -m integration -m performance
```

**Current Status:**
- ⚠️ Queue trigger behavior test failed (FM-005)
- ⊘ Performance tests not yet executed
- ✅ Most E2E tests pass with cloud resources

### 8.3 Test Markers

**Local Tests** (`@pytest.mark.local`):
- Run against Azurite and local Supabase
- Use `UseDevelopmentStorage=true` connection string
- Validate queue operations, database operations, status transitions

**Cloud Tests** (`@pytest.mark.cloud`):
- Run against real Azure Storage and cloud Supabase
- Require deployed Azure Functions
- Validate actual function processing, performance, cold starts

**Integration Tests** (`@pytest.mark.integration`):
- All Phase 5 integration tests
- Require real resources (local or cloud)

**Performance Tests** (`@pytest.mark.performance`):
- Throughput and latency validation
- Require deployed Azure Functions
- Currently skipped until functions are stable

---

## 9. Known Issues & Technical Debt

### 9.1 Critical Issues (From FRACAS)

**FM-005: Azure Functions Not Processing Queue Messages** (Critical)
- **Status**: Open
- **Impact**: Functions deployed but not processing messages from queues
- **Evidence**: 18 messages queued, 0 function executions
- **Location**: Cloud deployment
- **Next Steps**: Investigate queue trigger configuration, message encoding, function logs

### 9.2 Non-Blocking Issues

**FM-001: Batch Result Persistence Query Error** (Non-blocking)
- **Status**: Known issue
- **Impact**: `test_batch_result_persistence` fails locally
- **Error**: `DatabaseError: Query failed: tuple index out of range`
- **Location**: `rag_eval/services/workers/persistence.py:575` in `get_completed_batches()`
- **Priority**: Low (other batch operations work)

### 9.3 Technical Debt

- **Test Marker Registration**: Pytest markers (`@pytest.mark.local`, `@pytest.mark.cloud`) not registered in `pytest.ini` or `pyproject.toml` (causes warnings)
- **Performance Test Execution**: Performance tests skipped until cloud functions are stable
- **Environment Variable Validation**: No validation script to ensure all required variables are set

---

## 10. Goals for Initiative 002

### 10.1 Primary Goals (Codebase Consolidation)

1. **Eliminate Code Duplication**
   - Remove duplicate `backend/rag_eval/` code from `infra/azure/azure_functions/backend/`
   - Consolidate to single source of truth in `backend/rag_eval/`
   - Update build scripts and deployment process to reference source code directly
   - Ensure functions import from project root backend, not copied code

2. **Consolidate Configuration Management**
   - Unify environment variable loading strategy
   - Consolidate `.env.local` and `local.settings.json` usage
   - Standardize configuration across local and cloud environments
   - Create single source of truth for configuration

3. **Streamline Deployment Process**
   - Optimize Git-based deployment build script
   - Eliminate unnecessary code copying steps
   - Simplify function entry point path resolution
   - Improve deployment reliability and speed

4. **Consolidate Test Infrastructure**
   - Unify local and cloud test execution
   - Consolidate test fixtures and utilities
   - Register pytest markers to eliminate warnings
   - Create comprehensive test execution scripts

### 10.2 Secondary Goals (Testing & Production Readiness)

1. **Complete Phase 5 Cloud Testing**
   - Resolve FM-005 (Azure Functions queue trigger issue)
   - Execute all performance tests
   - Validate production readiness

2. **Fix Identified Issues**
   - Resolve batch result persistence query error (FM-001)
   - Fix any other issues discovered during testing
   - Ensure all tests pass in both local and cloud environments

3. **Production Validation**
   - Verify end-to-end pipeline works in production
   - Validate performance meets requirements
   - Ensure monitoring and observability work correctly

4. **Documentation Consolidation**
   - Consolidate deployment guides
   - Unify local and cloud testing documentation
   - Create comprehensive troubleshooting guides
   - Document consolidated codebase structure

### 10.3 Phase 0 Requirements (Scoping Phase)

**Critical**: During Phase 0 (when scoping documents are created), the following cursor rules files must be updated to reflect the worker-queue architecture transformation implemented in Initiative 001:

1. **Update `.cursor/rules/architecture_rules.md`**
   - Add worker-queue architecture layer to the architecture description
   - Document the worker infrastructure (`rag_eval/services/workers/`)
   - Update layer boundaries to include Azure Functions as the worker execution layer
   - Document queue-based communication patterns
   - Update development workflow to include local Azure Functions development with Azurite

2. **Update `.cursor/rules/scoping_document.md`**
   - Update "High-Level System Overview" to include worker-queue architecture
   - Add worker infrastructure to the codebase structure diagram
   - Document Azure Functions deployment structure
   - Update "Out of Scope" section to reflect that workers are now in scope (remove workers from out-of-scope list)
   - Add Azure Storage Queues to infrastructure architecture section
   - Update database scope to include worker status tracking (migrations 0019, 0020)

3. **Update `.cursor/rules/state_of_development.md`**
   - Update "CURRENT PHASE" to reflect worker-queue architecture is implemented
   - Remove workers from "OUT OF SCOPE FOR THIS PHASE" section
   - Add worker-queue architecture to "PRIMARY OBJECTIVES IN THIS PHASE" or create new section
   - Update "LOCAL DEVELOPMENT WORKFLOW" to include Azure Functions local development
   - Add validation gates for worker-queue functionality
   - Update meta-rule to reflect workers are now part of the system

**Rationale**: These cursor rules files serve as foundational documentation for the codebase. They currently describe the system as synchronous and explicitly exclude workers. Since Initiative 001 successfully implemented the worker-queue architecture, these rules must be updated to accurately reflect the current system architecture and prevent confusion during future development.

**Timing**: These updates should be completed during Phase 0 of Initiative 002, before implementation begins, to ensure all scoping documents and development guidelines accurately reflect the worker-queue architecture.

**Re-validation**: After consolidation work is complete (Phase 4), these cursor rules files should be re-validated to ensure they still accurately reflect the consolidated codebase structure. Specifically:
- Verify `architecture_rules.md` accurately describes the consolidated code structure (no duplicate code references)
- Verify `scoping_document.md` accurately reflects the simplified deployment process and consolidated configuration
- Verify `state_of_development.md` accurately describes the updated development workflow after consolidation
- **Note**: These files are documentation/guidance files and are NOT modified as part of the consolidation work itself. They are only updated in Phase 0 to reflect worker-queue architecture, then validated in Phase 4 to ensure they remain accurate after consolidation.

---

## 11. Evidence Links

### 11.1 Initiative 001 Documentation
- `001/scoping/context.md` - Original worker-queue architecture context
- `001/scoping/PRD001.md` - Product requirements
- `001/scoping/RFC001.md` - Technical design
- `001/fracas.md` - Failure tracking (FM-001, FM-005)

### 11.2 Phase 5 Testing Documentation
- `001/intermediate/phase_5_testing.md` - Complete testing overview
- `001/intermediate/PHASE_5_LOCAL_TEST_RESULTS.md` - Local test results
- `001/intermediate/phase_5_azure_functions_deployment.md` - Deployment guide
- `001/LOCAL_DEVELOPMENT.md` - Local development setup guide

### 11.3 Implementation Files
- `backend/tests/integration/test_phase5_e2e_pipeline.py` - E2E tests
- `backend/tests/integration/test_supabase_phase5.py` - Supabase tests
- `backend/tests/integration/test_phase5_performance.py` - Performance tests
- `backend/scripts/verify_phase5_migrations.py` - Migration verification
- `scripts/test_functions_local.sh` - Local test execution script
- `scripts/dev_functions_local.sh` - Local development script

### 11.4 Configuration Files
- `infra/azure/azure_functions/local.settings.json` - Local function settings
- `infra/azure/azure_functions/host.json` - Function host configuration
- `infra/azure/azure_functions/*/function.json` - Function trigger configurations

---

## 12. Deliverables

### 12.1 Testing Deliverables
- ✅ Local test execution scripts
- ✅ Local development environment setup
- ⚠️ Cloud test execution and validation (in progress)
- ⚠️ Performance test results (pending)

### 12.2 Documentation Deliverables
- ✅ Local development guide
- ✅ Local test results documentation
- ⚠️ Cloud test results documentation (pending)
- ⚠️ Production deployment verification guide (pending)

### 12.3 Code Deliverables
- ✅ Local development infrastructure
- ✅ Test markers and fixtures
- ⚠️ Issue fixes (FM-001, FM-005)
- ⚠️ Performance optimizations (if needed)

---

## 13. Success Metrics

### 13.1 Test Coverage
- **Target**: 100% of runnable tests passing
- **Current**: 96% (22/23 local, cloud testing in progress)
- **Blockers**: FM-005 (cloud), FM-001 (local, non-blocking)

### 13.2 Performance Metrics
- **Throughput**: Validate system maintains constant throughput despite Document Intelligence constraints
- **Latency**: Measure and validate worker processing times
- **Cold Starts**: Measure Azure Functions cold start latency
- **Queue Depth**: Validate queue depth handling under load

### 13.3 Production Readiness
- ✅ All functions deployed and enabled
- ✅ All environment variables configured
- ⚠️ Queue triggers processing messages (FM-005)
- ⚠️ Performance requirements met (pending validation)

---

## 14. Context Budget

- **Max**: 20k tokens
- **Allocation**: 40/40/20 (context/rollups/snippets)
- **Overflow**: Keep interface contracts and key signatures

---

## 15. Isolation

**Isolation**: false

**Justification**: Initiative 002_codebase_consolidation is a direct continuation of Initiative 001, focusing on codebase consolidation, testing, and production readiness. It builds upon the worker-queue architecture implemented in 001 and requires access to the same codebase, infrastructure, and documentation. The consolidation work involves eliminating duplication in the existing codebase structure, while the testing infrastructure and results from 001 are essential context for validating the consolidated codebase.

---

**Last Updated**: 2025-12-07  
**Related Initiative**: 001 (Worker-Queue Architecture Implementation)  
**Status**: Ready for 002_codebase_consolidation planning and execution

