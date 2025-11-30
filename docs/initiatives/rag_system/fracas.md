# FRACAS.md - Failure Reporting, Analysis, and Corrective Actions System

**Initiative:** RAG Lab (RAG System Testing Suite)  
**Status:** Active  
**Date Started:** 2025-01-27  
**Last Updated:** 2025-11-29  
**Maintainer:** Development Team

## 📋 **How to Use This Document**

This document serves as a comprehensive failure tracking system for the RAG Lab initiative. Use it to:

1. **Document new failures** as they occur during development/testing
2. **Track investigation progress** and findings
3. **Record root cause analysis** and solutions
4. **Maintain a knowledge base** of known issues and fixes

### **Documentation Guidelines:**
- **Be specific** about symptoms, timing, and context
- **Include evidence** (logs, error messages, screenshots)
- **Update status** as investigation progresses
- **Link related failures** when applicable
- **Record both successful and failed solutions**

---

## 🚨 **Active Failure Modes**

### **FM-001: Pytest Tests Hanging During Execution**
- **Severity**: High
- **Status**: ✅ Fixed
- **First Observed**: 2025-01-27
- **Last Updated**: 2025-01-27
- **Resolved**: 2025-01-27

**Symptoms:**
- Pytest tests for `test_rag_chunking.py` hang indefinitely during execution
- Tests can be collected successfully (`pytest --collect-only` works)
- Tests hang when actually running, even simple tests like `test_chunk_text_basic`
- Multiple pytest processes accumulate and consume CPU resources
- Tests for `test_rag_ingestion.py` run successfully (12 passed, 1 skipped)
- Issue appears specific to chunking tests
- **Critical Finding**: Even direct Python execution (not pytest) hangs when calling `chunk_text_fixed_size()`

**Observations:**
- Test collection completes successfully (26 tests collected in 0.52s)
- Tests hang immediately when execution begins
- No error messages or output before hanging
- **Key Finding**: The `__main__` block in `chunking.py` works correctly (logging disabled before import)
- **Key Finding**: Direct Python execution hangs when calling `chunk_text_fixed_size()` even with logging disabled
- Issue occurs in virtual environment with all dependencies installed
- Both direct pytest execution and via Python module show same behavior
- **Key Finding**: Hang occurs during function execution, not during import

**Investigation Notes:**
- ✅ Verified imports work correctly (`from rag_eval.services.rag.chunking import chunk_text_fixed_size` succeeds)
- ✅ Verified `__main__` block works (logging disabled before import)
- ✅ Verified test collection works (`pytest --collect-only` succeeds)
- ✅ Verified Chunk class creation works independently
- ❌ **Root Cause Identified**: Hang occurs when calling `chunk_text_fixed_size()` function, even with logger calls removed
- ❌ Attempted fixes:
  - Removed all `logger.info()` calls - still hangs
  - Added `logger.isEnabledFor()` checks - still hangs
  - Wrapped logger calls in try/except - still hangs
  - Patched `get_logger` in test file - still hangs
- 🔍 **Current Hypothesis**: Hang may be in Chunk dataclass creation within the loop, or in the while loop itself
- 🔍 **Alternative Hypothesis**: Output capture deadlock (pytest or Python stdout/stderr blocking)

**Root Cause:**
**✅ IDENTIFIED**: Infinite loop in `chunk_text_fixed_size()` function. The loop logic `start = end - overlap` can fail to advance `start` when:
1. `overlap >= chunk_size` causes `next_start <= start` (loop doesn't advance)
2. Missing break condition when `end >= len(text)` (loop continues even after processing all text)

**Evidence from Debug Output:**
- Loop iterated 633,000+ times with `start=1230` stuck (text length was 1250)
- `start` value never advanced past 1230, causing infinite loop
- Function created millions of identical chunks before being interrupted

**Solution:**
**✅ FIXED**: 
1. Added parameter validation: `if overlap >= chunk_size: raise ValueError()`
2. Added explicit break condition: `if end >= len(text): break`
3. Added safety check: `if next_start <= start: raise ValueError()` to catch edge cases
4. Removed debug print statements after fix verification

**Corrective Actions Taken:**
1. ✅ Fixed infinite loop in `chunk_text_fixed_size()` by adding break condition and validation
2. ✅ Added parameter validation to prevent `overlap >= chunk_size`
3. ✅ Added safety check to ensure loop always advances
4. ✅ Verified fix: Main function works, pytest tests now pass

**Test Results:**
- ✅ Main function (`python -m rag_eval.services.rag.chunking`) works correctly
- ✅ Pytest test `test_chunk_text_basic` passes (0.27s execution time)
- ✅ All chunking tests should now pass

**Related Issues:**
- None identified

---

### **FM-002: Supabase Client Creation Failure - ClientOptions AttributeError**
- **Severity**: High
- **Status**: ✅ Fixed
- **First Observed**: 2025-11-29
- **Last Updated**: 2025-11-29
- **Resolved**: 2025-11-29

**Symptoms:**
- File upload fails with `AttributeError: 'ClientOptions' object has no attribute 'storage'`
- Error occurs when attempting to upload document to Supabase Storage
- Upload endpoint returns 500 Internal Server Error
- Document status is updated to "uploaded" despite upload failure
- Error traceback shows failure in `_get_supabase_client()` function at line 78-89

**Observations:**
- Error occurs during Supabase client creation in `supabase_storage.py`
- Stack trace shows: `File "/Users/aq_home/1Projects/rag_evaluator/backend/rag_eval/services/rag/supabase_storage.py", line 78, in _get_supabase_client`
- The Supabase library internally tries to access `client_options.storage` attribute
- `ClientOptions` object created with `auto_refresh_token=True` and `persist_session=False` doesn't have `storage` attribute
- Using `supabase>=2.0.0` from requirements.txt
- File upload request successfully received (4933 bytes read from `healthguard_select_ppo_plan.pdf`)

**Investigation Notes:**
- 🔍 **Hypothesis**: Version incompatibility with Supabase Python client library
- 🔍 **Hypothesis**: `ClientOptions` API changed in newer versions of supabase-py
- 🔍 **Hypothesis**: Options should be passed differently or not at all
- Need to verify correct way to create Supabase client with supabase-py 2.0.0+
- May need to create client without options or pass options as dictionary

**Root Cause:**
✅ **IDENTIFIED**: Version compatibility issue with Supabase Python client library (supabase>=2.0.0). The `ClientOptions` class doesn't have a `storage` attribute that the library internally tries to access when initializing the auth client. The library's internal code at `supabase/_sync/client.py` line 146 attempts to access `client_options.storage`, but the `ClientOptions` object created with only `auto_refresh_token` and `persist_session` doesn't include this attribute.

**Solution:**
✅ **FIXED**: Removed the `ClientOptions` parameter from `create_client()` call. The default client options are sufficient for storage operations. Changes made:
1. Removed `ClientOptions` import (no longer needed)
2. Simplified `create_client()` call to use only `supabase_url` and `supabase_key`
3. Default client options will be used automatically

**Evidence:**
- Error log from terminal (lines 113-165):
  ```
  AttributeError: 'ClientOptions' object has no attribute 'storage'
  File "/Users/aq_home/1Projects/rag_evaluator/backend/rag_eval/services/rag/supabase_storage.py", line 78, in _get_supabase_client
  ```
- File: `backend/rag_eval/services/rag/supabase_storage.py` lines 77-85
- Requirements: `supabase>=2.0.0` in `backend/requirements.txt`

**Corrective Actions Taken:**
1. ✅ Removed `ClientOptions` import from `supabase_storage.py`
2. ✅ Simplified `create_client()` call to use default options
3. ✅ Updated all upload endpoint tests to include proper mocks for storage and database operations
4. ✅ Updated e2e tests to include storage and database mocks
5. ✅ Verified fix: All 25 upload-related tests pass (12 unit tests, 2 e2e tests, 11 storage tests)

**Files Modified:**
- `backend/rag_eval/services/rag/supabase_storage.py` - Removed ClientOptions usage
- `backend/tests/test_upload_endpoint.py` - Added mocks for storage and database operations
- `backend/tests/test_rag_e2e.py` - Added mocks for storage and database operations

**Test Results:**
- ✅ All 11 Supabase storage tests pass
- ✅ All 12 upload endpoint tests pass
- ✅ All 2 e2e upload pipeline tests pass
- ✅ Total: 25/25 tests passing

**Related Issues:**
- None identified

---

### **FM-003: Supabase Storage RLS Policy Violation**
- **Severity**: High
- **Status**: ✅ Fixed
- **First Observed**: 2025-11-29
- **Last Updated**: 2025-11-29
- **Resolved**: 2025-11-29

**Symptoms:**
- File upload fails with `403 Unauthorized` error: `new row violates row-level security policy`
- Error occurs when attempting to upload document to Supabase Storage bucket 'documents'
- Upload retries 4 times with exponential backoff, all attempts fail
- Error message: `{'statusCode': 403, 'error': Unauthorized, 'message': new row violates row-level security policy}`
- Document status is updated to "uploaded" despite upload failure

**Observations:**
- Error occurs during file upload to Supabase Storage
- Stack trace shows: `storage3.exceptions.StorageApiError: {'statusCode': 403, 'error': Unauthorized, 'message': new row violates row-level security policy}`
- The storage bucket exists but RLS policies are blocking the upload
- Using anon key or service role key - RLS policies need to be configured
- File upload request successfully received (4933 bytes read from `healthguard_select_ppo_plan.pdf`)
- Retry logic works correctly (4 attempts with exponential backoff)

**Investigation Notes:**
- 🔍 **Root Cause Identified**: Supabase Storage has Row Level Security (RLS) enabled on `storage.objects` table
- 🔍 **Issue**: No RLS policies exist to allow service role or anon key to upload files to the 'documents' bucket
- 🔍 **Solution**: Create migration to set up storage bucket and RLS policies
- Need to create policies for INSERT, SELECT, UPDATE, DELETE operations on storage.objects for 'documents' bucket

**Root Cause:**
✅ **IDENTIFIED**: Supabase Storage bucket 'documents' has Row Level Security (RLS) enabled, but no policies are configured to allow the service to upload files. The `storage.objects` table requires explicit RLS policies for INSERT operations, and the current setup doesn't have any policies allowing uploads.

**Solution:**
🔧 **IN PROGRESS**: Created migration `0004_setup_storage_bucket.sql` to:
1. Create the storage bucket 'documents' if it doesn't exist (public, 50MB limit)
2. Enable RLS on storage.objects (if not already enabled)
3. Create RLS policies for:
   - Public read access (SELECT)
   - Service role upload (INSERT)
   - Service role delete (DELETE)
   - Service role update (UPDATE)

**Evidence:**
- Error log from terminal (lines 445-519):
  ```
  storage3.exceptions.StorageApiError: {'statusCode': 403, 'error': Unauthorized, 'message': new row violates row-level security policy}
  ```
- File: `backend/rag_eval/services/rag/supabase_storage.py` line 134
- Supabase Storage API endpoint: `http://127.0.0.1:54321/storage/v1/object/documents/...`

**Corrective Actions Taken:**
1. ✅ Created migration file `infra/supabase/migrations/0004_setup_storage_bucket.sql`
2. ✅ Fixed migration (removed ALTER TABLE - RLS already enabled by default)
3. ✅ Created integration test file `backend/tests/test_storage_integration.py`
4. ✅ Added pytest marker for integration tests in `pyproject.toml`
5. ✅ Applied migration successfully (`make reset-db`)
6. ✅ Verified bucket and RLS policies created correctly
7. ✅ Fixed integration test for public URL generation (handles dict response)
8. ✅ All integration tests passing (5/5)

**Files Modified:**
- `infra/supabase/migrations/0004_setup_storage_bucket.sql` - Migration for storage bucket and RLS policies
- `backend/tests/test_storage_integration.py` - Integration tests for real Supabase storage
- `backend/pyproject.toml` - Added integration test marker

**Test Coverage:**
- ✅ **Unit Tests**: 11/11 passing (all mocked - test code logic)
- ✅ **Endpoint Tests**: 12/12 passing (all mocked - test endpoint integration)
- ✅ **E2E Tests**: 2/2 passing (all mocked - test full pipeline)
- ✅ **Integration Tests**: 5/5 passing (real Supabase - validates RLS policies)

**Validation Status:**
- ✅ Code logic validated with unit tests
- ✅ Integration points validated with mocked tests
- ✅ RLS policies validated with real Supabase integration tests
- ✅ Migration validated (applied and verified)
- ✅ Storage bucket created and verified
- ✅ All 4 RLS policies created and verified

**Verification Results:**
- ✅ Storage bucket `documents` exists (public, 50MB limit)
- ✅ RLS policy "Public read access for documents" (SELECT) - verified
- ✅ RLS policy "Service role upload for documents" (INSERT) - verified
- ✅ RLS policy "Service role delete for documents" (DELETE) - verified
- ✅ RLS policy "Service role update for documents" (UPDATE) - verified
- ✅ Integration test: Upload with RLS policies - PASSED
- ✅ Integration test: Download with RLS policies - PASSED
- ✅ Integration test: Delete with RLS policies - PASSED
- ✅ Integration test: Public URL generation - PASSED
- ✅ Integration test: Upload retry logic - PASSED

**Related Issues:**
- None identified

---

## 🔧 **Resolved Failure Modes**

### **FM-001: Pytest Tests Hanging During Execution** ✅ RESOLVED
- **Severity**: High
- **Status**: ✅ Fixed
- **First Observed**: 2025-01-27
- **Resolved**: 2025-01-27

**Root Cause**: Infinite loop in `chunk_text_fixed_size()` function due to missing break condition when `end >= len(text)` and lack of validation for `overlap >= chunk_size`.

**Solution Applied**:
1. Added parameter validation: `if overlap >= chunk_size: raise ValueError()`
2. Added explicit break condition: `if end >= len(text): break`
3. Added safety check: `if next_start <= start: raise ValueError()`

**Verification**:
- ✅ All 14 chunking tests pass (0.29s execution time)
- ✅ Main function works correctly
- ✅ No infinite loops observed

**Files Modified**:
- `backend/rag_eval/services/rag/chunking.py` - Fixed infinite loop logic
- `backend/tests/test_rag_chunking.py` - Cleaned up debug code

---

## 📝 **New Failure Documentation Template**

Use this template when documenting new failures:

```markdown
### **FM-XXX: [Failure Name]**
- **Severity**: [Low/Medium/High/Critical]
- **Status**: [🔍 Under Investigation | ⚠️ Known issue | 🔧 Fix in progress]
- **First Observed**: [YYYY-MM-DD]
- **Last Updated**: [YYYY-MM-DD]

**Symptoms:**
- [Specific error messages or behaviors]
- [When the failure occurs]
- [What functionality is affected]

**Observations:**
- [What you noticed during testing]
- [Patterns or timing of the failure]
- [Any error messages or logs]

**Investigation Notes:**
- [Steps taken to investigate]
- [Hypotheses about the cause]
- [Tests performed or attempted]
- [Files or components involved]

**Root Cause:**
[The actual cause once identified, or "Under investigation" if unknown]

**Solution:**
[How the issue was fixed, or "Pending" if not yet resolved]

**Evidence:**
- [Code changes made]
- [Log entries or error messages]
- [Test results or screenshots]

**Related Issues:**
- [Links to related failures or issues]
```

---

## 🧪 **Testing Scenarios**

### **Scenario 1: Azure Service Connection Validation**
- **Steps**: 
  1. Attempt to connect to each Azure service (Blob Storage, Document Intelligence, AI Foundry, AI Search)
  2. Verify credentials are configured correctly
  3. Test basic operations (upload, extract, embed, search)
- **Expected**: All services connect successfully with valid credentials
- **Current Status**: ⚠️ Credentials not yet configured (expected for Phase 0)
- **Last Tested**: 2025-01-27
- **Known Issues**: All Azure service credentials are missing (documented in Phase 0 validation)

### **Scenario 2: Supabase Database Connection**
- **Steps**: 
  1. Connect to Supabase Postgres database
  2. Verify schema matches expected structure
  3. Test query execution on key tables
- **Expected**: Database connects and schema is correct
- **Current Status**: ⚠️ Connection not yet validated (expected for Phase 0)
- **Last Tested**: 2025-01-27
- **Known Issues**: Database credentials not yet configured (documented in Phase 0 validation)

### **Scenario 3: Component Integration Testing**
- **Steps**: Test each component (ingestion, chunking, embeddings, search, generation) in isolation
- **Expected**: All components function correctly with mocked dependencies
- **Current Status**: 🔍 Not yet tested (pending implementation phases)
- **Last Tested**: N/A
- **Known Issues**: None

---

## 🔍 **Failure Tracking Guidelines**

### **When to Document a Failure:**
- Any unexpected behavior or error during development/testing
- Performance issues or slow responses
- Service unavailability or crashes
- Data inconsistencies or corruption
- Security concerns or vulnerabilities
- Configuration issues or missing credentials
- Integration failures between components

### **What to Include:**
1. **Immediate Documentation**: Record symptoms and context as soon as possible
2. **Evidence Collection**: Screenshots, logs, error messages, stack traces
3. **Reproduction Steps**: Detailed steps to reproduce the issue
4. **Environment Details**: OS, browser, service versions, configuration
5. **Impact Assessment**: What functionality is affected and severity

### **Investigation Process:**
1. **Initial Assessment**: Determine severity and impact
2. **Data Gathering**: Collect logs, error messages, and context
3. **Hypothesis Formation**: Develop theories about the root cause
4. **Testing**: Attempt to reproduce and isolate the issue
5. **Root Cause Analysis**: Identify the actual cause
6. **Solution Development**: Implement and test fixes
7. **Documentation**: Update the failure record with findings

### **Status Updates:**
- **🔍 Under Investigation**: Issue is being analyzed and tested
- **⚠️ Known issue**: Issue understood, workaround available
- **🔧 Fix in progress**: Solution being implemented
- **✅ Fixed**: Issue has been resolved and verified
- **Won't Fix**: Issue is known but not planned to be addressed

## 📈 **System Health Metrics**

### **Current Performance:**
- **Configuration Status**: All Azure and Supabase credentials missing (expected for Phase 0)
- **Codebase Status**: Core interfaces and structure in place, ingestion and chunking components implemented
- **Database Schema**: Schema defined in migrations, not yet validated against actual database
- **Test Coverage**: No unit tests yet (pending implementation phases)

### **Known Limitations:**
- Azure service credentials not yet configured (blocker for integration testing)
- Supabase database connection not yet validated
- Unit tests not yet implemented for existing components (ingestion, chunking)
- Pipeline orchestration not yet implemented

## 🔍 **Investigation Areas**

### **High Priority:**
1. Azure service configuration and credential setup
2. Supabase database connection and schema validation
3. Component unit test implementation

### **Medium Priority:**
1. Integration testing strategy
2. Error handling and retry logic validation
3. Performance optimization opportunities

### **Low Priority:**
1. Documentation completeness
2. Code quality improvements
3. Future enhancement planning

## 📝 **Testing Notes**

### **Phase 0 Validation (2025-01-27):**

#### **Documentation Review:**
- ✅ **PRD001.md**: Complete product requirements reviewed. All acceptance criteria clearly defined.
- ✅ **RFC001.md**: Technical architecture reviewed. All design decisions documented. Implementation phases clearly outlined.
- ✅ **TODO001.md**: Complete implementation breakdown reviewed. All phases and tasks identified.
- ✅ **context.md**: System context and scope boundaries reviewed. Adjacent components understood.

#### **Codebase Review:**
- ✅ **Core Interfaces** (`rag_eval/core/interfaces.py`): All required data structures present (Query, Chunk, RetrievalResult, ModelAnswer, EvaluationScore)
- ✅ **Configuration System** (`rag_eval/core/config.py`): Complete Config class with all Azure service and Supabase settings. Default values for embedding and generation models set.
- ✅ **Database Models** (`rag_eval/db/models.py`): All required database record types defined (QueryRecord, RetrievalLog, ModelAnswerRecord, PromptVersion, etc.)
- ✅ **Database Connection** (`rag_eval/db/connection.py`): Connection pool implementation present with proper error handling
- ✅ **Query Executor** (`rag_eval/db/queries.py`): QueryExecutor class implemented with execute_query and execute_insert methods
- ✅ **Ingestion Component** (`rag_eval/services/rag/ingestion.py`): Implemented with Azure Document Intelligence integration. Missing retry logic and unit tests.
- ✅ **Chunking Component** (`rag_eval/services/rag/chunking.py`): Implemented with both fixed-size and LLM-based chunking. Missing unit tests.
- ✅ **Pipeline Scaffold** (`rag_eval/services/rag/pipeline.py`): Scaffolded but not implemented (raises NotImplementedError)
- ✅ **Exceptions** (`rag_eval/core/exceptions.py`): All required exception types defined

#### **Configuration Validation:**
- ⚠️ **Azure AI Foundry Endpoint**: Configured in `.env.local` (not validated in Phase 0)
- ⚠️ **Azure AI Foundry API Key**: Configured in `.env.local` (not validated in Phase 0)
- ⚠️ **Azure Search Endpoint**: Configured in `.env.local` (not validated in Phase 0)
- ⚠️ **Azure Search API Key**: Configured in `.env.local` (not validated in Phase 0)
- ⚠️ **Azure Search Index Name**: Configured in `.env.local` (not validated in Phase 0)
- ⚠️ **Azure Document Intelligence Endpoint**: Configured in `.env.local` (not validated in Phase 0)
- ⚠️ **Azure Document Intelligence API Key**: Configured in `.env.local` (not validated in Phase 0)
- ⚠️ **Azure Blob Connection String**: Configured in `.env.local` (not validated in Phase 0)
- ⚠️ **Azure Blob Container Name**: Configured in `.env.local` (not validated in Phase 0)
- ⚠️ **Supabase URL**: Configured in `.env.local` (not validated in Phase 0)
- ⚠️ **Supabase Key**: Configured in `.env.local` (not validated in Phase 0)
- ⚠️ **Database URL**: Configured in `.env.local` (not validated in Phase 0)

**Note**: All credentials are configured in `.env.local` file. The `Config.from_env()` method automatically loads from `.env.local` (or a custom path if specified). Connection validation will occur during Phase 1+ implementation when connection tests are run.

#### **Database Schema Validation:**
- ✅ **Migration File** (`infra/supabase/migrations/0001_init.sql`): Schema matches RFC001.md requirements
- ✅ **Required Tables Present**: 
  - `prompt_versions` (version_id, version_name, prompt_text, created_at)
  - `queries` (query_id, query_text, timestamp, metadata)
  - `retrieval_logs` (log_id, query_id, chunk_id, similarity_score, timestamp)
  - `model_answers` (answer_id, query_id, answer_text, prompt_version, retrieved_chunk_ids, timestamp)
  - `eval_judgments` (for evaluator system, out of scope for RAG pipeline)
  - `meta_eval_summaries` (for meta-evaluator system, out of scope for RAG pipeline)
- ✅ **Indexes**: All required indexes defined
- ⚠️ **Connection Not Validated**: Database connection not yet tested (credentials missing)

#### **Dependencies Review:**
- ✅ **requirements.txt**: All required Azure SDKs present:
  - `azure-ai-inference==1.0.0b1` (for AI Foundry)
  - `azure-ai-documentintelligence==1.0.0` (for Document Intelligence)
  - `azure-search-documents==11.4.0` (for AI Search)
  - `azure-storage-blob==12.19.0` (for Blob Storage)
  - `psycopg2-binary>=2.9.9` (for Supabase Postgres)
  - `fastapi==0.104.1` (for API)
  - `pydantic>=2.11.7,<3.0.0` (for validation)

#### **Codebase Structure Validation:**
- ✅ **Architecture Matches RFC001.md**: Codebase structure aligns with RFC design
- ✅ **Component Isolation**: Ingestion and chunking are properly isolated modules
- ⚠️ **Missing Components**: Embeddings, search, generation, logging, and storage components not yet implemented
- ⚠️ **Missing Tests**: No unit tests for existing components (ingestion, chunking)

#### **Discrepancies Found:**
1. **Config Default Generation Model** - ✅ RESOLVED
   - **Original Issue**: Config defaulted to `gpt-4`, documentation specified `gpt-4o-mini`
   - **Resolution**: Standardized to `gpt-4o` across all documentation and config (2025-01-27)
   - **Updated Files**: config.py, RFC001.md, TODO001.md, prompt_phase_6_001.md

2. **Config Default Embedding Model** - ✅ RESOLVED
   - **Original Issue**: Config defaulted to `text-embedding-ada-002`, RFC had inconsistency
   - **Resolution**: Standardized to `text-embedding-3-small` across all documentation and config (2025-01-27)
   - **Updated Files**: config.py, RFC001.md

3. **Ingestion Component Missing Retry Logic**:
   - `extract_text_from_document()` does not implement retry logic with exponential backoff
   - RFC001.md requires retry logic for all Azure service calls
   - **Resolution**: Add retry logic in Phase 2 or document as technical debt

#### **Blockers Identified:**
- ✅ **Azure Credentials**: Credentials are configured in `.env.local` file (not a blocker)
- ⚠️ **Supabase Connection**: Database connection must be validated during Phase 1+ implementation
- ⚠️ **Unit Tests**: Existing components (ingestion, chunking) need unit tests before proceeding

#### **Environment Configuration:**
- **Configuration File**: `.env.local` in project root (or `backend/.env.local`)
- **Loading**: `Config.from_env()` automatically loads from `.env.local`
- **Custom Path**: Can specify custom env file: `Config.from_env(env_file="/path/to/custom.env")`
- **Testing/CLI**: When running tests or CLI commands, can specify env file via:
  - Environment variable: `ENV_FILE=/path/to/env.file python script.py`
  - Or modify code to pass `env_file` parameter to `Config.from_env()`

#### **Next Steps:**
1. ✅ Azure service credentials configured in `.env.local`
2. ✅ Supabase connection configured in `.env.local`
3. Run connection tests to validate all external services (Phase 1+)
4. Proceed to Phase 1 implementation (Azure Blob Storage upload)

### **Next Test Session:**
- [ ] Configure Azure service credentials
- [ ] Validate Supabase database connection
- [ ] Run connection tests for all external services
- [ ] Begin Phase 1 implementation (Azure Blob Storage upload)

---

**Last Updated**: 2025-01-27  
**Next Review**: After Phase 1 completion  
**Maintainer**: Development Team

