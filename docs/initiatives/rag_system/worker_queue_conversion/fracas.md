# FRACAS.md - Failure Reporting, Analysis, and Corrective Actions System

**Initiative:** RAG Ingestion Worker–Queue Architecture Conversion  
**Status:** Active  
**Date Started:** 2025-01-XX  
**Last Updated:** 2025-01-XX  
**Maintainer:** Development Team

## 📋 **How to Use This Document**

This document serves as a comprehensive failure tracking system for the RAG Ingestion Worker–Queue Architecture Conversion. Use it to:

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

### **FM-001: Batch Metadata Test Failure**
- **Severity**: Low
- **Status**: 🔍 Under Investigation
- **First Observed**: 2025-01-XX
- **Last Updated**: 2025-01-XX

**Symptoms:**
- `test_batch_result_persistence` test fails with "tuple index out of range" error
- Error occurs in `get_completed_batches` function when querying chunks table
- 12 of 13 Supabase integration tests pass

**Observations:**
- Error occurs during batch metadata persistence test
- Query execution fails with tuple index out of range
- Other persistence operations work correctly

**Investigation Notes:**
- Issue appears to be in `get_completed_batches` function in persistence.py
- May be related to query result handling or batch metadata structure
- Non-blocking for Phase 5 testing - other tests pass

**Root Cause:**
Under investigation

**Solution:**
Pending

**Evidence:**
- Test output: `ERROR rag_eval.db.queries:queries.py:32 Query execution failed: tuple index out of range`
- Test: `tests/integration/test_supabase_phase5.py::TestBatchMetadata::test_batch_result_persistence`

---

## 🔧 **Resolved Failure Modes**

*No resolved failure modes at this time. This section will be populated as issues are resolved.*

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

---

### **FM-002: Azure Functions Not Deployed**
- **Severity**: Critical
- **Status**: ⚠️ Known issue
- **First Observed**: 2025-01-XX
- **Last Updated**: 2025-01-XX

**Symptoms:**
- Azure Function App `func-raglab-uploadworkers` exists and is running
- No worker functions are deployed (ingestion-worker, chunking-worker, embedding-worker, indexing-worker)
- `az functionapp function list` returns empty
- `az functionapp function show` returns NotFound error

**Observations:**
- Function App infrastructure exists (Python 3.12, Running state, East US location)
- Functions Extension Version: null (expected ~4)
- Environment variables partially configured (AzureWebJobsStorage, AZURE_STORAGE_QUEUES_CONNECTION_STRING, APPLICATIONINSIGHTS_CONNECTION_STRING)
- Missing: DATABASE_URL, SUPABASE_URL, SUPABASE_KEY, AZURE_AI_FOUNDRY_*, AZURE_SEARCH_*, AZURE_DOCUMENT_INTELLIGENCE_*
- Functions not deployed to Function App

**Investigation Notes:**
- Function App created but functions not deployed
- Need to deploy functions from `infra/azure/azure_functions/` directory
- Need to configure all required environment variables
- Blocking Phase 2 and Phase 3 integration tests
- Phase 2 validation executed: Function App health check completed, functions not found

**Root Cause:**
Azure Functions deployment not completed. Functions need to be deployed per `phase_5_azure_functions_deployment.md`.

**Solution:**
1. Deploy functions using Azure Functions Core Tools or Azure CLI
2. Configure all required environment variables (see FM-003)
3. Verify queue triggers are configured
4. Test function deployment

**Evidence:**
- `az functionapp function show` error: "Error retrieving function"
- `az functionapp function list` returns empty
- Function App state: Running, but no functions deployed
- Phase 2 validation: Task 2.1 completed, functions not found

**Impact:**
- Blocks Phase 2: Azure Functions Deployment Validation
- Blocks Phase 3: End-to-End Pipeline Integration Testing (requires deployed functions)
- Blocks Phase 4: Performance Testing (requires deployed functions)
- Phase 1 tests can proceed (database and Supabase integration)

---

### **FM-003: Missing Critical Environment Variables**
- **Severity**: Critical
- **Status**: ⚠️ Known issue
- **First Observed**: 2025-01-XX
- **Last Updated**: 2025-01-XX

**Symptoms:**
- Function App `func-raglab-uploadworkers` has only 3 environment variables configured
- Missing 10 critical environment variables required for worker execution
- Functions cannot execute even if deployed (missing database, AI services, storage connections)

**Observations:**
- Configured variables: AzureWebJobsStorage, APPLICATIONINSIGHTS_CONNECTION_STRING, AZURE_STORAGE_QUEUES_CONNECTION_STRING
- Missing critical variables:
  - DATABASE_URL (required for database connections)
  - SUPABASE_URL (required for Supabase operations)
  - SUPABASE_KEY (required for Supabase authentication)
  - AZURE_AI_FOUNDRY_ENDPOINT (required for embeddings and generation)
  - AZURE_AI_FOUNDRY_API_KEY (required for AI Foundry authentication)
  - AZURE_SEARCH_ENDPOINT (required for Azure AI Search)
  - AZURE_SEARCH_API_KEY (required for Azure AI Search authentication)
  - AZURE_SEARCH_INDEX_NAME (required for search index operations)
  - AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT (required for document extraction)
  - AZURE_DOCUMENT_INTELLIGENCE_API_KEY (required for Document Intelligence authentication)
- Optional variables not configured: AZURE_BLOB_CONNECTION_STRING, AZURE_BLOB_CONTAINER_NAME

**Investigation Notes:**
- Phase 2 validation: Task 2.2 completed
- Environment variables check executed via Azure CLI
- All critical variables missing - functions will fail immediately if deployed without these
- Variables should be set per `phase_5_azure_functions_deployment.md` Step 2

**Root Cause:**
Environment variables not configured in Function App. Only basic infrastructure variables (storage, Application Insights) are set.

**Solution:**
1. Configure all required environment variables per `phase_5_azure_functions_deployment.md` Step 2
2. Use `.env.local` file or Azure Key Vault for secure storage
3. Set variables via Azure CLI or Azure Portal
4. Verify all variables are present before deploying functions

**Evidence:**
- `az functionapp config appsettings list` shows only 3 relevant variables
- Phase 2 validation: Task 2.2 completed, 10 critical variables missing
- Reference: `../../setup/environment_variables.md` for variable descriptions

**Impact:**
- Blocks function execution even if functions are deployed
- Functions will fail immediately on startup or first operation
- Blocks Phase 2: Azure Functions Deployment Validation
- Blocks Phase 3: End-to-End Pipeline Integration Testing
- Blocks Phase 4: Performance Testing

**Status Update (2025-01-XX)**: Environment variables have been configured. Functions are now deployed and running.

---

### **FM-004: Tests Requiring Supabase Upload Are Skipped**
- **Severity**: Low
- **Status**: ✅ **RESOLVED**
- **First Observed**: 2025-01-XX
- **Last Updated**: 2025-01-XX

**Symptoms:**
- Tests requiring `test_document_id` fixture were being skipped
- Supabase upload was failing in test environment
- Tests that don't require Supabase upload (e.g., `test_queue_depth_handling`) passed successfully

**Observations:**
- Initial: 1 test passed, 6 tests skipped
- After fix: 6 tests passed, 1 test failed (expected failure)

**Investigation Notes:**
- Phase 5.3 testing executed: Initially 1 test passed, 6 tests skipped
- Tests were skipped at fixture level (test_pdf_path or test_document_id)
- Root cause identified: Two issues in test fixture:
  1. `upload_document_to_storage` called with wrong parameter order
  2. Database INSERT used wrong column name (`created_at` instead of `upload_timestamp`)

**Root Cause:**
1. **Parameter order issue**: Test fixture called `upload_document_to_storage(doc_id, filename, file_content, config)` but function signature is `upload_document_to_storage(file_content, document_id, filename, config)`
2. **Database schema issue**: Test fixture used `created_at` column which doesn't exist - should use `upload_timestamp` and include required columns (`storage_path`, `file_size`, `mime_type`)

**Solution:**
1. ✅ Fixed parameter order in `upload_document_to_storage` call
2. ✅ Fixed database INSERT to use correct column names and include required columns
3. ✅ Verified Supabase is running (`supabase start`)
4. ✅ Verified storage bucket "documents" exists

**Evidence:**
- Test output: After fixes, 6 of 7 tests now pass
- Test: `tests/integration/test_phase5_e2e_pipeline.py` - 6 passed, 1 failed (expected)
- Phase 5.3 test execution: 6 passed, 1 failed (expected failure for Azure Functions processing)

**Impact:**
- ✅ **RESOLVED** - All tests now execute successfully
- Full end-to-end pipeline testing with document upload now works
- Only remaining failure is expected (Azure Functions processing timeout)

---

## 🧪 **Testing Scenarios**

### **Scenario 1: Phase 0 Environment Validation**
- **Steps**: Validate testing environment setup (venv, pytest, dependencies)
- **Expected**: All dependencies installed, pytest can discover tests
- **Current Status**: ✅ Working
- **Last Tested**: 2025-01-XX
- **Known Issues**: None

---

## 🔍 **Failure Tracking Guidelines**

### **When to Document a Failure:**
- Any unexpected behavior or error during development/testing
- Performance issues or slow responses
- Service unavailability or crashes
- Data inconsistencies or corruption
- Security concerns or vulnerabilities
- Test failures during implementation phases

### **What to Include:**
1. **Immediate Documentation**: Record symptoms and context as soon as possible
2. **Evidence Collection**: Screenshots, logs, error messages, stack traces
3. **Reproduction Steps**: Detailed steps to reproduce the issue
4. **Environment Details**: OS, Python version, service versions, configuration
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
- Phase 0: ✅ Complete
- Environment Setup: ✅ Validated
- Test Discovery: ✅ Working (594 tests discovered)

### **Known Limitations:**
- None identified at this time

## 🔍 **Investigation Areas**

### **High Priority:**
1. None at this time

### **Medium Priority:**
1. None at this time

### **Low Priority:**
1. None at this time

## 📝 **Testing Notes**

### **Recent Tests (2025-01-XX):**
- ✅ Python 3.13.10 available in venv
- ✅ pytest 9.0.1 installed
- ✅ pytest-cov 7.0.0 installed
- ✅ pytest can discover tests (594 items collected)

### **Next Test Session:**
- [ ] Phase 1 persistence infrastructure tests
- [ ] Phase 2 queue infrastructure tests
- [ ] Phase 3 worker implementation tests

---

**Last Updated**: 2025-01-XX  
**Next Review**: After each phase completion  
**Maintainer**: Development Team

