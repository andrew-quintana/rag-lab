# Prompt: Phase 5 Complete Integration & RAG Upload Testing

## Your Mission

Execute comprehensive integration testing for Phase 5: Integration Testing & Migration. You must validate the complete worker-queue architecture and RAG upload pipeline using **real Azure resources** (Azure Functions, Storage Queues, Supabase). Document all results, failures, and metrics as you progress.

## Context & References

**Key Documentation to Reference**:
- `phase_5_azure_functions_deployment.md` - Azure Functions deployment guide
- `phase_5_migration_guide.md` - Database migration instructions
- `phase_5_migration_strategy.md` - Migration execution plan
- `phase_5_testing.md` - Testing overview and test locations
- `phase_5_decisions.md` - Implementation decisions
- `phase_5_handoff.md` - Phase 5 completion summary
- `scoping/TODO001.md` - Task checklist (mark tasks complete)
- `../../setup/environment_variables.md` - Environment variable reference
- `fracas.md` - Document all failures immediately

## Prerequisites Check

**Before starting, verify**:
1. Database migrations 0019 and 0020 applied to production Supabase (see `phase_5_migration_guide.md`)
2. Azure Function App `func-raglab-uploadworkers` deployed and running (see `phase_5_azure_functions_deployment.md`)
3. All four worker functions deployed: `ingestion-worker`, `chunking-worker`, `embedding-worker`, `indexing-worker`
4. Environment variables configured in Function App (see `../../setup/environment_variables.md`)
5. Application Insights monitoring enabled
6. Azure Storage Queues accessible (`raglabqueues` storage account)
7. Test PDF available: `docs/inputs/scan_classic_hmo.pdf` (use only first 6 pages)

**If any prerequisite is missing, document it in `fracas.md` and stop execution.**

---

## Phase 1: Pre-Deployment Validation

### Task 1.1: Database Migration Verification

**Execute**:
```bash
cd backend
source venv/bin/activate
python scripts/verify_phase5_migrations.py
```

**Verify**:
- ✅ All status columns exist in `documents` table (`status`, `parsed_at`, `chunked_at`, `embedded_at`, `indexed_at`)
- ✅ `extracted_text` column exists in `documents` table
- ✅ `chunks` table exists with all required columns (`chunk_id`, `document_id`, `text`, `metadata`, `embedding`, `created_at`)
- ✅ Indexes created on `documents.status` and `chunks.document_id`
- ✅ No migration errors reported

**Document**: Record results in `phase_5_testing.md`. If any checks fail, document in `fracas.md` and stop.

**Reference**: `phase_5_migration_guide.md` for migration application instructions

---

### Task 1.2: Supabase Integration Tests

**Execute**:
```bash
cd backend
source venv/bin/activate
pytest tests/integration/test_supabase_phase5.py -v -m integration
```

**Verify all test classes pass**:
- `TestPersistenceOperations` - Load/persist extracted text, chunks, embeddings
- `TestStatusUpdates` - Status transitions (`uploaded` → `parsed` → `chunked` → `embedded` → `indexed`)
- `TestIdempotency` - Workers can safely retry operations
- `TestBatchMetadata` - Batch processing metadata storage
- `TestErrorHandling` - Error handling with real database connections
- `TestWorkerReadWrite` - All workers can read/write to Supabase

**Document**: 
- Record test results in `phase_5_testing.md`
- Document any failures immediately in `fracas.md`
- Include test execution time and any warnings

**Reference**: `phase_5_testing.md` section "2. Supabase Integration Tests"

---

## Phase 2: Azure Functions Deployment Validation

### Task 2.1: Function App Health Check

**Execute**:
```bash
# Check Function App status
az functionapp show \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --query "{Name:name, State:state, Location:location, Runtime:siteConfig.linuxFxVersion}"

# Verify all functions are deployed
az functionapp function list \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab
```

**Verify**:
- ✅ Function App state is "Running"
- ✅ All 4 worker functions listed: `ingestion-worker`, `chunking-worker`, `embedding-worker`, `indexing-worker`
- ✅ Python runtime version 3.12
- ✅ Functions Extension Version ~4

**Document**: Record status in `phase_5_testing.md`. If any function is missing, document in `fracas.md`.

**Reference**: `phase_5_azure_functions_deployment.md` - Step 7: Testing Deployment

---

### Task 2.2: Environment Variables Verification

**Execute**:
```bash
az functionapp config appsettings list \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --query "[].{Name:name}" -o table
```

**Verify all required variables are present**:
- `AzureWebJobsStorage` (storage connection string)
- `AZURE_STORAGE_QUEUES_CONNECTION_STRING`
- `DATABASE_URL`
- `SUPABASE_URL`
- `SUPABASE_KEY`
- `AZURE_AI_FOUNDRY_ENDPOINT`
- `AZURE_AI_FOUNDRY_API_KEY`
- `AZURE_SEARCH_ENDPOINT`
- `AZURE_SEARCH_API_KEY`
- `AZURE_SEARCH_INDEX_NAME`
- `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT`
- `AZURE_DOCUMENT_INTELLIGENCE_API_KEY`
- `AZURE_BLOB_CONNECTION_STRING` (if used)
- `AZURE_BLOB_CONTAINER_NAME` (if used)

**Document**: List any missing variables in `fracas.md`. If critical variables are missing, stop and fix before continuing.

**Reference**: 
- `../../setup/environment_variables.md` for variable descriptions
- `phase_5_azure_functions_deployment.md` - Step 2: Configure Environment Variables

---

### Task 2.3: Queue Trigger Configuration Test

**Execute**:
```bash
# Send test message to ingestion-uploads queue
az storage message put \
  --queue-name ingestion-uploads \
  --content '{"document_id":"test-trigger-001","source_storage":"supabase","filename":"test.pdf","attempt":1,"stage":"uploaded"}' \
  --account-name raglabqueues \
  --connection-string "$(az storage account show-connection-string --name raglabqueues --resource-group rag-lab --query connectionString -o tsv)"

# Monitor function execution (watch for 60 seconds)
az functionapp log tail \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab
```

**Verify**:
- ✅ Message successfully enqueued
- ✅ `ingestion-worker` function triggered within 30 seconds
- ✅ Function execution appears in logs
- ✅ No errors in function execution logs

**Document**: 
- Record trigger latency in `phase_5_testing.md`
- Document any errors in `fracas.md`
- If function doesn't trigger, document in `fracas.md` and stop

**Reference**: `phase_5_azure_functions_deployment.md` - Step 7.1: Test Queue Trigger

---

## Phase 3: End-to-End Pipeline Integration Testing

### Task 3.1: Complete RAG Upload Pipeline Test

**Execute**:
```bash
cd backend
source venv/bin/activate
pytest tests/integration/test_phase5_e2e_pipeline.py::TestEndToEndPipeline -v -m integration
```

**Verify the following scenarios**:

1. **Single Document Upload Flow**:
   - Upload PDF via API endpoint (or direct queue message)
   - Verify message enqueued to `ingestion-uploads`
   - Monitor status transitions in Supabase: `uploaded` → `parsed` → `chunked` → `embedded` → `indexed`
   - Verify extracted text persisted in `documents.extracted_text`
   - Verify chunks created in `chunks` table
   - Verify embeddings stored in `chunks.embedding`
   - Verify documents indexed in Azure AI Search
   - Verify timestamps updated: `parsed_at`, `chunked_at`, `embedded_at`, `indexed_at`

2. **Message Passing Between Stages**:
   - Verify `ingestion-worker` enqueues to `ingestion-chunking`
   - Verify `chunking-worker` enqueues to `ingestion-embeddings`
   - Verify `embedding-worker` enqueues to `ingestion-indexing`
   - Verify `indexing-worker` completes without enqueueing

3. **Concurrent Document Processing**:
   - Upload 3-5 documents simultaneously
   - Verify all documents process independently
   - Verify no cross-contamination of data
   - Verify all documents reach `indexed` status

**Success Criteria**:
- ✅ Complete pipeline executes end-to-end
- ✅ All status transitions occur correctly
- ✅ All data persisted to Supabase
- ✅ Documents searchable in Azure AI Search
- ✅ Concurrent processing works correctly
- ✅ Total pipeline time < 5 minutes per document

**Document**: 
- Record execution times, document IDs tested in `phase_5_testing.md`
- Document any failures in `fracas.md`
- Include screenshots of status transitions if possible

**Reference**: 
- `phase_5_testing.md` - Section "3. End-to-End Pipeline Tests"
- `scoping/RFC001.md` - Worker-queue architecture design
- `scoping/PRD001.md` - Product requirements

---

### Task 3.2: Failure Scenario Testing

**Execute**:
```bash
pytest tests/integration/test_phase5_e2e_pipeline.py::TestFailureScenarios -v -m integration
```

**Verify the following failure scenarios**:

1. **Transient Error Recovery**:
   - Simulate transient Azure service error
   - Verify worker retries automatically
   - Verify document eventually processes successfully

2. **Dead-Letter Queue Handling**:
   - Send invalid message format
   - Verify message moved to `ingestion-dead-letter` queue
   - Verify document status set to appropriate failure state (`failed_parsing`, `failed_chunking`, etc.)

3. **Idempotency Testing**:
   - Send duplicate message for already-processed document
   - Verify worker detects and skips processing (no-op)
   - Verify no duplicate data created

4. **Partial Failure Handling**:
   - Test scenario where some chunks fail to index
   - Verify partial success handled gracefully
   - Verify error details stored in document metadata

**Success Criteria**:
- ✅ Retry logic works correctly
- ✅ Dead-letter queue receives poison messages
- ✅ Idempotency prevents duplicate processing
- ✅ Partial failures handled gracefully
- ✅ Error states correctly recorded

**Document**: 
- Document all failure scenarios tested and recovery behavior in `phase_5_testing.md`
- Document any failures in `fracas.md`

**Reference**: 
- `phase_5_testing.md` - Section "3. End-to-End Pipeline Tests"
- `phase_5_decisions.md` - Error handling strategy decisions
- `scoping/RFC001.md` - Failure behavior specifications

---

### Task 3.3: Upload API Endpoint Integration Test

**Execute the following test steps**:

1. **Upload Document via API**:
   ```bash
   # Start API server if not running
   # Then upload document
   curl -X POST http://localhost:8000/api/documents/upload \
     -F "file=@docs/inputs/scan_classic_hmo.pdf" \
     -F "metadata={\"source\":\"test\"}"
   ```

2. **Verify Immediate Response**:
   - ✅ API returns immediately with `document_id`
   - ✅ Response includes status: `uploaded`
   - ✅ No timeout or blocking
   - ✅ Response time < 2 seconds

3. **Monitor Async Processing**:
   - Poll status endpoint: `GET /api/documents/{document_id}/status`
   - Verify status transitions: `uploaded` → `parsed` → `chunked` → `embedded` → `indexed`
   - Verify status updates occur asynchronously (not blocking API)

4. **Verify Final State**:
   - Query Supabase to verify complete processing
   - Verify document searchable in Azure AI Search
   - Verify chunks and embeddings persisted

**Success Criteria**:
- ✅ API responds immediately (< 2 seconds)
- ✅ Document processes asynchronously
- ✅ Status endpoint reflects current state
- ✅ Complete processing succeeds
- ✅ Document searchable after indexing

**Document**: 
- Record API response times, status polling intervals, total processing time in `phase_5_testing.md`
- Document any failures in `fracas.md`

**Reference**: 
- `phase_4_handoff.md` - Upload endpoint implementation details
- `scoping/RFC001.md` - API endpoint specifications

---

## Phase 4: Performance Testing

### Task 4.1: Worker Processing Time

**Execute**:
```bash
pytest tests/integration/test_phase5_performance.py::TestWorkerProcessingTime -v -m integration -m performance
```

**Measure and verify**:
- ⏱️ **Ingestion Worker**: < 30 seconds per document
- ⏱️ **Chunking Worker**: < 10 seconds per document
- ⏱️ **Embedding Worker**: < 60 seconds per document (depends on chunk count)
- ⏱️ **Indexing Worker**: < 30 seconds per document
- ⏱️ **Total Pipeline**: < 5 minutes per document

**Success Criteria**:
- ✅ All workers meet time requirements
- ✅ Processing time scales linearly with document size
- ✅ No performance degradation under load

**Document**: 
- Record processing times, document sizes in `phase_5_testing.md`
- Create performance graphs if possible
- Document any performance issues in `fracas.md`

**Reference**: `scoping/PRD001.md` - Performance requirements section

---

### Task 4.2: Queue Depth and Throughput

**Execute**:
```bash
pytest tests/integration/test_phase5_performance.py::TestQueueDepthHandling -v -m integration -m performance
```

**Test Scenarios**:
- Upload 10 documents simultaneously
- Monitor queue depths for each stage
- Verify queues drain efficiently
- Verify no queue backlog > 20 messages

**Success Criteria**:
- ✅ Average queue depth < 10 messages
- ✅ Maximum queue depth < 20 messages
- ✅ Queues drain within 10 minutes for 10 documents
- ✅ No message loss

**Document**: 
- Record queue depths over time, drain rates in `phase_5_testing.md`
- Document any queue backlog issues in `fracas.md`

---

### Task 4.3: Concurrent Processing

**Execute**:
```bash
pytest tests/integration/test_phase5_performance.py::TestConcurrentProcessing -v -m integration -m performance
```

**Test Scenarios**:
- Upload 5 documents simultaneously
- Verify multiple function instances process in parallel
- Verify no resource contention
- Verify all documents complete successfully

**Success Criteria**:
- ✅ Multiple function instances active simultaneously
- ✅ Concurrent processing improves throughput
- ✅ No data corruption or cross-contamination
- ✅ All documents process successfully

**Document**: 
- Record concurrent instance count, throughput improvement in `phase_5_testing.md`
- Document any concurrency issues in `fracas.md`

---

### Task 4.4: Cold Start Latency

**Execute**:
```bash
pytest tests/integration/test_phase5_performance.py::TestColdStartLatency -v -m integration -m performance
```

**Test Scenarios**:
- Wait 10 minutes (allow functions to scale to zero)
- Send message to trigger cold start
- Measure time from message enqueue to function execution start

**Success Criteria**:
- ✅ Cold start latency < 30 seconds
- ✅ Subsequent invocations have minimal latency (< 5 seconds)

**Document**: 
- Record cold start times, warm start times in `phase_5_testing.md`
- Document any cold start issues in `fracas.md`

---

## Phase 5: Application Insights Monitoring Validation

### Task 5.1: Telemetry Verification

**Execute**:
1. Navigate to Azure Portal → Application Insights → `ai-rag-evaluator`
2. Review Application Insights dashboard
3. Check for function execution logs
4. Verify custom metrics tracked

**Verify metrics**:
- ✅ Function execution count
- ✅ Function execution duration
- ✅ Error rate
- ✅ Queue depth (if custom metrics implemented)
- ✅ Status transition events (if custom events implemented)

**Success Criteria**:
- ✅ All function executions logged
- ✅ Error tracking functional
- ✅ Performance metrics available
- ✅ Custom events captured (if implemented)

**Document**: 
- Screenshot Application Insights dashboard
- Note any missing metrics in `phase_5_testing.md`
- Document monitoring gaps in `fracas.md`

**Reference**: `phase_5_azure_functions_deployment.md` - Step 1.3: Create Application Insights

---

### Task 5.2: Error Monitoring

**Execute**:
1. Review Application Insights for errors
2. Test error scenarios and verify they appear in logs
3. Verify error details include context (document_id, stage, etc.)

**Success Criteria**:
- ✅ Errors appear in Application Insights
- ✅ Error details include sufficient context
- ✅ Alerts configured (if applicable)

**Document**: 
- Document error tracking setup, alert configuration in `phase_5_testing.md`
- Document any monitoring gaps in `fracas.md`

---

## Phase 6: Production Readiness Validation

### Task 6.1: Data Integrity Verification

**Execute**:
1. Upload test document
2. Verify data at each stage:
   - Extracted text matches source
   - Chunks created correctly
   - Embeddings generated (non-zero vectors)
   - Search results return correct document

**Success Criteria**:
- ✅ No data loss through pipeline
- ✅ Data integrity maintained
- ✅ Search results accurate

**Document**: Record data integrity verification results in `phase_5_testing.md`

---

### Task 6.2: Rollback Testing

**Execute**:
1. Document current synchronous path still functional
2. Verify can disable async path if needed
3. Verify can re-enable synchronous processing

**Success Criteria**:
- ✅ Rollback plan documented
- ✅ Synchronous path still available (if applicable)
- ✅ Feature flags functional (if implemented)

**Document**: Record rollback testing results in `phase_5_testing.md`

**Reference**: `phase_5_migration_strategy.md` - Rollback procedures

---

## Test Data Constraints

**CRITICAL**: All tests processing actual PDFs must use only the **first 6 pages** of `docs/inputs/scan_classic_hmo.pdf` to avoid exceeding Azure Document Intelligence budget.

**Implementation**:
- Use `slice_pdf_to_batch()` helper function
- Document constraint in all test results
- Skip tests if PDF not available

**Reference**: `phase_5_testing.md` - "Test Data Constraints" section, `phase_5_decisions.md` - Decision 2: Test Data Constraint

---

## Success Criteria Summary

### Must Pass (Blocking) - All Required:
- [ ] Database migration verification passes
- [ ] All Supabase integration tests pass
- [ ] All end-to-end pipeline tests pass
- [ ] Upload API endpoint works correctly
- [ ] All workers process documents successfully
- [ ] Documents searchable in Azure AI Search

### Should Pass (Non-Blocking) - Document Results:
- [ ] Performance tests meet requirements
- [ ] Queue depth handling works under load
- [ ] Concurrent processing functional
- [ ] Cold start latency acceptable
- [ ] Application Insights monitoring complete

**If any "Must Pass" criteria fails, document in `fracas.md` and stop execution.**

---

## Documentation Requirements

**For each test phase, you MUST**:
1. **Record Results**: Update `phase_5_testing.md` with results, execution times, metrics
2. **Document Failures**: Immediately document any failures in `fracas.md` with:
   - Failure description
   - Steps to reproduce
   - Error messages
   - Impact assessment
3. **Screenshots**: Capture Application Insights dashboards, test outputs (save to appropriate location)
4. **Metrics**: Record all performance metrics, execution times in `phase_5_testing.md`
5. **Issues**: Document any issues, workarounds, or known limitations in `fracas.md`

**Reference Documents**:
- `phase_5_testing.md` - Test results summary template
- `fracas.md` - Failure analysis and corrective action system
- `phase_5_handoff.md` - Handoff document to update with results

---

## Final Steps After Testing

**Once all tests complete (pass or fail), you MUST**:

1. **Update Handoff Document**: 
   - Update `phase_5_handoff.md` with test results summary
   - Include overall status (Pass/Fail/Partial)
   - List all blocking issues

2. **Update Task Tracking**:
   - Mark Phase 5 testing tasks complete in `scoping/TODO001.md`
   - Note any incomplete tasks

3. **Create Summary**:
   - Summarize all test results
   - List all failures from `fracas.md`
   - Provide recommendations for next steps

4. **If All Tests Pass**:
   - Proceed with migration strategy execution (see `phase_5_migration_strategy.md`)
   - Update production documentation
   - Schedule stakeholder review

**Reference**: `phase_5_migration_strategy.md` - Complete migration execution plan

---

## Rollback Plan

**If critical issues found during testing**:

1. **Document Issues**: Document all issues in `fracas.md` with severity (Critical/High/Medium/Low)
2. **Review Rollback**: Review rollback procedures in `phase_5_migration_strategy.md`
3. **Stop Execution**: If critical blocking issues, stop testing and document
4. **Address Issues**: Work with team to address issues before retesting
5. **Re-deploy**: After fixes, re-deploy (see `phase_5_azure_functions_deployment.md`)
6. **Re-test**: Re-run failed tests after fixes

**Reference**: `phase_5_migration_strategy.md` - Rollback procedures section

---

## Quick Reference: Key Documents

**Core Phase 5 Documentation**:
- `phase_5_azure_functions_deployment.md` - Deployment guide and troubleshooting
- `phase_5_migration_guide.md` - Database migration instructions
- `phase_5_migration_strategy.md` - Migration execution plan
- `phase_5_testing.md` - Testing overview and test locations
- `phase_5_decisions.md` - Implementation decisions
- `phase_5_handoff.md` - Phase 5 completion summary

**Configuration & Setup**:
- `../../setup/environment_variables.md` - Environment variable reference
- `backend/scripts/set_function_app_env_vars.sh` - Function App env var script

**Test Files**:
- `backend/tests/integration/test_supabase_phase5.py` - Supabase integration tests
- `backend/tests/integration/test_phase5_e2e_pipeline.py` - End-to-end pipeline tests
- `backend/tests/integration/test_phase5_performance.py` - Performance tests
- `backend/scripts/verify_phase5_migrations.py` - Migration verification script

**Tracking & Documentation**:
- `scoping/TODO001.md` - Task checklist (mark tasks complete)
- `fracas.md` - Failure tracking (document all failures immediately)
- `scoping/PRD001.md` - Product requirements
- `scoping/RFC001.md` - Technical design specifications
- `scoping/context.md` - Project context

**Related Phase Documentation**:
- `phase_4_handoff.md` - Upload endpoint implementation
- `phase_2_azure_integration.md` - Azure Storage Queues setup

---

## Execution Instructions

1. **Read this entire prompt** before starting
2. **Verify all prerequisites** are met before proceeding
3. **Execute tasks in order** - do not skip phases
4. **Document as you go** - do not wait until the end
5. **Stop immediately** if critical blocking issues are found
6. **Update tracking documents** after each phase
7. **Create final summary** when all testing is complete

**Begin execution now. Good luck!**
