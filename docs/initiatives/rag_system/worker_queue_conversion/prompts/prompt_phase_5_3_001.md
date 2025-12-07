# Prompt: Phase 5.3 - End-to-End Pipeline Integration Testing

## Your Mission

Execute Phase 3 of Phase 5 integration testing: End-to-End Pipeline Integration Testing. You must validate the complete RAG upload pipeline from document upload through indexing, verifying message passing, status transitions, and data persistence using **real Azure resources** (Azure Functions, Storage Queues, Supabase, Azure AI Search). Document all results, failures, and metrics as you progress.

## Context & References

**Key Documentation to Reference**:
- `phase_5_testing.md` - Testing overview and test locations
- `phase_5_azure_functions_deployment.md` - Azure Functions deployment guide
- `phase_5_decisions.md` - Implementation decisions
- `phase_5_handoff.md` - Phase 5 completion summary
- `scoping/RFC001.md` - Worker-queue architecture design
- `scoping/PRD001.md` - Product requirements
- `fracas.md` - Document all failures immediately
- `scoping/TODO001.md` - Task checklist (mark tasks complete)

**Prerequisites** (from Phase 5.2):
- ✅ Azure Function App running
- ✅ All 4 worker functions deployed
- ✅ Environment variables configured
- ✅ Queue triggers verified

## Prerequisites Check

**Before starting, verify**:
1. Azure Functions deployed and running (from Phase 5.1)
2. All queues accessible: `ingestion-uploads`, `ingestion-chunking`, `ingestion-embeddings`, `ingestion-indexing`, `ingestion-dead-letter`
3. Supabase database accessible and migrations applied
4. Azure AI Search index accessible
5. Test PDF available: `docs/inputs/scan_classic_hmo.pdf` (use only first 6 pages)
6. API server can be started (for Task 3.3)

**If any prerequisite is missing, document it in `fracas.md` and stop execution.**

---

## Task 3.1: Complete RAG Upload Pipeline Test

**Execute**:
```bash
cd backend
source venv/bin/activate
pytest tests/integration/test_phase5_e2e_pipeline.py::TestEndToEndPipeline -v -m integration
```

**Verify the following scenarios**:

### 1. Single Document Upload Flow
- Upload PDF via API endpoint (or direct queue message)
- Verify message enqueued to `ingestion-uploads`
- Monitor status transitions in Supabase: `uploaded` → `parsed` → `chunked` → `embedded` → `indexed`
- Verify extracted text persisted in `documents.extracted_text`
- Verify chunks created in `chunks` table
- Verify embeddings stored in `chunks.embedding`
- Verify documents indexed in Azure AI Search
- Verify timestamps updated: `parsed_at`, `chunked_at`, `embedded_at`, `indexed_at`

### 2. Message Passing Between Stages
- Verify `ingestion-worker` enqueues to `ingestion-chunking`
- Verify `chunking-worker` enqueues to `ingestion-embeddings`
- Verify `embedding-worker` enqueues to `ingestion-indexing`
- Verify `indexing-worker` completes without enqueueing

### 3. Concurrent Document Processing
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
- Include status transition timeline
- Record queue message flow between stages

**Reference**: 
- `phase_5_testing.md` - Section "3. End-to-End Pipeline Tests"
- `scoping/RFC001.md` - Worker-queue architecture design
- `scoping/PRD001.md` - Product requirements

---

## Task 3.2: Failure Scenario Testing

**Execute**:
```bash
pytest tests/integration/test_phase5_e2e_pipeline.py::TestFailureScenarios -v -m integration
```

**Verify the following failure scenarios**:

### 1. Transient Error Recovery
- Simulate transient Azure service error
- Verify worker retries automatically
- Verify document eventually processes successfully
- Verify retry count tracked in message metadata

### 2. Dead-Letter Queue Handling
- Send invalid message format
- Verify message moved to `ingestion-dead-letter` queue
- Verify document status set to appropriate failure state (`failed_parsing`, `failed_chunking`, etc.)
- Verify error details stored in document metadata

### 3. Idempotency Testing
- Send duplicate message for already-processed document
- Verify worker detects and skips processing (no-op)
- Verify no duplicate data created
- Verify status remains unchanged

### 4. Partial Failure Handling
- Test scenario where some chunks fail to index
- Verify partial success handled gracefully
- Verify error details stored in document metadata
- Verify successfully indexed chunks remain searchable

**Success Criteria**:
- ✅ Retry logic works correctly
- ✅ Dead-letter queue receives poison messages
- ✅ Idempotency prevents duplicate processing
- ✅ Partial failures handled gracefully
- ✅ Error states correctly recorded

**Document**: 
- Document all failure scenarios tested and recovery behavior in `phase_5_testing.md`
- Document any failures in `fracas.md`
- Record retry counts and failure recovery times

**Reference**: 
- `phase_5_testing.md` - Section "3. End-to-End Pipeline Tests"
- `phase_5_decisions.md` - Error handling strategy decisions
- `scoping/RFC001.md` - Failure behavior specifications

---

## Task 3.3: Upload API Endpoint Integration Test

**Execute the following test steps**:

### 1. Start API Server
```bash
cd backend
source venv/bin/activate
# Start API server in background or separate terminal
uvicorn rag_eval.api.main:app --host 0.0.0.0 --port 8000
```

### 2. Upload Document via API
```bash
# Upload document (use only first 6 pages)
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@docs/inputs/scan_classic_hmo.pdf" \
  -F "metadata={\"source\":\"test\"}"
```

**Verify Immediate Response**:
- ✅ API returns immediately with `document_id`
- ✅ Response includes status: `uploaded`
- ✅ No timeout or blocking
- ✅ Response time < 2 seconds

### 3. Monitor Async Processing
```bash
# Poll status endpoint
curl http://localhost:8000/api/documents/{document_id}/status
```

**Verify**:
- ✅ Status transitions: `uploaded` → `parsed` → `chunked` → `embedded` → `indexed`
- ✅ Status updates occur asynchronously (not blocking API)
- ✅ Timestamps update as status changes
- ✅ Error details included if processing fails

### 4. Verify Final State
- Query Supabase to verify complete processing
- Verify document searchable in Azure AI Search
- Verify chunks and embeddings persisted
- Verify all timestamps populated

**Success Criteria**:
- ✅ API responds immediately (< 2 seconds)
- ✅ Document processes asynchronously
- ✅ Status endpoint reflects current state
- ✅ Complete processing succeeds
- ✅ Document searchable after indexing

**Document**: 
- Record API response times, status polling intervals, total processing time in `phase_5_testing.md`
- Document any failures in `fracas.md`
- Record status transition timeline
- Include API response examples

**Reference**: 
- `phase_4_handoff.md` - Upload endpoint implementation details
- `scoping/RFC001.md` - API endpoint specifications

---

## Test Data Constraints

**CRITICAL**: All tests processing actual PDFs must use only the **first 6 pages** of `docs/inputs/scan_classic_hmo.pdf` to avoid exceeding Azure Document Intelligence budget.

**Implementation**:
- Use `slice_pdf_to_batch()` helper function
- Document constraint in all test results
- Skip tests if PDF not available

**Reference**: `phase_5_testing.md` - "Test Data Constraints" section, `phase_5_decisions.md` - Decision 2: Test Data Constraint

---

## Success Criteria

### Must Pass (Blocking) - All Required:
- [ ] Complete pipeline executes end-to-end
- [ ] All status transitions occur correctly
- [ ] All data persisted to Supabase
- [ ] Documents searchable in Azure AI Search
- [ ] Upload API endpoint works correctly
- [ ] Status endpoint reflects current state

### Should Pass (Non-Blocking) - Document Results:
- [ ] Concurrent processing works correctly
- [ ] Failure scenarios handled gracefully
- [ ] Retry logic works correctly
- [ ] Idempotency prevents duplicate processing
- [ ] Total pipeline time < 5 minutes per document

**If any "Must Pass" criteria fails, document in `fracas.md` and stop execution.**

---

## Documentation Requirements

**You MUST**:
1. **Record Results**: Update `phase_5_testing.md` with:
   - End-to-end pipeline test results
   - Status transition timelines
   - Message passing verification
   - API endpoint test results
   - Failure scenario test results
   - Execution times and metrics

2. **Document Failures**: Immediately document any failures in `fracas.md` with:
   - Failure description
   - Steps to reproduce
   - Error messages
   - Impact assessment
   - Severity (Critical/High/Medium/Low)

3. **Update Tracking**: 
   - Mark Phase 3 tasks complete in `scoping/TODO001.md`
   - Update `phase_5_handoff.md` with Phase 3 status

---

## Next Steps

**If all Phase 3 tasks pass**:
- Proceed to Phase 5.4: Performance Testing
- Reference: `prompt_phase_5_4_001.md`

**If Phase 3 tasks fail**:
- Document all issues in `fracas.md`
- Address critical blockers before proceeding
- Re-run Phase 3 tests after fixes

---

## Quick Reference

**Test Files**:
- `backend/tests/integration/test_phase5_e2e_pipeline.py` - End-to-end pipeline tests

**Key Files**:
- `phase_5_testing.md` - Test results
- `fracas.md` - Failure tracking
- `phase_5_handoff.md` - Handoff document
- `scoping/TODO001.md` - Task tracking

---

**Begin execution now. Good luck!**

