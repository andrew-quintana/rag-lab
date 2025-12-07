# Phase 4 Prompt — API Integration

## Context

This prompt guides the implementation of **Phase 4: API Integration** for the RAG Ingestion Worker–Queue Architecture Conversion. This phase modifies the FastAPI upload endpoint to enqueue messages instead of processing synchronously, and adds a status query endpoint.

**Related Documents:**
- @docs/initiatives/rag_system/worker_queue_conversion/scoping/PRD001.md - Product requirements
- @docs/initiatives/rag_system/worker_queue_conversion/scoping/RFC001.md - Technical design (API Endpoints section)
- @docs/initiatives/rag_system/worker_queue_conversion/scoping/TODO001.md - Implementation tasks (Phase 4 section - check off tasks as completed)
- @docs/initiatives/rag_system/worker_queue_conversion/scoping/context.md - Project context

## Objectives

1. **Upload Endpoint**: Modify upload endpoint to enqueue messages instead of synchronous processing
2. **Status Endpoint**: Add status query endpoint to track document processing progress
3. **Response Models**: Update response models for new asynchronous flow
4. **Testing**: Create comprehensive unit tests for API endpoints

## Execution Requirements

### Checklist Modifications
- **REQUIRED**: Check off completed tasks in @docs/initiatives/rag_system/worker_queue_conversion/scoping/TODO001.md Phase 4 section as you complete them
- Update phase status when all validation requirements are met

### Validation
- **REQUIRED**: All unit tests for Phase 4 must pass before proceeding to Phase 5
- **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/api/test_upload_endpoint.py -v`
- **REQUIRED**: Test coverage must meet minimum 80% for API endpoints
- **REQUIRED**: All test assertions must pass (no failures, no errors)

### Documentation
- **REQUIRED**: Create `phase_4_decisions.md` if any decisions are made that aren't in PRD/RFC
- **REQUIRED**: Create `phase_4_testing.md` documenting testing results
- **REQUIRED**: Create `phase_4_handoff.md` summarizing what's needed for Phase 5

## Key References

### Existing API
- @backend/api/routes/upload.py - Existing upload endpoint (or equivalent location)
- Existing FastAPI application structure

### Queue Client
- @backend/rag_eval/services/workers/queue_client.py - Queue operations from Phase 2

### Persistence Layer
- @backend/rag_eval/services/workers/persistence.py - Status query functions from Phase 1

### Response Models
- `UploadResponse` with `document_id` and `status`
- `DocumentStatusResponse` with status, timestamps, error details

## Phase 4 Tasks

### Upload Endpoint Modification
1. Modify upload endpoint (`api/routes/upload.py` or equivalent):
   - Change from synchronous processing to enqueueing message
   - Upload file to storage (Supabase or Azure Blob) - keep existing logic
   - Create document record in database with `status = 'uploaded'`
   - Enqueue message to `ingestion-uploads` queue
   - Return immediately with `document_id` and `status='uploaded'`
   - Add backward compatibility flag (optional synchronous path) if needed
2. Update `UploadResponse` model:
   - `document_id: str`
   - `status: str` (default: "uploaded")
   - `message: str` (optional)

### Status Query Endpoint
1. Add status query endpoint:
   - `GET /documents/{document_id}/status`
   - Query document status from database
   - Return current status, timestamps (`parsed_at`, `chunked_at`, `embedded_at`, `indexed_at`), and error details if failed
2. Implement `DocumentStatusResponse` model:
   - `document_id: str`
   - `status: str`
   - `parsed_at: Optional[datetime]`
   - `chunked_at: Optional[datetime]`
   - `embedded_at: Optional[datetime]`
   - `indexed_at: Optional[datetime]`
   - `error_details: Optional[str]` (if status is `failed_*`)

### Testing
1. Create test file: `backend/tests/components/api/test_upload_endpoint.py`
2. Test upload endpoint enqueues message correctly
3. Test upload endpoint returns immediately with document_id
4. Test status query endpoint returns correct status
5. Test error handling for invalid requests
6. Test backward compatibility flag (synchronous path) if implemented
7. Test response model validation
8. Test API error responses
9. Test status transitions (uploaded → parsed → chunked → embedded → indexed)

### Documentation
1. Add docstrings to all API functions
2. Document API endpoint changes
3. Document response models
4. Document backward compatibility approach

## Success Criteria

- [ ] Upload endpoint modified to enqueue messages
- [ ] Status query endpoint implemented
- [ ] Response models updated
- [ ] All unit tests pass (minimum 80% coverage)
- [ ] All Phase 4 tasks in TODO001.md checked off
- [ ] Phase 4 handoff document created

## Important Notes

- **Asynchronous Flow**: Upload endpoint returns immediately; processing happens asynchronously via workers
- **Status Tracking**: Users can query document status to track progress through pipeline stages
- **Backward Compatibility**: Consider adding optional synchronous path for migration period
- **Error Handling**: Status endpoint should return error details if document processing failed

## Blockers

- **BLOCKER**: Phase 5 cannot proceed until Phase 4 validation complete
- **BLOCKER**: Workers (Phase 3) must be implemented before API integration

## Next Phase

After completing Phase 4, proceed to **Phase 5: Integration Testing & Migration** using @docs/initiatives/rag_system/worker_queue_conversion/prompts/prompt_phase_5_001.md

