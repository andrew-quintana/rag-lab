# Phase 1 Prompt — Document Upload (Azure Blob Storage)

## Purpose
Implement Azure Blob Storage integration for document persistence. This component enables the upload pipeline to store raw documents before processing.

## Context References

**Primary Documentation:**
- @docs/initiatives/rag_system/scoping/PRD001.md - Product requirements (FR1: Upload Pipeline)
- @docs/initiatives/rag_system/scoping/RFC001.md - Technical design (Phase 1: Document Upload)
- @docs/initiatives/rag_system/scoping/TODO001.md - Phase 1 implementation tasks (check off as completed)
- @docs/initiatives/rag_system/scoping/context.md - System context

**Codebase References:**
- @backend/rag_eval/core/config.py - Azure Blob Storage configuration
- @backend/rag_eval/core/exceptions.py - Error handling (AzureServiceError)
- @backend/rag_eval/services/rag/ingestion.py - Ingestion component (will use storage)
- @backend/rag_eval/api/routes/upload.py - Upload endpoint (will integrate storage)

**Implementation Target:**
- **NEW FILE**: `backend/rag_eval/services/rag/storage.py` - Azure Blob Storage operations

## Phase Objectives

1. **Implement Storage Module**: Create `storage.py` with blob upload/download functions
2. **Azure Integration**: Connect to Azure Blob Storage with retry logic
3. **Error Handling**: Implement proper error handling and raise `AzureServiceError`
4. **Testing**: Write comprehensive unit tests with mocked Azure services
5. **Connection Testing**: Verify actual Azure connection (warn if credentials missing)

## Execution Requirements

### Checklist Management
- **REQUIRED**: Check off all [ ] checkboxes in @docs/initiatives/rag_system/scoping/TODO001.md Phase 1 section as tasks are completed
- Update TODO001.md directly when each task is finished

### Validation Requirements
- **REQUIRED**: Validate all code with unit tests before marking complete
- Test error handling paths (100% coverage for error paths)
- Test retry logic with exponential backoff
- Connection tests should warn but not fail if credentials missing

### Documentation Requirements
- **REQUIRED**: Create `phase_1_decisions.md` - Document any decisions not in PRD/RFC
- **REQUIRED**: Create `phase_1_testing.md` - Document testing performed
- **REQUIRED**: Create `phase_1_handoff.md` - Document what's needed for Phase 2
- Add docstrings to all functions
- Document retry strategy and error handling

## Key Implementation Tasks

### Core Implementation
- Create `rag_eval/services/rag/storage.py` module
- Implement `upload_document_to_blob()` with:
  - Azure Blob Storage connection
  - Container creation (idempotent)
  - File upload with metadata
  - Retry logic (3 retries, exponential backoff)
  - Error handling (raise `AzureServiceError`)
- Implement `download_document_from_blob()` (optional, for future use)

### Testing
- Unit tests with mocked Azure Blob Storage
- Test container creation (idempotent)
- Test error handling and retries
- Connection test (warns if credentials missing, doesn't fail tests)

### Documentation
- Function docstrings
- Retry strategy documentation
- Error handling documentation
- Phase 1 testing summary

## Success Criteria

- [ ] `storage.py` module created and implemented
- [ ] `upload_document_to_blob()` function complete with retry logic
- [ ] All Phase 1 checkboxes in TODO001.md are checked
- [ ] Unit tests written and passing (mocked Azure)
- [ ] Connection test implemented (warns if credentials missing)
- [ ] All error paths tested (100% coverage)
- [ ] phase_1_decisions.md created
- [ ] phase_1_testing.md created
- [ ] phase_1_handoff.md created
- [ ] All failures documented in fracas.md (if any)

## Dependencies

- Azure Blob Storage credentials in config -> use virtual environment for testing
- `azure-storage-blob` Python package installed
- Container name configured

## Next Phase

Once Phase 1 is complete, proceed to **Phase 2 — Extraction, Preprocessing, and Chunking**.

---

**Document Status**: Active  
**Last Updated**: 2025-01-27  
**Related Documents**: [PRD001.md](./PRD001.md), [RFC001.md](./RFC001.md), [TODO001.md](./TODO001.md)

