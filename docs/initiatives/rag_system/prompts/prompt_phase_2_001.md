# Phase 2 Prompt — Extraction, Preprocessing, and Chunking

## Purpose
Complete unit tests for existing ingestion and chunking components. Validate and ensure deterministic behavior for document processing.

## Context References

**Primary Documentation:**
- @docs/initiatives/rag_system/scoping/PRD001.md - Product requirements (FR1: Upload Pipeline, NFR5: Determinism)
- @docs/initiatives/rag_system/scoping/RFC001.md - Technical design (Decision 1: Deterministic Chunking Strategy)
- @docs/initiatives/rag_system/scoping/TODO001.md - Phase 2 implementation tasks (check off as completed)
- @docs/initiatives/rag_system/scoping/context.md - System context

**Codebase References:**
- @backend/rag_eval/services/rag/ingestion.py - **EXISTING** - Ingestion component (needs unit tests)
- @backend/rag_eval/services/rag/chunking.py - **EXISTING** - Chunking component (needs unit tests)
- @backend/rag_eval/core/config.py - Azure Document Intelligence configuration
- @backend/rag_eval/core/exceptions.py - Error handling
- @backend/tests/fixtures/sample_documents/ - Sample documents for testing

**Test Files:**
- Create unit tests in `backend/tests/test_rag_ingestion.py`
- Create unit tests in `backend/tests/test_rag_chunking.py`

## Phase Objectives

1. **Review Existing Code**: Validate ingestion and chunking implementations
2. **Unit Testing**: Write comprehensive unit tests for both components
3. **Determinism Validation**: Ensure chunking is deterministic and reproducible
4. **Error Handling**: Validate error handling and retry logic
5. **Connection Testing**: Verify Azure Document Intelligence connection

## Execution Requirements

### Checklist Management
- **REQUIRED**: Check off all [ ] checkboxes in @docs/initiatives/rag_system/scoping/TODO001.md Phase 2 section as tasks are completed
- Update TODO001.md directly when each task is finished

### Validation Requirements
- **REQUIRED**: Validate all code with unit tests before marking complete
- Test deterministic behavior (same input = same output)
- Test all error handling paths (100% coverage for error paths)
- Connection tests should warn but not fail if credentials missing

### Documentation Requirements
- **REQUIRED**: Create `phase_2_decisions.md` - Document any decisions not in PRD/RFC
- **REQUIRED**: Create `phase_2_testing.md` - Document testing performed
- **REQUIRED**: Create `phase_2_handoff.md` - Document what's needed for Phase 3
- Review and update docstrings for all functions
- Document deterministic behavior requirements

## Key Implementation Tasks

### Code Review & Validation
- Review `extract_text_from_document()` implementation
- Review `ingest_document()` implementation
- Review `chunk_text_fixed_size()` implementation (ensure deterministic)
- Review `chunk_text_with_llm()` implementation (ensure fallback)
- Review `chunk_text()` implementation (ensure default behavior)

### Testing
- Unit tests for `extract_text_from_document()` (mocked Azure Document Intelligence)
- Unit tests for `ingest_document()`
- Unit tests for `chunk_text_fixed_size()` (deterministic, reproducible)
- Unit tests for `chunk_text_with_llm()` (mocked, with fallback)
- Unit tests for `chunk_text()` (default behavior)
- Connection test for Azure Document Intelligence (warns if credentials missing)
- Connection test for Azure AI Foundry (for LLM chunking, warns if credentials missing)

### Documentation
- Review and update function docstrings
- Document extraction capabilities and limitations
- Document chunking strategies and when to use each
- Document deterministic behavior requirements
- Phase 2 testing summary

## Success Criteria

- [ ] All existing code reviewed and validated
- [ ] All Phase 2 checkboxes in TODO001.md are checked
- [ ] Unit tests written and passing for ingestion component
- [ ] Unit tests written and passing for chunking component
- [ ] Deterministic behavior validated (same input = same chunks)
- [ ] All error paths tested (100% coverage)
- [ ] Connection tests implemented (warn if credentials missing)
- [ ] phase_2_decisions.md created
- [ ] phase_2_testing.md created
- [ ] phase_2_handoff.md created
- [ ] All failures documented in fracas.md (if any)

## Dependencies

- Azure Document Intelligence credentials in config -> use virtual environment for testing
- Azure AI Foundry credentials in config (for LLM chunking) -> use virtual environment for testing
- Sample documents in `backend/tests/fixtures/sample_documents/`

## Next Phase

Once Phase 2 is complete, proceed to **Phase 3 — Embedding Generation**.

---

**Document Status**: Active  
**Last Updated**: 2025-01-27  
**Related Documents**: [PRD001.md](./PRD001.md), [RFC001.md](./RFC001.md), [TODO001.md](./TODO001.md)

