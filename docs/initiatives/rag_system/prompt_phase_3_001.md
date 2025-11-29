# Phase 3 Prompt — Embedding Generation

## Purpose
Implement embedding generation for chunks and queries using Azure AI Foundry. This enables vector representation for similarity search.

## Context References

**Primary Documentation:**
- @docs/initiatives/rag_system/PRD001.md - Product requirements (FR1: Upload Pipeline, FR2: Query Pipeline)
- @docs/initiatives/rag_system/RFC001.md - Technical design (Phase 1: Embedding Generation, Decision 2: Single Embedding Model)
- @docs/initiatives/rag_system/TODO001.md - Phase 3 implementation tasks (check off as completed)
- @docs/initiatives/rag_system/context.md - System context

**Codebase References:**
- @backend/rag_eval/core/config.py - Azure AI Foundry configuration
- @backend/rag_eval/core/interfaces.py - Chunk and Query interfaces
- @backend/rag_eval/core/exceptions.py - Error handling (AzureServiceError)
- @backend/rag_eval/services/rag/chunking.py - Chunk generation (Phase 2)

**Implementation Target:**
- **NEW FILE**: `backend/rag_eval/services/rag/embeddings.py` - Embedding generation

## Phase Objectives

1. **Implement Embeddings Module**: Create `embeddings.py` with chunk and query embedding functions
2. **Azure Integration**: Connect to Azure AI Foundry embedding API
3. **Model Consistency**: Enforce same embedding model for chunks and queries
4. **Batch Processing**: Implement batch embedding generation for efficiency
5. **Testing**: Write comprehensive unit tests with mocked Azure services

## Execution Requirements

### Checklist Management
- **REQUIRED**: Check off all [ ] checkboxes in @docs/initiatives/rag_system/TODO001.md Phase 3 section as tasks are completed
- Update TODO001.md directly when each task is finished

### Validation Requirements
- **REQUIRED**: Validate all code with unit tests before marking complete
- Test model consistency validation (chunks and queries use same model)
- Test embedding dimension validation
- Test all error handling paths (100% coverage for error paths)
- Connection tests should warn but not fail if credentials missing

### Documentation Requirements
- **REQUIRED**: Create `phase_3_decisions.md` - Document any decisions not in PRD/RFC
- **REQUIRED**: Create `phase_3_testing.md` - Document testing performed
- **REQUIRED**: Create `phase_3_handoff.md` - Document what's needed for Phase 4
- Add docstrings to all functions
- Document embedding model requirements
- Document retry strategy and error handling

## Key Implementation Tasks

### Core Implementation
- Create `rag_eval/services/rag/embeddings.py` module
- Implement `generate_embeddings(chunks: List[Chunk], config) -> List[List[float]]`:
  - Connect to Azure AI Foundry embedding API
  - Batch embedding generation for efficiency
  - Validate embedding dimensions match expected model output
  - Handle empty chunk list
  - Retry logic (3 retries, exponential backoff)
  - Error handling (raise `AzureServiceError`)
- Implement `generate_query_embedding(query: Query, config) -> List[float]`:
  - Use same embedding model as chunks (enforced via config)
  - Retry logic (3 retries, exponential backoff)
  - Validate embedding dimensions
  - Error handling

### Testing
- Unit tests for `generate_embeddings()` (mocked Azure AI Foundry)
- Unit tests for `generate_query_embedding()` (mocked Azure AI Foundry)
- Test batch processing
- Test empty chunk list handling
- Test embedding dimension validation
- Test model consistency validation
- Test error handling and retries
- Connection test for Azure AI Foundry (warns if credentials missing)

### Documentation
- Function docstrings
- Embedding model requirements documentation
- Retry strategy documentation
- Error handling documentation
- Phase 3 testing summary

## Success Criteria

- [x] `embeddings.py` module created and implemented
- [x] `generate_embeddings()` function complete with batch processing
- [x] `generate_query_embedding()` function complete
- [x] Model consistency enforced (same model for chunks and queries)
- [x] All Phase 3 checkboxes in TODO001.md are checked
- [x] Unit tests written and passing (20 tests)
- [x] Connection tests implemented and passing (2 tests - Azure AI Foundry validated)
- [x] All error paths tested (100% coverage)
- [x] phase_3_decisions.md created
- [x] phase_3_testing.md created
- [x] phase_3_handoff.md created
- [x] All failures documented in fracas.md (if any) - No failures encountered

**Status**: ✅ **Phase 3 Complete** (2025-01-27)
- All 22 tests passing
- Azure AI Foundry connection validated
- Endpoint format fix applied

## Dependencies

- Azure AI Foundry credentials in config -> use virtual environment for testing
- Embedding model name in config (default: "text-embedding-3-small")
- `azure-ai-inference` or OpenAI-compatible client library installed
- Chunking component (Phase 2) complete

## Next Phase

Once Phase 3 is complete, proceed to **Phase 4 — Azure AI Search Integration**.

---

**Document Status**: Active  
**Last Updated**: 2025-01-27  
**Related Documents**: [PRD001.md](./PRD001.md), [RFC001.md](./RFC001.md), [TODO001.md](./TODO001.md)

