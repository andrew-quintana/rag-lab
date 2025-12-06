# Phase 4 Prompt — Azure AI Search Integration

## Purpose
Implement Azure AI Search integration for vector similarity search. This enables chunk indexing and retrieval for the RAG pipeline.

## Context References

**Primary Documentation:**
- @docs/initiatives/rag_system/scoping/PRD001.md - Product requirements (FR1: Upload Pipeline, FR2: Query Pipeline)
- @docs/initiatives/rag_system/scoping/RFC001.md - Technical design (Phase 2: Azure AI Search Integration, Decision 6: Top-K Retrieval Default)
- @docs/initiatives/rag_system/scoping/TODO001.md - Phase 4 implementation tasks (check off as completed)
- @docs/initiatives/rag_system/scoping/context.md - System context

**Codebase References:**
- @backend/rag_eval/core/config.py - Azure AI Search configuration
- @backend/rag_eval/core/interfaces.py - Chunk, Query, RetrievalResult interfaces
- @backend/rag_eval/core/exceptions.py - Error handling (AzureServiceError)
- @backend/rag_eval/services/rag/embeddings.py - Embedding generation (Phase 3)

**Implementation Target:**
- **EXISTING FILE**: `backend/rag_eval/services/rag/search.py` - Azure AI Search operations

## Phase Objectives

1. **Implement Search Module**: Complete `search.py` with indexing and retrieval functions
2. **Index Management**: Implement idempotent index creation with lightweight schema
3. **Vector Search**: Implement vector similarity search (cosine similarity)
4. **Top-K Retrieval**: Implement configurable top-k retrieval (default: 5)
5. **Testing**: Write comprehensive unit tests with mocked Azure services

## Execution Requirements

### Checklist Management
- **REQUIRED**: Check off all [ ] checkboxes in @docs/initiatives/rag_system/scoping/TODO001.md Phase 4 section as tasks are completed
- Update TODO001.md directly when each task is finished

### Validation Requirements
- **REQUIRED**: Validate all code with unit tests before marking complete
- Test idempotent index creation (never destructive)
- Test vector similarity search and ranking
- Test reproducibility (same query = same results)
- Test all error handling paths (100% coverage for error paths)
- Connection tests should warn but not fail if credentials missing

### Documentation Requirements
- **REQUIRED**: Create `phase_4_decisions.md` - Document any decisions not in PRD/RFC
- **REQUIRED**: Create `phase_4_testing.md` - Document testing performed
- **REQUIRED**: Create `phase_4_handoff.md` - Document what's needed for Phase 5
- Add docstrings to all functions
- Document index schema and requirements
- Document retrieval parameters (top_k default: 5)
- Document idempotent index creation strategy

## Key Implementation Tasks

### Core Implementation
- Implement index creation logic (idempotent):
  - Check if index exists before creation
  - Create index with lightweight schema if not exists:
    - `id`: Chunk ID (string, key)
    - `chunk_text`: Chunk content (string, searchable)
    - `embedding`: Vector embedding (Collection(Edm.Single), vectorizable)
    - `document_id`: Source document ID (string)
    - `metadata`: Additional metadata (JSON)
  - Never perform destructive resets (preserve existing data)
  - Handle index creation errors gracefully
- Implement `index_chunks(chunks: List[Chunk], embeddings: List[List[float]], config) -> None`:
  - Ensure index exists (create if needed, idempotent)
  - Batch index chunks with embeddings and metadata
  - Validate chunks and embeddings match in length
  - Handle empty chunk list
  - Retry logic with exponential backoff
  - Error handling (raise `AzureServiceError`)
- Implement `retrieve_chunks(query: Query, top_k: int = 5, config=None) -> List[RetrievalResult]`:
  - Generate query embedding (use Phase 3 implementation)
  - Perform vector similarity search (cosine similarity)
  - Retrieve top-k chunks with similarity scores
  - Return `RetrievalResult` objects with metadata
  - Handle empty index gracefully (return empty list)
  - Handle query validation errors
  - Retry logic with exponential backoff
  - Error handling

### Testing
- Unit tests for index creation (mocked Azure AI Search)
- Unit tests for `index_chunks()` (mocked Azure AI Search)
- Unit tests for `retrieve_chunks()` (mocked Azure AI Search)
- Test idempotent index creation
- Test vector similarity search and ranking
- Test top-k retrieval logic
- Test similarity score calculation
- Test metadata retrieval with chunks
- Test empty index handling
- Test reproducibility (same query = same results)
- Connection test for Azure AI Search (warns if credentials missing)

### Documentation
- Function docstrings
- Index schema documentation
- Retrieval parameters documentation (top_k default: 5)
- Idempotent index creation strategy documentation
- Phase 4 testing summary

## Success Criteria

- [ ] Index creation logic implemented (idempotent)
- [ ] `index_chunks()` function complete
- [ ] `retrieve_chunks()` function complete with vector search
- [ ] All Phase 4 checkboxes in TODO001.md are checked
- [ ] Unit tests written and passing (mocked Azure)
- [ ] Connection test implemented (warns if credentials missing)
- [ ] All error paths tested (100% coverage)
- [ ] Reproducibility validated (same query = same results)
- [ ] phase_4_decisions.md created
- [ ] phase_4_testing.md created
- [ ] phase_4_handoff.md created
- [ ] All failures documented in fracas.md (if any)

## Dependencies

- Azure AI Search credentials in config -> use virtual environment for testing
- `azure-search-documents` client library installed
- Index name in config
- Embedding generation (Phase 3) complete

## Next Phase

Once Phase 4 is complete, proceed to **Phase 5 — Prompt Template System**.

---

**Document Status**: Active  
**Last Updated**: 2025-01-27  
**Related Documents**: [PRD001.md](./PRD001.md), [RFC001.md](./RFC001.md), [TODO001.md](./TODO001.md)

