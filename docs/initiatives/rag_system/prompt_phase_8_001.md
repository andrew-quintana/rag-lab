# Phase 8 Prompt — Supabase Logging

## Purpose
Implement Supabase logging for query pipeline operations. This enables observability and analysis of RAG pipeline behavior.

## Context References

**Primary Documentation:**
- @docs/initiatives/rag_system/PRD001.md - Product requirements (FR3: Logging and Observability)
- @docs/initiatives/rag_system/RFC001.md - Technical design (Phase 6: Supabase Logging)
- @docs/initiatives/rag_system/TODO001.md - Phase 8 implementation tasks (check off as completed)
- @docs/initiatives/rag_system/context.md - System context

**Codebase References:**
- @backend/rag_eval/db/queries.py - QueryExecutor interface
- @backend/rag_eval/db/connection.py - Supabase connection
- @backend/rag_eval/core/interfaces.py - Query, RetrievalResult, ModelAnswer interfaces
- @backend/rag_eval/services/rag/pipeline.py - Pipeline orchestration (Phase 7)
- @infra/supabase/migrations/0001_init.sql - Database schema (queries, retrieval_logs, model_answers tables)

**Implementation Target:**
- **NEW FILE**: `backend/rag_eval/services/rag/logging.py` - Supabase logging functions

## Phase Objectives

1. **Implement Logging Module**: Create `logging.py` with query, retrieval, and answer logging functions
2. **Database Integration**: Log to Supabase tables (queries, retrieval_logs, model_answers)
3. **Error Resilience**: Ensure logging failures don't break pipeline
4. **Batch Operations**: Implement batch insertion for retrieval logs
5. **Testing**: Write comprehensive unit tests with mocked Supabase

## Execution Requirements

### Checklist Management
- **REQUIRED**: Check off all [ ] checkboxes in @docs/initiatives/rag_system/TODO001.md Phase 8 section as tasks are completed
- Update TODO001.md directly when each task is finished

### Validation Requirements
- **REQUIRED**: Validate all code with unit tests before marking complete
- Test that logging failures don't break pipeline
- Test batch insertion for retrieval logs
- Test query ID and answer ID generation
- Test all error handling paths (100% coverage for error paths)
- Connection tests should warn but not fail if credentials missing

### Documentation Requirements
- **REQUIRED**: Create `phase_8_decisions.md` - Document any decisions not in PRD/RFC
- **REQUIRED**: Create `phase_8_testing.md` - Document testing performed
- **REQUIRED**: Create `phase_8_handoff.md` - Document what's needed for Phase 9
- Add docstrings to all functions
- Document database schema and table relationships
- Document logging failure handling strategy
- Document batch insertion strategy

## Key Implementation Tasks

### Core Implementation
- Implement `log_query(query: Query, query_executor: QueryExecutor) -> str`:
  - Insert query into `queries` table
  - Generate query ID if missing
  - Return query ID
  - Handle logging failures gracefully (don't fail pipeline)
  - Error handling
- Implement `log_retrieval(query_id: str, retrieval_results: List[RetrievalResult], query_executor: QueryExecutor) -> None`:
  - Batch insert retrieval logs into `retrieval_logs` table
  - Log chunk_id, similarity_score for each retrieval result
  - Handle empty retrieval results
  - Handle logging failures gracefully (don't fail pipeline)
  - Error handling
- Implement `log_model_answer(answer: ModelAnswer, query_executor: QueryExecutor) -> str`:
  - Insert model answer into `model_answers` table
  - Generate answer ID if missing
  - Return answer ID
  - Handle logging failures gracefully (don't fail pipeline)
  - Error handling
- Update `run_rag()` in pipeline.py to call logging functions:
  - Log query at start of pipeline
  - Log retrieval results after retrieval
  - Log model answer after generation

### Testing
- Ensure to test using the virtual environment
- Unit tests for `log_query()` (mocked Supabase)
- Unit tests for `log_retrieval()` (mocked Supabase)
- Unit tests for `log_model_answer()` (mocked Supabase)
- Test query ID generation
- Test answer ID generation
- Test batch insertion for retrieval logs
- Test empty retrieval results
- Test error handling (logging failures shouldn't break pipeline)
- Connection test for Supabase (warns if credentials missing)
- Test that logging failures don't break pipeline

### Documentation
- Function docstrings
- Database schema and table relationships documentation
- Logging failure handling strategy documentation
- Batch insertion strategy documentation
- Phase 8 testing summary

## Success Criteria

- [ ] `logging.py` module created and implemented
- [ ] `log_query()` function complete
- [ ] `log_retrieval()` function complete with batch insertion
- [ ] `log_model_answer()` function complete
- [ ] `run_rag()` updated to call logging functions
- [ ] All Phase 8 checkboxes in TODO001.md are checked
- [ ] Unit tests written and passing (mocked Supabase)
- [ ] Connection test implemented (warns if credentials missing)
- [ ] All error paths tested (100% coverage)
- [ ] Logging failures don't break pipeline (validated)
- [ ] phase_8_decisions.md created
- [ ] phase_8_testing.md created
- [ ] phase_8_handoff.md created
- [ ] All failures documented in fracas.md (if any)

## Dependencies

- Supabase connection and database schema
- `QueryExecutor` from `rag_eval/db/queries.py` available
- Pipeline orchestration (Phase 7) complete

## Next Phase

Once Phase 8 is complete, proceed to **Phase 9 — Upload Pipeline Integration**.

---

**Document Status**: Active  
**Last Updated**: 2025-01-27  
**Related Documents**: [PRD001.md](./PRD001.md), [RFC001.md](./RFC001.md), [TODO001.md](./TODO001.md)

