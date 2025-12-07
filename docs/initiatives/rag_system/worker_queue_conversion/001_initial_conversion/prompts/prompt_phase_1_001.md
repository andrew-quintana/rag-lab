# Phase 1 Prompt — Persistence Infrastructure

## Context

This prompt guides the implementation of **Phase 1: Persistence Infrastructure** for the RAG Ingestion Worker–Queue Architecture Conversion. This phase implements the persistence layer needed to store intermediate data (extracted text, chunks, embeddings) between pipeline stages, enabling worker independence.

**Related Documents:**
- @docs/initiatives/rag_system/worker_queue_conversion/scoping/PRD001.md - Product requirements
- @docs/initiatives/rag_system/worker_queue_conversion/scoping/RFC001.md - Technical design (Persistence Infrastructure section)
- @docs/initiatives/rag_system/worker_queue_conversion/scoping/TODO001.md - Implementation tasks (Phase 1 section - check off tasks as completed)
- @docs/initiatives/rag_system/worker_queue_conversion/scoping/context.md - Project context

## Objectives

1. **Database Schema Changes**: Add status columns, timestamps, and new tables (chunks table)
2. **Persistence Functions**: Implement load/persist helper functions for extracted text, chunks, and embeddings
3. **Idempotency Checks**: Implement status-based idempotency checks
4. **Testing**: Create comprehensive unit tests for all persistence operations

## Execution Requirements

### Checklist Modifications
- **REQUIRED**: Check off completed tasks in @docs/initiatives/rag_system/worker_queue_conversion/scoping/TODO001.md Phase 1 section as you complete them
- Update phase status when all validation requirements are met

### Validation
- **REQUIRED**: All unit tests for Phase 1 must pass before proceeding to Phase 2
- **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/workers/test_persistence.py -v`
- **REQUIRED**: Test coverage must meet minimum 80% for persistence.py module
- **REQUIRED**: All test assertions must pass (no failures, no errors)

### Documentation
- **REQUIRED**: Create `phase_1_decisions.md` if any decisions are made that aren't in PRD/RFC
- **REQUIRED**: Create `phase_1_testing.md` documenting testing results
- **REQUIRED**: Create `phase_1_handoff.md` summarizing what's needed for Phase 2

## Key References

### Database Schema
- @infra/supabase/migrations/ - Existing database migrations
- RFC001.md Persistence Infrastructure section - Schema design decisions

### Existing Components
- @backend/rag_eval/services/rag/ - Existing service modules (unchanged)
- @backend/rag_eval/db/ - Database connection and query utilities

### Data Structures
- `Chunk` dataclass (from chunking.py)
- Document status enum: `uploaded`, `parsed`, `chunked`, `embedded`, `indexed`, `failed_*`

## Phase 1 Tasks

### Database Schema Changes
1. Create migration file for schema changes
2. Add status column to `documents` table (if not exists)
3. Add timestamp columns (`parsed_at`, `chunked_at`, `embedded_at`, `indexed_at`)
4. Create `chunks` table with required columns
5. Add `extracted_text` column to `documents` table (or implement storage-based approach)
6. Create indexes for performance
7. Test migration files execute successfully

### Core Implementation
1. Create `rag_eval/services/workers/persistence.py` module
2. Implement load/persist functions:
   - `load_extracted_text(document_id: str, config) -> str`
   - `persist_extracted_text(document_id: str, text: str, config) -> None`
   - `load_chunks(document_id: str, config) -> List[Chunk]`
   - `persist_chunks(document_id: str, chunks: List[Chunk], config) -> None`
   - `load_embeddings(document_id: str, config) -> List[List[float]]`
   - `persist_embeddings(document_id: str, chunks: List[Chunk], embeddings: List[List[float]], config) -> None`
   - `update_document_status(document_id: str, status: str, timestamp_field: Optional[str] = None, config) -> None`
3. Implement idempotency checks:
   - `check_document_status(document_id: str, config) -> str`
   - `should_process_document(document_id: str, target_status: str, config) -> bool`

### Testing
1. Create test file: `backend/tests/components/workers/test_persistence.py`
2. Test load operations for extracted text (database and storage approaches)
3. Test persist operations with various data sizes
4. Test error handling (missing data, invalid IDs)
5. Test idempotency of load/persist operations
6. Test database transaction handling
7. Test edge cases (empty data, null values, large payloads)
8. Test chunk loading and persistence
9. Test embedding loading and persistence
10. Test status update operations
11. Test idempotency checks (status-based)

### Documentation
1. Add docstrings to all functions
2. Document persistence layer design decisions
3. Document database schema changes
4. Document storage approach for extracted text (database vs. storage)

## Success Criteria

- [ ] Database schema changes implemented and tested
- [ ] All persistence functions implemented
- [ ] All unit tests pass (minimum 80% coverage)
- [ ] All Phase 1 tasks in TODO001.md checked off
- [ ] Phase 1 handoff document created

## Important Notes

- **Storage Decision**: RFC recommends starting with database column for extracted text; migrate to storage if size becomes an issue
- **Chunks Table**: Use database table for queryability and metadata support
- **Embeddings Storage**: Use column in chunks table for simplicity; migrate to separate table if needed
- **Idempotency**: Workers check document status before processing to enable safe retries

## Blockers

- **BLOCKER**: Phase 2 cannot proceed until Phase 1 validation complete
- **BLOCKER**: Database migrations must be tested and applied before proceeding

## Next Phase

After completing Phase 1, proceed to **Phase 2: Queue Infrastructure** using @docs/initiatives/rag_system/worker_queue_conversion/prompts/prompt_phase_2_001.md

