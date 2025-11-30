# Phase 11 Prompt — Logging and Persistence (Optional)

## Context

This prompt guides the implementation of **Phase 11: Logging and Persistence (Optional)** for the RAG Evaluation MVP system. This phase implements optional logging of evaluation results to Supabase Postgres database.

**Related Documents:**
- @docs/initiatives/eval_system/PRD001.md - Product requirements (optional logging mentioned)
- @docs/initiatives/eval_system/RFC001.md - Technical design (Phase 10: Logging and Persistence, Interface Contracts)
- @docs/initiatives/eval_system/TODO001.md - Implementation tasks (Phase 11 section - check off tasks as completed)
- @docs/initiatives/eval_system/context.md - Project context

## Objectives

1. **Implement Logging Functions**: Create functions to log evaluation results to Supabase
2. **Optional Logging**: Support both local-only and database logging modes
3. **Graceful Degradation**: Handle logging failures gracefully (don't fail evaluation pipeline)
4. **Test Thoroughly**: Achieve 80%+ test coverage with comprehensive unit tests

## Execution Requirements

### Checklist Modifications
- **REQUIRED**: Check off completed tasks in @docs/initiatives/eval_system/TODO001.md Phase 11 section as you complete them
- Update phase status when all validation requirements are met

### Validation
- **REQUIRED**: All unit tests for Phase 11 must pass
- **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/evaluator/test_evaluator_logging.py -v`
- **REQUIRED**: Test coverage must meet minimum 80% for logging.py module
- **REQUIRED**: All test assertions must pass (no failures, no errors)

### Documentation
- **REQUIRED**: Create `phase_11_decisions.md` if any decisions are made that aren't in PRD/RFC
- **REQUIRED**: Create `phase_11_testing.md` documenting testing results
- **REQUIRED**: Create `phase_11_handoff.md` for final validation

## Key References

### Interface Contract (from RFC001.md)
```python
def log_evaluation_result(
    result: EvaluationResult,
    query_executor: Optional[QueryExecutor] = None
) -> Optional[str]

def log_evaluation_batch(
    results: List[EvaluationResult],
    query_executor: Optional[QueryExecutor] = None
) -> None
```

### Implementation Location
- `rag_eval/services/evaluator/logging.py`

### Test Location
- `backend/tests/components/evaluator/test_evaluator_logging.py`

### Database Components
- `QueryExecutor` from @backend/rag_eval/db/queries.py
- Supabase Postgres database schema (review @infra/supabase/migrations/)

### Dependencies
- Phase 10: Evaluation Pipeline Orchestration (for `EvaluationResult`)

## Phase 11 Tasks

### Setup
1. **REQUIRED**: Activate backend venv: `cd backend && source venv/bin/activate`
2. Create `rag_eval/services/evaluator/logging.py` module
3. Review Supabase database schema for evaluation results table
4. Review existing `QueryExecutor` from `rag_eval/db/queries.py`
5. Set up test fixtures for database operations
6. Create test file: `backend/tests/components/evaluator/test_evaluator_logging.py`

### Core Implementation
1. Implement `log_evaluation_result()` function matching RFC001 interface
2. If query_executor is None, skip logging (local-only mode)
3. Serialize EvaluationResult to JSON
4. Insert into Supabase Postgres `evaluation_results` table
5. Handle logging failures gracefully (don't fail evaluation pipeline)
6. Return result_id if successful, None otherwise
7. Implement `log_evaluation_batch()` function
8. Batch insert evaluation results
9. Handle partial failures gracefully
10. Log batch operation status

### Testing
1. Unit tests for `log_evaluation_result()`:
   - Test logging with query_executor (mocked)
   - Test local-only mode (query_executor is None)
   - Test error handling for database failures
   - Test that logging failures don't fail evaluation pipeline
2. Unit tests for `log_evaluation_batch()`:
   - Test batch logging with query_executor (mocked)
   - Test local-only mode (query_executor is None)
   - Test partial failure handling
3. Connection test for Supabase (warns if credentials missing, doesn't fail tests)
4. **Document any failures** in fracas.md immediately when encountered

### Documentation
1. Add docstrings to all functions
2. Document optional nature of logging
3. Document database schema requirements
4. Document local-only vs. database logging modes

## Success Criteria

- [ ] `log_evaluation_result()` function implemented matching RFC001 interface
- [ ] `log_evaluation_batch()` function implemented
- [ ] Local-only mode supported (query_executor is None)
- [ ] All unit tests pass with 80%+ coverage
- [ ] Tests verify graceful degradation on logging failures
- [ ] All error handling implemented
- [ ] All Phase 11 tasks in TODO001.md checked off
- [ ] Phase 11 handoff document created

## Important Notes

- **Optional**: This phase is optional - can be deferred if not needed for MVP
- **Graceful Degradation**: Logging failures must not fail the evaluation pipeline
- **Local-Only Mode**: Must support local-only mode when query_executor is None
- **Test Coverage**: Minimum 80% coverage required

## Blockers

- **BLOCKER**: Initiative completion cannot proceed until Phase 11 validation complete (if implemented)
- **BLOCKER**: All tests must pass before proceeding

## Next Steps

After completing Phase 11, proceed to **Initiative Completion** validation as specified in @docs/initiatives/eval_system/TODO001.md

