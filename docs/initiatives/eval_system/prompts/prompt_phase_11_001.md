# Phase 11 Prompt — Logging and Persistence (Optional)

## Context

This prompt guides the implementation of **Phase 11: Logging and Persistence (Optional)** for the RAG Evaluation MVP system. This phase implements optional logging of evaluation results to Supabase Postgres database.

**Related Documents:**
- @docs/initiatives/eval_system/scoping/PRD001.md - Product requirements (optional logging mentioned)
- @docs/initiatives/eval_system/scoping/RFC001.md - Technical design (Phase 10: Logging and Persistence, Interface Contracts)
- @docs/initiatives/eval_system/scoping/TODO001.md - Implementation tasks (Phase 11 section - check off tasks as completed)
- @docs/initiatives/eval_system/scoping/context.md - Project context

## Objectives

1. **Implement Logging Functions**: Create functions to log evaluation results to Supabase
2. **JSON Storage**: Store evaluation metrics as JSONB for flexibility and extensibility
3. **Database Schema Updates**: Create/update database tables to support JSON-based evaluation metrics
4. **Optional Logging**: Support both local-only and database logging modes
5. **Graceful Degradation**: Handle logging failures gracefully (don't fail evaluation pipeline)
6. **Test Thoroughly**: Achieve 80%+ test coverage with comprehensive unit tests

## Execution Requirements

### Checklist Modifications
- **REQUIRED**: Check off completed tasks in @docs/initiatives/eval_system/scoping/TODO001.md Phase 11 section as you complete them
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
- **New Table**: `evaluation_results` with JSONB columns for flexible metric storage

### Dependencies
- Phase 10: Evaluation Pipeline Orchestration (for `EvaluationResult`)

## Phase 11 Tasks

### Setup
1. **REQUIRED**: Activate backend venv: `cd backend && source venv/bin/activate`
2. Create `rag_eval/services/evaluator/logging.py` module
3. Review Supabase database schema and existing migrations
4. Review existing `QueryExecutor` from `rag_eval/db/queries.py`
5. Set up test fixtures for database operations
6. Create test file: `backend/tests/components/evaluator/test_evaluator_logging.py`
7. **Database Schema**: Create migration file for `evaluation_results` table

### Database Schema Implementation
1. **Create Migration File**: `infra/supabase/migrations/0011_add_evaluation_results_table.sql`
2. **Design Table Schema**:
   - `result_id` (VARCHAR(255) PRIMARY KEY)
   - `example_id` (VARCHAR(255) NOT NULL) - Reference to evaluation example
   - `timestamp` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
   - `judge_output` (JSONB) - Store `JudgeEvaluationResult` as JSON
   - `meta_eval_output` (JSONB) - Store `MetaEvaluationResult` as JSON (includes ground truth)
   - `beir_metrics` (JSONB) - Store `BEIRMetricsResult` as JSON
   - `judge_performance_metrics` (JSONB, NULLABLE) - Store `JudgePerformanceMetrics` as JSON (if calculated)
   - `metadata` (JSONB, NULLABLE) - Additional flexible metadata
3. **Create Indexes**:
   - Index on `example_id` for lookups
   - Index on `timestamp` for time-based queries
   - GIN index on JSONB columns for efficient JSON queries
4. **Test Migration**: Verify migration runs successfully

### Core Implementation
1. Implement `log_evaluation_result()` function matching RFC001 interface
2. If query_executor is None, skip logging (local-only mode)
3. **Serialize to JSON**: Convert all evaluation result components to JSON:
   - `JudgeEvaluationResult` → JSON (correctness_binary, hallucination_binary, risk_direction, risk_impact, reasoning, failure_mode)
   - `MetaEvaluationResult` → JSON (judge_correct, explanation, ground_truth_* fields)
   - `BEIRMetricsResult` → JSON (recall_at_k, precision_at_k, ndcg_at_k)
   - `JudgePerformanceMetrics` → JSON (if provided, optional)
4. Insert into Supabase Postgres `evaluation_results` table with JSONB columns
5. Handle logging failures gracefully (don't fail evaluation pipeline)
6. Return result_id if successful, None otherwise
7. Implement `log_evaluation_batch()` function
8. Batch insert evaluation results (all as JSONB)
9. Handle partial failures gracefully
10. Log batch operation status
11. **JSON Serialization Helpers**: Create helper functions to serialize dataclasses to JSON-compatible dicts

### Testing
1. **Database Schema Tests**:
   - Test migration file executes successfully
   - Verify table structure matches specification
   - Verify indexes are created correctly
   - Test JSONB column queries work correctly
2. Unit tests for `log_evaluation_result()`:
   - Test logging with query_executor (mocked)
   - Test local-only mode (query_executor is None)
   - Test JSON serialization of all result components
   - Test JSONB insertion into database
   - Test error handling for database failures
   - Test that logging failures don't fail evaluation pipeline
   - Test logging of `MetaEvaluationResult` with ground truth fields (for metrics calculation)
   - Test logging of `JudgePerformanceMetrics` (optional field)
3. Unit tests for `log_evaluation_batch()`:
   - Test batch logging with query_executor (mocked)
   - Test local-only mode (query_executor is None)
   - Test partial failure handling
   - Test batch logging includes all meta-evaluation ground truth data
   - Test batch JSON serialization and insertion
4. Integration tests for JSON retrieval and deserialization:
   - Test retrieving logged evaluation results from database
   - Test deserializing JSONB back to Python objects
   - Test that all fields are preserved in JSON round-trip
   - Test metrics recalculation from retrieved JSON data
5. Integration tests for judge performance metrics logging:
   - Test logging of `JudgePerformanceMetrics` results (if metrics are calculated)
   - Test that metrics can be retrieved and recalculated from logged evaluation results
   - Test metrics calculation from logged batch results
6. Connection test for Supabase (warns if credentials missing, doesn't fail tests)
7. **Document any failures** in fracas.md immediately when encountered

### Documentation
1. Add docstrings to all functions
2. Document optional nature of logging
3. Document database schema requirements (JSONB structure)
4. Document local-only vs. database logging modes
5. Document JSON serialization format for each result type
6. Document how to query JSONB columns for analysis
7. Document migration file location and usage

## Success Criteria

- [ ] Database migration file created and tested (`0011_add_evaluation_results_table.sql`)
- [ ] `evaluation_results` table created with JSONB columns
- [ ] Indexes created for efficient queries
- [ ] `log_evaluation_result()` function implemented matching RFC001 interface
- [ ] `log_evaluation_batch()` function implemented
- [ ] JSON serialization helpers implemented for all result types
- [ ] All evaluation metrics stored as JSONB (flexible and extensible)
- [ ] Local-only mode supported (query_executor is None)
- [ ] Ground truth fields from `MetaEvaluationResult` are properly logged as JSON
- [ ] Judge performance metrics integration tested (if metrics calculated)
- [ ] JSON retrieval and deserialization tested
- [ ] All unit tests pass with 80%+ coverage
- [ ] Tests verify graceful degradation on logging failures
- [ ] Tests verify metrics can be recalculated from logged JSON results
- [ ] All error handling implemented
- [ ] All Phase 11 tasks in TODO001.md checked off
- [ ] Phase 11 handoff document created

## Important Notes

- **Optional**: This phase is optional - can be deferred if not needed for MVP
- **JSON Storage**: All evaluation metrics stored as JSONB for flexibility and future extensibility
- **Database Migration**: Must create migration file in `infra/supabase/migrations/` directory
- **JSON Serialization**: Use Python's `json` module with custom encoders for dataclasses and datetime objects
- **Graceful Degradation**: Logging failures must not fail the evaluation pipeline
- **Local-Only Mode**: Must support local-only mode when query_executor is None
- **Test Coverage**: Minimum 80% coverage required
- **JSONB Queries**: Consider documenting example queries for retrieving and analyzing logged JSON data

## Blockers

- **BLOCKER**: Initiative completion cannot proceed until Phase 11 validation complete (if implemented)
- **BLOCKER**: All tests must pass before proceeding

## Next Steps

After completing Phase 11, proceed to **Initiative Completion** validation as specified in @docs/initiatives/eval_system/scoping/TODO001.md

