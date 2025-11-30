# Phase 9 Prompt — BEIR Metrics Evaluator

## Context

This prompt guides the implementation of **Phase 9: BEIR Metrics Evaluator** for the RAG Evaluation MVP system. This phase implements the BEIR-style retrieval metrics computation (recall@k, precision@k, nDCG@k) as pure Python functions.

**Related Documents:**
- @docs/initiatives/eval_system/PRD001.md - Product requirements (FR7: BEIR-Style Retrieval Metrics)
- @docs/initiatives/eval_system/RFC001.md - Technical design (Phase 8: BEIR Metrics Evaluator, Interface Contracts)
- @docs/initiatives/eval_system/TODO001.md - Implementation tasks (Phase 9 section - check off tasks as completed)
- @docs/initiatives/eval_system/context.md - Project context

## Objectives

1. **Implement BEIR Metrics**: Create functions to compute recall@k, precision@k, nDCG@k
2. **Pure Python Implementation**: No LLM calls - standard information retrieval metrics
3. **Handle Edge Cases**: Zero relevant passages, all relevant passages retrieved
4. **Test Thoroughly**: Achieve 80%+ test coverage with comprehensive unit tests

## Execution Requirements

### Checklist Modifications
- **REQUIRED**: Check off completed tasks in @docs/initiatives/eval_system/TODO001.md Phase 9 section as you complete them
- Update phase status when all validation requirements are met

### Validation
- **REQUIRED**: All unit tests for Phase 9 must pass before proceeding to Phase 10
- **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/evaluator/test_evaluator_beir_metrics.py -v`
- **REQUIRED**: Test coverage must meet minimum 80% for beir_metrics.py module
- **REQUIRED**: All test assertions must pass (no failures, no errors)

### Documentation
- **REQUIRED**: Create `phase_9_decisions.md` if any decisions are made that aren't in PRD/RFC
- **REQUIRED**: Create `phase_9_testing.md` documenting testing results
- **REQUIRED**: Create `phase_9_handoff.md` summarizing what's needed for Phase 10

## Key References

### Interface Contract (from RFC001.md)
```python
def compute_beir_metrics(
    retrieved_chunks: List[RetrievalResult],
    ground_truth_chunk_ids: List[str],
    k: int = 5
) -> BEIRMetricsResult
```

### Data Structures
- `BEIRMetricsResult` dataclass (from RFC001.md):
  - `recall_at_k: float`
  - `precision_at_k: float`
  - `ndcg_at_k: float`

### Metric Formulas (from RFC001.md)
- **Recall@k**: (Number of relevant chunks in top-k) / (Total number of relevant chunks)
- **Precision@k**: (Number of relevant chunks in top-k) / k
- **nDCG@k**: Normalized discounted cumulative gain, using relevance scores (1 for relevant, 0 for irrelevant)

### Implementation Location
- `rag_eval/services/evaluator/beir_metrics.py`

### Test Location
- `backend/tests/components/evaluator/test_evaluator_beir_metrics.py`

## Phase 9 Tasks

### Setup
1. **REQUIRED**: Activate backend venv: `cd backend && source venv/bin/activate`
2. Create `rag_eval/services/evaluator/beir_metrics.py` module
3. Ensure package `__init__.py` files are properly configured with exports
4. Verify imports work correctly
5. Review data structures: `BEIRMetricsResult` dataclass
6. Review BEIR metric formulas (recall@k, precision@k, nDCG@k)
7. Set up test fixtures for retrieval results and ground-truth chunk IDs
8. Create test file: `backend/tests/components/evaluator/test_evaluator_beir_metrics.py`

### Core Implementation
1. Implement `compute_beir_metrics()` function matching RFC001 interface
2. Validate inputs are non-empty
3. Extract chunk IDs from retrieved chunks (top-k)
4. Compute recall@k using formula: (Number of relevant chunks in top-k) / (Total number of relevant chunks)
5. Compute precision@k using formula: (Number of relevant chunks in top-k) / k
6. Compute nDCG@k using normalized discounted cumulative gain formula
7. Handle edge cases: zero relevant passages, all relevant passages retrieved
8. Return `BEIRMetricsResult` object
9. Implement helper functions:
   - `_compute_recall_at_k(retrieved_chunk_ids: List[str], ground_truth_chunk_ids: List[str], k: int) -> float`
   - `_compute_precision_at_k(retrieved_chunk_ids: List[str], ground_truth_chunk_ids: List[str], k: int) -> float`
   - `_compute_ndcg_at_k(retrieved_chunks: List[RetrievalResult], ground_truth_chunk_ids: List[str], k: int) -> float`

### Testing
1. Unit tests for `compute_beir_metrics()`:
   - Test recall@k calculation
   - Test precision@k calculation
   - Test nDCG@k calculation
   - Test metrics with ground-truth passage IDs
   - Test edge case: zero relevant passages retrieved
   - Test edge case: all relevant passages retrieved
   - Test edge case: k larger than number of retrieved chunks
   - Test edge case: empty ground-truth chunk IDs list
2. **Document any failures** in fracas.md immediately when encountered

### Documentation
1. Add docstrings to all functions
2. Document BEIR metric formulas
3. Document edge case handling

## Success Criteria

- [ ] `compute_beir_metrics()` function implemented matching RFC001 interface
- [ ] All three metrics (recall@k, precision@k, nDCG@k) computed correctly
- [ ] All edge cases handled
- [ ] All unit tests pass with 80%+ coverage
- [ ] Tests verify metric calculations match formulas
- [ ] All error handling implemented
- [ ] All Phase 9 tasks in TODO001.md checked off
- [ ] Phase 9 handoff document created

## Important Notes

- **No LLM Calls**: Pure Python implementation - standard information retrieval metrics
- **Metric Formulas**: Must match formulas specified in RFC001.md
- **Edge Cases**: Handle zero relevant passages, all relevant passages, k > retrieved chunks
- **Test Coverage**: Minimum 80% coverage required

## Blockers

- **BLOCKER**: Phase 10 cannot proceed until Phase 9 validation complete
- **BLOCKER**: All tests must pass before proceeding

## Next Phase

After completing Phase 9, proceed to **Phase 10: Evaluation Pipeline Orchestration** using @docs/initiatives/eval_system/prompt_phase_10_001.md

