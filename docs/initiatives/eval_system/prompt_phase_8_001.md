# Phase 8 Prompt — Meta-Evaluator (Deterministic Validation)

## Context

This prompt guides the implementation of **Phase 8: Meta-Evaluator** for the RAG Evaluation MVP system. This phase implements the deterministic meta-evaluator that validates judge verdicts against ground truth (no LLM calls - pure Python function).

**Related Documents:**
- @docs/initiatives/eval_system/PRD001.md - Product requirements (FR6: Meta-Evaluator)
- @docs/initiatives/eval_system/RFC001.md - Technical design (Phase 7: Meta-Evaluator, Interface Contracts)
- @docs/initiatives/eval_system/TODO001.md - Implementation tasks (Phase 8 section - check off tasks as completed)
- @docs/initiatives/eval_system/context.md - Project context

## Objectives

1. **Implement Meta-Evaluator**: Create deterministic function to validate judge verdicts
2. **Implement Validation Logic**: Rule-based validation for correctness, hallucination, cost, and impact
3. **No LLM Calls**: Pure Python deterministic function (no Azure Foundry calls)
4. **Test Thoroughly**: Achieve 80%+ test coverage with comprehensive unit tests

## Execution Requirements

### Checklist Modifications
- **REQUIRED**: Check off completed tasks in @docs/initiatives/eval_system/TODO001.md Phase 8 section as you complete them
- Update phase status when all validation requirements are met

### Validation
- **REQUIRED**: All unit tests for Phase 8 must pass before proceeding to Phase 9
- **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/meta_eval/test_evaluator_meta_eval.py -v`
- **REQUIRED**: Test coverage must meet minimum 80% for meta_eval.py module
- **REQUIRED**: All test assertions must pass (no failures, no errors)

### Documentation
- **REQUIRED**: Create `phase_8_decisions.md` if any decisions are made that aren't in PRD/RFC
- **REQUIRED**: Create `phase_8_testing.md` documenting testing results
- **REQUIRED**: Create `phase_8_handoff.md` summarizing what's needed for Phase 9

## Key References

### Interface Contract (from RFC001.md)
```python
def meta_evaluate_judge(
    judge_output: JudgeEvaluationResult,
    retrieved_context: List[RetrievalResult],
    model_answer: str,
    reference_answer: str,
    extracted_costs: Optional[Dict[str, Any]] = None,
    actual_costs: Optional[Dict[str, Any]] = None
) -> MetaEvaluationResult
```

### Data Structures
- `MetaEvaluationResult` dataclass (from RFC001.md):
  - `judge_correct: bool`
  - `explanation: Optional[str]`

### Implementation Location
- `rag_eval/services/evaluator/meta_eval.py`

### Test Location
- `backend/tests/components/meta_eval/test_evaluator_meta_eval.py`

### Dependencies
- Phase 7: LLM-as-Judge Orchestrator (for `JudgeEvaluationResult`)

## Phase 8 Tasks

### Setup
1. **REQUIRED**: Activate backend venv: `cd backend && source venv/bin/activate`
2. Create `rag_eval/services/evaluator/meta_eval.py` module
3. Ensure package `__init__.py` files are properly configured with exports
4. Verify imports work correctly
5. Review data structures: `MetaEvaluationResult` dataclass
6. Set up test fixtures for validation scenarios
7. Create test file: `backend/tests/components/meta_eval/test_evaluator_meta_eval.py`

### Core Implementation
1. Implement `meta_evaluate_judge()` function matching RFC001 interface
2. Validate inputs are non-empty
3. **Validation 1**: Validate correctness_binary
   - Compare model answer to reference answer (exact or semantic similarity)
   - If judge says `correctness_binary: true`, verify model answer matches reference
   - If judge says `correctness_binary: false`, verify model answer differs from reference
4. **Validation 2**: Validate hallucination_binary
   - Check if model answer claims are supported by retrieved chunks
   - If judge says `hallucination_binary: true`, verify model answer contains unsupported claims
   - If judge says `hallucination_binary: false`, verify all claims are supported
5. **Validation 3**: Validate hallucination_cost (if costs available and hallucination detected)
   - Compare extracted costs vs actual costs to determine expected cost direction
   - Validate judge's `hallucination_cost` matches expected direction
6. **Validation 4**: Validate hallucination_impact (if costs available and hallucination detected)
   - Calculate expected impact magnitude based on cost differences
   - Validate judge's `hallucination_impact` is within reasonable range of expected impact
7. Determine overall judge correctness (all validations pass)
8. Generate deterministic explanation of validation results
9. Return `MetaEvaluationResult` object
10. Implement helper functions:
    - `_validate_correctness(judge_correctness: bool, model_answer: str, reference_answer: str) -> bool`
    - `_validate_hallucination(judge_hallucination: bool, model_answer: str, retrieved_context: List[RetrievalResult]) -> bool`
    - `_validate_cost_classification(judge_cost: Optional[int], extracted_costs: Dict[str, Any], actual_costs: Dict[str, Any]) -> bool`
    - `_validate_impact_magnitude(judge_impact: Optional[float], extracted_costs: Dict[str, Any], actual_costs: Dict[str, Any]) -> bool`
    - `_generate_explanation(validation_results: Dict[str, bool]) -> str`

### Testing
1. Unit tests for `meta_evaluate_judge()`:
   - Test judge_correct classification: correct correctness_binary verdict
   - Test judge_incorrect classification: incorrect correctness_binary verdict
   - Test judge_correct classification: correct hallucination_binary verdict
   - Test judge_incorrect classification: incorrect hallucination_binary verdict
   - Test validation of hallucination_cost against ground truth costs
   - Test validation of hallucination_impact against ground truth costs
   - Test deterministic explanation generation
   - Test edge case: partial judge correctness (some verdicts correct, others incorrect)
   - Test edge case: missing ground truth information
   - Test edge case: zero retrieved chunks
2. **Document any failures** in fracas.md immediately when encountered

### Documentation
1. Add docstrings to all functions
2. Document validation logic for each judge verdict type
3. Document deterministic nature (no LLM calls)
4. Document explanation generation approach

## Success Criteria

- [ ] `meta_evaluate_judge()` function implemented matching RFC001 interface
- [ ] All validation logic implemented (correctness, hallucination, cost, impact)
- [ ] No LLM calls (pure Python deterministic function)
- [ ] All unit tests pass with 80%+ coverage
- [ ] Tests verify all validation scenarios
- [ ] All error handling implemented
- [ ] All Phase 8 tasks in TODO001.md checked off
- [ ] Phase 8 handoff document created

## Important Notes

- **No LLM Calls**: This is a pure Python deterministic function - no Azure Foundry calls
- **Rule-Based Validation**: Uses rule-based logic to compare judge output against ground truth
- **Semantic Similarity**: May need semantic similarity for correctness validation (consider using simple string comparison or basic similarity metrics)
- **Test Coverage**: Minimum 80% coverage required

## Blockers

- **BLOCKER**: Phase 9 cannot proceed until Phase 8 validation complete
- **BLOCKER**: All tests must pass before proceeding

## Next Phase

After completing Phase 8, proceed to **Phase 9: BEIR Metrics Evaluator** using @docs/initiatives/eval_system/prompt_phase_9_001.md

