# Phase 7 Prompt — LLM-as-Judge Orchestrator

## Context

This prompt guides the implementation of **Phase 7: LLM-as-Judge Orchestrator** for the RAG Evaluation MVP system. This phase implements the deterministic Python script orchestrator that coordinates all LLM nodes (correctness, hallucination, cost, impact) to produce the complete judge evaluation result.

**Related Documents:**
- @docs/initiatives/eval_system/PRD001.md - Product requirements (FR1: LLM-as-Judge Orchestrator)
- @docs/initiatives/eval_system/RFC001.md - Technical design (Phase 6: LLM-as-Judge Orchestrator, Interface Contracts)
- @docs/initiatives/eval_system/TODO001.md - Implementation tasks (Phase 7 section - check off tasks as completed)
- @docs/initiatives/eval_system/context.md - Project context

## Objectives

1. **Implement Judge Orchestrator**: Create deterministic script that orchestrates all LLM nodes
2. **Implement Conditional Branching**: Call cost/impact nodes only when hallucination detected
3. **Construct Reasoning Trace**: Combine reasoning from all invoked LLM nodes
4. **Test Thoroughly**: Achieve 80%+ test coverage with comprehensive unit tests

## Execution Requirements

### Checklist Modifications
- **REQUIRED**: Check off completed tasks in @docs/initiatives/eval_system/TODO001.md Phase 7 section as you complete them
- Update phase status when all validation requirements are met

### Validation
- **REQUIRED**: All unit tests for Phase 7 must pass before proceeding to Phase 8
- **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/evaluator/test_evaluator_judge.py -v`
- **REQUIRED**: Test coverage must meet minimum 80% for judge.py module
- **REQUIRED**: All test assertions must pass (no failures, no errors)

### Documentation
- **REQUIRED**: Create `phase_7_decisions.md` if any decisions are made that aren't in PRD/RFC
- **REQUIRED**: Create `phase_7_testing.md` documenting testing results
- **REQUIRED**: Create `phase_7_handoff.md` summarizing what's needed for Phase 8

## Key References

### Interface Contract (from RFC001.md)
```python
def evaluate_answer_with_judge(
    query: str,
    retrieved_context: List[RetrievalResult],
    model_answer: str,
    reference_answer: str,
    config: Optional[Config] = None
) -> JudgeEvaluationResult
```

### Data Structures
- `JudgeEvaluationResult` dataclass (from RFC001.md):
  - `correctness_binary: bool`
  - `hallucination_binary: bool`
  - `risk_direction: Optional[int]` (-1, 0, or 1)
  - `risk_impact: Optional[int]` (0, 1, 2, or 3)
  - `reasoning: str`
  - `failure_mode: Optional[str]`

### Implementation Location
- `rag_eval/services/evaluator/judge.py`

### Test Location
- `backend/tests/components/evaluator/test_evaluator_judge.py`

### Dependencies
- Phase 2: Correctness LLM-Node
- Phase 3: Hallucination LLM-Node
- Phase 4: Hallucination Cost LLM-Node
- Phase 5: Cost Extraction LLM-Node
- Phase 6: Risk Impact LLM-Node

## Phase 7 Tasks

### Setup
1. **REQUIRED**: Activate backend venv: `cd backend && source venv/bin/activate`
2. Create `rag_eval/services/evaluator/judge.py` module
3. Ensure package `__init__.py` files are properly configured with exports
4. Verify imports work correctly (correctness, hallucination, cost, impact, cost_extraction nodes)
5. Review data structures: `JudgeEvaluationResult` dataclass
6. Set up test fixtures for mocked LLM node responses
7. Create test file: `backend/tests/components/evaluator/test_evaluator_judge.py`

### Core Implementation
1. Implement `evaluate_answer_with_judge()` function matching RFC001 interface
2. Validate all inputs are non-empty
3. **Step 1**: Call correctness LLM-node (always)
   - `correctness_binary = classify_correctness(query, model_answer, reference_answer, config)`
4. **Step 2**: Call hallucination LLM-node (always)
   - `hallucination_binary = classify_hallucination(retrieved_context, model_answer, config)`
5. **Step 3**: Conditional - if correctness is True, call cost classification node
   - `risk_direction = None`
   - `if correctness_binary: risk_direction = classify_risk_direction(model_answer, retrieved_context, config)`
6. **Step 4**: Conditional - if correctness is True, extract costs and calculate impact
   - `risk_impact = None`
   - `if correctness_binary:`
     - Extract costs from model answer: `model_answer_cost = extract_costs(model_answer, config)`
     - Extract costs from retrieved chunks: `chunks_text = " ".join([chunk.chunk_text for chunk in retrieved_context])`
     - `actual_cost = extract_costs(chunks_text, config)`
     - Calculate impact: `risk_impact = calculate_risk_impact(model_answer_cost, actual_cost, config)`
7. **Step 5**: Construct reasoning trace from all LLM node outputs
   - Collect reasoning from correctness node
   - Collect reasoning from hallucination node
   - Collect reasoning from cost node (if called)
   - Collect reasoning from impact node (if called)
   - Construct combined reasoning trace
8. **Step 6**: Assemble `JudgeEvaluationResult` with all fields
   - `correctness_binary`
   - `hallucination_binary`
   - `risk_direction` (optional)
   - `risk_impact` (optional)
   - `reasoning` (combined trace)
   - `failure_mode` (optional, extracted from reasoning or LLM output)
9. Handle edge cases: zero chunks, empty answers, LLM failures
10. Return `JudgeEvaluationResult` object
11. Implement `_orchestrate_judge_evaluation()` helper function (if needed for organization)
12. Implement `_construct_reasoning_trace()` helper function

### Testing
1. Unit tests for `evaluate_answer_with_judge()`:
   - Test deterministic script orchestration with mocked LLM calls
   - Test correctness LLM-node invocation (always called)
   - Test hallucination LLM-node invocation (always called)
   - Test conditional branching: correctness_binary true path (cost and impact nodes called)
   - Test conditional branching: correctness_binary false path (cost and impact nodes NOT called)
   - Test invocation of cost classification node when correctness is True
   - Test invocation of cost extraction node when correctness is True
   - Test invocation of impact node when correctness is True
   - Test that cost/impact nodes are NOT called when correctness is False
   - Test output schema validation (all required fields present including correctness fields)
   - Test reasoning trace construction from LLM node outputs
   - Test error handling when LLM calls fail
   - Test edge case: zero retrieved chunks
   - Test edge case: empty model answer
   - Test edge case: empty reference answer
2. Integration tests with mocked LLM nodes
3. **Document any failures** in fracas.md immediately when encountered

### Documentation
1. Add docstrings to all functions
2. Document orchestration logic and conditional branching
3. Document reasoning trace construction
4. Document edge case handling

## Success Criteria

- [ ] `evaluate_answer_with_judge()` function implemented matching RFC001 interface
- [ ] All LLM nodes properly orchestrated with conditional branching
- [ ] Reasoning trace constructed from all invoked nodes
- [ ] All unit tests pass with 80%+ coverage
- [ ] Tests verify conditional branching logic
- [ ] All error handling implemented
- [ ] All Phase 7 tasks in TODO001.md checked off
- [ ] Phase 7 handoff document created

## Important Notes

- **Conditional Branching**: Cost and impact nodes are ONLY called when `correctness_binary: true`
- **Always Called**: Correctness and hallucination nodes are ALWAYS called
- **Reasoning Trace**: Must combine reasoning from all invoked nodes
- **Test Coverage**: Minimum 80% coverage required

## Blockers

- **BLOCKER**: Phase 8 cannot proceed until Phase 7 validation complete
- **BLOCKER**: All tests must pass before proceeding

## Next Phase

After completing Phase 7, proceed to **Phase 8: Meta-Evaluator** using @docs/initiatives/eval_system/prompt_phase_8_001.md

