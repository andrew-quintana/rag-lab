# Phase 10 Prompt — Evaluation Pipeline Orchestration

## Context

This prompt guides the implementation of **Phase 10: Evaluation Pipeline Orchestration** for the RAG Evaluation MVP system. This phase implements the complete evaluation pipeline that integrates all components (retrieval, RAG generation, judge, meta-eval, BEIR metrics) to evaluate the RAG system.

**Related Documents:**
- @docs/initiatives/eval_system/PRD001.md - Product requirements (FR9: Evaluation Pipeline Integration)
- @docs/initiatives/eval_system/RFC001.md - Technical design (Phase 9: Evaluation Pipeline Orchestration, Interface Contracts)
- @docs/initiatives/eval_system/TODO001.md - Implementation tasks (Phase 10 section - check off tasks as completed)
- @docs/initiatives/eval_system/context.md - Project context

## Objectives

1. **Implement Pipeline Orchestrator**: Create function to orchestrate complete evaluation pipeline
2. **Integrate RAG Components**: Integrate with existing RAG system (`retrieve_chunks`, `generate_answer`)
3. **Execute Full Pipeline**: Retrieval → RAG generation → Judge → Meta-Eval → BEIR Metrics
4. **Test Thoroughly**: Achieve 80%+ test coverage with comprehensive unit tests

## Execution Requirements

### Checklist Modifications
- **REQUIRED**: Check off completed tasks in @docs/initiatives/eval_system/TODO001.md Phase 10 section as you complete them
- Update phase status when all validation requirements are met

### Validation
- **REQUIRED**: All unit tests for Phase 10 must pass before proceeding to Phase 11
- **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/evaluator/test_evaluator_orchestrator.py -v`
- **REQUIRED**: Test coverage must meet minimum 80% for orchestrator.py module
- **REQUIRED**: All test assertions must pass (no failures, no errors)

### Documentation
- **REQUIRED**: Create `phase_10_decisions.md` if any decisions are made that aren't in PRD/RFC
- **REQUIRED**: Create `phase_10_testing.md` documenting testing results
- **REQUIRED**: Create `phase_10_handoff.md` summarizing what's needed for Phase 11

## Key References

### Interface Contract (from RFC001.md)
```python
def evaluate_rag_system(
    evaluation_dataset: List[EvaluationExample],
    rag_retriever: Callable[[str, int], List[RetrievalResult]],
    rag_generator: Callable[[str, List[RetrievalResult]], ModelAnswer],
    config: Optional[Config] = None
) -> List[EvaluationResult]
```

### Data Structures
- `EvaluationExample` dataclass (from RFC001.md)
- `EvaluationResult` dataclass (from RFC001.md):
  - `example_id: str`
  - `judge_output: JudgeEvaluationResult`
  - `meta_eval_output: MetaEvaluationResult`
  - `beir_metrics: BEIRMetricsResult`
  - `timestamp: datetime`

### Implementation Location
- `rag_eval/services/evaluator/orchestrator.py`

### Test Location
- `backend/tests/components/evaluator/test_evaluator_orchestrator.py`

### RAG System Components
- `retrieve_chunks(query: str, k: int) -> List[RetrievalResult]` (from existing RAG system)
- `generate_answer(query: str, retrieved_context: List[RetrievalResult]) -> ModelAnswer` (from existing RAG system)

### Dependencies
- Phase 1: Evaluation Dataset
- Phase 7: LLM-as-Judge Orchestrator
- Phase 8: Meta-Evaluator
- Phase 9: BEIR Metrics Evaluator

## Phase 10 Tasks

### Setup
1. **REQUIRED**: Activate backend venv: `cd backend && source venv/bin/activate`
2. Create `rag_eval/services/evaluator/orchestrator.py` module
3. Ensure package `__init__.py` files are properly configured with exports
4. Verify imports work correctly (judge, meta_eval, beir_metrics)
5. Review data structures: `EvaluationExample`, `EvaluationResult` dataclasses
6. Review RAG system components: `retrieve_chunks`, `generate_answer`
7. Set up test fixtures for evaluation dataset and mocked RAG components
8. Create test file: `backend/tests/components/evaluator/test_evaluator_orchestrator.py`

### Core Implementation
1. Implement `evaluate_rag_system()` function matching RFC001 interface
2. Validate evaluation dataset is non-empty
3. Load evaluation dataset (if passed as file path, otherwise use list directly)
4. For each example in evaluation dataset:
   - **Step 1**: Retrieve chunks using RAG retriever
     - `retrieved_chunks = rag_retriever(example.question, k=5)`
   - **Step 2**: Generate answer using RAG generator
     - `model_answer = rag_generator(example.question, retrieved_chunks)`
   - **Step 3**: Evaluate with LLM-as-Judge
     - `judge_output = evaluate_answer_with_judge(example.question, retrieved_chunks, model_answer.text, example.reference_answer, config)`
   - **Step 4**: Meta-evaluate judge verdict
     - Extract costs if needed (for meta-evaluation)
     - `meta_eval_output = meta_evaluate_judge(judge_output, retrieved_chunks, model_answer.text, example.reference_answer, extracted_costs, actual_costs)`
   - **Step 5**: Compute BEIR metrics
     - `beir_metrics = compute_beir_metrics(retrieved_chunks, example.ground_truth_chunk_ids, k=5)`
   - **Step 6**: Assemble EvaluationResult
     - `result = EvaluationResult(example_id=example.example_id, judge_output=judge_output, meta_eval_output=meta_eval_output, beir_metrics=beir_metrics, timestamp=datetime.now())`
   - Handle errors gracefully with proper logging
   - Measure and log latency metrics
5. Return list of EvaluationResult objects
6. Implement `_evaluate_single_example()` helper function
   - Extract single example evaluation logic
   - Handle errors for individual examples (don't fail entire pipeline)

### Testing
1. Unit tests for `evaluate_rag_system()`:
   - Test full pipeline: Retrieval → RAG generation → Judge → Meta-Eval → Metrics
   - Test pipeline with mocked RAG components
   - Test pipeline error handling and propagation
   - Test pipeline logging and observability
   - Test edge case: empty evaluation dataset
   - Test edge case: RAG retriever failure
   - Test edge case: RAG generator failure
   - Test edge case: Judge evaluation failure
2. Integration tests with real RAG components (optional, requires full system setup)
3. **Document any failures** in fracas.md immediately when encountered

### Documentation
1. Add docstrings to all functions
2. Document pipeline flow and error handling
3. Document integration with RAG system components
4. Document latency measurement approach

## Success Criteria

- [ ] `evaluate_rag_system()` function implemented matching RFC001 interface
- [ ] Full pipeline executes: Retrieval → RAG generation → Judge → Meta-Eval → Metrics
- [ ] All components properly integrated
- [ ] All unit tests pass with 80%+ coverage
- [ ] Tests verify full pipeline execution
- [ ] All error handling implemented
- [ ] All Phase 10 tasks in TODO001.md checked off
- [ ] Phase 10 handoff document created

## Important Notes

- **RAG Integration**: Must integrate with existing RAG system components without modifying their interfaces
- **Error Handling**: Handle errors gracefully - don't fail entire pipeline for individual example failures
- **Latency Measurement**: Measure and log latency metrics for performance monitoring
- **Test Coverage**: Minimum 80% coverage required

## Blockers

- **BLOCKER**: Phase 11 cannot proceed until Phase 10 validation complete
- **BLOCKER**: All tests must pass before proceeding

## Next Phase

After completing Phase 10, proceed to **Phase 11: Logging and Persistence (Optional)** using @docs/initiatives/eval_system/prompt_phase_11_001.md

