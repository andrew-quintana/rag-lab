# Phase 3.5 Prompt — LangGraph Infrastructure Setup

## Context

This prompt guides the implementation of **Phase 3.5: LangGraph Infrastructure Setup** for the RAG Evaluation MVP system. This phase sets up LangGraph infrastructure for orchestration, preparing the foundation for Phase 7 (LLM-as-Judge Orchestrator) to use LangGraph instead of a deterministic script.

**Related Documents:**
- @docs/initiatives/eval_system/PRD001.md - Product requirements
- @docs/initiatives/eval_system/RFC001.md - Technical design (Decision 5: Base Evaluator Class Pattern)
- @docs/initiatives/eval_system/TODO001.md - Implementation tasks (Phase 3.5 section - check off tasks as completed)
- @docs/initiatives/eval_system/context.md - Project context

## Objectives

1. **Install LangGraph Dependencies**: Add LangGraph and required dependencies to requirements.txt
2. **Create Graph Infrastructure**: Set up basic LangGraph state management using TypedDict
3. **Create Node Functions**: Build pure function nodes that wrap existing evaluator classes (following LangGraph best practices)
4. **Create Test Graph**: Build a simple test workflow using node functions to validate LangGraph setup
5. **Test Thoroughly**: Achieve 80%+ test coverage for LangGraph infrastructure
6. **Document Integration**: Document LangGraph setup, state management, node function pattern, and usage

## Execution Requirements

### Checklist Modifications
- **REQUIRED**: Check off completed tasks in @docs/initiatives/eval_system/TODO001.md Phase 3.5 section as you complete them
- Update phase status when all validation requirements are met

### Validation
- **REQUIRED**: All unit tests for Phase 3.5 must pass before proceeding to Phase 4
- **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/evaluator/test_graph_setup.py -v`
- **REQUIRED**: Test coverage must meet minimum 80% for graph_base.py and test_graph.py modules
- **REQUIRED**: All test assertions must pass (no failures, no errors)

### Documentation
- **REQUIRED**: Create `phase_3_5_decisions.md` if any decisions are made that aren't in PRD/RFC
- **REQUIRED**: Create `phase_3_5_testing.md` documenting testing results
- **REQUIRED**: Create `phase_3_5_handoff.md` summarizing what's needed for Phase 7

## Key References

### LangGraph Best Practices
- Nodes should be pure functions that take state and return partial state updates
- Use TypedDict for state schema (not Pydantic)
- Maintain immutability: don't mutate input state
- Validate state at boundaries

### Implementation Location
- `rag_eval/services/evaluator/graph_base.py` - Graph infrastructure and state management
- `rag_eval/services/evaluator/test_graph.py` - Test graph implementation
- `backend/tests/components/evaluator/test_graph_setup.py` - Test suite

### Existing Components
- @backend/rag_eval/services/evaluator/correctness.py - CorrectnessEvaluator class
- @backend/rag_eval/services/evaluator/hallucination.py - HallucinationEvaluator class
- @backend/rag_eval/core/config.py - Config class
- @backend/rag_eval/core/interfaces.py - RetrievalResult dataclass

## Phase 3.5 Tasks

### Setup
1. **REQUIRED**: Activate backend venv: `cd backend && source venv/bin/activate`
2. Add LangGraph dependencies to `backend/requirements.txt`:
   - `langgraph>=0.2.0`
   - `langchain-core>=0.3.0`
3. Install dependencies: `pip install -r requirements.txt`
4. Verify LangGraph can be imported: `python -c "from langgraph.graph import StateGraph"`

### Graph Infrastructure
1. Create `rag_eval/services/evaluator/graph_base.py` module
2. Define `JudgeEvaluationState` TypedDict with fields:
   - `query: str` (required)
   - `retrieved_context: List[RetrievalResult]` (required)
   - `model_answer: str` (required)
   - `reference_answer: str` (required)
   - `correctness_binary: Optional[bool]`
   - `hallucination_binary: Optional[bool]`
   - `hallucination_cost: Optional[int]`
   - `hallucination_impact: Optional[float]`
   - `reasoning: List[str]`
   - `config: Optional[Config]`
3. Implement `validate_initial_state(state: JudgeEvaluationState) -> None`:
   - Check all required fields are present
   - Validate non-empty strings
   - Raise ValueError with clear messages
4. Implement `get_config_from_state(state: JudgeEvaluationState) -> Config`:
   - Return config from state if present
   - Default to `Config.from_env()` if not present

### Test Graph Implementation
1. Create `rag_eval/services/evaluator/test_graph.py` module
2. Implement `correctness_node(state: JudgeEvaluationState) -> dict`:
   - Extract config from state using `get_config_from_state()`
   - Create `CorrectnessEvaluator` instance
   - Call `classify_correctness()` with state fields
   - Return partial state update: `{"correctness_binary": result, "reasoning": [...]}`
   - Append reasoning to existing reasoning list
3. Implement `hallucination_node(state: JudgeEvaluationState) -> dict`:
   - Extract config from state
   - Create `HallucinationEvaluator` instance
   - Call `classify_hallucination()` with state fields
   - Return partial state update: `{"hallucination_binary": result, "reasoning": [...]}`
   - Append reasoning to existing reasoning list
4. Implement `should_continue(state: JudgeEvaluationState) -> Literal["end"]`:
   - For test graph, always return "end"
   - (Phase 7 will implement conditional routing)
5. Implement `create_test_graph() -> StateGraph`:
   - Create StateGraph with `JudgeEvaluationState`
   - Add nodes: "correctness" and "hallucination"
   - Set entry point: "correctness"
   - Add edge: correctness → hallucination
   - Add conditional edge: hallucination → END
   - Return compiled graph
6. Implement `run_test_evaluation(...) -> JudgeEvaluationState`:
   - Create initial state from parameters
   - Validate initial state
   - Create and invoke graph
   - Return final state

### Package Exports
1. Update `backend/rag_eval/services/evaluator/__init__.py`:
   - Import from `graph_base`: `JudgeEvaluationState`, `validate_initial_state`, `get_config_from_state`
   - Add to `__all__` list
   - Keep existing exports

### Testing
1. Create `backend/tests/components/evaluator/test_graph_setup.py`
2. Test LangGraph import and basic functionality
3. Test state class creation and validation
4. Test state validation functions (success and error cases)
5. Test simple graph construction and execution
6. Test node functions with mocked evaluators
7. Test conditional edge routing
8. Test integration with existing evaluator classes (mocked)
9. Test error handling in graph execution
10. Test graph execution order
11. Test config handling in state
12. **Document any failures** in fracas.md immediately when encountered

### Documentation
1. Add comprehensive docstrings to all functions
2. Document LangGraph setup and state management patterns
3. Document node function pattern (pure functions wrapping evaluators)
4. Document immutability pattern (partial state updates)

## Success Criteria

- [ ] LangGraph dependencies installed and importable
- [ ] Graph infrastructure module created with state management
- [ ] Test graph successfully executes with existing evaluators (mocked)
- [ ] All unit tests pass with 80%+ coverage
- [ ] Tests verify node function pattern works correctly
- [ ] All error handling implemented
- [ ] All Phase 3.5 tasks in TODO001.md checked off
- [ ] Phase 3.5 handoff document created

## Important Notes

- **Pure Function Pattern**: Nodes must be pure functions that don't mutate input state
- **Immutability**: Return partial state updates, don't modify input state
- **Existing Evaluators**: Keep existing evaluator classes unchanged; wrap them in node functions
- **Test Graph**: Keep test graph simple - just enough to validate setup works
- **Test Coverage**: Minimum 80% coverage required

## Blockers

- **BLOCKER**: Phase 4 can proceed independently (doesn't depend on LangGraph)
- **BLOCKER**: Phase 7 cannot proceed until Phase 3.5 validation complete
- **BLOCKER**: All tests must pass before proceeding

## Next Phase

After completing Phase 3.5, continue with **Phase 4: Hallucination Cost LLM-Node** using @docs/initiatives/eval_system/prompt_phase_4_001.md. Phase 7 will use the LangGraph infrastructure set up in this phase.

