# Phase 3.5 Handoff — LangGraph Infrastructure Setup

## Status

✅ **Phase 3.5 Complete**: LangGraph infrastructure set up and validated, ready for Phase 7

## Summary

Phase 3.5 successfully sets up LangGraph infrastructure for orchestration, preparing the foundation for Phase 7 (LLM-as-Judge Orchestrator) to use LangGraph instead of a deterministic script. The implementation includes:

- ✅ LangGraph and langchain-core dependencies added
- ✅ `JudgeEvaluationState` TypedDict defined for state management
- ✅ Pure function node pattern implemented (wrapping existing evaluators)
- ✅ Test graph created and validated with existing evaluators
- ✅ Comprehensive test suite (target: 80%+ coverage)
- ✅ State validation and utility functions

## Implementation Details

### Files Created

1. **`backend/rag_eval/services/evaluator/graph_base.py`**
   - `JudgeEvaluationState` TypedDict definition
   - `validate_initial_state()` function for state validation
   - `get_config_from_state()` helper function
   - State management utilities

2. **`backend/tests/components/evaluator/test_graph.py`**
   - `correctness_node()` - Pure function node wrapping CorrectnessEvaluator
   - `hallucination_node()` - Pure function node wrapping HallucinationEvaluator
   - `should_continue()` - Conditional edge function (always ends for test graph)
   - `create_test_graph()` - Creates simple 2-node test graph
   - `run_test_evaluation()` - Convenience function to run test evaluation
   - **Note**: This is a test utility file, not production code

3. **`backend/tests/components/evaluator/test_graph_setup.py`**
   - Comprehensive test suite for LangGraph infrastructure
   - Tests for state management, validation, graph construction, node functions, execution, and error handling
   - Target: 80%+ coverage

### Files Modified

1. **`backend/requirements.txt`**
   - Added `langgraph>=0.2.0`
   - Added `langchain-core>=0.3.0`

2. **`backend/rag_eval/services/evaluator/__init__.py`**
   - Exported `JudgeEvaluationState`, `validate_initial_state`, `get_config_from_state`
   - Maintained existing exports

## Key Patterns Established

### Pure Function Node Pattern

LangGraph nodes are implemented as pure functions that:
- Take `JudgeEvaluationState` as input
- Extract needed data from state
- Call existing evaluator class methods
- Return partial state update (immutable)
- Do not mutate input state

Example:
```python
def correctness_node(state: JudgeEvaluationState) -> dict:
    config = get_config_from_state(state)
    evaluator = CorrectnessEvaluator(config=config)
    result = evaluator.classify_correctness(...)
    return {"correctness_binary": result, "reasoning": [...]}
```

### State Schema

`JudgeEvaluationState` includes:
- Required fields: `query`, `retrieved_context`, `model_answer`, `reference_answer`
- Optional results: `correctness_binary`, `hallucination_binary`, `hallucination_cost`, `hallucination_impact`
- `reasoning`: List of strings accumulated from all nodes
- `config`: Optional Config instance (defaults to `Config.from_env()`)

## Test Results

- **Tests**: 22/22 passed
- **Coverage**: 93% (exceeds 80% requirement)
- **Execution time**: ~0.20 seconds

## Configuration

### Required Dependencies

Install LangGraph dependencies:
```bash
cd backend && source venv/bin/activate
pip install -r requirements.txt
```

### No Additional Environment Variables

LangGraph setup uses existing configuration (Azure credentials, etc.).

## Usage Example

```python
from rag_eval.services.evaluator.test_graph import run_test_evaluation
from rag_eval.core.interfaces import RetrievalResult

retrieval_result = RetrievalResult(
    chunk_id="chunk1",
    similarity_score=0.9,
    chunk_text="Test chunk text"
)

final_state = run_test_evaluation(
    query="What is the copay?",
    retrieved_context=[retrieval_result],
    model_answer="The copay is $50.",
    reference_answer="Copay is $50.",
    config=None
)

print(f"Correctness: {final_state['correctness_binary']}")
print(f"Hallucination: {final_state['hallucination_binary']}")
```

## Decisions Made

See `phase_3_5_decisions.md` for detailed decisions:
1. LangGraph version selection (0.2.0+)
2. State schema design (TypedDict)
3. Pure function node pattern
4. State validation at boundaries
5. Config handling in state
6. Test graph design
7. Reasoning collection pattern
8. File organization

## What's Needed for Phase 7

### Prerequisites
- ✅ Phase 3.5 complete and validated
- ✅ LangGraph infrastructure available
- ✅ State management patterns established
- ✅ Node function pattern documented

### Phase 7 Requirements

1. **LLM-as-Judge Orchestrator**: Implement full judge evaluation graph
   - Use `JudgeEvaluationState` from `graph_base.py`
   - Create node functions for all evaluators (correctness, hallucination, cost, impact, cost_extraction)
   - Implement conditional edges: route to cost/impact nodes only when `hallucination_binary: true`
   - Construct reasoning trace from all invoked nodes
   - Assemble `JudgeEvaluationResult` from final state

2. **Graph Structure**:
   - Entry: correctness node (always executes)
   - correctness → hallucination (always executes)
   - hallucination → conditional edge:
     - If `hallucination_binary: true` → cost node → cost_extraction → impact node → END
     - If `hallucination_binary: false` → END
   - Final: assemble `JudgeEvaluationResult` from state

3. **Interface Contract**: Maintain RFC001 interface
   ```python
   def evaluate_answer_with_judge(
       query: str,
       retrieved_context: List[RetrievalResult],
       model_answer: str,
       reference_answer: str,
       config: Optional[Config] = None
   ) -> JudgeEvaluationResult
   ```
   - This function will create initial state, invoke graph, and convert final state to `JudgeEvaluationResult`

4. **Test File**: `backend/tests/components/evaluator/test_evaluator_judge.py`
   - Test graph execution with mocked evaluators
   - Test conditional branching (hallucination true/false paths)
   - Test reasoning trace construction
   - Test error handling

### Key Differences from Test Graph

- **Conditional Branching**: Test graph always ends after hallucination; Phase 7 routes to cost/impact nodes
- **Additional Nodes**: Phase 7 includes cost, cost_extraction, and impact nodes
- **Output Conversion**: Phase 7 converts final state to `JudgeEvaluationResult` dataclass
- **Reasoning Trace**: Phase 7 combines reasoning into single string for `JudgeEvaluationResult`

## Blockers Removed

- ✅ LangGraph infrastructure set up
- ✅ State management patterns established
- ✅ Node function pattern validated
- ✅ Integration with existing evaluators confirmed

## Next Steps

1. **Continue with Phase 4**: Proceed to **Phase 4: Hallucination Cost LLM-Node** using `@docs/initiatives/eval_system/prompt_phase_4_001.md`
2. **Phase 7 Preparation**: When ready for Phase 7, use LangGraph infrastructure from this phase to implement the full judge orchestrator

## Notes

- Test graph (`tests/components/evaluator/test_graph.py`) is a test utility and can be kept as reference for Phase 7
- `graph_base.py` contains reusable infrastructure for Phase 7
- All existing evaluator classes work as-is; no changes needed
- Phase 7 will follow the same node function pattern established here
- Test graph demonstrates the pattern but Phase 7 will implement the full orchestrator with conditional branching

