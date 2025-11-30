# Phase 3 Handoff — Hallucination LLM-Node

## Status

✅ **Phase 3 Complete**: All validation requirements met, ready for Phase 4

## Summary

Phase 3 successfully implements the hallucination classification LLM node that performs strict grounding analysis against retrieved evidence. The implementation includes:

- ✅ `classify_hallucination()` function matching RFC001 interface
- ✅ Prompt template with emphasis on NOT using reference answer
- ✅ Azure Foundry GPT-4o-mini integration with structured output
- ✅ Comprehensive test suite (27 tests, 86% coverage)
- ✅ Error handling for AzureServiceError and ValueError
- ✅ Input validation for empty context and empty answers
- ✅ **Critical**: Reference answer is NOT used in hallucination detection

## Implementation Details

### Files Created
1. **`backend/rag_eval/services/evaluator/hallucination.py`**
   - Main implementation module
   - Classes: `HallucinationEvaluator` (inherits from `BaseEvaluatorNode`)
   - Functions: `classify_hallucination()`, `_construct_prompt()`, `_format_retrieved_context()`
   - Error handling: AzureServiceError, ValueError
   - Temperature: 0.1 (for reproducibility)

2. **`backend/rag_eval/prompts/evaluation/hallucination_prompt.md`**
   - Prompt template with system instructions
   - Placeholders: `{retrieved_context}`, `{model_answer}`
   - **Critical**: Explicitly states reference answer is NOT used
   - Output format: JSON with `hallucination_binary` (bool) and `reasoning` (str)
   - Examples included for grounded vs. ungrounded claims

3. **`backend/tests/components/evaluator/test_evaluator_hallucination.py`**
   - 27 comprehensive unit tests
   - Coverage: 86% (exceeds 80% requirement)
   - Tests: input validation, API calls, error handling, grounding analysis, integration
   - **Critical test**: Verifies reference answer is NOT used

### Package Exports
- Updated `backend/rag_eval/services/evaluator/__init__.py` to export `classify_hallucination`

## Interface Contract (RFC001 Compliance)

```python
def classify_hallucination(
    retrieved_context: List[RetrievalResult],
    model_answer: str,
    config: Optional[Config] = None
) -> bool
```

**Status**: ✅ Fully implemented and tested

**Key Difference from Correctness Node**: 
- Uses `retrieved_context` instead of `reference_answer`
- Performs grounding analysis (not correctness comparison)
- Reference answer is explicitly NOT used

## Test Results

- **Tests**: 27/27 passed
- **Coverage**: 86% (exceeds 80% requirement)
- **Execution time**: ~1.17 seconds
- **Integration test**: ✅ Passed (when credentials available)
- **Critical test**: ✅ Reference answer NOT used verification passes

## Configuration

### Required Environment Variables
- `AZURE_AI_FOUNDRY_ENDPOINT`: Azure AI Foundry endpoint URL
- `AZURE_AI_FOUNDRY_API_KEY`: Azure AI Foundry API key

### Optional Configuration
- `AZURE_AI_FOUNDRY_EVALUATION_MODEL`: Override default model (defaults to "gpt-4o-mini")

### Model and Parameters
- **Model**: `gpt-4o-mini` (as specified in RFC001)
- **Temperature**: `0.1` (for reproducibility)
- **Max tokens**: `500` (sufficient for classification responses)

## Usage Example

```python
from rag_eval.core.config import Config
from rag_eval.core.interfaces import RetrievalResult
from rag_eval.services.evaluator.hallucination import classify_hallucination

config = Config.from_env()

retrieved_context = [
    RetrievalResult(
        chunk_id="chunk_001",
        similarity_score=0.95,
        chunk_text="The copay for a specialist visit is $50."
    )
]

model_answer = "The copay for specialist visits is $50."

has_hallucination = classify_hallucination(
    retrieved_context=retrieved_context,
    model_answer=model_answer,
    config=config
)
print(f"Hallucination: {has_hallucination}")  # False (grounded)
```

## Error Handling

### Exceptions Raised
- **ValueError**: Invalid inputs (empty context, empty model answer, missing fields in response)
- **AzureServiceError**: Azure API failures (network errors, API errors, retry exhaustion)

### Error Handling Strategy
- Input validation before API calls
- Retry logic with exponential backoff (handled by base class)
- Detailed error messages for debugging
- Graceful handling of missing credentials in tests

## Critical Design Decision: Reference Answer NOT Used

**This is a critical requirement for Phase 3**: Hallucination detection uses ONLY retrieved context as ground truth. Reference answer is NOT used because:

1. **Hallucination is about grounding, not correctness**: A model answer can be incorrect compared to reference but still grounded in retrieved chunks (retrieval failure, not hallucination)
2. **Strict grounding analysis**: Model answer must be supported by what was actually retrieved
3. **Separation of concerns**: Correctness node handles reference comparison; hallucination node handles grounding

**Implementation verification**:
- Prompt template explicitly states "reference answer is NOT used"
- Function signature does not include `reference_answer` parameter
- Tests verify reference answer is not in prompt construction
- Docstrings document this critical requirement

## Decisions Made

See `phase_3_decisions.md` for detailed decisions:
1. File-based prompt storage (not database)
2. Retrieved context formatting with chunk IDs
3. Reference answer explicitly NOT used
4. Empty retrieved context handling (raises ValueError)
5. Temperature setting (0.1)
6. Error handling strategy
7. Max tokens (500)
8. BaseEvaluatorNode inheritance pattern

## What's Needed for Phase 4

### Prerequisites
- ✅ Phase 3 complete and validated
- ✅ Hallucination node available for orchestrator integration
- ✅ Test patterns established for LLM node testing
- ✅ Grounding analysis pattern established

### Phase 4 Requirements
1. **Hallucination Cost LLM-Node**: Implement `classify_hallucination_cost()` function
   - Interface: `classify_hallucination_cost(model_answer: str, retrieved_context: List[RetrievalResult], config: Optional[Config] = None) -> int`
   - Returns: -1 (opportunity cost) or +1 (resource cost)
   - Similar structure to hallucination node
   - Uses retrieved context (not reference answer)
   - Prompt template: `backend/rag_eval/prompts/evaluation/hallucination_cost_prompt.md`

2. **Test File**: `backend/tests/components/evaluator/test_evaluator_hallucination_cost.py`
   - Follow same test patterns as hallucination tests
   - Test cost classification (-1 vs. +1)
   - Test that reference answer is NOT used
   - Test edge cases (ambiguous cost direction)

3. **Documentation**: Create phase_4_decisions.md, phase_4_testing.md, phase_4_handoff.md

### Key Differences from Phase 3
- **Output**: Returns integer (-1 or +1) instead of boolean
- **Purpose**: Classifies cost direction of hallucinations (overestimate vs. underestimate)
- **Conditional**: Only called when hallucination is detected (in orchestrator)
- **Prompt**: Different prompt template focusing on cost direction analysis

## Blockers Removed

- ✅ Phase 3 validation complete
- ✅ All tests passing
- ✅ Coverage requirement met (86%)
- ✅ Interface contract implemented
- ✅ Critical requirement verified (reference answer NOT used)

## Next Steps

Proceed to **Phase 4: Hallucination Cost LLM-Node** using `@docs/initiatives/eval_system/prompt_phase_4_001.md`.

