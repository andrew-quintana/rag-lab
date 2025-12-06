# Phase 2 Handoff — Correctness LLM-Node

## Status

✅ **Phase 2 Complete**: All validation requirements met, ready for Phase 3

## Summary

Phase 2 successfully implements the correctness classification LLM node that directly compares model answers to gold reference answers. The implementation includes:

- ✅ `classify_correctness()` function matching RFC001 interface
- ✅ Prompt template with examples and JSON output format
- ✅ Azure Foundry GPT-4o-mini integration with structured output
- ✅ Comprehensive test suite (23 tests, 83% coverage)
- ✅ Error handling for AzureServiceError and ValueError
- ✅ Input validation for empty strings

## Implementation Details

### Files Created
1. **`backend/rag_eval/services/evaluator/correctness.py`**
   - Main implementation module
   - Functions: `classify_correctness()`, `_construct_correctness_prompt()`, `_call_correctness_api()`, `_load_prompt_template()`
   - Error handling: AzureServiceError, ValueError
   - Temperature: 0.1 (for reproducibility)

2. **`backend/rag_eval/prompts/evaluation/correctness_prompt.md`**
   - Prompt template with system instructions
   - Placeholders: `{query}`, `{model_answer}`, `{reference_answer}`
   - Output format: JSON with `correctness_binary` (bool) and `reasoning` (str)
   - Examples included for correct/incorrect classifications

3. **`backend/tests/components/evaluator/test_evaluator_correctness.py`**
   - 23 comprehensive unit tests
   - Coverage: 83% (exceeds 80% requirement)
   - Tests: input validation, API calls, error handling, integration

### Package Exports
- Updated `backend/rag_eval/services/evaluator/__init__.py` to export `classify_correctness`

## Interface Contract (RFC001 Compliance)

```python
def classify_correctness(
    query: str,
    model_answer: str,
    reference_answer: str,
    config: Optional[Config] = None
) -> bool
```

**Status**: ✅ Fully implemented and tested

## Test Results

- **Tests**: 23/23 passed
- **Coverage**: 83% (exceeds 80% requirement)
- **Execution time**: ~4 seconds
- **Integration test**: ✅ Passed (when credentials available)

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
from rag_eval.services.evaluator.correctness import classify_correctness

config = Config.from_env()
query = "What is the copay for a specialist visit?"
model_answer = "The copay for a specialist visit is $50."
reference_answer = "Specialist visits have a $50 copay."

is_correct = classify_correctness(query, model_answer, reference_answer, config)
print(f"Correctness: {is_correct}")  # True
```

## Error Handling

### Exceptions Raised
- **ValueError**: Invalid inputs (empty strings, missing fields in response)
- **AzureServiceError**: Azure API failures (network errors, API errors, retry exhaustion)

### Error Handling Strategy
- Input validation before API calls
- Retry logic with exponential backoff (3 retries)
- Detailed error messages for debugging
- Graceful handling of missing credentials in tests

## Decisions Made

See `phase_2_decisions.md` for detailed decisions:
1. File-based prompt storage (not database)
2. JSON parsing with markdown code block support
3. Model selection (gpt-4o-mini with fallback)
4. Temperature setting (0.1)
5. Error handling strategy
6. Max tokens (500)
7. Retry logic (3 retries with exponential backoff)

## What's Needed for Phase 3

### Prerequisites
- ✅ Phase 2 complete and validated
- ✅ Correctness node available for orchestrator integration
- ✅ Test patterns established for LLM node testing

### Phase 3 Requirements
1. **Hallucination LLM-Node**: Implement `classify_hallucination()` function
   - Interface: `classify_hallucination(retrieved_context: List[RetrievalResult], model_answer: str, config: Optional[Config] = None) -> bool`
   - Similar structure to correctness node
   - Uses retrieved context (not reference answer)
   - Prompt template: `backend/rag_eval/prompts/evaluation/hallucination_prompt.md`

2. **Test File**: `backend/tests/components/evaluator/test_evaluator_hallucination.py`
   - Follow same test patterns as correctness tests
   - Test grounding analysis against retrieved context
   - Test edge cases (empty context, empty answer)

3. **Documentation**: Create phase_3_decisions.md, phase_3_testing.md, phase_3_handoff.md

### Key Differences from Phase 2
- **Input**: Uses `retrieved_context` (List[RetrievalResult]) instead of `reference_answer`
- **Purpose**: Detects hallucinations (ungrounded claims) rather than correctness
- **Prompt**: Different prompt template focusing on grounding analysis

## Blockers Removed

- ✅ Phase 2 validation complete
- ✅ All tests passing
- ✅ Coverage requirement met
- ✅ Interface contract implemented

## Next Steps

Proceed to **Phase 3: Hallucination LLM-Node** using `@docs/initiatives/eval_system/prompt_phase_3_001.md`.


