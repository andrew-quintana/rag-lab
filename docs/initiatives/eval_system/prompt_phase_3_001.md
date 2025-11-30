# Phase 3 Prompt — Hallucination LLM-Node

## Context

This prompt guides the implementation of **Phase 3: Hallucination LLM-Node** for the RAG Evaluation MVP system. This phase implements the hallucination binary classification LLM node that performs strict grounding analysis against retrieved evidence.

**Related Documents:**
- @docs/initiatives/eval_system/PRD001.md - Product requirements (FR3: Hallucination LLM-Node)
- @docs/initiatives/eval_system/RFC001.md - Technical design (Phase 3: Hallucination LLM-Node, Interface Contracts)
- @docs/initiatives/eval_system/TODO001.md - Implementation tasks (Phase 3 section - check off tasks as completed)
- @docs/initiatives/eval_system/context.md - Project context

## Objectives

1. **Implement Hallucination Node**: Create LLM node for binary hallucination classification
2. **Create Prompt Template**: Design prompt emphasizing grounding analysis (reference answer NOT used)
3. **Integrate Azure Foundry**: Use Azure Foundry GPT-4o-mini with structured output
4. **Test Thoroughly**: Achieve 80%+ test coverage with comprehensive unit tests

## Execution Requirements

### Checklist Modifications
- **REQUIRED**: Check off completed tasks in @docs/initiatives/eval_system/TODO001.md Phase 3 section as you complete them
- Update phase status when all validation requirements are met

### Validation
- **REQUIRED**: All unit tests for Phase 3 must pass before proceeding to Phase 4
- **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/evaluator/test_evaluator_hallucination.py -v`
- **REQUIRED**: Test coverage must meet minimum 80% for hallucination.py module
- **REQUIRED**: All test assertions must pass (no failures, no errors)

### Documentation
- **REQUIRED**: Create `phase_3_decisions.md` if any decisions are made that aren't in PRD/RFC
- **REQUIRED**: Create `phase_3_testing.md` documenting testing results
- **REQUIRED**: Create `phase_3_handoff.md` summarizing what's needed for Phase 4

## Key References

### Interface Contract (from RFC001.md)
```python
def classify_hallucination(
    retrieved_context: List[RetrievalResult],
    model_answer: str,
    config: Optional[Config] = None
) -> bool
```

### Implementation Location
- `rag_eval/services/evaluator/hallucination.py`
- **Base Class**: `rag_eval/services/evaluator/base_evaluator.py` - Inherit from `BaseEvaluatorNode`
- **LLM Provider**: `rag_eval/services/shared/llm_providers.py` - Use `LLMProvider` abstraction

### Test Location
- `backend/tests/components/evaluator/test_evaluator_hallucination.py`

### Prompt Location
- `backend/rag_eval/prompts/evaluation/hallucination_prompt.md` (or store in database)

### Data Structures
- `RetrievalResult` from @backend/rag_eval/core/interfaces.py

### Reference Implementation
- See `rag_eval/services/evaluator/correctness.py` for example of `BaseEvaluatorNode` usage

## Phase 3 Tasks

### Setup
1. **REQUIRED**: Activate backend venv: `cd backend && source venv/bin/activate`
2. Create `rag_eval/services/evaluator/hallucination.py` module
3. Ensure package `__init__.py` files are properly configured with exports
4. Verify imports work correctly
5. Review Azure Foundry API configuration
6. Set up test fixtures for mock LLM responses
7. Create test file: `backend/tests/components/evaluator/test_evaluator_hallucination.py`

### Prompt Creation
1. Create prompt template for hallucination classification
2. Prompt design:
   - System instruction: "You are an expert evaluator analyzing whether a model answer contains hallucinations based on grounding in retrieved evidence."
   - Input placeholders: `{retrieved_context}`, `{model_answer}`
   - Output format: JSON with `hallucination_binary` (bool) and `reasoning` (str)
   - **CRITICAL**: Emphasize that reference answer is NOT used in hallucination detection
   - Include examples of hallucination detection (grounded vs. ungrounded claims)
3. Test prompt template with sample inputs

### Core Implementation
1. Create `HallucinationEvaluator` class inheriting from `BaseEvaluatorNode`:
   - Override `_construct_prompt()` method to build hallucination-specific prompt
   - Implement `classify_hallucination()` method using base class `_call_llm()` and `_parse_json_response()`
   - Format retrieved context (concatenate chunk texts with chunk IDs)
   - Validate `hallucination_binary` field in parsed JSON response
   - Return boolean classification
   - Validate inputs are non-empty
2. Implement module-level `classify_hallucination()` function for backward compatibility:
   - Create `HallucinationEvaluator` instance
   - Call `classify_hallucination()` method
   - Return result
3. **Note**: Base class handles prompt loading, LLM calls (via provider), JSON parsing, and error handling

### Testing
1. Unit tests for `classify_hallucination()`:
   - Test binary classification: hallucination detected (true)
   - Test binary classification: no hallucination (false)
   - Test grounding analysis: information not in retrieved evidence
   - Test grounding analysis: information supported by evidence
   - Test edge case: ambiguous grounding scenarios
   - **CRITICAL**: Test that reference answer is NOT used in hallucination detection
   - Test edge case: zero retrieved chunks
   - Test error handling for LLM failures
2. Connection test for Azure Foundry API
3. **Document any failures** in fracas.md immediately when encountered

### Documentation
1. Add docstrings to all functions
2. **CRITICAL**: Document that reference answer is NOT used in hallucination detection
3. Document prompt design and grounding analysis approach

## Success Criteria

- [ ] `classify_hallucination()` function implemented matching RFC001 interface
- [ ] Prompt template created with emphasis on NOT using reference answer
- [ ] All unit tests pass with 80%+ coverage
- [ ] Tests verify reference answer is NOT used
- [ ] All error handling implemented
- [ ] All Phase 3 tasks in TODO001.md checked off
- [ ] Phase 3 handoff document created

## Important Notes

- **CRITICAL**: Reference answer must NOT be used in hallucination detection (only retrieved chunks)
- **Temperature**: Use temperature=0.1 for all LLM calls
- **Grounding Analysis**: Strict grounding analysis against retrieved evidence only
- **Test Coverage**: Minimum 80% coverage required

## Blockers

- **BLOCKER**: Phase 4 cannot proceed until Phase 3 validation complete
- **BLOCKER**: All tests must pass before proceeding

## Next Phase

After completing Phase 3, proceed to **Phase 4: Hallucination Cost LLM-Node** using @docs/initiatives/eval_system/prompt_phase_4_001.md

