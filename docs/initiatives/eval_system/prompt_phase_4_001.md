# Phase 4 Prompt — Hallucination Cost LLM-Node

## Context

This prompt guides the implementation of **Phase 4: Hallucination Cost LLM-Node** for the RAG Evaluation MVP system. This phase implements the cost type classification LLM node that classifies hallucinations as opportunity cost (-1) or resource cost (+1).

**Related Documents:**
- @docs/initiatives/eval_system/PRD001.md - Product requirements (FR4: Hallucination Cost LLM-Node)
- @docs/initiatives/eval_system/RFC001.md - Technical design (Phase 4: Hallucination Cost LLM-Node, Interface Contracts)
- @docs/initiatives/eval_system/TODO001.md - Implementation tasks (Phase 4 section - check off tasks as completed)
- @docs/initiatives/eval_system/context.md - Project context

## Objectives

1. **Implement Cost Classification Node**: Create LLM node for cost type classification (-1 or +1)
2. **Create Prompt Template**: Design prompt explaining opportunity cost vs. resource cost
3. **Integrate Azure Foundry**: Use Azure Foundry GPT-4o-mini with structured output
4. **Test Thoroughly**: Achieve 80%+ test coverage with comprehensive unit tests

## Execution Requirements

### Checklist Modifications
- **REQUIRED**: Check off completed tasks in @docs/initiatives/eval_system/TODO001.md Phase 4 section as you complete them
- Update phase status when all validation requirements are met

### Validation
- **REQUIRED**: All unit tests for Phase 4 must pass before proceeding to Phase 5
- **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/evaluator/test_evaluator_hallucination_cost.py -v`
- **REQUIRED**: Test coverage must meet minimum 80% for hallucination_cost.py module
- **REQUIRED**: All test assertions must pass (no failures, no errors)

### Documentation
- **REQUIRED**: Create `phase_4_decisions.md` if any decisions are made that aren't in PRD/RFC
- **REQUIRED**: Create `phase_4_testing.md` documenting testing results
- **REQUIRED**: Create `phase_4_handoff.md` summarizing what's needed for Phase 5

## Key References

### Interface Contract (from RFC001.md)
```python
def classify_hallucination_cost(
    model_answer: str,
    retrieved_context: List[RetrievalResult],
    config: Optional[Config] = None
) -> int
```

### Cost Classification Values
- **-1 = Opportunity Cost**: Model overestimated cost, dissuading user from seeking care
- **+1 = Resource Cost**: Model underestimated cost, persuading user to pursue care

### Implementation Location
- `rag_eval/services/evaluator/hallucination_cost.py`
- **Base Class**: `rag_eval/services/evaluator/base_evaluator.py` - Inherit from `BaseEvaluatorNode`
- **LLM Provider**: `rag_eval/services/shared/llm_providers.py` - Use `LLMProvider` abstraction

### Test Location
- `backend/tests/components/evaluator/test_evaluator_hallucination_cost.py`

### Prompt Location
- `backend/rag_eval/prompts/evaluation/hallucination_cost_prompt.md` (or store in database)

### Reference Implementation
- See `rag_eval/services/evaluator/correctness.py` for example of `BaseEvaluatorNode` usage

## Phase 4 Tasks

### Setup
1. **REQUIRED**: Activate backend venv: `cd backend && source venv/bin/activate`
2. Create `rag_eval/services/evaluator/hallucination_cost.py` module
3. Ensure package `__init__.py` files are properly configured with exports
4. Verify imports work correctly
5. Review Azure Foundry API configuration
6. Set up test fixtures for mock LLM responses
7. Create test file: `backend/tests/components/evaluator/test_evaluator_hallucination_cost.py`

### Prompt Creation
1. Create prompt template for cost classification
2. Prompt design:
   - System instruction: "You are an expert evaluator classifying the cost impact direction of hallucinations."
   - Input placeholders: `{model_answer}`, `{retrieved_context}`
   - Output format: JSON with `hallucination_cost` (-1 or +1) and `reasoning` (str)
   - Explain cost classification:
     - -1 = Opportunity Cost: Model overestimated cost, dissuading user from seeking care
     - +1 = Resource Cost: Model underestimated cost, persuading user to pursue care
   - **CRITICAL**: Emphasize that reference answer is NOT used (only retrieved chunks for ground truth)
   - Include examples of opportunity cost vs. resource cost classifications
3. Test prompt template with sample inputs

### Core Implementation
1. Create `HallucinationCostEvaluator` class inheriting from `BaseEvaluatorNode`:
   - Override `_construct_prompt()` method to build cost classification-specific prompt
   - Implement `classify_hallucination_cost()` method using base class `_call_llm()` and `_parse_json_response()`
   - Format retrieved context
   - Validate `hallucination_cost` field in parsed JSON response (-1 or +1)
   - Return integer classification
   - Validate inputs are non-empty
   - Handle ambiguous cost direction cases
2. Implement module-level `classify_hallucination_cost()` function for backward compatibility:
   - Create `HallucinationCostEvaluator` instance
   - Call `classify_hallucination_cost()` method
   - Return result
3. **Note**: Base class handles prompt loading, LLM calls (via provider), JSON parsing, and error handling

### Testing
1. Unit tests for `classify_hallucination_cost()`:
   - Test opportunity cost classification (-1): overestimated cost
   - Test resource cost classification (+1): underestimated cost
   - Test cost analysis for quantitative hallucinations
   - Test cost analysis for non-quantitative hallucinations
   - Test edge case: ambiguous cost direction
   - **CRITICAL**: Test that reference answer is NOT used in cost classification
   - Test error handling for LLM failures
2. Connection test for Azure Foundry API
3. **Document any failures** in fracas.md immediately when encountered

### Documentation
1. Add docstrings to all functions
2. Document cost classification logic (-1 vs. +1)
3. **CRITICAL**: Document that reference answer is NOT used
4. Document prompt design and cost direction analysis

## Success Criteria

- [ ] `classify_hallucination_cost()` function implemented matching RFC001 interface
- [ ] Prompt template created with clear cost classification explanation
- [ ] All unit tests pass with 80%+ coverage
- [ ] Tests verify reference answer is NOT used
- [ ] All error handling implemented
- [ ] All Phase 4 tasks in TODO001.md checked off
- [ ] Phase 4 handoff document created

## Important Notes

- **CRITICAL**: Reference answer must NOT be used in cost classification (only retrieved chunks)
- **Temperature**: Use temperature=0.1 for all LLM calls
- **Cost Classification**: Applies to all hallucinations (quantitative and non-quantitative)
- **Test Coverage**: Minimum 80% coverage required

## Blockers

- **BLOCKER**: Phase 5 cannot proceed until Phase 4 validation complete
- **BLOCKER**: All tests must pass before proceeding

## Next Phase

After completing Phase 4, proceed to **Phase 5: Cost Extraction LLM-Node** using @docs/initiatives/eval_system/prompt_phase_5_001.md

