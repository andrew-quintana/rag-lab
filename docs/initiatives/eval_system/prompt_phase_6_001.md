# Phase 6 Prompt — Hallucination Impact LLM-Node

## Context

This prompt guides the implementation of **Phase 6: Hallucination Impact LLM-Node** for the RAG Evaluation MVP system. This phase implements the impact magnitude calculation LLM node that computes hallucination impact on a 0-3 scale, handling mixed resource types (time, money, steps).

**Related Documents:**
- @docs/initiatives/eval_system/PRD001.md - Product requirements (FR5: Hallucination Impact LLM-Node)
- @docs/initiatives/eval_system/RFC001.md - Technical design (Phase 5: Hallucination Impact LLM-Node, Interface Contracts)
- @docs/initiatives/eval_system/TODO001.md - Implementation tasks (Phase 6 section - check off tasks as completed)
- @docs/initiatives/eval_system/context.md - Project context

## Objectives

1. **Implement Impact Calculation Node**: Create LLM node for impact magnitude calculation (0-3 scale)
2. **Create Prompt Template**: Design prompt for handling mixed resource types and relative importance
3. **Integrate Azure Foundry**: Use Azure Foundry GPT-4o-mini with structured output
4. **Test Thoroughly**: Achieve 80%+ test coverage with comprehensive unit tests

## Execution Requirements

### Checklist Modifications
- **REQUIRED**: Check off completed tasks in @docs/initiatives/eval_system/TODO001.md Phase 6 section as you complete them
- Update phase status when all validation requirements are met

### Validation
- **REQUIRED**: All unit tests for Phase 6 must pass before proceeding to Phase 7
- **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/evaluator/test_evaluator_risk_impact.py -v`
- **REQUIRED**: Test coverage must meet minimum 80% for risk_impact.py module
- **REQUIRED**: All test assertions must pass (no failures, no errors)

### Documentation
- **REQUIRED**: Create `phase_6_decisions.md` if any decisions are made that aren't in PRD/RFC
- **REQUIRED**: Create `phase_6_testing.md` documenting testing results
- **REQUIRED**: Create `phase_6_handoff.md` summarizing what's needed for Phase 7

## Key References

### Interface Contract (from RFC001.md)
```python
def calculate_risk_impact(
    model_answer_cost: Dict[str, Any],  # {"time": ..., "money": ..., "steps": ...}
    actual_cost: Dict[str, Any],  # From retrieved chunks (ground truth)
    config: Optional[Config] = None
) -> float
```

### Impact Scale
- **0**: Minimal/no impact
- **1**: Low impact
- **2**: Moderate impact
- **3**: High/severe impact

### Implementation Location
- `rag_eval/services/evaluator/risk_impact.py`
- **Base Class**: `rag_eval/services/evaluator/base_evaluator.py` - Inherit from `BaseEvaluatorNode`
- **LLM Provider**: `rag_eval/services/shared/llm_providers.py` - Use `LLMProvider` abstraction

### Test Location
- `backend/tests/components/evaluator/test_evaluator_risk_impact.py`

### Prompt Location
- `backend/rag_eval/prompts/evaluation/risk_impact_prompt.md` (or store in database)

### Dependencies
- Phase 5: Cost Extraction LLM-Node (for extracting costs from text)

### Reference Implementation
- See `rag_eval/services/evaluator/correctness.py` for example of `BaseEvaluatorNode` usage

## Phase 6 Tasks

### Setup
1. **REQUIRED**: Activate backend venv: `cd backend && source venv/bin/activate`
2. Create `rag_eval/services/evaluator/risk_impact.py` module
3. Ensure package `__init__.py` files are properly configured with exports
4. Verify imports work correctly
5. Review Azure Foundry API configuration
6. Set up test fixtures for mock LLM responses
7. Create test file: `backend/tests/components/evaluator/test_evaluator_risk_impact.py`

### Prompt Creation
1. Create prompt template for impact calculation
2. Prompt design:
   - System instruction: "You are an expert evaluator calculating the real-world impact magnitude of hallucinations."
   - Input placeholders: `{model_answer_cost}`, `{actual_cost}`
   - Output format: JSON with `risk_impact` (float 0-3) and `reasoning` (str)
   - Explain impact scale:
     - 0: Minimal/no impact
     - 1: Low impact
     - 2: Moderate impact
     - 3: High/severe impact
   - **CRITICAL**: Emphasize considering mixed resource types (time, money, steps) and their relative importance
   - Include examples of impact calculations for different cost differences
3. Test prompt template with sample inputs

### Core Implementation
1. Create `HallucinationImpactEvaluator` class inheriting from `BaseEvaluatorNode`:
   - Override `_construct_prompt()` method to build impact calculation-specific prompt
   - Implement `calculate_risk_impact()` method using base class `_call_llm()` and `_parse_json_response()`
   - Format cost dictionaries for prompt (JSON representation)
   - Parse JSON response to extract `risk_impact` (0-3)
   - Validate impact is in range [0, 3]
   - Return float impact magnitude
   - Validate inputs are non-empty dictionaries
2. Implement module-level `calculate_risk_impact()` function for backward compatibility:
   - Create `HallucinationImpactEvaluator` instance
   - Call `calculate_risk_impact()` method
   - Return result
3. **Note**: Base class handles prompt loading, LLM calls (via provider), JSON parsing, and error handling

### Testing
1. Unit tests for `calculate_risk_impact()`:
   - Test impact calculation for time-based costs
   - Test impact calculation for money-based costs
   - Test impact calculation for step-based costs
   - Test impact calculation for mixed resource types
   - Test impact scaling factor range [0, 3]
   - Test edge case: zero impact scenarios
   - Test edge case: maximum impact scenarios
   - Test error handling for LLM failures
2. Connection test for Azure Foundry API
3. **Document any failures** in fracas.md immediately when encountered

### Documentation
1. Add docstrings to all functions
2. Document impact scale (0-3) and rationale
3. Document handling of mixed resource types
4. Document why LLM node is used (not deterministic function)

## Success Criteria

- [ ] `calculate_risk_impact()` function implemented matching RFC001 interface
- [ ] Prompt template created with emphasis on mixed resource types
- [ ] All unit tests pass with 80%+ coverage
- [ ] Tests verify impact range [0, 3]
- [ ] All error handling implemented
- [ ] All Phase 6 tasks in TODO001.md checked off
- [ ] Phase 6 handoff document created

## Important Notes

- **Impact Range**: Must validate impact is in range [0, 3]
- **Temperature**: Use temperature=0.1 for all LLM calls
- **Mixed Resources**: Must handle time, money, and steps with relative importance assessment
- **LLM Rationale**: Uses LLM node (not deterministic function) because it requires nuanced reasoning about mixed resource types
- **Test Coverage**: Minimum 80% coverage required

## Blockers

- **BLOCKER**: Phase 7 cannot proceed until Phase 6 validation complete
- **BLOCKER**: All tests must pass before proceeding

## Next Phase

After completing Phase 6, proceed to **Phase 7: LLM-as-Judge Orchestrator** using @docs/initiatives/eval_system/prompt_phase_7_001.md

