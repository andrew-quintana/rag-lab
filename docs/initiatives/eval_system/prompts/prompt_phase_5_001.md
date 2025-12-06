# Phase 5 Prompt — Cost Extraction LLM-Node

## Context

This prompt guides the implementation of **Phase 5: Cost Extraction LLM-Node** for the RAG Evaluation MVP system. This phase implements the cost extraction LLM node that parses structured cost information (time, money, steps) from unstructured text.

**Related Documents:**
- @docs/initiatives/eval_system/scoping/PRD001.md - Product requirements (FR5: Risk Impact LLM-Node - requires cost extraction)
- @docs/initiatives/eval_system/scoping/RFC001.md - Technical design (Phase 4: Cost Extraction LLM-Node, Interface Contracts)
- @docs/initiatives/eval_system/scoping/TODO001.md - Implementation tasks (Phase 5 section - check off tasks as completed)
- @docs/initiatives/eval_system/scoping/context.md - Project context

## Objectives

1. **Implement Cost Extraction Node**: Create LLM node to extract cost information from text
2. **Create Prompt Template**: Design prompt for parsing time, money, and steps from natural language
3. **Integrate Azure Foundry**: Use Azure Foundry GPT-4o-mini with structured output
4. **Test Thoroughly**: Achieve 80%+ test coverage with comprehensive unit tests

## Execution Requirements

### Checklist Modifications
- **REQUIRED**: Check off completed tasks in @docs/initiatives/eval_system/scoping/TODO001.md Phase 5 section as you complete them
- Update phase status when all validation requirements are met

### Validation
- **REQUIRED**: All unit tests for Phase 5 must pass before proceeding to Phase 6
- **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/evaluator/test_evaluator_cost_extraction.py -v`
- **REQUIRED**: Test coverage must meet minimum 80% for cost_extraction.py module
- **REQUIRED**: All test assertions must pass (no failures, no errors)

### Documentation
- **REQUIRED**: Create `phase_5_decisions.md` if any decisions are made that aren't in PRD/RFC
- **REQUIRED**: Create `phase_5_testing.md` documenting testing results
- **REQUIRED**: Create `phase_5_handoff.md` summarizing what's needed for Phase 6

## Key References

### Interface Contract (from RFC001.md)
```python
def extract_costs(text: str, config: Optional[Config] = None) -> Dict[str, Any]
```

### Output Format
- Dictionary with optional fields: `time` (optional float/str), `money` (optional float/str), `steps` (optional int/str), and `reasoning` (str)
- Example: `{"time": "2 hours", "money": 500.0, "steps": 3, "reasoning": "..."}`

### Implementation Location
- `rag_eval/services/evaluator/cost_extraction.py`
- **Base Class**: `rag_eval/services/evaluator/base_evaluator.py` - Inherit from `BaseEvaluatorNode`
- **LLM Provider**: `rag_eval/services/shared/llm_providers.py` - Use `LLMProvider` abstraction

### Test Location
- `backend/tests/components/evaluator/test_evaluator_cost_extraction.py`

### Prompt Location
- **Database**: Prompts are stored in `public.prompts` table with `prompt_type='evaluation'` and `name='cost_extraction_evaluator'`
- **Testing**: Use `prompt_path` parameter pointing to test fixtures in `tests/fixtures/prompts/` for unit tests
- **Production**: Use `query_executor` parameter to load from database (required)

### Reference Implementation
- See `rag_eval/services/evaluator/correctness.py` for example of `BaseEvaluatorNode` usage

## Phase 5 Tasks

### Setup
1. **REQUIRED**: Activate backend venv: `cd backend && source venv/bin/activate`
2. Create `rag_eval/services/evaluator/cost_extraction.py` module
3. Ensure package `__init__.py` files are properly configured with exports
4. Verify imports work correctly
5. Review Azure Foundry API configuration
6. Set up test fixtures for mock LLM responses
7. Create test file: `backend/tests/components/evaluator/test_evaluator_cost_extraction.py`

### Prompt Creation
1. Create prompt template for cost extraction
2. Prompt design:
   - System instruction: "You are an expert parser extracting cost information (time, money, steps) from text."
   - Input placeholder: `{text}`
   - Output format: JSON with `time` (optional float/str), `money` (optional float/str), `steps` (optional int/str), and `reasoning` (str)
   - Include examples of cost extraction from various formats:
     - Time: "2 hours", "30 minutes", "1 day"
     - Money: "$500", "500 dollars", "500.00"
     - Steps: "3 steps", "step 3", "third step"
   - Handle missing cost information (optional fields)
3. Test prompt template with sample inputs

### Core Implementation
1. Create `CostExtractionEvaluator` class inheriting from `BaseEvaluatorNode`:
   - Override `_construct_prompt()` method to build cost extraction-specific prompt
   - Implement `extract_costs()` method using base class `_call_llm()` and `_parse_json_response()`
   - Parse JSON response to extract cost fields (time, money, steps)
   - Return dictionary with optional cost fields
   - Validate input is non-empty
2. Implement module-level `extract_costs()` function for backward compatibility:
   - Create `CostExtractionEvaluator` instance
   - Call `extract_costs()` method
   - Return result
3. **Note**: Base class handles prompt loading, LLM calls (via provider), JSON parsing, and error handling

### Testing
1. Unit tests for `extract_costs()`:
   - Test extraction of time-based costs (e.g., "2 hours", "30 minutes")
   - Test extraction of money-based costs (e.g., "$500", "500 dollars", "500.00")
   - Test extraction of step-based costs (e.g., "3 steps", "step 3")
   - Test extraction of mixed cost types from same text
   - Test handling of missing cost information (optional fields)
   - Test edge case: no cost information in text
   - Test edge case: ambiguous cost expressions
   - Test error handling for LLM failures
2. Connection test for Azure Foundry API
3. **Document any failures** in fracas.md immediately when encountered

### Documentation
1. Add docstrings to all functions
2. Document cost extraction format and supported expressions
3. Document optional fields and handling of missing information

## Success Criteria

- [ ] `extract_costs()` function implemented matching RFC001 interface
- [ ] Prompt template created with examples of various cost formats
- [ ] All unit tests pass with 80%+ coverage
- [ ] Tests cover time, money, and steps extraction
- [ ] All error handling implemented
- [ ] All Phase 5 tasks in TODO001.md checked off
- [ ] Phase 5 handoff document created

## Important Notes

- **Optional Fields**: All cost fields (time, money, steps) are optional - handle missing information gracefully
- **Temperature**: Use temperature=0.1 for all LLM calls
- **Format Flexibility**: Must handle various formats (e.g., "$500", "500 dollars", "500.00")
- **Test Coverage**: Minimum 80% coverage required

## Blockers

- **BLOCKER**: Phase 6 cannot proceed until Phase 5 validation complete
- **BLOCKER**: All tests must pass before proceeding

## Next Phase

After completing Phase 5, proceed to **Phase 6: Risk Impact LLM-Node** using @docs/initiatives/eval_system/prompt_phase_6_001.md

