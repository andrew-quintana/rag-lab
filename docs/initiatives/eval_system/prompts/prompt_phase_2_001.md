# Phase 2 Prompt — Correctness LLM-Node

## Context

This prompt guides the implementation of **Phase 2: Correctness LLM-Node** for the RAG Evaluation MVP system. This phase implements the correctness classification LLM node that directly compares model answers to gold reference answers.

**Related Documents:**
- @docs/initiatives/eval_system/scoping/PRD001.md - Product requirements (FR2: Correctness LLM-Node)
- @docs/initiatives/eval_system/scoping/RFC001.md - Technical design (Phase 2: Correctness LLM-Node, Interface Contracts)
- @docs/initiatives/eval_system/scoping/TODO001.md - Implementation tasks (Phase 2 section - check off tasks as completed)
- @docs/initiatives/eval_system/scoping/context.md - Project context

## Objectives

1. **Implement Correctness Node**: Create LLM node for direct comparison of model answer to reference answer
2. **Create Prompt Template**: Design and implement prompt for correctness classification
3. **Integrate Azure Foundry**: Use Azure Foundry GPT-4o-mini with structured output
4. **Test Thoroughly**: Achieve 80%+ test coverage with comprehensive unit tests

## Execution Requirements

### Checklist Modifications
- **REQUIRED**: Check off completed tasks in @docs/initiatives/eval_system/scoping/TODO001.md Phase 2 section as you complete them
- Update phase status when all validation requirements are met

### Validation
- **REQUIRED**: All unit tests for Phase 2 must pass before proceeding to Phase 3
- **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/evaluator/test_evaluator_correctness.py -v`
- **REQUIRED**: Test coverage must meet minimum 80% for correctness.py module
- **REQUIRED**: All test assertions must pass (no failures, no errors)

### Documentation
- **REQUIRED**: Create `phase_2_decisions.md` if any decisions are made that aren't in PRD/RFC
- **REQUIRED**: Create `phase_2_testing.md` documenting testing results
- **REQUIRED**: Create `phase_2_handoff.md` summarizing what's needed for Phase 3

## Key References

### Interface Contract (from RFC001.md)
```python
def classify_correctness(
    query: str,
    model_answer: str,
    reference_answer: str,
    config: Optional[Config] = None
) -> bool
```

### Implementation Location
- `rag_eval/services/evaluator/correctness.py`

### Test Location
- `backend/tests/components/evaluator/test_evaluator_correctness.py`

### Prompt Location
- `backend/rag_eval/prompts/evaluation/correctness_prompt.md` (or store in database with `prompt_type="evaluation"`)

### Existing Components
- @backend/rag_eval/core/config.py - Configuration management
- @backend/rag_eval/core/interfaces.py - Data structures
- @backend/rag_eval/services/rag/generation.py - Example of Azure Foundry LLM integration

## Phase 2 Tasks

### Setup
1. **REQUIRED**: Activate backend venv: `cd backend && source venv/bin/activate`
2. Create `rag_eval/services/evaluator/correctness.py` module
3. Ensure package `__init__.py` files are properly configured with exports
4. Verify imports work correctly
5. Review Azure Foundry API configuration
6. Set up test fixtures for mock LLM responses
7. Create test file: `backend/tests/components/evaluator/test_evaluator_correctness.py`

### Prompt Creation
1. Create prompt template for correctness classification
2. Prompt design:
   - System instruction: "You are an expert evaluator comparing a model answer directly to a gold reference answer."
   - Input placeholders: `{query}`, `{model_answer}`, `{reference_answer}`
   - Output format: JSON with `correctness_binary` (bool) and `reasoning` (str)
   - Include examples of correct/incorrect classifications
3. Test prompt template with sample inputs

### Core Implementation
1. Implement `classify_correctness()` function matching RFC001 interface
2. Load prompt template (from file or database)
3. Construct prompt with query, model answer, reference answer
4. Call Azure Foundry GPT-4o-mini with structured output (JSON)
5. Set temperature=0.1 for reproducibility
6. Parse JSON response to extract `correctness_binary`
7. Return boolean classification
8. Handle LLM failures with proper error handling (`AzureServiceError`)
9. Validate inputs are non-empty (`ValueError`)
10. Implement `_construct_correctness_prompt()` helper function

### Testing
1. Unit tests for `classify_correctness()`:
   - Test binary classification: correct (true)
   - Test binary classification: incorrect (false)
   - Test direct comparison between model answer and reference answer
   - Test edge case: empty model answer
   - Test edge case: empty reference answer
   - Test error handling for LLM failures
   - Test input validation (empty strings)
2. Connection test for Azure Foundry API (warns if credentials missing, doesn't fail tests)
3. **Document any failures** in fracas.md immediately when encountered

### Documentation
1. Add docstrings to all functions matching RFC001 interface contract
2. Document return types and error conditions
3. Document prompt design and rationale
4. Document temperature setting (0.1) for reproducibility

## Success Criteria

- [ ] `classify_correctness()` function implemented matching RFC001 interface
- [ ] Prompt template created and tested
- [ ] All unit tests pass with 80%+ coverage
- [ ] All error handling implemented
- [ ] All Phase 2 tasks in TODO001.md checked off
- [ ] Phase 2 handoff document created

## Important Notes

- **Temperature**: Use temperature=0.1 for all LLM calls (reproducibility)
- **Error Handling**: Must handle `AzureServiceError` and `ValueError` appropriately
- **Interface Compliance**: Function signature must match RFC001 exactly
- **Test Coverage**: Minimum 80% coverage required

## Blockers

- **BLOCKER**: Phase 3 cannot proceed until Phase 2 validation complete
- **BLOCKER**: All tests must pass before proceeding

## Next Phase

After completing Phase 2, proceed to **Phase 3: Hallucination LLM-Node** using @docs/initiatives/eval_system/prompt_phase_3_001.md

