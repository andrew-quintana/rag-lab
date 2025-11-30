# Phase 6 Decisions — Risk Impact LLM-Node

**Date**: 2024-12-19  
**Status**: Complete

## Implementation Decisions

### Decision 1: Cost Dictionary Formatting

**Decision**: Format cost dictionaries as JSON strings in the prompt using `json.dumps()` with indentation.

**Rationale**:
- JSON formatting provides clear, structured representation of cost data
- Indentation improves readability for LLM
- Handles mixed types (strings, numbers) gracefully
- Fallback to `str()` if JSON serialization fails (edge case)

**Implementation**: `_format_cost_dict()` method in `RiskImpactEvaluator` class.

### Decision 2: Impact Range Validation

**Decision**: Validate `risk_impact` is in range [0, 3] after parsing from LLM response.

**Rationale**:
- Ensures output conforms to specification
- Prevents invalid impact values from propagating
- Provides clear error messages for debugging
- Validates both integer and float types (converts to float)

**Implementation**: Validation in `calculate_risk_impact()` method after JSON parsing.

### Decision 3: Input Validation

**Decision**: Validate that both `model_answer_cost` and `actual_cost` are non-empty dictionaries.

**Rationale**:
- Prevents invalid inputs from reaching LLM
- Provides clear error messages
- Ensures prompt construction has valid data
- Catches errors early in the execution flow

**Implementation**: Input validation at the start of `calculate_risk_impact()` method.

### Decision 4: Prompt Template Storage

**Decision**: Store prompt template in database (`public.prompts` table) with:
- `prompt_type='evaluation'`
- `name='risk_impact_evaluator'`
- `version='0.1'`
- `live=true`

**Rationale**:
- Consistent with other evaluation prompts
- Enables versioning and updates
- Supports live version management
- Allows prompt updates without code changes

**Implementation**: SQL migration file `0010_insert_risk_impact_prompt_v0_1.sql`.

### Decision 5: Test Prompt File

**Decision**: Create test prompt file at `backend/tests/fixtures/prompts/risk_impact_prompt.md` for unit testing.

**Rationale**:
- Enables testing without database connection
- Provides reference documentation
- Supports offline development
- Maintains backward compatibility with `prompt_path` parameter

**Implementation**: Test prompt file created with full prompt template.

### Decision 6: Module-Level Function

**Decision**: Implement module-level `calculate_risk_impact()` function for backward compatibility.

**Rationale**:
- Maintains API consistency with other evaluator modules
- Supports existing code that may call the function directly
- Provides convenience wrapper
- Follows established pattern from other evaluators

**Implementation**: Module-level function that creates `RiskImpactEvaluator` instance and delegates.

## No Decisions Required

The following aspects followed established patterns from previous phases:
- Base class inheritance (`BaseEvaluatorNode`)
- LLM provider abstraction (`LLMProvider`)
- Error handling patterns (`AzureServiceError`, `ValueError`)
- Temperature setting (0.1 for reproducibility)
- Test structure and organization

## Future Considerations

1. **Prompt Versioning**: Consider adding support for prompt version selection in module-level function
2. **Cost Dictionary Schema**: May need to standardize cost dictionary schema if used across multiple components
3. **Impact Calibration**: May need to calibrate impact scale based on real-world evaluation results

