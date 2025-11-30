# Phase 6 Handoff — Risk Impact LLM-Node

**Date**: 2024-12-19  
**Status**: ✅ Complete  
**Next Phase**: Phase 7 — LLM-as-Judge Orchestrator

## Summary

Phase 6 successfully implements the Risk Impact LLM-Node for calculating system-level risk impact magnitude as discrete values (0, 1, 2, or 3). The implementation handles mixed resource types (time, money, steps) and evaluates impact regardless of deviation origin.

## Deliverables

### 1. Implementation
- ✅ `rag_eval/services/evaluator/risk_impact.py` - Main implementation module
- ✅ `RiskImpactEvaluator` class inheriting from `BaseEvaluatorNode`
- ✅ Module-level `calculate_risk_impact()` function for backward compatibility
- ✅ Updated `rag_eval/services/evaluator/__init__.py` with exports

### 2. Prompt Template
- ✅ Database migration: `infra/supabase/migrations/0010_insert_risk_impact_prompt_v0_1.sql`
- ✅ Test prompt file: `backend/tests/fixtures/prompts/risk_impact_prompt.md`
- ✅ Prompt stored in database with:
  - `prompt_type='evaluation'`
  - `name='risk_impact_evaluator'`
  - `version='0.1'`
  - `live=true`

### 3. Tests
- ✅ Test file: `backend/tests/components/evaluator/test_evaluator_risk_impact.py`
- ✅ 21 tests, all passing
- ✅ 86% test coverage (exceeds 80% requirement)
- ✅ Comprehensive test coverage including edge cases

### 4. Documentation
- ✅ Complete docstrings for all functions
- ✅ `phase_6_testing.md` - Testing summary
- ✅ `phase_6_decisions.md` - Implementation decisions
- ✅ `phase_6_handoff.md` - This document
- ✅ Updated `TODO001.md` with Phase 6 completion status

## Interface Contract

### Function Signature
```python
def calculate_risk_impact(
    model_answer_cost: Dict[str, Any],  # {"time": ..., "money": ..., "steps": ...}
    actual_cost: Dict[str, Any],  # From retrieved chunks (ground truth)
    config: Optional[Config] = None
) -> float
```

### Return Value
- **Type**: `float`
- **Discrete Values**: {0, 1, 2, 3}
- **Scale**:
  - 0: Minimal/no impact
  - 1: Low impact
  - 2: Moderate impact
  - 3: High/severe impact

### Usage Example
```python
from rag_eval.services.evaluator.risk_impact import calculate_risk_impact
from rag_eval.core.config import Config

config = Config.from_env()
model_cost = {"money": 500.0, "time": "2 hours"}
actual_cost = {"money": 50.0, "time": "30 minutes"}

impact = calculate_risk_impact(model_cost, actual_cost, config)
# Returns: int in {0, 1, 2, 3}
```

## Dependencies

### Required
- Phase 5: Cost Extraction LLM-Node (for extracting costs from text)
- `BaseEvaluatorNode` base class
- `LLMProvider` abstraction
- Azure Foundry API access (for LLM calls)

### Database
- Prompt template stored in `public.prompts` table
- Migration `0010_insert_risk_impact_prompt_v0_1.sql` must be applied

## Integration Points

### For Phase 7 (LLM-as-Judge Orchestrator)
The `calculate_risk_impact()` function will be called by the judge orchestrator when:
1. `correctness_binary` is `True`
2. Costs have been extracted from model answer and retrieved chunks
3. Impact magnitude calculation is needed

### Integration Pattern
```python
# In judge orchestrator (Phase 7)
if correctness_binary:
    # Extract costs
    model_answer_cost = extract_costs(model_answer, config)
    actual_cost = extract_costs(chunks_text, config)
    
    # Calculate impact
    risk_impact = calculate_risk_impact(model_answer_cost, actual_cost, config)
```

## Testing

### Test Execution
```bash
cd backend && source venv/bin/activate && pytest tests/components/evaluator/test_evaluator_risk_impact.py -v
```

### Coverage Verification
```bash
cd backend && source venv/bin/activate && pytest tests/components/evaluator/test_evaluator_risk_impact.py --cov=rag_eval.services.evaluator.risk_impact --cov-report=term-missing
```

### Test Results
- ✅ 21/21 tests passing
- ✅ 86% coverage (exceeds 80% requirement)
- ✅ All edge cases covered
- ✅ Error handling validated

## Known Limitations

1. **LLM Non-Determinism**: Impact calculations may vary slightly across calls due to LLM non-determinism (mitigated by temperature=0.1)
2. **Cost Dictionary Schema**: Cost dictionaries are flexible (Dict[str, Any]) - no strict schema validation
3. **Missing Coverage**: 10 lines not covered (edge case error handling) - acceptable given 86% overall coverage

## Next Steps for Phase 7

1. **Review Phase 7 Prompt**: `@docs/initiatives/eval_system/prompt_phase_7_001.md`
2. **Integrate Risk Impact**: Use `calculate_risk_impact()` in judge orchestrator
3. **Test Integration**: Ensure risk impact calculation works in full judge pipeline
4. **Update Judge Output**: Include `risk_impact` in `JudgeEvaluationResult`

## Validation Checklist

- [x] All tests pass (21/21)
- [x] Test coverage exceeds 80% (86%)
- [x] Prompt template stored in database
- [x] Module exports configured
- [x] Documentation complete
- [x] TODO001.md updated
- [x] No linting errors
- [x] Interface contract matches RFC001.md

## Status

✅ **Phase 6 Complete** - Ready for Phase 7

All validation requirements met. The Risk Impact LLM-Node is fully implemented, tested, and documented. The implementation follows established patterns from previous phases and integrates seamlessly with the evaluation system architecture.

