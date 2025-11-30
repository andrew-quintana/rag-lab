# Phase 4 Implementation Decisions — Hallucination Cost LLM-Node

**Date**: 2024-12-19  
**Phase**: Phase 4 — Hallucination Cost LLM-Node  
**Component**: `rag_eval/services/evaluator/risk_direction.py`

## Overview

This document captures implementation decisions made during Phase 4 that are not explicitly specified in PRD001.md or RFC001.md.

## Decisions

### 1. Cost Classification Default for Ambiguous Cases

**Decision**: When cost direction is ambiguous or cannot be determined, default to Resource Cost (+1).

**Rationale**:
- Conservative approach: err on the side of caution for user protection
- Resource cost (+1) represents underestimation, which could lead to unexpected expenses
- Better to flag potential resource costs than miss them
- Documented in prompt template as guidance for LLM

**Location**: `risk_direction_prompt.md` - Example 5

### 2. Prompt Template Structure

**Decision**: Use markdown file format for prompt template, stored in `backend/rag_eval/prompts/evaluation/risk_direction_prompt.md`.

**Rationale**:
- Consistent with existing prompt templates (correctness_prompt.md, hallucination_prompt.md)
- Easy to edit and version control
- Can be loaded at runtime without database dependency
- Follows established pattern from Phase 2 and Phase 3

**Location**: `rag_eval/prompts/evaluation/risk_direction_prompt.md`

### 3. Error Handling for Invalid Cost Values

**Decision**: Validate that `risk_direction` is exactly -1 or +1, raising ValueError for any other integer value.

**Rationale**:
- Strict validation ensures only valid cost classifications are returned
- Prevents invalid values from propagating through the system
- Clear error messages help with debugging
- Consistent with type safety principles

**Location**: `risk_direction.py` lines 220-225

### 4. Module-Level Function for Backward Compatibility

**Decision**: Implement module-level `classify_risk_direction()` function that creates an evaluator instance internally.

**Rationale**:
- Maintains backward compatibility with RFC001 interface contract
- Allows direct function calls without instantiating evaluator class
- Consistent with Phase 2 (correctness) and Phase 3 (hallucination) patterns
- Simplifies usage for simple cases

**Location**: `risk_direction.py` lines 237-316

### 5. Temperature Setting

**Decision**: Use temperature=0.1 for all LLM calls, as specified in requirements.

**Rationale**:
- Reproducibility: lower temperature reduces variance in LLM responses
- Consistency: matches Phase 2 and Phase 3 implementations
- Explicitly documented in requirements
- Trade-off: slightly less creative but more deterministic

**Location**: `risk_direction.py` line 200

### 6. Reference Answer Exclusion

**Decision**: Explicitly exclude reference answer from cost classification prompt and logic.

**Rationale**:
- **CRITICAL REQUIREMENT**: Cost classification must be based on retrieved context only
- Cost classification is about direction of error relative to ground truth evidence
- Reference answer is not ground truth for cost information (retrieved chunks are)
- Explicitly tested and verified in test suite

**Location**: 
- Prompt template: `risk_direction_prompt.md` - emphasizes reference answer NOT used
- Implementation: `risk_direction.py` - no reference answer parameter
- Tests: `test_evaluator_risk_direction.py` - `test_reference_answer_not_used`

### 7. Cost Classification Applies to All Hallucinations

**Decision**: Cost classification applies to both quantitative and non-quantitative hallucinations.

**Rationale**:
- Quantitative hallucinations: direct cost comparisons (e.g., "$500" vs "$300")
- Non-quantitative hallucinations: indirect cost impacts (e.g., "covered" vs "not covered", "required" vs "optional")
- Both types can have cost direction implications
- Prompt template includes examples for both types

**Location**: `risk_direction_prompt.md` - Instructions section, Examples 1-4

### 8. Logging and Observability

**Decision**: Include comprehensive logging at info and debug levels for cost classification operations.

**Rationale**:
- Debugging: helps trace cost classification decisions
- Observability: enables monitoring of cost classification performance
- Consistency: matches logging patterns from Phase 2 and Phase 3
- Logs include model name, answer length, chunk count, and classification result

**Location**: `risk_direction.py` lines 186-189, 225-228

### 9. Formatting Retrieved Context

**Decision**: Format retrieved context with chunk IDs in format: `[Chunk ID: {chunk_id}] {chunk_text}`.

**Rationale**:
- Traceability: chunk IDs enable tracking which chunks contributed to classification
- Consistency: matches formatting from Phase 3 (hallucination evaluator)
- Readability: clear separation between chunk ID and text
- Empty context handling: returns "[No retrieved context available]" for empty lists

**Location**: `risk_direction.py` lines 38-59

### 10. Package Exports

**Decision**: Export `classify_risk_direction` from `rag_eval.services.evaluator` package.

**Rationale**:
- Public API: makes function available for use in orchestrator (Phase 7)
- Consistency: matches exports from Phase 2 and Phase 3
- Discoverability: users can import from package level
- Maintains backward compatibility

**Location**: `rag_eval/services/evaluator/__init__.py`

## No Decisions Required

The following aspects were already clearly specified in PRD001.md or RFC001.md:
- Interface contract: `classify_risk_direction(model_answer, retrieved_context, config) -> int`
- Cost classification values: -1 (opportunity cost) or +1 (resource cost)
- Base class: `BaseEvaluatorNode`
- LLM provider: `LLMProvider` abstraction
- Test location: `backend/tests/components/evaluator/test_evaluator_risk_direction.py`
- Test coverage requirement: minimum 80%

## Future Considerations

1. **Prompt Template Storage**: Consider database storage for prompt templates if dynamic updates are needed
2. **Cost Classification Refinement**: May need to refine ambiguous case handling based on real-world usage
3. **Performance**: Monitor LLM call latency and consider caching if needed
4. **Error Recovery**: Consider retry logic for transient LLM failures (currently handled by base class)

## Conclusion

All implementation decisions align with PRD001.md and RFC001.md requirements. The decisions prioritize correctness, consistency with existing patterns, and maintainability.

