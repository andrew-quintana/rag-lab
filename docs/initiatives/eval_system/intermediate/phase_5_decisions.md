# Phase 5 Decisions — Cost Extraction LLM-Node

## Overview

This document captures implementation decisions made during Phase 5 that are not explicitly covered in PRD001.md or RFC001.md.

## Decision 1: Optional Field Handling

**Decision**: All cost fields (time, money, steps) are optional and are only included in the result dictionary if they are present in the LLM response and not null.

**Rationale**:
- Matches the RFC001 interface contract which specifies optional fields
- Provides clean output without null/None values cluttering the result
- Makes it easy to check for presence of cost types using `in` operator

**Implementation**: The `extract_costs()` method only adds fields to the result dictionary if they exist in the parsed JSON response and are not None.

```python
# Only add fields that exist and are not None
if "time" in parsed and parsed["time"] is not None:
    result["time"] = parsed["time"]
```

## Decision 2: Flexible Cost Value Types

**Decision**: Cost values can be strings or numbers (float/int) as returned by the LLM, without strict type coercion.

**Rationale**:
- LLM may return "2 hours" (string) or 2.5 (float) depending on context
- Flexibility allows the LLM to choose the most appropriate representation
- Type coercion can be handled downstream if needed

**Implementation**: The result dictionary preserves the type returned by the LLM (string, float, or int).

## Decision 3: Reasoning Field is Required

**Decision**: The `reasoning` field is always required in the output, even when no costs are found.

**Rationale**:
- Provides transparency into why costs were or weren't extracted
- Helps with debugging and understanding LLM decisions
- Consistent with other evaluator nodes (correctness, hallucination, risk_direction)

**Implementation**: The method validates that `reasoning` is present and raises `ValueError` if missing.

## Decision 4: Single Text Input

**Decision**: The cost extraction node takes a single `text` parameter, not separate fields for different contexts.

**Rationale**:
- Simpler interface matches the use case (extracting from model answers or chunk text)
- Can be called separately for model answer and chunks text
- Matches the RFC001 interface contract

**Implementation**: The `extract_costs(text: str)` function signature matches RFC001 exactly.

## Decision 5: Temperature Setting

**Decision**: Use temperature=0.1 for all LLM calls, consistent with other evaluator nodes.

**Rationale**:
- Matches the requirement in PRD001 and RFC001
- Ensures reproducibility across evaluation runs
- Consistent with other evaluator nodes (correctness, hallucination, risk_direction)

**Implementation**: All `_call_llm()` calls use `temperature=0.1`.

## Decision 6: Prompt Template Location

**Decision**: Prompt template stored in `backend/rag_eval/prompts/evaluation/cost_extraction_prompt.md` with database loading support.

**Rationale**:
- Consistent with other evaluator prompts (correctness_prompt.md, hallucination_prompt.md, risk_direction_prompt.md)
- Supports both file-based (testing) and database-based (production) loading
- Follows the BaseEvaluatorNode pattern

**Implementation**: Uses `_DEFAULT_PROMPT_PATH` for backward compatibility, supports `query_executor` for database loading.

## Decision 7: Error Handling

**Decision**: Re-raise `AzureServiceError` and `ValueError` as-is, wrap other exceptions in `AzureServiceError`.

**Rationale**:
- Consistent with other evaluator nodes
- Preserves error type information for callers
- Follows the BaseEvaluatorNode error handling pattern

**Implementation**: Exception handling in `extract_costs()` method matches the pattern used in `correctness.py` and `risk_direction.py`.

## Decision 8: Module-Level Function

**Decision**: Provide module-level `extract_costs()` function for backward compatibility, following the pattern from other evaluators.

**Rationale**:
- Maintains consistency with `classify_correctness()`, `classify_hallucination()`, `classify_risk_direction()`
- Provides simple interface for callers who don't need custom configuration
- Supports both class-based and function-based usage

**Implementation**: Module-level function creates `CostExtractionEvaluator` instance and delegates to `extract_costs()` method.

---

**Document Status**: Complete  
**Last Updated**: 2024-12-19  
**Author**: Implementation Agent

