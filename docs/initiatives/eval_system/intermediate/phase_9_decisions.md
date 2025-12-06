# Phase 9 Decisions — BEIR Metrics Evaluator

## Overview

This document captures implementation decisions made during Phase 9: BEIR Metrics Evaluator that are not explicitly covered in PRD001.md or RFC001.md.

**Date**: 2024-12-19  
**Component**: `rag_eval/services/evaluator/beir_metrics.py`

## Decision 1: BEIRMetricsResult Dataclass Location

**Decision**: Added `BEIRMetricsResult` dataclass to `rag_eval/core/interfaces.py` alongside other evaluation result dataclasses.

**Rationale**:
- Consistent with existing pattern (JudgeEvaluationResult, MetaEvaluationResult in same file)
- Central location for all interface definitions
- Easy to import and use across the codebase
- Follows established codebase conventions

**Implementation**: Added to `backend/rag_eval/core/interfaces.py` with fields:
- `recall_at_k: float`
- `precision_at_k: float`
- `ndcg_at_k: float`

## Decision 2: Pure Python Implementation (No LLM Calls)

**Decision**: Implemented all BEIR metrics as pure Python functions with no LLM calls.

**Rationale**:
- BEIR metrics are standard information retrieval metrics with well-defined formulas
- No need for LLM reasoning - these are deterministic calculations
- Faster execution (no API calls)
- Lower cost (no LLM usage)
- Reproducible results (deterministic)
- Matches BEIR benchmark framework approach

**Implementation**: All three metrics (recall@k, precision@k, nDCG@k) implemented as pure Python functions using standard mathematical formulas.

## Decision 3: Helper Function Visibility

**Decision**: Implemented helper functions (`_compute_recall_at_k`, `_compute_precision_at_k`, `_compute_ndcg_at_k`) as private functions (prefixed with `_`).

**Rationale**:
- Helper functions are implementation details
- Main function `compute_beir_metrics()` is the public API
- Private functions allow for internal refactoring without breaking external code
- Follows Python naming conventions

**Implementation**: All helper functions are private (prefixed with `_`) but are tested directly for comprehensive coverage.

## Decision 4: Edge Case Handling for Empty Ground Truth

**Decision**: Return 0.0 for all metrics when ground truth is empty, rather than raising an error.

**Rationale**:
- Empty ground truth is a valid edge case (no relevant chunks exist)
- Returning 0.0 is mathematically reasonable (no relevant chunks = 0 recall, 0 precision, 0 nDCG)
- Allows evaluation pipeline to continue without errors
- Consistent with BEIR framework behavior

**Implementation**: All three helper functions check for empty ground truth and return 0.0.

## Decision 5: k Parameter Handling

**Decision**: When k is larger than the number of retrieved chunks, use the actual number of retrieved chunks for precision calculation.

**Rationale**:
- Precision@k = (relevant in top-k) / k
- If k > retrieved chunks, we should use actual number of retrieved chunks
- This prevents division by a larger k when fewer chunks are available
- Matches standard IR evaluation practice

**Implementation**: `_compute_precision_at_k` uses `min(k, len(retrieved_chunk_ids))` for the denominator.

## Decision 6: nDCG Relevance Scoring

**Decision**: Use binary relevance scores (1 for relevant, 0 for irrelevant) for nDCG calculation.

**Rationale**:
- Matches BEIR framework standard (binary relevance)
- Simpler implementation (no need for graded relevance)
- Consistent with recall@k and precision@k (binary classification)
- Standard practice in information retrieval evaluation

**Implementation**: Relevance is 1.0 if chunk_id is in ground_truth_set, 0.0 otherwise.

## Decision 7: Input Validation Strictness

**Decision**: Raise `ValueError` for invalid inputs (empty lists, k <= 0) rather than returning default values.

**Rationale**:
- Fail fast principle - catch errors early
- Clear error messages help debugging
- Prevents silent failures that could lead to incorrect metrics
- Matches Python best practices

**Implementation**: `compute_beir_metrics()` validates all inputs and raises `ValueError` with descriptive messages.

## Decision 8: Module Documentation

**Decision**: Include BEIR repository reference in module docstring.

**Rationale**:
- Provides context for BEIR framework
- Helps developers understand the standard being followed
- Links to authoritative source for metric definitions
- Matches requirement in Phase 9 prompt

**Implementation**: Module docstring includes reference to BEIR GitHub repository.

## Decision 9: Test Coverage Strategy

**Decision**: Test helper functions directly in addition to main function.

**Rationale**:
- Ensures comprehensive coverage of all code paths
- Makes it easier to test edge cases in isolation
- Helps identify bugs in specific metric calculations
- Achieves 96% coverage (exceeds 80% requirement)

**Implementation**: Separate test classes for each helper function with comprehensive test cases.

## Decision 10: Floating-Point Comparison Tolerance

**Decision**: Use tolerance-based comparisons (`abs(value - expected) < 1e-6`) for floating-point assertions in tests.

**Rationale**:
- Floating-point arithmetic can have small rounding errors
- Tolerance-based comparison is standard practice for float comparisons
- Prevents test failures due to minor numerical differences
- More robust than exact equality checks

**Implementation**: All floating-point metric assertions use tolerance-based comparisons.

## Summary

All decisions align with:
- BEIR benchmark framework standards
- Python best practices
- Codebase conventions
- Phase 9 requirements

No deviations from PRD001.md or RFC001.md were necessary. All decisions support the goal of implementing standard BEIR-style retrieval metrics as pure Python functions.

