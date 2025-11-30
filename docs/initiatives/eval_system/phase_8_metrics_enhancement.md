# Phase 8 Enhancement — Data Science Metrics for Judge Performance

## Overview

Added data science components to the meta-evaluator to calculate recall, precision, and F1 scores for the LLM-as-Judge across all output metrics. This enables quantitative evaluation of judge reliability and performance.

## New Components

### Data Structures

#### `JudgeMetricScores` (in `rag_eval/core/interfaces.py`)
```python
@dataclass
class JudgeMetricScores:
    """Performance metrics for a single judge output metric."""
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    total_samples: int
```

#### `JudgePerformanceMetrics` (in `rag_eval/core/interfaces.py`)
```python
@dataclass
class JudgePerformanceMetrics:
    """Comprehensive performance metrics for LLM-as-Judge across all output metrics."""
    correctness: JudgeMetricScores
    hallucination: JudgeMetricScores
    risk_direction: Optional[JudgeMetricScores] = None
    risk_impact: Optional[JudgeMetricScores] = None
```

### Enhanced `MetaEvaluationResult`

Added ground truth fields to enable metrics calculation:
- `ground_truth_correctness: Optional[bool]`
- `ground_truth_hallucination: Optional[bool]`
- `ground_truth_risk_direction: Optional[int]`
- `ground_truth_risk_impact: Optional[int]`

### New Functions

#### `calculate_judge_metrics()`
```python
def calculate_judge_metrics(
    evaluation_results: List[Tuple[JudgeEvaluationResult, MetaEvaluationResult]]
) -> JudgePerformanceMetrics
```

**Purpose**: Calculate recall, precision, and F1 scores for all judge output metrics by comparing judge verdicts against ground truth values.

**Input**: List of (judge_output, meta_eval_result) pairs from batch evaluation.

**Output**: `JudgePerformanceMetrics` object containing:
- Correctness metrics (always present)
- Hallucination metrics (always present)
- Risk direction metrics (if cost data available)
- Risk impact metrics (if cost data available)

### Ground Truth Computation Functions

The meta-evaluator now computes and stores ground truth values:

1. **`_compute_ground_truth_correctness()`**: Determines if model answer matches reference
2. **`_compute_ground_truth_hallucination()`**: Determines if model answer contains hallucinations
3. **`_compute_ground_truth_risk_direction()`**: Determines expected risk direction from costs
4. **`_compute_ground_truth_risk_impact()`**: Determines expected risk impact from costs

### Metrics Calculation

#### Binary Classification Metrics
For `correctness` and `hallucination` (binary: True/False):
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)

#### Multiclass Classification Metrics
For `risk_direction` (-1, 0, 1) and `risk_impact` (0, 1, 2, 3):
- Uses **macro-averaging** across all classes
- Calculates metrics for each class (treating it as positive)
- Averages precision, recall, and F1 across classes

## Usage Example

```python
from rag_eval.services.evaluator.meta_eval import (
    meta_evaluate_judge,
    calculate_judge_metrics
)
from rag_eval.core.interfaces import JudgeEvaluationResult

# Collect evaluation results
results = []
for example in evaluation_dataset:
    # ... run judge evaluation ...
    judge_output = evaluate_answer_with_judge(...)
    
    # Meta-evaluate (now includes ground truth)
    meta_eval = meta_evaluate_judge(
        judge_output=judge_output,
        retrieved_context=retrieved_context,
        model_answer=model_answer,
        reference_answer=reference_answer,
        extracted_costs=extracted_costs,
        actual_costs=actual_costs
    )
    
    results.append((judge_output, meta_eval))

# Calculate performance metrics
metrics = calculate_judge_metrics(results)

# Access metrics
print(f"Correctness Precision: {metrics.correctness.precision:.3f}")
print(f"Correctness Recall: {metrics.correctness.recall:.3f}")
print(f"Correctness F1: {metrics.correctness.f1_score:.3f}")
print(f"Hallucination Precision: {metrics.hallucination.precision:.3f}")
print(f"Hallucination Recall: {metrics.hallucination.recall:.3f}")
print(f"Hallucination F1: {metrics.hallucination.f1_score:.3f}")

if metrics.risk_direction:
    print(f"Risk Direction F1: {metrics.risk_direction.f1_score:.3f}")

if metrics.risk_impact:
    print(f"Risk Impact F1: {metrics.risk_impact.f1_score:.3f}")
```

## Metrics Interpretation

### Correctness Metrics
- **High Precision**: Judge rarely incorrectly marks correct answers as incorrect
- **High Recall**: Judge rarely misses incorrect answers
- **High F1**: Balanced precision and recall for correctness detection

### Hallucination Metrics
- **High Precision**: Judge rarely incorrectly flags grounded content as hallucination
- **High Recall**: Judge rarely misses actual hallucinations
- **High F1**: Balanced precision and recall for hallucination detection

### Risk Direction Metrics
- **Macro-averaged F1**: Performance across all three classes (-1, 0, 1)
- Useful for understanding judge's ability to classify cost direction

### Risk Impact Metrics
- **Macro-averaged F1**: Performance across all four impact levels (0, 1, 2, 3)
- Useful for understanding judge's ability to assess impact magnitude

## Test Coverage

**New Tests**: 8 comprehensive tests for metrics calculation
- Perfect score scenarios
- Error scenarios
- Missing optional metrics
- Mixed scenarios
- All metric types (correctness, hallucination, risk_direction, risk_impact)

**Total Tests**: 44 tests (36 original + 8 new)
**Coverage**: 86% (maintained from original implementation)

## Integration Points

### Phase 10 (Orchestrator)
The orchestrator can now collect evaluation results and calculate judge performance metrics:

```python
# In orchestrator.py
evaluation_results = []
for example in evaluation_dataset:
    # ... run evaluation pipeline ...
    result = EvaluationResult(...)
    evaluation_results.append((result.judge_output, result.meta_eval_output))

# Calculate judge performance
judge_metrics = calculate_judge_metrics(evaluation_results)
```

### Reporting
Metrics can be used for:
- Judge reliability assessment
- Performance monitoring over time
- Comparison across different judge configurations
- Identifying areas where judge needs improvement

## Benefits

1. **Quantitative Evaluation**: Provides numerical metrics for judge performance
2. **Comprehensive Coverage**: Metrics for all judge output types
3. **Batch Analysis**: Efficient calculation from multiple evaluation results
4. **Ground Truth Integration**: Automatically computes ground truth during meta-evaluation
5. **Flexible**: Handles missing optional metrics gracefully

## Files Modified

1. `backend/rag_eval/core/interfaces.py` - Added data structures
2. `backend/rag_eval/services/evaluator/meta_eval.py` - Added metrics calculation
3. `backend/rag_eval/services/evaluator/__init__.py` - Exported new function
4. `backend/tests/components/meta_eval/test_evaluator_meta_eval.py` - Added tests

## Next Steps

- Integrate metrics calculation into Phase 10 (Orchestrator)
- Add metrics reporting/logging capabilities
- Consider adding additional metrics (accuracy, confusion matrices, etc.)

