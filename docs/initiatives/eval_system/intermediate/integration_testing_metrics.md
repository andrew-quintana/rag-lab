# Integration Testing Plan — Judge Performance Metrics

## Overview

This document outlines the integration testing plan for the judge performance metrics functionality added to Phase 8. The metrics calculation (`calculate_judge_metrics()`) will be tested as part of Phase 10 (Orchestrator) and Phase 11 (Logging) integration tests.

## Background

Phase 8 was enhanced with data science components to calculate recall, precision, and F1 scores for the LLM-as-Judge across all output metrics:
- Correctness (binary classification)
- Hallucination (binary classification)
- Risk direction (multiclass: -1, 0, 1)
- Risk impact (multiclass: 0, 1, 2, 3)

See `phase_8_metrics_enhancement.md` for implementation details.

## Integration Points

### Phase 10: Evaluation Pipeline Orchestration

**Integration Point**: After pipeline execution, calculate judge performance metrics from batch results.

**Test Scenarios**:

1. **Basic Metrics Calculation**
   - Run full evaluation pipeline on test dataset
   - Collect all (judge_output, meta_eval_output) pairs
   - Call `calculate_judge_metrics()` on collected results
   - Verify metrics are calculated correctly:
     - Correctness metrics (precision, recall, F1)
     - Hallucination metrics (precision, recall, F1)
     - Risk direction metrics (if cost data available)
     - Risk impact metrics (if cost data available)

2. **Mixed Scenarios**
   - Test with evaluation dataset containing:
     - Examples with cost data (for risk metrics)
     - Examples without cost data (risk metrics should be None)
   - Verify metrics calculation handles missing optional metrics gracefully

3. **Metrics Accuracy**
   - Create test dataset with known judge performance
   - Run pipeline and calculate metrics
   - Verify calculated metrics match expected values:
     - Perfect judge performance → precision=1.0, recall=1.0, F1=1.0
     - Known error rates → verify metrics reflect actual performance

4. **Edge Cases**
   - Empty evaluation results (should raise ValueError)
   - Single evaluation result
   - All results missing cost data (risk metrics should be None)
   - All results with cost data (all metrics should be present)

### Phase 11: Logging and Persistence

**Integration Point**: Log evaluation results with ground truth data, enable metrics recalculation from logged data.

**Test Scenarios**:

1. **Ground Truth Logging**
   - Log evaluation results with `MetaEvaluationResult` containing ground truth fields
   - Verify ground truth data is properly serialized and stored
   - Verify ground truth fields can be retrieved from database

2. **Metrics Recalculation**
   - Log batch of evaluation results
   - Retrieve logged results from database
   - Reconstruct (judge_output, meta_eval_output) pairs
   - Call `calculate_judge_metrics()` on retrieved results
   - Verify recalculated metrics match original metrics

3. **Metrics Logging** (Optional)
   - If `JudgePerformanceMetrics` are calculated, test logging them separately
   - Verify metrics can be retrieved and used for reporting

## Test Implementation

### Phase 10 Integration Tests

**File**: `backend/tests/components/evaluator/test_evaluator_orchestrator.py`

**Test Class**: `TestJudgeMetricsIntegration`

```python
class TestJudgeMetricsIntegration:
    """Integration tests for judge performance metrics with orchestrator"""
    
    def test_metrics_calculation_from_pipeline_results(self):
        """Test calculate_judge_metrics() with pipeline results"""
        # Run pipeline on test dataset
        # Collect results
        # Calculate metrics
        # Verify metrics structure and values
    
    def test_metrics_with_mixed_cost_data(self):
        """Test metrics calculation with mixed scenarios"""
        # Test with some examples having cost data, some not
        # Verify optional metrics handled correctly
    
    def test_metrics_accuracy(self):
        """Test metrics calculation accuracy"""
        # Create known performance scenarios
        # Verify metrics match expected values
```

### Phase 11 Integration Tests

**File**: `backend/tests/components/evaluator/test_evaluator_logging.py`

**Test Class**: `TestMetricsLoggingIntegration`

```python
class TestMetricsLoggingIntegration:
    """Integration tests for metrics logging and recalculation"""
    
    def test_ground_truth_logging(self):
        """Test logging of ground truth data"""
        # Log results with ground truth
        # Verify ground truth is stored correctly
    
    def test_metrics_recalculation_from_logged_data(self):
        """Test metrics recalculation from logged results"""
        # Log batch results
        # Retrieve and recalculate metrics
        # Verify metrics match original
```

## Success Criteria

### Phase 10 Integration
- [ ] Metrics calculation works correctly with pipeline results
- [ ] All metric types (correctness, hallucination, risk_direction, risk_impact) calculated when data available
- [ ] Optional metrics (risk_direction, risk_impact) handled gracefully when data missing
- [ ] Metrics values are accurate (match expected performance)

### Phase 11 Integration
- [ ] Ground truth data properly logged and retrievable
- [ ] Metrics can be recalculated from logged results
- [ ] Recalculated metrics match original metrics

## Dependencies

- Phase 8: Meta-Evaluator with metrics calculation (✅ Complete)
- Phase 10: Orchestrator implementation (⏳ Pending)
- Phase 11: Logging implementation (⏳ Pending)

## Documentation Updates

- Phase 10 prompt updated with metrics integration testing requirements
- Phase 11 prompt updated with metrics logging integration testing requirements
- TODO001.md updated with metrics integration testing tasks

## Notes

- Metrics calculation is deterministic and should produce identical results for same inputs
- Ground truth computation happens during meta-evaluation, so no additional computation needed
- Metrics can be calculated at any time from collected evaluation results
- Metrics are useful for judge reliability assessment and performance monitoring

