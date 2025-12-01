"""Unit tests for Evaluation Result Logging"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import List

from rag_eval.core.interfaces import (
    EvaluationResult,
    JudgeEvaluationResult,
    MetaEvaluationResult,
    BEIRMetricsResult,
    JudgePerformanceMetrics,
    JudgeMetricScores
)
from rag_eval.db.queries import QueryExecutor
from rag_eval.db.connection import DatabaseConnection
from rag_eval.services.evaluator.logging import (
    log_evaluation_result,
    log_evaluation_batch,
    _serialize_judge_output,
    _serialize_meta_eval_output,
    _serialize_beir_metrics,
    _serialize_judge_performance_metrics,
    _serialize_to_json,
    JSONEncoder
)


@pytest.fixture
def sample_judge_output():
    """Create a sample judge output"""
    return JudgeEvaluationResult(
        correctness_binary=True,
        hallucination_binary=False,
        risk_direction=0,
        risk_impact=1,
        reasoning="The answer is correct and grounded in the retrieved context.",
        failure_mode=None
    )


@pytest.fixture
def sample_meta_eval_output():
    """Create a sample meta-evaluation output with ground truth"""
    return MetaEvaluationResult(
        judge_correct=True,
        explanation="Judge verdicts are correct.",
        ground_truth_correctness=True,
        ground_truth_hallucination=False,
        ground_truth_risk_direction=0,
        ground_truth_risk_impact=1
    )


@pytest.fixture
def sample_beir_metrics():
    """Create sample BEIR metrics"""
    return BEIRMetricsResult(
        recall_at_k=0.5,
        precision_at_k=0.5,
        ndcg_at_k=0.6
    )


@pytest.fixture
def sample_evaluation_result(sample_judge_output, sample_meta_eval_output, sample_beir_metrics):
    """Create a sample evaluation result"""
    return EvaluationResult(
        example_id="val_001",
        judge_output=sample_judge_output,
        meta_eval_output=sample_meta_eval_output,
        beir_metrics=sample_beir_metrics,
        timestamp=datetime(2024, 1, 1, 12, 0, 0)
    )


@pytest.fixture
def sample_judge_performance_metrics():
    """Create sample judge performance metrics"""
    return JudgePerformanceMetrics(
        correctness=JudgeMetricScores(
            precision=0.9,
            recall=0.85,
            f1_score=0.875,
            true_positives=17,
            true_negatives=3,
            false_positives=2,
            false_negatives=3,
            total_samples=25
        ),
        hallucination=JudgeMetricScores(
            precision=0.88,
            recall=0.90,
            f1_score=0.89,
            true_positives=18,
            true_negatives=4,
            false_positives=2,
            false_negatives=1,
            total_samples=25
        ),
        risk_direction=JudgeMetricScores(
            precision=0.75,
            recall=0.80,
            f1_score=0.774,
            true_positives=12,
            true_negatives=5,
            false_positives=4,
            false_negatives=3,
            total_samples=24
        ),
        risk_impact=JudgeMetricScores(
            precision=0.70,
            recall=0.75,
            f1_score=0.724,
            true_positives=9,
            true_negatives=8,
            false_positives=4,
            false_negatives=3,
            total_samples=24
        )
    )


@pytest.fixture
def mock_query_executor():
    """Create a mock QueryExecutor"""
    executor = Mock(spec=QueryExecutor)
    executor.execute_insert = Mock(return_value="test_result_id_123")
    executor.execute_query = Mock(return_value=[])
    return executor


class TestJSONSerialization:
    """Test JSON serialization helpers"""
    
    def test_serialize_judge_output(self, sample_judge_output):
        """Test serialization of JudgeEvaluationResult"""
        result = _serialize_judge_output(sample_judge_output)
        
        assert isinstance(result, dict)
        assert result["correctness_binary"] == True
        assert result["hallucination_binary"] == False
        assert result["risk_direction"] == 0
        assert result["risk_impact"] == 1
        assert result["reasoning"] == "The answer is correct and grounded in the retrieved context."
        assert result["failure_mode"] is None
    
    def test_serialize_meta_eval_output(self, sample_meta_eval_output):
        """Test serialization of MetaEvaluationResult with ground truth"""
        result = _serialize_meta_eval_output(sample_meta_eval_output)
        
        assert isinstance(result, dict)
        assert result["judge_correct"] == True
        assert result["explanation"] == "Judge verdicts are correct."
        assert result["ground_truth_correctness"] == True
        assert result["ground_truth_hallucination"] == False
        assert result["ground_truth_risk_direction"] == 0
        assert result["ground_truth_risk_impact"] == 1
    
    def test_serialize_beir_metrics(self, sample_beir_metrics):
        """Test serialization of BEIRMetricsResult"""
        result = _serialize_beir_metrics(sample_beir_metrics)
        
        assert isinstance(result, dict)
        assert result["recall_at_k"] == 0.5
        assert result["precision_at_k"] == 0.5
        assert result["ndcg_at_k"] == 0.6
    
    def test_serialize_judge_performance_metrics(self, sample_judge_performance_metrics):
        """Test serialization of JudgePerformanceMetrics"""
        result = _serialize_judge_performance_metrics(sample_judge_performance_metrics)
        
        assert isinstance(result, dict)
        assert "correctness" in result
        assert "hallucination" in result
        assert "risk_direction" in result
        assert "risk_impact" in result
        
        # Check correctness metrics
        assert result["correctness"]["precision"] == 0.9
        assert result["correctness"]["recall"] == 0.85
        assert result["correctness"]["f1_score"] == 0.875
    
    def test_serialize_judge_performance_metrics_partial(self):
        """Test serialization of JudgePerformanceMetrics with optional fields None"""
        metrics = JudgePerformanceMetrics(
            correctness=JudgeMetricScores(
                precision=0.9,
                recall=0.85,
                f1_score=0.875,
                true_positives=17,
                true_negatives=3,
                false_positives=2,
                false_negatives=3,
                total_samples=25
            ),
            hallucination=JudgeMetricScores(
                precision=0.88,
                recall=0.90,
                f1_score=0.89,
                true_positives=18,
                true_negatives=4,
                false_positives=2,
                false_negatives=1,
                total_samples=25
            ),
            risk_direction=None,
            risk_impact=None
        )
        
        result = _serialize_judge_performance_metrics(metrics)
        
        assert isinstance(result, dict)
        assert "correctness" in result
        assert "hallucination" in result
        assert "risk_direction" not in result
        assert "risk_impact" not in result
    
    def test_serialize_to_json_datetime(self):
        """Test serialization of datetime objects"""
        dt = datetime(2024, 1, 1, 12, 0, 0)
        result = _serialize_to_json(dt)
        
        assert isinstance(result, str)
        assert result == "2024-01-01T12:00:00"
    
    def test_serialize_to_json_none(self):
        """Test serialization of None"""
        result = _serialize_to_json(None)
        assert result is None
    
    def test_serialize_to_json_dict(self):
        """Test serialization of dict"""
        data = {"key1": "value1", "key2": 42, "nested": {"inner": True}}
        result = _serialize_to_json(data)
        assert isinstance(result, dict)
        assert result["key1"] == "value1"
        assert result["key2"] == 42
        assert result["nested"]["inner"] == True
    
    def test_serialize_to_json_list(self):
        """Test serialization of list"""
        data = [1, 2, 3, {"nested": "value"}]
        result = _serialize_to_json(data)
        assert isinstance(result, list)
        assert result[0] == 1
        assert result[3]["nested"] == "value"
    
    def test_serialize_to_json_fallback(self):
        """Test serialization fallback for unsupported types"""
        class UnsupportedType:
            def __init__(self):
                self.value = "test"
        
        obj = UnsupportedType()
        result = _serialize_to_json(obj)
        # Should fall back to string representation
        assert isinstance(result, str)
    
    def test_json_encoder(self):
        """Test JSONEncoder class"""
        encoder = JSONEncoder()
        
        # Test with dataclass
        from dataclasses import dataclass
        @dataclass
        class TestClass:
            value: str
        
        obj = TestClass(value="test")
        result = encoder.default(obj)
        assert isinstance(result, dict)
        assert result["value"] == "test"
        
        # Test with datetime
        dt = datetime(2024, 1, 1, 12, 0, 0)
        result = encoder.default(dt)
        assert result == "2024-01-01T12:00:00"
        
        # Test with primitive
        assert encoder.default(42) == 42
        assert encoder.default("string") == "string"
        assert encoder.default(True) == True


class TestLogEvaluationResult:
    """Test log_evaluation_result function"""
    
    def test_log_evaluation_result_success(self, sample_evaluation_result, mock_query_executor):
        """Test successful logging of evaluation result"""
        result_id = log_evaluation_result(sample_evaluation_result, mock_query_executor)
        
        assert result_id is not None
        assert result_id == "test_result_id_123"
        mock_query_executor.execute_insert.assert_called_once()
        
        # Verify the insert query was called with correct parameters
        call_args = mock_query_executor.execute_insert.call_args
        assert call_args is not None
        query = call_args[0][0]
        params = call_args[0][1]
        
        assert "INSERT INTO evaluation_results" in query
        assert params[1] == "val_001"  # example_id
        assert isinstance(params[3], str)  # judge_output JSON
        assert isinstance(params[4], str)  # meta_eval_output JSON
        assert isinstance(params[5], str)  # beir_metrics JSON
    
    def test_log_evaluation_result_local_only_mode(self, sample_evaluation_result):
        """Test local-only mode when query_executor is None"""
        result_id = log_evaluation_result(sample_evaluation_result, None)
        
        assert result_id is None
    
    def test_log_evaluation_result_database_failure(self, sample_evaluation_result, mock_query_executor):
        """Test graceful handling of database failures"""
        mock_query_executor.execute_insert.side_effect = Exception("Database connection failed")
        
        # Should not raise exception, should return None
        result_id = log_evaluation_result(sample_evaluation_result, mock_query_executor)
        
        assert result_id is None
    
    def test_log_evaluation_result_with_all_fields(self, mock_query_executor):
        """Test logging with all optional fields populated"""
        judge_output = JudgeEvaluationResult(
            correctness_binary=True,
            hallucination_binary=False,
            risk_direction=-1,
            risk_impact=3,
            reasoning="Test reasoning",
            failure_mode="retrieval_miss"
        )
        
        meta_eval = MetaEvaluationResult(
            judge_correct=True,
            explanation="All validations passed",
            ground_truth_correctness=True,
            ground_truth_hallucination=False,
            ground_truth_risk_direction=-1,
            ground_truth_risk_impact=3
        )
        
        beir_metrics = BEIRMetricsResult(
            recall_at_k=0.8,
            precision_at_k=0.6,
            ndcg_at_k=0.75
        )
        
        result = EvaluationResult(
            example_id="val_002",
            judge_output=judge_output,
            meta_eval_output=meta_eval,
            beir_metrics=beir_metrics,
            timestamp=datetime.now()
        )
        
        result_id = log_evaluation_result(result, mock_query_executor)
        
        assert result_id is not None
        mock_query_executor.execute_insert.assert_called_once()
    
    def test_log_evaluation_result_no_returning_id(self, sample_evaluation_result, mock_query_executor):
        """Test logging when execute_insert returns None (no RETURNING clause result)"""
        mock_query_executor.execute_insert.return_value = None
        
        result_id = log_evaluation_result(sample_evaluation_result, mock_query_executor)
        
        # Should return generated UUID even if RETURNING didn't work
        assert result_id is not None
        assert isinstance(result_id, str)
        mock_query_executor.execute_insert.assert_called_once()


class TestLogEvaluationBatch:
    """Test log_evaluation_batch function"""
    
    def test_log_evaluation_batch_success(
        self, 
        sample_evaluation_result, 
        mock_query_executor
    ):
        """Test successful batch logging"""
        results = [
            sample_evaluation_result,
            EvaluationResult(
                example_id="val_002",
                judge_output=sample_evaluation_result.judge_output,
                meta_eval_output=sample_evaluation_result.meta_eval_output,
                beir_metrics=sample_evaluation_result.beir_metrics,
                timestamp=datetime.now()
            )
        ]
        
        log_evaluation_batch(results, mock_query_executor)
        
        # Should call execute_insert for each result
        assert mock_query_executor.execute_insert.call_count == 2
    
    def test_log_evaluation_batch_local_only_mode(self, sample_evaluation_result):
        """Test local-only mode when query_executor is None"""
        results = [sample_evaluation_result]
        
        # Should not raise exception
        log_evaluation_batch(results, None)
    
    def test_log_evaluation_batch_empty_list(self, mock_query_executor):
        """Test batch logging with empty list"""
        log_evaluation_batch([], mock_query_executor)
        
        # Should not call execute_insert
        mock_query_executor.execute_insert.assert_not_called()
    
    def test_log_evaluation_batch_partial_failure(
        self, 
        sample_evaluation_result, 
        mock_query_executor
    ):
        """Test graceful handling of partial batch failures"""
        results = [
            sample_evaluation_result,
            EvaluationResult(
                example_id="val_002",
                judge_output=sample_evaluation_result.judge_output,
                meta_eval_output=sample_evaluation_result.meta_eval_output,
                beir_metrics=sample_evaluation_result.beir_metrics,
                timestamp=datetime.now()
            )
        ]
        
        # First call succeeds, second fails
        mock_query_executor.execute_insert.side_effect = [
            "result_id_1",
            Exception("Database error")
        ]
        
        # Should not raise exception
        log_evaluation_batch(results, mock_query_executor)
        
        # Both should have been attempted
        assert mock_query_executor.execute_insert.call_count == 2
    
    def test_log_evaluation_batch_with_judge_metrics(
        self,
        sample_evaluation_result,
        sample_judge_performance_metrics,
        mock_query_executor
    ):
        """Test batch logging with judge performance metrics"""
        results = [sample_evaluation_result]
        
        log_evaluation_batch(
            results, 
            mock_query_executor,
            judge_performance_metrics=sample_judge_performance_metrics
        )
        
        # Should call execute_insert for result
        assert mock_query_executor.execute_insert.call_count == 1
        
        # Should call execute_query to update metrics
        assert mock_query_executor.execute_query.call_count == 1
        
        # Verify update query
        update_call = mock_query_executor.execute_query.call_args
        assert update_call is not None
        query = update_call[0][0]
        assert "UPDATE evaluation_results" in query
        assert "judge_performance_metrics" in query
    
    def test_log_evaluation_batch_metrics_failure(
        self,
        sample_evaluation_result,
        sample_judge_performance_metrics,
        mock_query_executor
    ):
        """Test graceful handling of metrics logging failure"""
        results = [sample_evaluation_result]
        
        # Make metrics update fail
        mock_query_executor.execute_query.side_effect = Exception("Update failed")
        
        # Should not raise exception
        log_evaluation_batch(
            results,
            mock_query_executor,
            judge_performance_metrics=sample_judge_performance_metrics
        )
        
        # Should still have attempted to insert result
        assert mock_query_executor.execute_insert.call_count == 1


class TestJSONRoundTrip:
    """Test JSON serialization and deserialization round-trip"""
    
    def test_judge_output_json_round_trip(self, sample_judge_output):
        """Test that judge output can be serialized and deserialized"""
        serialized = _serialize_judge_output(sample_judge_output)
        json_str = json.dumps(serialized)
        deserialized = json.loads(json_str)
        
        assert deserialized["correctness_binary"] == sample_judge_output.correctness_binary
        assert deserialized["hallucination_binary"] == sample_judge_output.hallucination_binary
        assert deserialized["risk_direction"] == sample_judge_output.risk_direction
        assert deserialized["risk_impact"] == sample_judge_output.risk_impact
        assert deserialized["reasoning"] == sample_judge_output.reasoning
    
    def test_meta_eval_output_json_round_trip(self, sample_meta_eval_output):
        """Test that meta eval output with ground truth can be serialized and deserialized"""
        serialized = _serialize_meta_eval_output(sample_meta_eval_output)
        json_str = json.dumps(serialized)
        deserialized = json.loads(json_str)
        
        assert deserialized["judge_correct"] == sample_meta_eval_output.judge_correct
        assert deserialized["ground_truth_correctness"] == sample_meta_eval_output.ground_truth_correctness
        assert deserialized["ground_truth_hallucination"] == sample_meta_eval_output.ground_truth_hallucination
        assert deserialized["ground_truth_risk_direction"] == sample_meta_eval_output.ground_truth_risk_direction
        assert deserialized["ground_truth_risk_impact"] == sample_meta_eval_output.ground_truth_risk_impact
    
    def test_beir_metrics_json_round_trip(self, sample_beir_metrics):
        """Test that BEIR metrics can be serialized and deserialized"""
        serialized = _serialize_beir_metrics(sample_beir_metrics)
        json_str = json.dumps(serialized)
        deserialized = json.loads(json_str)
        
        assert deserialized["recall_at_k"] == sample_beir_metrics.recall_at_k
        assert deserialized["precision_at_k"] == sample_beir_metrics.precision_at_k
        assert deserialized["ndcg_at_k"] == sample_beir_metrics.ndcg_at_k
    
    def test_judge_performance_metrics_json_round_trip(self, sample_judge_performance_metrics):
        """Test that judge performance metrics can be serialized and deserialized"""
        serialized = _serialize_judge_performance_metrics(sample_judge_performance_metrics)
        json_str = json.dumps(serialized)
        deserialized = json.loads(json_str)
        
        assert deserialized["correctness"]["precision"] == sample_judge_performance_metrics.correctness.precision
        assert deserialized["hallucination"]["f1_score"] == sample_judge_performance_metrics.hallucination.f1_score
        assert deserialized["risk_direction"]["recall"] == sample_judge_performance_metrics.risk_direction.recall
        assert deserialized["risk_impact"]["precision"] == sample_judge_performance_metrics.risk_impact.precision


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_log_result_with_none_fields(self, mock_query_executor):
        """Test logging result with None optional fields"""
        judge_output = JudgeEvaluationResult(
            correctness_binary=True,
            hallucination_binary=False,
            risk_direction=None,
            risk_impact=None,
            reasoning="Test",
            failure_mode=None
        )
        
        meta_eval = MetaEvaluationResult(
            judge_correct=True,
            explanation=None,
            ground_truth_correctness=None,
            ground_truth_hallucination=None,
            ground_truth_risk_direction=None,
            ground_truth_risk_impact=None
        )
        
        result = EvaluationResult(
            example_id="val_003",
            judge_output=judge_output,
            meta_eval_output=meta_eval,
            beir_metrics=BEIRMetricsResult(recall_at_k=0.5, precision_at_k=0.5, ndcg_at_k=0.6),
            timestamp=datetime.now()
        )
        
        result_id = log_evaluation_result(result, mock_query_executor)
        
        assert result_id is not None
        mock_query_executor.execute_insert.assert_called_once()
    
    def test_batch_with_serialization_error(self, mock_query_executor):
        """Test batch logging when serialization fails for one result"""
        # Create a result that might cause serialization issues
        # (though our serialization should handle all cases)
        results = [
            EvaluationResult(
                example_id="val_001",
                judge_output=JudgeEvaluationResult(
                    correctness_binary=True,
                    hallucination_binary=False,
                    risk_direction=0,
                    risk_impact=1,
                    reasoning="Test",
                    failure_mode=None
                ),
                meta_eval_output=MetaEvaluationResult(
                    judge_correct=True,
                    explanation="Test"
                ),
                beir_metrics=BEIRMetricsResult(recall_at_k=0.5, precision_at_k=0.5, ndcg_at_k=0.6),
                timestamp=datetime.now()
            )
        ]
        
        # Should handle gracefully
        log_evaluation_batch(results, mock_query_executor)
        
        # Should have attempted to insert
        assert mock_query_executor.execute_insert.call_count == 1
    
    def test_batch_all_serialization_failures(self, mock_query_executor):
        """Test batch logging when all results fail serialization"""
        # Mock a result that will fail serialization
        class BadResult:
            example_id = "val_001"
            judge_output = object()  # Not a dataclass, will fail
            meta_eval_output = object()
            beir_metrics = object()
            timestamp = datetime.now()
        
        # This will cause serialization to fail
        # We need to patch json.dumps to raise an exception
        with patch('rag_eval.services.evaluator.logging.json.dumps', side_effect=Exception("Serialization failed")):
            results = [
                EvaluationResult(
                    example_id="val_001",
                    judge_output=JudgeEvaluationResult(
                        correctness_binary=True,
                        hallucination_binary=False,
                        risk_direction=0,
                        risk_impact=1,
                        reasoning="Test",
                        failure_mode=None
                    ),
                    meta_eval_output=MetaEvaluationResult(judge_correct=True),
                    beir_metrics=BEIRMetricsResult(recall_at_k=0.5, precision_at_k=0.5, ndcg_at_k=0.6),
                    timestamp=datetime.now()
                )
            ]
            
            log_evaluation_batch(results, mock_query_executor)
            
            # Should not have attempted to insert (all serializations failed)
            mock_query_executor.execute_insert.assert_not_called()
    
    def test_batch_outer_exception(self, sample_evaluation_result, mock_query_executor):
        """Test batch logging when outer exception occurs"""
        # Make execute_insert raise exception
        mock_query_executor.execute_insert.side_effect = Exception("Database error")
        
        # Should handle gracefully
        log_evaluation_batch([sample_evaluation_result], mock_query_executor)
        
        # Should have attempted to insert
        assert mock_query_executor.execute_insert.call_count == 1

