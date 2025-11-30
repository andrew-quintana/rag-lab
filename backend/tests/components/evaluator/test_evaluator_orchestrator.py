"""Unit tests for Evaluation Pipeline Orchestrator"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import List

from rag_eval.core.config import Config
from rag_eval.core.exceptions import AzureServiceError, EvaluationError
from rag_eval.core.interfaces import (
    EvaluationExample,
    EvaluationResult,
    RetrievalResult,
    ModelAnswer,
    JudgeEvaluationResult,
    MetaEvaluationResult,
    BEIRMetricsResult,
    JudgePerformanceMetrics,
    JudgeMetricScores
)
from rag_eval.services.evaluator.orchestrator import (
    evaluate_rag_system,
    _evaluate_single_example
)


@pytest.fixture
def mock_config():
    """Create a mock Config object"""
    config = Mock(spec=Config)
    config.azure_ai_foundry_endpoint = "https://test-endpoint"
    config.azure_ai_foundry_api_key = "test-key"
    config.azure_ai_foundry_deployment = "gpt-4o-mini"
    return config


@pytest.fixture
def sample_evaluation_example():
    """Create a sample evaluation example"""
    return EvaluationExample(
        example_id="val_001",
        question="What is the copay for specialist visits?",
        reference_answer="The copay for specialist visits is $50 per visit.",
        ground_truth_chunk_ids=["chunk_1", "chunk_2"],
        beir_failure_scale_factor=0.3
    )


@pytest.fixture
def sample_evaluation_dataset(sample_evaluation_example):
    """Create a sample evaluation dataset"""
    return [
        sample_evaluation_example,
        EvaluationExample(
            example_id="val_002",
            question="What is the deductible?",
            reference_answer="The deductible is $1,500.",
            ground_truth_chunk_ids=["chunk_0"],
            beir_failure_scale_factor=0.2
        )
    ]


@pytest.fixture
def sample_retrieved_chunks():
    """Create sample retrieved chunks"""
    return [
        RetrievalResult(
            chunk_id="chunk_1",
            similarity_score=0.95,
            chunk_text="The copay for specialist visits is $50."
        ),
        RetrievalResult(
            chunk_id="chunk_2",
            similarity_score=0.90,
            chunk_text="Deductible is $1,500 per year."
        )
    ]


@pytest.fixture
def sample_model_answer():
    """Create a sample model answer"""
    return ModelAnswer(
        text="The copay for specialist visits is $50 per visit.",
        query_id="test_query_123",
        prompt_version="v1",
        retrieved_chunk_ids=["chunk_1", "chunk_2"],
        timestamp=datetime.now()
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
    """Create a sample meta-evaluation output"""
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
def mock_rag_retriever(sample_retrieved_chunks):
    """Create a mock RAG retriever function"""
    def retriever(query: str, k: int) -> List[RetrievalResult]:
        return sample_retrieved_chunks[:k]
    return retriever


@pytest.fixture
def mock_rag_generator(sample_model_answer):
    """Create a mock RAG generator function"""
    def generator(query: str, chunks: List[RetrievalResult]) -> ModelAnswer:
        return sample_model_answer
    return generator


class TestEvaluateRAGSystem:
    """Test the main evaluate_rag_system function"""
    
    def test_empty_evaluation_dataset(self, mock_config, mock_rag_retriever, mock_rag_generator):
        """Test that empty evaluation dataset raises ValueError"""
        with pytest.raises(ValueError, match="evaluation_dataset cannot be empty"):
            evaluate_rag_system(
                evaluation_dataset=[],
                rag_retriever=mock_rag_retriever,
                rag_generator=mock_rag_generator,
                config=mock_config
            )
    
    @patch('rag_eval.services.evaluator.orchestrator.compute_beir_metrics')
    @patch('rag_eval.services.evaluator.orchestrator.meta_evaluate_judge')
    @patch('rag_eval.services.evaluator.orchestrator.extract_costs')
    @patch('rag_eval.services.evaluator.orchestrator.evaluate_answer_with_judge')
    def test_full_pipeline_execution(
        self,
        mock_evaluate_judge,
        mock_extract_costs,
        mock_meta_eval,
        mock_beir_metrics,
        mock_config,
        sample_evaluation_dataset,
        mock_rag_retriever,
        mock_rag_generator,
        sample_judge_output,
        sample_meta_eval_output,
        sample_beir_metrics
    ):
        """Test full pipeline: Retrieval → RAG generation → Judge → Meta-Eval → Metrics"""
        # Setup mocks
        mock_evaluate_judge.return_value = sample_judge_output
        mock_extract_costs.return_value = {"money": 50.0}
        mock_meta_eval.return_value = sample_meta_eval_output
        mock_beir_metrics.return_value = sample_beir_metrics
        
        # Run pipeline
        results = evaluate_rag_system(
            evaluation_dataset=sample_evaluation_dataset,
            rag_retriever=mock_rag_retriever,
            rag_generator=mock_rag_generator,
            config=mock_config
        )
        
        # Verify results
        assert len(results) == 2
        assert all(isinstance(r, EvaluationResult) for r in results)
        assert results[0].example_id == "val_001"
        assert results[1].example_id == "val_002"
        
        # Verify judge was called for each example
        assert mock_evaluate_judge.call_count == 2
        
        # Verify meta-eval was called for each example
        assert mock_meta_eval.call_count == 2
        
        # Verify BEIR metrics were computed for each example
        assert mock_beir_metrics.call_count == 2
    
    @patch('rag_eval.services.evaluator.orchestrator.compute_beir_metrics')
    @patch('rag_eval.services.evaluator.orchestrator.meta_evaluate_judge')
    @patch('rag_eval.services.evaluator.orchestrator.extract_costs')
    @patch('rag_eval.services.evaluator.orchestrator.evaluate_answer_with_judge')
    def test_pipeline_with_mocked_rag_components(
        self,
        mock_evaluate_judge,
        mock_extract_costs,
        mock_meta_eval,
        mock_beir_metrics,
        mock_config,
        sample_evaluation_example,
        sample_judge_output,
        sample_meta_eval_output,
        sample_beir_metrics
    ):
        """Test pipeline with mocked RAG components"""
        # Setup mocks
        mock_evaluate_judge.return_value = sample_judge_output
        mock_extract_costs.return_value = {"money": 50.0}
        mock_meta_eval.return_value = sample_meta_eval_output
        mock_beir_metrics.return_value = sample_beir_metrics
        
        # Create mock RAG functions
        mock_retriever = Mock(return_value=[
            RetrievalResult(chunk_id="chunk_1", similarity_score=0.9, chunk_text="Test chunk")
        ])
        mock_generator = Mock(return_value=ModelAnswer(
            text="Test answer",
            query_id="test",
            prompt_version="v1",
            retrieved_chunk_ids=["chunk_1"]
        ))
        
        # Run pipeline
        results = evaluate_rag_system(
            evaluation_dataset=[sample_evaluation_example],
            rag_retriever=mock_retriever,
            rag_generator=mock_generator,
            config=mock_config
        )
        
        # Verify RAG components were called
        mock_retriever.assert_called_once_with(sample_evaluation_example.question, k=5)
        mock_generator.assert_called_once()
        
        # Verify results
        assert len(results) == 1
        assert results[0].example_id == sample_evaluation_example.example_id
    
    @patch('rag_eval.services.evaluator.orchestrator.compute_beir_metrics')
    @patch('rag_eval.services.evaluator.orchestrator.meta_evaluate_judge')
    @patch('rag_eval.services.evaluator.orchestrator.extract_costs')
    @patch('rag_eval.services.evaluator.orchestrator.evaluate_answer_with_judge')
    def test_pipeline_error_handling_rag_retriever_failure(
        self,
        mock_evaluate_judge,
        mock_extract_costs,
        mock_meta_eval,
        mock_beir_metrics,
        mock_config,
        sample_evaluation_dataset,
        mock_rag_generator,
        sample_judge_output,
        sample_meta_eval_output,
        sample_beir_metrics
    ):
        """Test pipeline error handling when RAG retriever fails"""
        # Setup mocks
        mock_evaluate_judge.return_value = sample_judge_output
        mock_extract_costs.return_value = {"money": 50.0}
        mock_meta_eval.return_value = sample_meta_eval_output
        mock_beir_metrics.return_value = sample_beir_metrics
        
        # Create failing retriever
        def failing_retriever(query: str, k: int):
            raise AzureServiceError("Retrieval failed")
        
        # Run pipeline - should handle error gracefully
        with pytest.raises(EvaluationError, match="All .* examples failed evaluation"):
            evaluate_rag_system(
                evaluation_dataset=sample_evaluation_dataset,
                rag_retriever=failing_retriever,
                rag_generator=mock_rag_generator,
                config=mock_config
            )
    
    @patch('rag_eval.services.evaluator.orchestrator.compute_beir_metrics')
    @patch('rag_eval.services.evaluator.orchestrator.meta_evaluate_judge')
    @patch('rag_eval.services.evaluator.orchestrator.extract_costs')
    @patch('rag_eval.services.evaluator.orchestrator.evaluate_answer_with_judge')
    def test_pipeline_error_handling_rag_generator_failure(
        self,
        mock_evaluate_judge,
        mock_extract_costs,
        mock_meta_eval,
        mock_beir_metrics,
        mock_config,
        sample_evaluation_dataset,
        mock_rag_retriever,
        sample_judge_output,
        sample_meta_eval_output,
        sample_beir_metrics
    ):
        """Test pipeline error handling when RAG generator fails"""
        # Setup mocks
        mock_evaluate_judge.return_value = sample_judge_output
        mock_extract_costs.return_value = {"money": 50.0}
        mock_meta_eval.return_value = sample_meta_eval_output
        mock_beir_metrics.return_value = sample_beir_metrics
        
        # Create failing generator
        def failing_generator(query: str, chunks: List[RetrievalResult]):
            raise AzureServiceError("Generation failed")
        
        # Run pipeline - should handle error gracefully
        with pytest.raises(EvaluationError, match="All .* examples failed evaluation"):
            evaluate_rag_system(
                evaluation_dataset=sample_evaluation_dataset,
                rag_retriever=mock_rag_retriever,
                rag_generator=failing_generator,
                config=mock_config
            )
    
    @patch('rag_eval.services.evaluator.orchestrator.compute_beir_metrics')
    @patch('rag_eval.services.evaluator.orchestrator.meta_evaluate_judge')
    @patch('rag_eval.services.evaluator.orchestrator.extract_costs')
    @patch('rag_eval.services.evaluator.orchestrator.evaluate_answer_with_judge')
    def test_pipeline_error_handling_judge_failure(
        self,
        mock_evaluate_judge,
        mock_extract_costs,
        mock_meta_eval,
        mock_beir_metrics,
        mock_config,
        sample_evaluation_dataset,
        mock_rag_retriever,
        mock_rag_generator,
        sample_meta_eval_output,
        sample_beir_metrics
    ):
        """Test pipeline error handling when judge evaluation fails"""
        # Setup mocks
        mock_evaluate_judge.side_effect = AzureServiceError("Judge evaluation failed")
        mock_extract_costs.return_value = {"money": 50.0}
        mock_meta_eval.return_value = sample_meta_eval_output
        mock_beir_metrics.return_value = sample_beir_metrics
        
        # Run pipeline - should handle error gracefully
        with pytest.raises(EvaluationError, match="All .* examples failed evaluation"):
            evaluate_rag_system(
                evaluation_dataset=sample_evaluation_dataset,
                rag_retriever=mock_rag_retriever,
                rag_generator=mock_rag_generator,
                config=mock_config
            )
    
    @patch('rag_eval.services.evaluator.orchestrator.compute_beir_metrics')
    @patch('rag_eval.services.evaluator.orchestrator.meta_evaluate_judge')
    @patch('rag_eval.services.evaluator.orchestrator.extract_costs')
    @patch('rag_eval.services.evaluator.orchestrator.evaluate_answer_with_judge')
    def test_pipeline_partial_failure_continues(
        self,
        mock_evaluate_judge,
        mock_extract_costs,
        mock_meta_eval,
        mock_beir_metrics,
        mock_config,
        sample_evaluation_dataset,
        mock_rag_retriever,
        mock_rag_generator,
        sample_judge_output,
        sample_meta_eval_output,
        sample_beir_metrics
    ):
        """Test that pipeline continues when some examples fail"""
        # Setup mocks - first example succeeds, second fails
        def side_effect(*args, **kwargs):
            if mock_evaluate_judge.call_count == 1:
                return sample_judge_output
            else:
                raise AzureServiceError("Judge failed")
        
        mock_evaluate_judge.side_effect = side_effect
        mock_extract_costs.return_value = {"money": 50.0}
        mock_meta_eval.return_value = sample_meta_eval_output
        mock_beir_metrics.return_value = sample_beir_metrics
        
        # Run pipeline - should return results for successful examples
        results = evaluate_rag_system(
            evaluation_dataset=sample_evaluation_dataset,
            rag_retriever=mock_rag_retriever,
            rag_generator=mock_rag_generator,
            config=mock_config
        )
        
        # Should have one successful result
        assert len(results) == 1
        assert results[0].example_id == "val_001"
    
    @patch('rag_eval.services.evaluator.orchestrator.compute_beir_metrics')
    @patch('rag_eval.services.evaluator.orchestrator.meta_evaluate_judge')
    @patch('rag_eval.services.evaluator.orchestrator.extract_costs')
    @patch('rag_eval.services.evaluator.orchestrator.evaluate_answer_with_judge')
    def test_pipeline_logging_and_observability(
        self,
        mock_evaluate_judge,
        mock_extract_costs,
        mock_meta_eval,
        mock_beir_metrics,
        mock_config,
        sample_evaluation_example,
        mock_rag_retriever,
        mock_rag_generator,
        sample_judge_output,
        sample_meta_eval_output,
        sample_beir_metrics
    ):
        """Test pipeline logging and observability"""
        # Setup mocks
        mock_evaluate_judge.return_value = sample_judge_output
        mock_extract_costs.return_value = {"money": 50.0}
        mock_meta_eval.return_value = sample_meta_eval_output
        mock_beir_metrics.return_value = sample_beir_metrics
        
        # Run pipeline
        results = evaluate_rag_system(
            evaluation_dataset=[sample_evaluation_example],
            rag_retriever=mock_rag_retriever,
            rag_generator=mock_rag_generator,
            config=mock_config
        )
        
        # Verify results contain timestamps
        assert len(results) == 1
        assert results[0].timestamp is not None
        assert isinstance(results[0].timestamp, datetime)


class TestEvaluateSingleExample:
    """Test the _evaluate_single_example helper function"""
    
    @patch('rag_eval.services.evaluator.orchestrator.compute_beir_metrics')
    @patch('rag_eval.services.evaluator.orchestrator.meta_evaluate_judge')
    @patch('rag_eval.services.evaluator.orchestrator.extract_costs')
    @patch('rag_eval.services.evaluator.orchestrator.evaluate_answer_with_judge')
    def test_evaluate_single_example_success(
        self,
        mock_evaluate_judge,
        mock_extract_costs,
        mock_meta_eval,
        mock_beir_metrics,
        mock_config,
        sample_evaluation_example,
        sample_retrieved_chunks,
        sample_model_answer,
        sample_judge_output,
        sample_meta_eval_output,
        sample_beir_metrics
    ):
        """Test successful single example evaluation"""
        # Setup mocks
        mock_evaluate_judge.return_value = sample_judge_output
        mock_extract_costs.return_value = {"money": 50.0}
        mock_meta_eval.return_value = sample_meta_eval_output
        mock_beir_metrics.return_value = sample_beir_metrics
        
        # Create Mock objects for RAG functions
        mock_rag_retriever = Mock(return_value=sample_retrieved_chunks)
        mock_rag_generator = Mock(return_value=sample_model_answer)
        
        # Run evaluation
        result = _evaluate_single_example(
            example=sample_evaluation_example,
            rag_retriever=mock_rag_retriever,
            rag_generator=mock_rag_generator,
            config=mock_config
        )
        
        # Verify result
        assert isinstance(result, EvaluationResult)
        assert result.example_id == sample_evaluation_example.example_id
        assert result.judge_output == sample_judge_output
        assert result.meta_eval_output == sample_meta_eval_output
        assert result.beir_metrics == sample_beir_metrics
        assert result.timestamp is not None
        
        # Verify all components were called
        mock_rag_retriever.assert_called_once_with(sample_evaluation_example.question, k=5)
        mock_rag_generator.assert_called_once()
        mock_evaluate_judge.assert_called_once()
        mock_meta_eval.assert_called_once()
        mock_beir_metrics.assert_called_once()
    
    @patch('rag_eval.services.evaluator.orchestrator.compute_beir_metrics')
    @patch('rag_eval.services.evaluator.orchestrator.meta_evaluate_judge')
    @patch('rag_eval.services.evaluator.orchestrator.extract_costs')
    @patch('rag_eval.services.evaluator.orchestrator.evaluate_answer_with_judge')
    def test_evaluate_single_example_cost_extraction_when_correctness_true(
        self,
        mock_evaluate_judge,
        mock_extract_costs,
        mock_meta_eval,
        mock_beir_metrics,
        mock_config,
        sample_evaluation_example,
        mock_rag_retriever,
        mock_rag_generator,
        sample_meta_eval_output,
        sample_beir_metrics
    ):
        """Test that cost extraction is called when correctness is True"""
        # Setup judge output with correctness=True
        judge_output = JudgeEvaluationResult(
            correctness_binary=True,
            hallucination_binary=False,
            risk_direction=0,
            risk_impact=1,
            reasoning="Correct answer",
            failure_mode=None
        )
        mock_evaluate_judge.return_value = judge_output
        mock_extract_costs.return_value = {"money": 50.0}
        mock_meta_eval.return_value = sample_meta_eval_output
        mock_beir_metrics.return_value = sample_beir_metrics
        
        # Run evaluation
        result = _evaluate_single_example(
            example=sample_evaluation_example,
            rag_retriever=mock_rag_retriever,
            rag_generator=mock_rag_generator,
            config=mock_config
        )
        
        # Verify cost extraction was called (twice: model answer and chunks)
        assert mock_extract_costs.call_count == 2
    
    @patch('rag_eval.services.evaluator.orchestrator.compute_beir_metrics')
    @patch('rag_eval.services.evaluator.orchestrator.meta_evaluate_judge')
    @patch('rag_eval.services.evaluator.orchestrator.extract_costs')
    @patch('rag_eval.services.evaluator.orchestrator.evaluate_answer_with_judge')
    def test_evaluate_single_example_no_cost_extraction_when_correctness_false(
        self,
        mock_evaluate_judge,
        mock_extract_costs,
        mock_meta_eval,
        mock_beir_metrics,
        mock_config,
        sample_evaluation_example,
        mock_rag_retriever,
        mock_rag_generator,
        sample_meta_eval_output,
        sample_beir_metrics
    ):
        """Test that cost extraction is NOT called when correctness is False"""
        # Setup judge output with correctness=False
        judge_output = JudgeEvaluationResult(
            correctness_binary=False,
            hallucination_binary=False,
            risk_direction=None,
            risk_impact=None,
            reasoning="Incorrect answer",
            failure_mode=None
        )
        mock_evaluate_judge.return_value = judge_output
        mock_meta_eval.return_value = sample_meta_eval_output
        mock_beir_metrics.return_value = sample_beir_metrics
        
        # Run evaluation
        result = _evaluate_single_example(
            example=sample_evaluation_example,
            rag_retriever=mock_rag_retriever,
            rag_generator=mock_rag_generator,
            config=mock_config
        )
        
        # Verify cost extraction was NOT called
        mock_extract_costs.assert_not_called()


class TestJudgePerformanceMetricsIntegration:
    """Test integration with judge performance metrics calculation"""
    
    @patch('rag_eval.services.evaluator.orchestrator.compute_beir_metrics')
    @patch('rag_eval.services.evaluator.orchestrator.meta_evaluate_judge')
    @patch('rag_eval.services.evaluator.orchestrator.extract_costs')
    @patch('rag_eval.services.evaluator.orchestrator.evaluate_answer_with_judge')
    @patch('rag_eval.services.evaluator.orchestrator.calculate_judge_metrics')
    def test_judge_metrics_calculation_from_pipeline_results(
        self,
        mock_calculate_metrics,
        mock_evaluate_judge,
        mock_extract_costs,
        mock_meta_eval,
        mock_beir_metrics,
        mock_config,
        sample_evaluation_dataset,
        mock_rag_retriever,
        mock_rag_generator,
        sample_judge_output,
        sample_meta_eval_output,
        sample_beir_metrics
    ):
        """Test calculate_judge_metrics integration with pipeline results"""
        # Setup mocks
        mock_evaluate_judge.return_value = sample_judge_output
        mock_extract_costs.return_value = {"money": 50.0}
        mock_meta_eval.return_value = sample_meta_eval_output
        mock_beir_metrics.return_value = sample_beir_metrics
        
        # Mock metrics calculation
        mock_metrics = JudgePerformanceMetrics(
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
                precision=0.8,
                recall=0.75,
                f1_score=0.775,
                true_positives=15,
                true_negatives=5,
                false_positives=4,
                false_negatives=5,
                total_samples=29
            )
        )
        mock_calculate_metrics.return_value = mock_metrics
        
        # Run pipeline
        results = evaluate_rag_system(
            evaluation_dataset=sample_evaluation_dataset,
            rag_retriever=mock_rag_retriever,
            rag_generator=mock_rag_generator,
            config=mock_config
        )
        
        # Manually calculate metrics from results
        from rag_eval.services.evaluator.meta_eval import calculate_judge_metrics
        
        evaluation_pairs = [
            (r.judge_output, r.meta_eval_output) for r in results
        ]
        metrics = calculate_judge_metrics(evaluation_pairs)
        
        # Verify metrics are calculated correctly
        assert isinstance(metrics, JudgePerformanceMetrics)
        assert metrics.correctness is not None
        assert metrics.hallucination is not None
        assert metrics.correctness.total_samples == len(results)
        assert metrics.hallucination.total_samples == len(results)
    
    @patch('rag_eval.services.evaluator.orchestrator.compute_beir_metrics')
    @patch('rag_eval.services.evaluator.orchestrator.meta_evaluate_judge')
    @patch('rag_eval.services.evaluator.orchestrator.extract_costs')
    @patch('rag_eval.services.evaluator.orchestrator.evaluate_answer_with_judge')
    def test_judge_metrics_with_mixed_scenarios(
        self,
        mock_evaluate_judge,
        mock_extract_costs,
        mock_meta_eval,
        mock_beir_metrics,
        mock_config,
        sample_evaluation_dataset,
        mock_rag_retriever,
        mock_rag_generator,
        sample_beir_metrics
    ):
        """Test metrics calculation with mixed scenarios (some with costs, some without)"""
        # Setup mixed judge outputs
        judge_output_1 = JudgeEvaluationResult(
            correctness_binary=True,
            hallucination_binary=False,
            risk_direction=0,
            risk_impact=1,
            reasoning="Correct with costs",
            failure_mode=None
        )
        judge_output_2 = JudgeEvaluationResult(
            correctness_binary=False,
            hallucination_binary=True,
            risk_direction=None,
            risk_impact=None,
            reasoning="Incorrect, no costs",
            failure_mode=None
        )
        
        meta_eval_1 = MetaEvaluationResult(
            judge_correct=True,
            explanation="Correct",
            ground_truth_correctness=True,
            ground_truth_hallucination=False,
            ground_truth_risk_direction=0,
            ground_truth_risk_impact=1
        )
        meta_eval_2 = MetaEvaluationResult(
            judge_correct=True,
            explanation="Correct",
            ground_truth_correctness=False,
            ground_truth_hallucination=True,
            ground_truth_risk_direction=None,
            ground_truth_risk_impact=None
        )
        
        # Setup mocks to return different outputs
        def judge_side_effect(*args, **kwargs):
            if mock_evaluate_judge.call_count == 1:
                return judge_output_1
            else:
                return judge_output_2
        
        def meta_side_effect(*args, **kwargs):
            if mock_meta_eval.call_count == 1:
                return meta_eval_1
            else:
                return meta_eval_2
        
        mock_evaluate_judge.side_effect = judge_side_effect
        mock_extract_costs.return_value = {"money": 50.0}
        mock_meta_eval.side_effect = meta_side_effect
        mock_beir_metrics.return_value = sample_beir_metrics
        
        # Run pipeline
        results = evaluate_rag_system(
            evaluation_dataset=sample_evaluation_dataset,
            rag_retriever=mock_rag_retriever,
            rag_generator=mock_rag_generator,
            config=mock_config
        )
        
        # Calculate metrics
        from rag_eval.services.evaluator.meta_eval import calculate_judge_metrics
        
        evaluation_pairs = [
            (r.judge_output, r.meta_eval_output) for r in results
        ]
        metrics = calculate_judge_metrics(evaluation_pairs)
        
        # Verify metrics handle mixed scenarios
        assert isinstance(metrics, JudgePerformanceMetrics)
        assert metrics.correctness is not None
        assert metrics.hallucination is not None
        # Risk direction and impact may be None if insufficient data
        # This is expected behavior for mixed scenarios

