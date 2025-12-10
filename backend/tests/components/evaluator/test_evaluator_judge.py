"""Unit tests for LLM-as-Judge orchestrator"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List

from src.core.config import Config
from src.core.exceptions import AzureServiceError
from src.core.interfaces import RetrievalResult, JudgeEvaluationResult
from src.services.evaluator.judge import (
    evaluate_answer_with_judge,
    _construct_reasoning_trace,
    _extract_failure_mode,
    _call_evaluator_with_reasoning
)
from src.services.evaluator.correctness import CorrectnessEvaluator
from src.services.evaluator.hallucination import HallucinationEvaluator
from src.services.evaluator.risk_direction import RiskDirectionEvaluator
from src.services.evaluator.cost_extraction import CostExtractionEvaluator
from src.services.evaluator.risk_impact import RiskImpactEvaluator


@pytest.fixture
def mock_config():
    """Create a mock Config object"""
    config = Mock(spec=Config)
    config.azure_ai_foundry_endpoint = "https://test-endpoint"
    config.azure_ai_foundry_api_key = "test-key"
    config.azure_ai_foundry_deployment = "gpt-4o-mini"
    return config


@pytest.fixture
def sample_retrieved_context():
    """Create sample retrieved context"""
    return [
        RetrievalResult(
            chunk_id="chunk_001",
            similarity_score=0.95,
            chunk_text="The copay for a specialist visit is $50."
        ),
        RetrievalResult(
            chunk_id="chunk_002",
            similarity_score=0.90,
            chunk_text="Deductible is $1000 per year."
        )
    ]


@pytest.fixture
def sample_query():
    """Sample query"""
    return "What is the copay for specialist visits?"


@pytest.fixture
def sample_model_answer():
    """Sample model answer"""
    return "The copay for specialist visits is $50."


@pytest.fixture
def sample_reference_answer():
    """Sample reference answer"""
    return "Specialist visits have a $50 copay."


class TestEvaluateAnswerWithJudge:
    """Test the main evaluate_answer_with_judge function"""
    
    def test_input_validation_empty_query(self, mock_config, sample_retrieved_context, 
                                         sample_model_answer, sample_reference_answer):
        """Test that empty query raises ValueError"""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            evaluate_answer_with_judge(
                query="",
                retrieved_context=sample_retrieved_context,
                model_answer=sample_model_answer,
                reference_answer=sample_reference_answer,
                config=mock_config
            )
    
    def test_input_validation_empty_retrieved_context(self, mock_config, sample_query,
                                                    sample_model_answer, sample_reference_answer):
        """Test that empty retrieved context raises ValueError"""
        with pytest.raises(ValueError, match="Retrieved context cannot be empty"):
            evaluate_answer_with_judge(
                query=sample_query,
                retrieved_context=[],
                model_answer=sample_model_answer,
                reference_answer=sample_reference_answer,
                config=mock_config
            )
    
    def test_input_validation_empty_model_answer(self, mock_config, sample_query,
                                                sample_retrieved_context, sample_reference_answer):
        """Test that empty model answer raises ValueError"""
        with pytest.raises(ValueError, match="Model answer cannot be empty"):
            evaluate_answer_with_judge(
                query=sample_query,
                retrieved_context=sample_retrieved_context,
                model_answer="",
                reference_answer=sample_reference_answer,
                config=mock_config
            )
    
    def test_input_validation_empty_reference_answer(self, mock_config, sample_query,
                                                     sample_retrieved_context, sample_model_answer):
        """Test that empty reference answer raises ValueError"""
        with pytest.raises(ValueError, match="Reference answer cannot be empty"):
            evaluate_answer_with_judge(
                query=sample_query,
                retrieved_context=sample_retrieved_context,
                model_answer=sample_model_answer,
                reference_answer="",
                config=mock_config
            )
    
    @patch('src.services.evaluator.judge.CorrectnessEvaluator')
    @patch('src.services.evaluator.judge.HallucinationEvaluator')
    def test_correctness_false_path_no_cost_nodes_called(
        self, mock_hallucination_class, mock_correctness_class,
        mock_config, sample_query, sample_retrieved_context,
        sample_model_answer, sample_reference_answer
    ):
        """Test that cost/impact nodes are NOT called when correctness is False"""
        # Setup mocks
        mock_correctness = Mock()
        mock_correctness._construct_prompt.return_value = "test prompt"
        mock_correctness._call_llm.return_value = '{"correctness_binary": false, "reasoning": "Not correct"}'
        mock_correctness._parse_json_response.return_value = {
            "correctness_binary": False,
            "reasoning": "Not correct"
        }
        mock_correctness_class.return_value = mock_correctness
        
        mock_hallucination = Mock()
        mock_hallucination._construct_prompt.return_value = "test prompt"
        mock_hallucination._call_llm.return_value = '{"hallucination_binary": false, "reasoning": "No hallucination"}'
        mock_hallucination._parse_json_response.return_value = {
            "hallucination_binary": False,
            "reasoning": "No hallucination"
        }
        mock_hallucination_class.return_value = mock_hallucination
        
        # Call function
        result = evaluate_answer_with_judge(
            query=sample_query,
            retrieved_context=sample_retrieved_context,
            model_answer=sample_model_answer,
            reference_answer=sample_reference_answer,
            config=mock_config
        )
        
        # Verify correctness and hallucination were called
        assert mock_correctness._call_llm.called
        assert mock_hallucination._call_llm.called
        
        # Verify result
        assert isinstance(result, JudgeEvaluationResult)
        assert result.correctness_binary is False
        assert result.hallucination_binary is False
        assert result.risk_direction is None
        assert result.risk_impact is None
        assert "Not correct" in result.reasoning
        assert "No hallucination" in result.reasoning
    
    @patch('src.services.evaluator.judge.RiskImpactEvaluator')
    @patch('src.services.evaluator.judge.CostExtractionEvaluator')
    @patch('src.services.evaluator.judge.RiskDirectionEvaluator')
    @patch('src.services.evaluator.judge.HallucinationEvaluator')
    @patch('src.services.evaluator.judge.CorrectnessEvaluator')
    def test_correctness_true_path_all_nodes_called(
        self, mock_correctness_class, mock_hallucination_class,
        mock_risk_direction_class, mock_cost_extraction_class, mock_risk_impact_class,
        mock_config, sample_query, sample_retrieved_context,
        sample_model_answer, sample_reference_answer
    ):
        """Test that all nodes are called when correctness is True"""
        # Setup correctness mock
        mock_correctness = Mock()
        mock_correctness._construct_prompt.return_value = "test prompt"
        mock_correctness._call_llm.return_value = '{"correctness_binary": true, "reasoning": "Correct"}'
        mock_correctness._parse_json_response.return_value = {
            "correctness_binary": True,
            "reasoning": "Correct"
        }
        mock_correctness_class.return_value = mock_correctness
        
        # Setup hallucination mock
        mock_hallucination = Mock()
        mock_hallucination._construct_prompt.return_value = "test prompt"
        mock_hallucination._call_llm.return_value = '{"hallucination_binary": false, "reasoning": "No hallucination"}'
        mock_hallucination._parse_json_response.return_value = {
            "hallucination_binary": False,
            "reasoning": "No hallucination"
        }
        mock_hallucination_class.return_value = mock_hallucination
        
        # Setup risk direction mock
        mock_risk_direction = Mock()
        mock_risk_direction._construct_prompt.return_value = "test prompt"
        mock_risk_direction._call_llm.return_value = '{"risk_direction": -1, "reasoning": "Care avoidance"}'
        mock_risk_direction._parse_json_response.return_value = {
            "risk_direction": -1,
            "reasoning": "Care avoidance"
        }
        mock_risk_direction_class.return_value = mock_risk_direction
        
        # Setup cost extraction mock
        mock_cost_extractor = Mock()
        mock_cost_extractor.extract_costs.side_effect = [
            {"money": 50.0, "reasoning": "Found $50"},
            {"money": 50.0, "reasoning": "Found $50"}
        ]
        mock_cost_extraction_class.return_value = mock_cost_extractor
        
        # Setup risk impact mock
        mock_risk_impact = Mock()
        mock_risk_impact._construct_prompt.return_value = "test prompt"
        mock_risk_impact._call_llm.return_value = '{"risk_impact": 1.5, "reasoning": "Low impact"}'
        mock_risk_impact._parse_json_response.return_value = {
            "risk_impact": 1.5,
            "reasoning": "Low impact"
        }
        mock_risk_impact_class.return_value = mock_risk_impact
        
        # Call function
        result = evaluate_answer_with_judge(
            query=sample_query,
            retrieved_context=sample_retrieved_context,
            model_answer=sample_model_answer,
            reference_answer=sample_reference_answer,
            config=mock_config
        )
        
        # Verify all nodes were called
        assert mock_correctness._call_llm.called
        assert mock_hallucination._call_llm.called
        assert mock_risk_direction._call_llm.called
        assert mock_cost_extractor.extract_costs.call_count == 2  # Called for model answer and chunks
        assert mock_risk_impact._call_llm.called
        
        # Verify result
        assert isinstance(result, JudgeEvaluationResult)
        assert result.correctness_binary is True
        assert result.hallucination_binary is False
        assert result.risk_direction == -1
        assert result.risk_impact == 1.5
        assert "Correct" in result.reasoning
        assert "No hallucination" in result.reasoning
        assert "Care avoidance" in result.reasoning
        assert "Low impact" in result.reasoning
    
    @patch('src.services.evaluator.judge.CorrectnessEvaluator')
    def test_llm_failure_propagates(self, mock_correctness_class, mock_config,
                                   sample_query, sample_retrieved_context,
                                   sample_model_answer, sample_reference_answer):
        """Test that LLM failures are properly propagated"""
        # Setup mock to raise AzureServiceError
        mock_correctness = Mock()
        mock_correctness._construct_prompt.return_value = "test prompt"
        mock_correctness._call_llm.side_effect = AzureServiceError("LLM call failed")
        mock_correctness_class.return_value = mock_correctness
        
        # Call function and expect error
        with pytest.raises(AzureServiceError, match="LLM call failed"):
            evaluate_answer_with_judge(
                query=sample_query,
                retrieved_context=sample_retrieved_context,
                model_answer=sample_model_answer,
                reference_answer=sample_reference_answer,
                config=mock_config
            )
    
    @patch('src.services.evaluator.judge.RiskImpactEvaluator')
    @patch('src.services.evaluator.judge.CostExtractionEvaluator')
    @patch('src.services.evaluator.judge.RiskDirectionEvaluator')
    @patch('src.services.evaluator.judge.CorrectnessEvaluator')
    @patch('src.services.evaluator.judge.HallucinationEvaluator')
    def test_output_schema_validation(
        self, mock_hallucination_class, mock_correctness_class,
        mock_risk_direction_class, mock_cost_extraction_class, mock_risk_impact_class,
        mock_config, sample_query, sample_retrieved_context,
        sample_model_answer, sample_reference_answer
    ):
        """Test that output schema is correct with all required fields"""
        # Setup mocks
        mock_correctness = Mock()
        mock_correctness._construct_prompt.return_value = "test prompt"
        mock_correctness._call_llm.return_value = '{"correctness_binary": true, "reasoning": "Correct"}'
        mock_correctness._parse_json_response.return_value = {
            "correctness_binary": True,
            "reasoning": "Correct"
        }
        mock_correctness_class.return_value = mock_correctness
        
        mock_hallucination = Mock()
        mock_hallucination._construct_prompt.return_value = "test prompt"
        mock_hallucination._call_llm.return_value = '{"hallucination_binary": false, "reasoning": "No hallucination"}'
        mock_hallucination._parse_json_response.return_value = {
            "hallucination_binary": False,
            "reasoning": "No hallucination"
        }
        mock_hallucination_class.return_value = mock_hallucination
        
        # Setup risk direction mock (called when correctness is True)
        mock_risk_direction = Mock()
        mock_risk_direction._construct_prompt.return_value = "test prompt"
        mock_risk_direction._call_llm.return_value = '{"risk_direction": -1, "reasoning": "Care avoidance"}'
        mock_risk_direction._parse_json_response.return_value = {
            "risk_direction": -1,
            "reasoning": "Care avoidance"
        }
        mock_risk_direction_class.return_value = mock_risk_direction
        
        # Setup cost extraction mock
        mock_cost_extractor = Mock()
        mock_cost_extractor.extract_costs.side_effect = [
            {"money": 50.0, "reasoning": "Found $50"},
            {"money": 50.0, "reasoning": "Found $50"}
        ]
        mock_cost_extraction_class.return_value = mock_cost_extractor
        
        # Setup risk impact mock
        mock_risk_impact = Mock()
        mock_risk_impact._construct_prompt.return_value = "test prompt"
        mock_risk_impact._call_llm.return_value = '{"risk_impact": 1.5, "reasoning": "Low impact"}'
        mock_risk_impact._parse_json_response.return_value = {
            "risk_impact": 1.5,
            "reasoning": "Low impact"
        }
        mock_risk_impact_class.return_value = mock_risk_impact
        
        # Call function
        result = evaluate_answer_with_judge(
            query=sample_query,
            retrieved_context=sample_retrieved_context,
            model_answer=sample_model_answer,
            reference_answer=sample_reference_answer,
            config=mock_config
        )
        
        # Verify output schema
        assert isinstance(result, JudgeEvaluationResult)
        assert hasattr(result, 'correctness_binary')
        assert hasattr(result, 'hallucination_binary')
        assert hasattr(result, 'risk_direction')
        assert hasattr(result, 'risk_impact')
        assert hasattr(result, 'reasoning')
        assert hasattr(result, 'failure_mode')
        
        assert isinstance(result.correctness_binary, bool)
        assert isinstance(result.hallucination_binary, bool)
        assert isinstance(result.reasoning, str)
        assert result.reasoning  # Not empty


class TestConstructReasoningTrace:
    """Test reasoning trace construction"""
    
    def test_reasoning_trace_includes_all_parts(self):
        """Test that reasoning trace includes all parts"""
        trace = _construct_reasoning_trace(
            correctness_result=True,
            hallucination_result=False,
            risk_direction=-1,
            risk_impact=1.5,
            correctness_reasoning="Correct answer",
            hallucination_reasoning="No hallucination",
            cost_reasoning="Care avoidance risk",
            impact_reasoning="Low impact"
        )
        
        assert "Correct answer" in trace
        assert "No hallucination" in trace
        assert "Care avoidance risk" in trace
        assert "Low impact" in trace
        assert "True" in trace
        assert "False" in trace
        assert "-1" in trace or "care avoidance" in trace.lower()
        assert "1.5" in trace
    
    def test_reasoning_trace_omits_conditional_parts(self):
        """Test that reasoning trace omits cost/impact when not available"""
        trace = _construct_reasoning_trace(
            correctness_result=False,
            hallucination_result=True,
            risk_direction=None,
            risk_impact=None,
            correctness_reasoning="Incorrect answer",
            hallucination_reasoning="Hallucination detected",
            cost_reasoning=None,
            impact_reasoning=None
        )
        
        assert "Incorrect answer" in trace
        assert "Hallucination detected" in trace
        assert "care avoidance" not in trace.lower()
        assert "unexpected cost" not in trace.lower()
        assert "Risk Impact" not in trace


class TestExtractFailureMode:
    """Test failure mode extraction"""
    
    def test_extract_failure_mode_cost_misstatement(self):
        """Test extraction of cost misstatement failure mode"""
        reasoning = "The model made a cost misstatement about the copay."
        failure_mode = _extract_failure_mode(reasoning)
        assert failure_mode == "Cost Misstatement"
    
    def test_extract_failure_mode_omitted_deductible(self):
        """Test extraction of omitted deductible failure mode"""
        reasoning = "The answer omitted deductible information."
        failure_mode = _extract_failure_mode(reasoning)
        assert failure_mode == "Omitted Deductible"
    
    def test_extract_failure_mode_no_match(self):
        """Test that None is returned when no failure mode matches"""
        reasoning = "The answer is generally correct."
        failure_mode = _extract_failure_mode(reasoning)
        assert failure_mode is None


class TestCallEvaluatorWithReasoning:
    """Test the _call_evaluator_with_reasoning helper function"""
    
    def test_extracts_result_and_reasoning(self, mock_config):
        """Test that function extracts both result and reasoning"""
        evaluator = Mock()
        evaluator._construct_prompt.return_value = "test prompt"
        evaluator._call_llm.return_value = '{"correctness_binary": true, "reasoning": "Correct answer"}'
        evaluator._parse_json_response.return_value = {
            "correctness_binary": True,
            "reasoning": "Correct answer"
        }
        
        result, reasoning = _call_evaluator_with_reasoning(
            evaluator,
            lambda: "test prompt",
            "correctness_binary"
        )
        
        assert result is True
        assert reasoning == "Correct answer"
        assert evaluator._call_llm.called
        assert evaluator._parse_json_response.called
    
    def test_missing_result_field_raises_error(self, mock_config):
        """Test that missing result field raises ValueError"""
        evaluator = Mock()
        evaluator._construct_prompt.return_value = "test prompt"
        evaluator._call_llm.return_value = '{"reasoning": "No result field"}'
        evaluator._parse_json_response.return_value = {
            "reasoning": "No result field"
        }
        
        with pytest.raises(ValueError, match="missing 'correctness_binary' field"):
            _call_evaluator_with_reasoning(
                evaluator,
                lambda: "test prompt",
                "correctness_binary"
            )
    
    def test_default_reasoning_when_missing(self, mock_config):
        """Test that default reasoning is used when missing"""
        evaluator = Mock()
        evaluator._construct_prompt.return_value = "test prompt"
        evaluator._call_llm.return_value = '{"correctness_binary": true}'
        evaluator._parse_json_response.return_value = {
            "correctness_binary": True
        }
        
        result, reasoning = _call_evaluator_with_reasoning(
            evaluator,
            lambda: "test prompt",
            "correctness_binary"
        )
        
        assert result is True
        assert reasoning == "No reasoning provided"


class TestEdgeCases:
    """Test edge cases"""
    
    @patch('src.services.evaluator.judge.CorrectnessEvaluator')
    @patch('src.services.evaluator.judge.HallucinationEvaluator')
    def test_zero_retrieved_chunks_raises_error(
        self, mock_hallucination_class, mock_correctness_class,
        mock_config, sample_query, sample_model_answer, sample_reference_answer
    ):
        """Test that zero retrieved chunks raises ValueError"""
        with pytest.raises(ValueError, match="Retrieved context cannot be empty"):
            evaluate_answer_with_judge(
                query=sample_query,
                retrieved_context=[],
                model_answer=sample_model_answer,
                reference_answer=sample_reference_answer,
                config=mock_config
            )
    
    @patch('src.services.evaluator.judge.CorrectnessEvaluator')
    @patch('src.services.evaluator.judge.HallucinationEvaluator')
    def test_whitespace_only_inputs_raise_error(
        self, mock_hallucination_class, mock_correctness_class,
        mock_config, sample_retrieved_context
    ):
        """Test that whitespace-only inputs raise ValueError"""
        with pytest.raises(ValueError):
            evaluate_answer_with_judge(
                query="   ",
                retrieved_context=sample_retrieved_context,
                model_answer="   ",
                reference_answer="   ",
                config=mock_config
            )


class TestIntegration:
    """Integration tests with mocked LLM nodes"""
    
    @patch('src.services.evaluator.judge.CorrectnessEvaluator')
    @patch('src.services.evaluator.judge.HallucinationEvaluator')
    @patch('src.services.evaluator.judge.RiskDirectionEvaluator')
    @patch('src.services.evaluator.judge.CostExtractionEvaluator')
    @patch('src.services.evaluator.judge.RiskImpactEvaluator')
    def test_full_orchestration_flow(
        self, mock_impact_class, mock_cost_class, mock_direction_class,
        mock_hallucination_class, mock_correctness_class,
        mock_config, sample_query, sample_retrieved_context,
        sample_model_answer, sample_reference_answer
    ):
        """Test full orchestration flow with all nodes"""
        # Setup all mocks
        mock_correctness = Mock()
        mock_correctness._construct_prompt.return_value = "correctness prompt"
        mock_correctness._call_llm.return_value = '{"correctness_binary": true, "reasoning": "Correct"}'
        mock_correctness._parse_json_response.return_value = {
            "correctness_binary": True,
            "reasoning": "Correct"
        }
        mock_correctness_class.return_value = mock_correctness
        
        mock_hallucination = Mock()
        mock_hallucination._construct_prompt.return_value = "hallucination prompt"
        mock_hallucination._call_llm.return_value = '{"hallucination_binary": false, "reasoning": "No hallucination"}'
        mock_hallucination._parse_json_response.return_value = {
            "hallucination_binary": False,
            "reasoning": "No hallucination"
        }
        mock_hallucination_class.return_value = mock_hallucination
        
        mock_direction = Mock()
        mock_direction._construct_prompt.return_value = "direction prompt"
        mock_direction._call_llm.return_value = '{"risk_direction": 1, "reasoning": "Unexpected cost"}'
        mock_direction._parse_json_response.return_value = {
            "risk_direction": 1,
            "reasoning": "Unexpected cost"
        }
        mock_direction_class.return_value = mock_direction
        
        mock_cost = Mock()
        mock_cost.extract_costs.side_effect = [
            {"money": 75.0, "reasoning": "Model cost"},
            {"money": 50.0, "reasoning": "Actual cost"}
        ]
        mock_cost_class.return_value = mock_cost
        
        mock_impact = Mock()
        mock_impact._construct_prompt.return_value = "impact prompt"
        mock_impact._call_llm.return_value = '{"risk_impact": 2.0, "reasoning": "Moderate impact"}'
        mock_impact._parse_json_response.return_value = {
            "risk_impact": 2.0,
            "reasoning": "Moderate impact"
        }
        mock_impact_class.return_value = mock_impact
        
        # Call function
        result = evaluate_answer_with_judge(
            query=sample_query,
            retrieved_context=sample_retrieved_context,
            model_answer=sample_model_answer,
            reference_answer=sample_reference_answer,
            config=mock_config
        )
        
        # Verify all nodes were called in correct order
        assert mock_correctness._call_llm.called
        assert mock_hallucination._call_llm.called
        assert mock_direction._call_llm.called
        assert mock_cost.extract_costs.call_count == 2
        assert mock_impact._call_llm.called
        
        # Verify result
        assert result.correctness_binary is True
        assert result.hallucination_binary is False
        assert result.risk_direction == 1
        assert result.risk_impact == 2.0
        assert len(result.reasoning) > 0

