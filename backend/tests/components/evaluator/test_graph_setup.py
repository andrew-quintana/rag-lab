"""Tests for LangGraph infrastructure setup"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List

from rag_eval.services.evaluator.graph_base import (
    JudgeEvaluationState,
    validate_initial_state,
    get_config_from_state,
)
from tests.components.evaluator.test_graph import (
    create_test_graph,
    run_test_evaluation,
    correctness_node,
    hallucination_node,
    should_continue,
)
from rag_eval.core.config import Config
from rag_eval.core.interfaces import RetrievalResult


class TestLangGraphImport:
    """Tests for LangGraph import and basic functionality"""
    
    def test_langgraph_importable(self):
        """Test that LangGraph can be imported"""
        from langgraph.graph import StateGraph, END
        assert StateGraph is not None
        assert END is not None
    
    def test_langchain_core_importable(self):
        """Test that langchain-core can be imported"""
        import langchain_core
        assert langchain_core is not None


class TestJudgeEvaluationState:
    """Tests for JudgeEvaluationState TypedDict"""
    
    def test_state_creation(self):
        """Test that state can be created with required fields"""
        state: JudgeEvaluationState = {
            "query": "What is the copay?",
            "retrieved_context": [],
            "model_answer": "The copay is $50.",
            "reference_answer": "Copay is $50.",
            "correctness_binary": None,
            "hallucination_binary": None,
            "hallucination_cost": None,
            "hallucination_impact": None,
            "reasoning": [],
            "config": None,
        }
        assert state["query"] == "What is the copay?"
        assert isinstance(state["retrieved_context"], list)
    
    def test_state_with_optional_fields(self):
        """Test that state can include optional fields"""
        state: JudgeEvaluationState = {
            "query": "What is the copay?",
            "retrieved_context": [],
            "model_answer": "The copay is $50.",
            "reference_answer": "Copay is $50.",
            "correctness_binary": True,
            "hallucination_binary": False,
            "hallucination_cost": None,
            "hallucination_impact": None,
            "reasoning": ["Test reasoning"],
            "config": None,
        }
        assert state["correctness_binary"] is True
        assert state["hallucination_binary"] is False
        assert len(state["reasoning"]) == 1


class TestStateValidation:
    """Tests for state validation functions"""
    
    def test_validate_initial_state_success(self):
        """Test successful state validation"""
        state: JudgeEvaluationState = {
            "query": "What is the copay?",
            "retrieved_context": [],
            "model_answer": "The copay is $50.",
            "reference_answer": "Copay is $50.",
            "correctness_binary": None,
            "hallucination_binary": None,
            "hallucination_cost": None,
            "hallucination_impact": None,
            "reasoning": [],
            "config": None,
        }
        # Should not raise
        validate_initial_state(state)
    
    def test_validate_initial_state_missing_field(self):
        """Test that missing required field raises ValueError"""
        state: JudgeEvaluationState = {
            "query": "What is the copay?",
            "retrieved_context": [],
            "model_answer": "The copay is $50.",
            # Missing reference_answer
            "correctness_binary": None,
            "hallucination_binary": None,
            "hallucination_cost": None,
            "hallucination_impact": None,
            "reasoning": [],
            "config": None,
        }
        with pytest.raises(ValueError, match="Missing required fields"):
            validate_initial_state(state)
    
    def test_validate_initial_state_empty_query(self):
        """Test that empty query raises ValueError"""
        state: JudgeEvaluationState = {
            "query": "",
            "retrieved_context": [],
            "model_answer": "The copay is $50.",
            "reference_answer": "Copay is $50.",
            "correctness_binary": None,
            "hallucination_binary": None,
            "hallucination_cost": None,
            "hallucination_impact": None,
            "reasoning": [],
            "config": None,
        }
        with pytest.raises(ValueError, match="Query cannot be empty"):
            validate_initial_state(state)
    
    def test_validate_initial_state_empty_model_answer(self):
        """Test that empty model_answer raises ValueError"""
        state: JudgeEvaluationState = {
            "query": "What is the copay?",
            "retrieved_context": [],
            "model_answer": "",
            "reference_answer": "Copay is $50.",
            "correctness_binary": None,
            "hallucination_binary": None,
            "hallucination_cost": None,
            "hallucination_impact": None,
            "reasoning": [],
            "config": None,
        }
        with pytest.raises(ValueError, match="Model answer cannot be empty"):
            validate_initial_state(state)
    
    def test_validate_initial_state_whitespace_only(self):
        """Test that whitespace-only fields raise ValueError"""
        state: JudgeEvaluationState = {
            "query": "   ",
            "retrieved_context": [],
            "model_answer": "The copay is $50.",
            "reference_answer": "Copay is $50.",
            "correctness_binary": None,
            "hallucination_binary": None,
            "hallucination_cost": None,
            "hallucination_impact": None,
            "reasoning": [],
            "config": None,
        }
        with pytest.raises(ValueError, match="Query cannot be empty"):
            validate_initial_state(state)
    
    def test_get_config_from_state_with_config(self):
        """Test getting config from state when present"""
        config = Config.from_env()
        state: JudgeEvaluationState = {
            "query": "test",
            "retrieved_context": [],
            "model_answer": "test",
            "reference_answer": "test",
            "correctness_binary": None,
            "hallucination_binary": None,
            "hallucination_cost": None,
            "hallucination_impact": None,
            "reasoning": [],
            "config": config,
        }
        result = get_config_from_state(state)
        assert result is config
    
    def test_get_config_from_state_without_config(self):
        """Test getting config from state when not present (defaults to from_env)"""
        state: JudgeEvaluationState = {
            "query": "test",
            "retrieved_context": [],
            "model_answer": "test",
            "reference_answer": "test",
            "correctness_binary": None,
            "hallucination_binary": None,
            "hallucination_cost": None,
            "hallucination_impact": None,
            "reasoning": [],
            "config": None,
        }
        result = get_config_from_state(state)
        assert isinstance(result, Config)


class TestGraphConstruction:
    """Tests for graph construction"""
    
    def test_create_test_graph(self):
        """Test that test graph can be created"""
        graph = create_test_graph()
        assert graph is not None
    
    def test_graph_has_nodes(self):
        """Test that graph has expected nodes"""
        graph = create_test_graph()
        # Graph is compiled, so we can't directly inspect nodes
        # But we can verify it's callable
        assert callable(graph.invoke)


class TestNodeFunctions:
    """Tests for individual node functions"""
    
    @patch('tests.components.evaluator.test_graph.CorrectnessEvaluator')
    def test_correctness_node(self, mock_evaluator_class):
        """Test correctness node function"""
        # Setup mock
        mock_evaluator = Mock()
        mock_evaluator.classify_correctness.return_value = True
        mock_evaluator_class.return_value = mock_evaluator
        
        state: JudgeEvaluationState = {
            "query": "What is the copay?",
            "retrieved_context": [],
            "model_answer": "The copay is $50.",
            "reference_answer": "Copay is $50.",
            "correctness_binary": None,
            "hallucination_binary": None,
            "hallucination_cost": None,
            "hallucination_impact": None,
            "reasoning": [],
            "config": None,
        }
        
        result = correctness_node(state)
        
        assert result["correctness_binary"] is True
        assert len(result["reasoning"]) == 1
        assert "Correctness classification: True" in result["reasoning"][0]
        mock_evaluator.classify_correctness.assert_called_once()
    
    @patch('tests.components.evaluator.test_graph.HallucinationEvaluator')
    def test_hallucination_node(self, mock_evaluator_class):
        """Test hallucination node function"""
        # Setup mock
        mock_evaluator = Mock()
        mock_evaluator.classify_hallucination.return_value = False
        mock_evaluator_class.return_value = mock_evaluator
        
        retrieval_result = RetrievalResult(
            chunk_id="chunk1",
            similarity_score=0.9,
            chunk_text="Test chunk text"
        )
        
        state: JudgeEvaluationState = {
            "query": "What is the copay?",
            "retrieved_context": [retrieval_result],
            "model_answer": "The copay is $50.",
            "reference_answer": "Copay is $50.",
            "correctness_binary": True,
            "hallucination_binary": None,
            "hallucination_cost": None,
            "hallucination_impact": None,
            "reasoning": ["Previous reasoning"],
            "config": None,
        }
        
        result = hallucination_node(state)
        
        assert result["hallucination_binary"] is False
        assert len(result["reasoning"]) == 2
        assert "Hallucination classification: False" in result["reasoning"][1]
        mock_evaluator.classify_hallucination.assert_called_once()
    
    def test_should_continue_always_ends(self):
        """Test that should_continue always returns 'end' for test graph"""
        state: JudgeEvaluationState = {
            "query": "test",
            "retrieved_context": [],
            "model_answer": "test",
            "reference_answer": "test",
            "correctness_binary": True,
            "hallucination_binary": False,
            "hallucination_cost": None,
            "hallucination_impact": None,
            "reasoning": [],
            "config": None,
        }
        result = should_continue(state)
        assert result == "end"


class TestGraphExecution:
    """Tests for graph execution"""
    
    @patch('tests.components.evaluator.test_graph.CorrectnessEvaluator')
    @patch('tests.components.evaluator.test_graph.HallucinationEvaluator')
    def test_run_test_evaluation_success(self, mock_hallucination_class, mock_correctness_class):
        """Test running test evaluation with mocked evaluators"""
        # Setup mocks
        mock_correctness = Mock()
        mock_correctness.classify_correctness.return_value = True
        mock_correctness_class.return_value = mock_correctness
        
        mock_hallucination = Mock()
        mock_hallucination.classify_hallucination.return_value = False
        mock_hallucination_class.return_value = mock_hallucination
        
        retrieval_result = RetrievalResult(
            chunk_id="chunk1",
            similarity_score=0.9,
            chunk_text="Test chunk text"
        )
        
        final_state = run_test_evaluation(
            query="What is the copay?",
            retrieved_context=[retrieval_result],
            model_answer="The copay is $50.",
            reference_answer="Copay is $50.",
            config=None
        )
        
        assert final_state["correctness_binary"] is True
        assert final_state["hallucination_binary"] is False
        assert len(final_state["reasoning"]) == 2
        assert "Correctness classification: True" in final_state["reasoning"][0]
        assert "Hallucination classification: False" in final_state["reasoning"][1]
    
    @patch('tests.components.evaluator.test_graph.CorrectnessEvaluator')
    @patch('tests.components.evaluator.test_graph.HallucinationEvaluator')
    def test_run_test_evaluation_with_config(self, mock_hallucination_class, mock_correctness_class):
        """Test running test evaluation with explicit config"""
        # Setup mocks
        mock_correctness = Mock()
        mock_correctness.classify_correctness.return_value = True
        mock_correctness_class.return_value = mock_correctness
        
        mock_hallucination = Mock()
        mock_hallucination.classify_hallucination.return_value = False
        mock_hallucination_class.return_value = mock_hallucination
        
        config = Config.from_env()
        retrieval_result = RetrievalResult(
            chunk_id="chunk1",
            similarity_score=0.9,
            chunk_text="Test chunk text"
        )
        
        final_state = run_test_evaluation(
            query="What is the copay?",
            retrieved_context=[retrieval_result],
            model_answer="The copay is $50.",
            reference_answer="Copay is $50.",
            config=config
        )
        
        assert final_state["config"] is config
        assert final_state["correctness_binary"] is True
    
    def test_run_test_evaluation_invalid_state(self):
        """Test that invalid initial state raises ValueError"""
        with pytest.raises(ValueError):
            run_test_evaluation(
                query="",  # Empty query
                retrieved_context=[],
                model_answer="test",
                reference_answer="test",
                config=None
            )
    
    @patch('tests.components.evaluator.test_graph.CorrectnessEvaluator')
    def test_graph_execution_order(self, mock_correctness_class):
        """Test that graph executes nodes in correct order"""
        call_order = []
        
        def track_correctness(*args, **kwargs):
            call_order.append("correctness")
            return True
        
        mock_correctness = Mock()
        mock_correctness.classify_correctness.side_effect = track_correctness
        mock_correctness_class.return_value = mock_correctness
        
        with patch('tests.components.evaluator.test_graph.HallucinationEvaluator') as mock_hallucination_class:
            def track_hallucination(*args, **kwargs):
                call_order.append("hallucination")
                return False
            
            mock_hallucination = Mock()
            mock_hallucination.classify_hallucination.side_effect = track_hallucination
            mock_hallucination_class.return_value = mock_hallucination
            
            retrieval_result = RetrievalResult(
                chunk_id="chunk1",
                similarity_score=0.9,
                chunk_text="Test chunk text"
            )
            
            run_test_evaluation(
                query="What is the copay?",
                retrieved_context=[retrieval_result],
                model_answer="The copay is $50.",
                reference_answer="Copay is $50.",
                config=None
            )
            
            assert call_order == ["correctness", "hallucination"]


class TestErrorHandling:
    """Tests for error handling in graph execution"""
    
    @patch('tests.components.evaluator.test_graph.CorrectnessEvaluator')
    def test_correctness_node_error_handling(self, mock_correctness_class):
        """Test that errors in correctness node are propagated"""
        mock_correctness = Mock()
        mock_correctness.classify_correctness.side_effect = ValueError("Test error")
        mock_correctness_class.return_value = mock_correctness
        
        state: JudgeEvaluationState = {
            "query": "What is the copay?",
            "retrieved_context": [],
            "model_answer": "The copay is $50.",
            "reference_answer": "Copay is $50.",
            "correctness_binary": None,
            "hallucination_binary": None,
            "hallucination_cost": None,
            "hallucination_impact": None,
            "reasoning": [],
            "config": None,
        }
        
        with pytest.raises(ValueError, match="Test error"):
            correctness_node(state)
    
    @patch('tests.components.evaluator.test_graph.CorrectnessEvaluator')
    @patch('tests.components.evaluator.test_graph.HallucinationEvaluator')
    def test_graph_execution_with_error(self, mock_hallucination_class, mock_correctness_class):
        """Test that graph execution handles errors appropriately"""
        mock_correctness = Mock()
        mock_correctness.classify_correctness.side_effect = ValueError("Test error")
        mock_correctness_class.return_value = mock_correctness
        
        retrieval_result = RetrievalResult(
            chunk_id="chunk1",
            similarity_score=0.9,
            chunk_text="Test chunk text"
        )
        
        with pytest.raises(ValueError, match="Test error"):
            run_test_evaluation(
                query="What is the copay?",
                retrieved_context=[retrieval_result],
                model_answer="The copay is $50.",
                reference_answer="Copay is $50.",
                config=None
            )

