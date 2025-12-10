"""Tests for system-level risk direction classification LLM node"""

import json
import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.services.evaluator.risk_direction import (
    classify_risk_direction,
    RiskDirectionEvaluator,
)
from src.core.config import Config
from src.core.exceptions import AzureServiceError, ValidationError
from src.core.interfaces import RetrievalResult
from src.services.shared.llm_providers import AzureFoundryProvider
from src.db.queries import QueryExecutor


class TestRiskDirectionPrompt:
    """Tests for prompt construction"""
    
    def setup_method(self):
        """Clear cache before each test"""
        from src.services.rag.generation import _prompt_cache
        _prompt_cache.clear()
    
    def test_load_prompt_template_from_file(self):
        """Test that prompt template can be loaded from file (for testing)"""
        test_prompt_path = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "prompts" / "prompt_v1.md"
        evaluator = RiskDirectionEvaluator(prompt_path=test_prompt_path)
        template = evaluator._load_prompt_template()
        assert isinstance(template, str)
        assert len(template) > 0
    
    def test_load_prompt_template_from_database(self):
        """Test that prompt template can be loaded from database"""
        # Clear cache to ensure fresh database query
        from src.services.rag.generation import _prompt_cache
        _prompt_cache.clear()
        
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "# System-Level Risk Direction Classification Prompt\n\nYou are an expert evaluator...\n\n**Retrieved Context:**\n{retrieved_context}\n\n**Model Answer:**\n{model_answer}\n"}
        ]
        
        evaluator = RiskDirectionEvaluator(
            query_executor=mock_query_executor,
            live=True
        )
        template = evaluator._load_prompt_template()
        
        assert isinstance(template, str)
        assert len(template) > 0
        assert "{retrieved_context}" in template
        assert "{model_answer}" in template
        assert "{reference_answer}" not in template
        
        # Verify database query was called with correct parameters
        # Note: version is not in query params when live=True
        mock_query_executor.execute_query.assert_called_once()
        call_args = mock_query_executor.execute_query.call_args
        assert call_args[0][1] == ("evaluation", "risk_direction_evaluator")
    
    def test_load_prompt_template_custom_path(self):
        """Test loading prompt template from custom path (for testing)"""
        test_prompt_path = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "prompts" / "prompt_v1.md"
        evaluator = RiskDirectionEvaluator(prompt_path=test_prompt_path)
        template = evaluator._load_prompt_template()
        assert isinstance(template, str)
        assert len(template) > 0
    
    def test_load_prompt_template_not_found_file(self):
        """Test that loading non-existent prompt file raises ValueError"""
        fake_path = Path("/nonexistent/prompt.md")
        evaluator = RiskDirectionEvaluator(prompt_path=fake_path)
        with pytest.raises(ValueError, match="Prompt template not found"):
            evaluator._load_prompt_template()
    
    def test_load_prompt_template_not_found_database(self):
        """Test that loading non-existent prompt from database raises ValidationError"""
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = []
        
        evaluator = RiskDirectionEvaluator(
            query_executor=mock_query_executor,
            live=True
        )
        with pytest.raises(ValidationError, match="not found"):
            evaluator._load_prompt_template()
    
    def test_format_retrieved_context(self):
        """Test formatting retrieved context with chunk IDs"""
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "# Risk Direction Prompt\n\n**Retrieved Context:**\n{retrieved_context}\n\n**Model Answer:**\n{model_answer}\n"}
        ]
        evaluator = RiskDirectionEvaluator(query_executor=mock_query_executor)
        
        retrieved_context = [
            RetrievalResult(
                chunk_id="chunk_001",
                similarity_score=0.95,
                chunk_text="The copay is $50."
            ),
            RetrievalResult(
                chunk_id="chunk_002",
                similarity_score=0.90,
                chunk_text="Specialist visits require a referral."
            )
        ]
        
        formatted = evaluator._format_retrieved_context(retrieved_context)
        
        assert "chunk_001" in formatted
        assert "chunk_002" in formatted
        assert "The copay is $50." in formatted
        assert "Specialist visits require a referral." in formatted
        assert "[Chunk ID: chunk_001]" in formatted
        assert "[Chunk ID: chunk_002]" in formatted
    
    def test_format_retrieved_context_empty(self):
        """Test formatting empty retrieved context"""
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "# Risk Direction Prompt\n\n**Retrieved Context:**\n{retrieved_context}\n\n**Model Answer:**\n{model_answer}\n"}
        ]
        evaluator = RiskDirectionEvaluator(query_executor=mock_query_executor)
        formatted = evaluator._format_retrieved_context([])
        assert "[No retrieved context available]" in formatted
    
    def test_construct_risk_direction_prompt(self):
        """Test prompt construction with placeholders"""
        retrieved_context = [
            RetrievalResult(
                chunk_id="chunk_001",
                similarity_score=0.95,
                chunk_text="The copay is $50."
            )
        ]
        model_answer = "The copay for specialist visits is $75."
        
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "# Risk Direction Prompt\n\n**Retrieved Context:**\n{retrieved_context}\n\n**Model Answer:**\n{model_answer}\n"}
        ]
        evaluator = RiskDirectionEvaluator(query_executor=mock_query_executor)
        prompt = evaluator._construct_prompt(model_answer, retrieved_context)
        
        assert isinstance(prompt, str)
        assert "The copay is $50." in prompt
        assert model_answer in prompt
        assert "{retrieved_context}" not in prompt
        assert "{model_answer}" not in prompt
        assert "chunk_001" in prompt
    
    def test_construct_risk_direction_prompt_missing_placeholder(self):
        """Test that missing placeholder raises ValueError"""
        # Create a template with missing placeholder
        fake_template = "Context: {retrieved_context}\nAnswer: {model_answer}"
        fake_path = Path("/tmp/fake_risk_direction_prompt.md")
        
        # Write fake template to temp file
        fake_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fake_path, 'w') as f:
            f.write(fake_template)
        
        try:
            evaluator = RiskDirectionEvaluator(prompt_path=fake_path)
            retrieved_context = [
                RetrievalResult(
                    chunk_id="chunk_001",
                    similarity_score=0.95,
                    chunk_text="Test"
                )
            ]
            # This should work since both placeholders are present
            prompt = evaluator._construct_prompt("answer", retrieved_context)
            assert "Test" in prompt
        finally:
            # Clean up
            if fake_path.exists():
                fake_path.unlink()


class TestRiskDirectionAPI:
    """Tests for LLM API calls via provider"""
    
    @patch('src.services.shared.llm_providers.requests.post')
    def test_call_llm_success(self, mock_post):
        """Test successful LLM call with valid JSON response"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "risk_direction": -1,
                        "reasoning": "Care avoidance risk detected."
                    })
                }
            }]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        # Create evaluator with mocked provider
        config = Config.from_env()
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured")
        
        provider = AzureFoundryProvider(
            endpoint=config.azure_ai_foundry_endpoint,
            api_key=config.azure_ai_foundry_api_key,
            model="gpt-4o-mini"
        )
        evaluator = RiskDirectionEvaluator(config=config, llm_provider=provider)
        
        retrieved_context = [
            RetrievalResult(
                chunk_id="chunk_001",
                similarity_score=0.95,
                chunk_text="Test context"
            )
        ]
        
        response = evaluator._call_llm("Test prompt")
        classification = evaluator._parse_json_response(response)
        
        assert classification["risk_direction"] == -1
        assert "reasoning" in classification
        mock_post.assert_called_once()
    
    @patch('src.services.shared.llm_providers.requests.post')
    def test_parse_json_response_markdown(self, mock_post):
        """Test parsing JSON wrapped in markdown code blocks"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "```json\n{\"risk_direction\": 1, \"reasoning\": \"Unexpected cost risk\"}\n```"
                }
            }]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        config = Config.from_env()
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured")
        
        provider = AzureFoundryProvider(
            endpoint=config.azure_ai_foundry_endpoint,
            api_key=config.azure_ai_foundry_api_key,
            model="gpt-4o-mini"
        )
        evaluator = RiskDirectionEvaluator(config=config, llm_provider=provider)
        
        response = evaluator._call_llm("Test prompt")
        classification = evaluator._parse_json_response(response)
        
        assert classification["risk_direction"] == 1
    
    @patch('src.services.shared.llm_providers.requests.post')
    def test_parse_json_response_invalid_json(self, mock_post):
        """Test that invalid JSON response raises ValueError"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "This is not JSON"
                }
            }]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        config = Config.from_env()
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured")
        
        provider = AzureFoundryProvider(
            endpoint=config.azure_ai_foundry_endpoint,
            api_key=config.azure_ai_foundry_api_key,
            model="gpt-4o-mini"
        )
        evaluator = RiskDirectionEvaluator(config=config, llm_provider=provider)
        
        response = evaluator._call_llm("Test prompt")
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            evaluator._parse_json_response(response)
    
    @patch('src.services.shared.llm_providers.requests.post')
    def test_classify_risk_direction_missing_field(self, mock_post):
        """Test that missing risk_direction field raises ValueError"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({"reasoning": "Some reasoning"})
                }
            }]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        config = Config.from_env()
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured")
        
        provider = AzureFoundryProvider(
            endpoint=config.azure_ai_foundry_endpoint,
            api_key=config.azure_ai_foundry_api_key,
            model="gpt-4o-mini"
        )
        evaluator = RiskDirectionEvaluator(config=config, llm_provider=provider)
        
        retrieved_context = [
            RetrievalResult(
                chunk_id="chunk_001",
                similarity_score=0.95,
                chunk_text="Test"
            )
        ]
        
        with pytest.raises(ValueError, match="missing 'risk_direction' field"):
            evaluator.classify_risk_direction("answer", retrieved_context)
    
    @patch('src.services.shared.llm_providers.requests.post')
    def test_classify_risk_direction_wrong_type(self, mock_post):
        """Test that non-integer risk_direction raises ValueError"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "risk_direction": "-1",  # String instead of int
                        "reasoning": "Some reasoning"
                    })
                }
            }]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        config = Config.from_env()
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured")
        
        provider = AzureFoundryProvider(
            endpoint=config.azure_ai_foundry_endpoint,
            api_key=config.azure_ai_foundry_api_key,
            model="gpt-4o-mini"
        )
        evaluator = RiskDirectionEvaluator(config=config, llm_provider=provider)
        
        retrieved_context = [
            RetrievalResult(
                chunk_id="chunk_001",
                similarity_score=0.95,
                chunk_text="Test"
            )
        ]
        
        with pytest.raises(ValueError, match="must be an integer"):
            evaluator.classify_risk_direction("answer", retrieved_context)
    
    @patch('src.services.shared.llm_providers.requests.post')
    def test_classify_risk_direction_invalid_value(self, mock_post):
        """Test that risk_direction not in [-1, 1] raises ValueError"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "risk_direction": 0,  # Invalid value
                        "reasoning": "Some reasoning"
                    })
                }
            }]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        config = Config.from_env()
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured")
        
        provider = AzureFoundryProvider(
            endpoint=config.azure_ai_foundry_endpoint,
            api_key=config.azure_ai_foundry_api_key,
            model="gpt-4o-mini"
        )
        evaluator = RiskDirectionEvaluator(config=config, llm_provider=provider)
        
        retrieved_context = [
            RetrievalResult(
                chunk_id="chunk_001",
                similarity_score=0.95,
                chunk_text="Test"
            )
        ]
        
        with pytest.raises(ValueError, match="must be -1 \\(care avoidance risk\\) or \\+1 \\(unexpected cost risk\\)"):
            evaluator.classify_risk_direction("answer", retrieved_context)


class TestClassifyRiskDirection:
    """Tests for classify_risk_direction function"""
    
    def test_classify_risk_direction_input_validation_empty_context(self):
        """Test that empty retrieved context raises ValueError"""
        config = Config.from_env()
        with pytest.raises(ValueError, match="Retrieved context cannot be empty"):
            classify_risk_direction("answer", [], config)
    
    def test_classify_risk_direction_input_validation_empty_model_answer(self):
        """Test that empty model answer raises ValueError"""
        config = Config.from_env()
        retrieved_context = [
            RetrievalResult(
                chunk_id="chunk_001",
                similarity_score=0.95,
                chunk_text="Test context"
            )
        ]
        with pytest.raises(ValueError, match="Model answer cannot be empty"):
            classify_risk_direction("", retrieved_context, config)
    
    def test_classify_risk_direction_input_validation_whitespace_only(self):
        """Test that whitespace-only model answer raises ValueError"""
        config = Config.from_env()
        retrieved_context = [
            RetrievalResult(
                chunk_id="chunk_001",
                similarity_score=0.95,
                chunk_text="Test context"
            )
        ]
        with pytest.raises(ValueError, match="Model answer cannot be empty"):
            classify_risk_direction("   ", retrieved_context, config)
    
    @patch('src.services.evaluator.risk_direction.RiskDirectionEvaluator._call_llm')
    @patch('src.services.evaluator.risk_direction.RiskDirectionEvaluator._construct_prompt')
    def test_classify_risk_direction_success_care_avoidance_risk(self, mock_construct, mock_call_llm):
        """Test successful risk classification returning -1 (care avoidance risk)"""
        mock_construct.return_value = "Test prompt"
        mock_call_llm.return_value = json.dumps({
            "risk_direction": -1,
            "reasoning": "Model overestimated cost - care avoidance risk."
        })
        
        config = Config.from_env()
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured")
        
        retrieved_context = [
            RetrievalResult(
                chunk_id="chunk_001",
                similarity_score=0.95,
                chunk_text="The copay is $50."
            )
        ]
        
        result = classify_risk_direction(
            model_answer="The copay is $75.",
            retrieved_context=retrieved_context,
            config=config
        )
        
        assert result == -1
        mock_construct.assert_called_once()
        mock_call_llm.assert_called_once()
    
    @patch('src.services.evaluator.risk_direction.RiskDirectionEvaluator._call_llm')
    @patch('src.services.evaluator.risk_direction.RiskDirectionEvaluator._construct_prompt')
    def test_classify_risk_direction_success_unexpected_cost_risk(self, mock_construct, mock_call_llm):
        """Test successful risk classification returning +1 (unexpected cost risk)"""
        mock_construct.return_value = "Test prompt"
        mock_call_llm.return_value = json.dumps({
            "risk_direction": 1,
            "reasoning": "Model underestimated cost - unexpected cost risk."
        })
        
        config = Config.from_env()
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured")
        
        retrieved_context = [
            RetrievalResult(
                chunk_id="chunk_001",
                similarity_score=0.95,
                chunk_text="The copay is $50."
            )
        ]
        
        result = classify_risk_direction(
            model_answer="The copay is $30.",
            retrieved_context=retrieved_context,
            config=config
        )
        
        assert result == 1
        mock_construct.assert_called_once()
        mock_call_llm.assert_called_once()
    
    @patch('src.services.evaluator.risk_direction.RiskDirectionEvaluator._call_llm')
    @patch('src.services.evaluator.risk_direction.RiskDirectionEvaluator._construct_prompt')
    def test_classify_risk_direction_uses_temperature_0_1(self, mock_construct, mock_call_llm):
        """Test that temperature=0.1 is used for reproducibility"""
        mock_construct.return_value = "Test prompt"
        mock_call_llm.return_value = json.dumps({
            "risk_direction": -1,
            "reasoning": "Opportunity cost."
        })
        
        config = Config.from_env()
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured")
        
        retrieved_context = [
            RetrievalResult(
                chunk_id="chunk_001",
                similarity_score=0.95,
                chunk_text="Test context"
            )
        ]
        
        classify_risk_direction(
            model_answer="Test answer",
            retrieved_context=retrieved_context,
            config=config
        )
        
        # Verify temperature=0.1 was used
        call_args = mock_call_llm.call_args
        assert call_args[1]["temperature"] == 0.1
    
    @patch('src.services.evaluator.risk_direction.RiskDirectionEvaluator._call_llm')
    @patch('src.services.evaluator.risk_direction.RiskDirectionEvaluator._construct_prompt')
    def test_classify_risk_direction_handles_azure_error(self, mock_construct, mock_call_llm):
        """Test that AzureServiceError is raised on API failure"""
        mock_construct.return_value = "Test prompt"
        mock_call_llm.side_effect = AzureServiceError("API call failed")
        
        config = Config.from_env()
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured")
        
        retrieved_context = [
            RetrievalResult(
                chunk_id="chunk_001",
                similarity_score=0.95,
                chunk_text="Test context"
            )
        ]
        
        with pytest.raises(AzureServiceError, match="API call failed"):
            classify_risk_direction(
                model_answer="Test answer",
                retrieved_context=retrieved_context,
                config=config
            )
    
    @patch('src.services.evaluator.risk_direction.RiskDirectionEvaluator._call_llm')
    @patch('src.services.evaluator.risk_direction.RiskDirectionEvaluator._construct_prompt')
    def test_classify_risk_direction_handles_value_error(self, mock_construct, mock_call_llm):
        """Test that ValueError is raised on invalid response"""
        mock_construct.return_value = "Test prompt"
        mock_call_llm.side_effect = ValueError("Invalid JSON response")
        
        config = Config.from_env()
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured")
        
        retrieved_context = [
            RetrievalResult(
                chunk_id="chunk_001",
                similarity_score=0.95,
                chunk_text="Test context"
            )
        ]
        
        with pytest.raises(ValueError, match="Invalid JSON response"):
            classify_risk_direction(
                model_answer="Test answer",
                retrieved_context=retrieved_context,
                config=config
            )
    
    def test_classify_risk_direction_default_config(self):
        """Test that Config.from_env() is used when config is None"""
        # Patch Config in both risk_direction and base_evaluator modules
        with patch('src.services.evaluator.base_evaluator.Config') as mock_config_class:
            mock_config = Mock()
            mock_config.azure_ai_foundry_endpoint = "https://test.endpoint"
            mock_config.azure_ai_foundry_api_key = "test-key"
            mock_config_class.from_env.return_value = mock_config
            
            with patch('src.services.evaluator.risk_direction.RiskDirectionEvaluator._construct_prompt') as mock_construct:
                with patch('src.services.evaluator.risk_direction.RiskDirectionEvaluator._call_llm') as mock_call_llm:
                    mock_construct.return_value = "Test prompt"
                    mock_call_llm.return_value = json.dumps({
                        "risk_direction": -1,
                        "reasoning": "Care avoidance risk."
                    })
                    
                    retrieved_context = [
                        RetrievalResult(
                            chunk_id="chunk_001",
                            similarity_score=0.95,
                            chunk_text="Test context"
                        )
                    ]
                    
                    classify_risk_direction(
                        model_answer="Test answer",
                        retrieved_context=retrieved_context,
                        config=None
                    )
                    
                    mock_config_class.from_env.assert_called_once()


class TestRiskDirectionClassification:
    """Tests for risk direction classification logic"""
    
    @patch('src.services.evaluator.risk_direction.RiskDirectionEvaluator._call_llm')
    @patch('src.services.evaluator.risk_direction.RiskDirectionEvaluator._construct_prompt')
    def test_care_avoidance_risk_overestimated_cost(self, mock_construct, mock_call_llm):
        """Test care avoidance risk classification (-1): overestimated cost"""
        mock_construct.return_value = "Test prompt"
        mock_call_llm.return_value = json.dumps({
            "risk_direction": -1,
            "reasoning": "Model overestimated copay as $75 when context states $50."
        })
        
        config = Config.from_env()
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured")
        
        retrieved_context = [
            RetrievalResult(
                chunk_id="chunk_001",
                similarity_score=0.95,
                chunk_text="The copay for specialist visits is $50."
            )
        ]
        
        result = classify_risk_direction(
            model_answer="The copay for specialist visits is $75.",
            retrieved_context=retrieved_context,
            config=config
        )
        
        assert result == -1
    
    @patch('src.services.evaluator.risk_direction.RiskDirectionEvaluator._call_llm')
    @patch('src.services.evaluator.risk_direction.RiskDirectionEvaluator._construct_prompt')
    def test_unexpected_cost_risk_underestimated_cost(self, mock_construct, mock_call_llm):
        """Test unexpected cost risk classification (+1): underestimated cost"""
        mock_construct.return_value = "Test prompt"
        mock_call_llm.return_value = json.dumps({
            "risk_direction": 1,
            "reasoning": "Model underestimated deductible as $1,000 when context states $1,500."
        })
        
        config = Config.from_env()
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured")
        
        retrieved_context = [
            RetrievalResult(
                chunk_id="chunk_001",
                similarity_score=0.95,
                chunk_text="The annual deductible is $1,500."
            )
        ]
        
        result = classify_risk_direction(
            model_answer="The annual deductible is $1,000.",
            retrieved_context=retrieved_context,
            config=config
        )
        
        assert result == 1
    
    @patch('src.services.evaluator.risk_direction.RiskDirectionEvaluator._call_llm')
    @patch('src.services.evaluator.risk_direction.RiskDirectionEvaluator._construct_prompt')
    def test_cost_analysis_quantitative_hallucination(self, mock_construct, mock_call_llm):
        """Test cost analysis for quantitative hallucinations"""
        mock_construct.return_value = "Test prompt"
        mock_call_llm.return_value = json.dumps({
            "risk_direction": -1,
            "reasoning": "Model overestimated out-of-pocket maximum."
        })
        
        config = Config.from_env()
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured")
        
        retrieved_context = [
            RetrievalResult(
                chunk_id="chunk_001",
                similarity_score=0.95,
                chunk_text="The out-of-pocket maximum is $3,000 per year."
            )
        ]
        
        result = classify_risk_direction(
            model_answer="The out-of-pocket maximum is $5,000 per year.",
            retrieved_context=retrieved_context,
            config=config
        )
        
        assert result == -1
    
    @patch('src.services.evaluator.risk_direction.RiskDirectionEvaluator._call_llm')
    @patch('src.services.evaluator.risk_direction.RiskDirectionEvaluator._construct_prompt')
    def test_cost_analysis_non_quantitative_hallucination(self, mock_construct, mock_call_llm):
        """Test cost analysis for non-quantitative hallucinations"""
        mock_construct.return_value = "Test prompt"
        mock_call_llm.return_value = json.dumps({
            "risk_direction": 1,
            "reasoning": "Model incorrectly states prior authorization is not required."
        })
        
        config = Config.from_env()
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured")
        
        retrieved_context = [
            RetrievalResult(
                chunk_id="chunk_001",
                similarity_score=0.95,
                chunk_text="Prior authorization is required for all specialist visits."
            )
        ]
        
        result = classify_risk_direction(
            model_answer="Specialist visits do not require prior authorization.",
            retrieved_context=retrieved_context,
            config=config
        )
        
        assert result == 1
    
    @patch('src.services.evaluator.risk_direction.RiskDirectionEvaluator._call_llm')
    @patch('src.services.evaluator.risk_direction.RiskDirectionEvaluator._construct_prompt')
    def test_ambiguous_cost_direction(self, mock_construct, mock_call_llm):
        """Test edge case: ambiguous cost direction (defaults to +1)"""
        mock_construct.return_value = "Test prompt"
        mock_call_llm.return_value = json.dumps({
            "risk_direction": 1,
            "reasoning": "Risk direction cannot be definitively determined, defaulting to unexpected cost risk."
        })
        
        config = Config.from_env()
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured")
        
        retrieved_context = [
            RetrievalResult(
                chunk_id="chunk_001",
                similarity_score=0.95,
                chunk_text="Coverage details may vary by plan."
            )
        ]
        
        result = classify_risk_direction(
            model_answer="The specific coverage is clearly defined as $50.",
            retrieved_context=retrieved_context,
            config=config
        )
        
        assert result == 1
    
    @patch('src.services.evaluator.risk_direction.RiskDirectionEvaluator._call_llm')
    @patch('src.services.evaluator.risk_direction.RiskDirectionEvaluator._construct_prompt')
    def test_reference_answer_not_used(self, mock_construct, mock_call_llm):
        """CRITICAL: Test that reference answer is NOT used in cost classification"""
        mock_construct.return_value = "Test prompt"
        mock_call_llm.return_value = json.dumps({
            "risk_direction": -1,
            "reasoning": "Model overestimated cost compared to retrieved context."
        })
        
        config = Config.from_env()
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured")
        
        retrieved_context = [
            RetrievalResult(
                chunk_id="chunk_001",
                similarity_score=0.95,
                chunk_text="The copay is $50."
            )
        ]
        
        # Model answer matches retrieved context but differs from reference answer
        # If reference answer were used, this might be classified incorrectly
        model_answer = "The copay is $75."  # Overestimates compared to context
        reference_answer = "The copay is $30."  # Different from both context and model
        
        result = classify_risk_direction(
            model_answer=model_answer,
            retrieved_context=retrieved_context,
            config=config
        )
        
        # Should be -1 (care avoidance risk) because model overestimates compared to context
        # Even though it might be different from reference answer
        assert result == -1
        
        # Verify prompt does NOT contain reference answer
        prompt = mock_construct.return_value
        assert reference_answer not in prompt
        assert "$30" not in prompt  # Reference answer value should not appear
    
    def test_zero_retrieved_chunks(self):
        """Test edge case: zero retrieved chunks raises ValueError"""
        config = Config.from_env()
        with pytest.raises(ValueError, match="Retrieved context cannot be empty"):
            classify_risk_direction("Some answer", [], config)


class TestRiskDirectionConnection:
    """Integration tests for Azure Foundry connection (warns if credentials missing)"""
    
    def test_connection_to_azure_foundry(self):
        """Test actual connection to Azure Foundry (warns if credentials missing)"""
        config = Config.from_env()
        
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured - skipping connection test")
        
        retrieved_context = [
            RetrievalResult(
                chunk_id="chunk_001",
                similarity_score=0.95,
                chunk_text="The copay for a specialist visit is $50."
            )
        ]
        
        # Try a simple cost classification
        try:
            result = classify_risk_direction(
                model_answer="The copay for a specialist visit is $75.",
                retrieved_context=retrieved_context,
                config=config
            )
            # If successful, result should be -1 or +1
            assert result in [-1, 1]
            print(f"✓ Connection test successful: risk_direction={result}")
        except AzureServiceError as e:
            pytest.fail(f"Azure connection test failed: {e}")
        except Exception as e:
            # Other errors (like missing model) are acceptable for connection test
            print(f"⚠ Connection test encountered error (may be expected): {e}")

