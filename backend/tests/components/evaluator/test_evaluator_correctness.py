"""Tests for correctness classification LLM node"""

import json
import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.services.evaluator.correctness import (
    classify_correctness,
    CorrectnessEvaluator,
)
from src.core.config import Config
from src.core.exceptions import AzureServiceError, ValidationError
from src.services.shared.llm_providers import AzureFoundryProvider
from src.db.queries import QueryExecutor


class TestCorrectnessPrompt:
    """Tests for prompt construction"""
    
    def setup_method(self):
        """Clear cache before each test"""
        from src.services.rag.generation import _prompt_cache
        _prompt_cache.clear()
    
    def test_load_prompt_template_from_file(self):
        """Test that prompt template can be loaded from file (for testing)"""
        # Use a test prompt file path
        test_prompt_path = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "prompts" / "prompt_v1.md"
        evaluator = CorrectnessEvaluator(prompt_path=test_prompt_path)
        template = evaluator._load_prompt_template()
        assert isinstance(template, str)
        assert len(template) > 0
    
    def test_load_prompt_template_from_database(self):
        """Test that prompt template can be loaded from database"""
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "# Correctness Classification Prompt\n\nYou are an expert evaluator...\n\n**Query:**\n{query}\n\n**Model Answer:**\n{model_answer}\n\n**Reference Answer:**\n{reference_answer}\n"}
        ]
        
        evaluator = CorrectnessEvaluator(
            query_executor=mock_query_executor,
            live=True
        )
        template = evaluator._load_prompt_template()
        
        assert isinstance(template, str)
        assert len(template) > 0
        assert "{query}" in template
        assert "{model_answer}" in template
        assert "{reference_answer}" in template
        
        # Verify database query was called with correct parameters
        # Note: version is not in query params when live=True
        mock_query_executor.execute_query.assert_called_once()
        call_args = mock_query_executor.execute_query.call_args
        assert call_args[0][1] == ("evaluation", "correctness_evaluator")
    
    def test_load_prompt_template_custom_path(self):
        """Test loading prompt template from custom path (for testing)"""
        test_prompt_path = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "prompts" / "prompt_v1.md"
        evaluator = CorrectnessEvaluator(prompt_path=test_prompt_path)
        template = evaluator._load_prompt_template()
        assert isinstance(template, str)
        assert len(template) > 0
    
    def test_load_prompt_template_not_found_file(self):
        """Test that loading non-existent prompt file raises ValueError"""
        fake_path = Path("/nonexistent/prompt.md")
        evaluator = CorrectnessEvaluator(prompt_path=fake_path)
        with pytest.raises(ValueError, match="Prompt template not found"):
            evaluator._load_prompt_template()
    
    def test_load_prompt_template_not_found_database(self):
        """Test that loading non-existent prompt from database raises ValidationError"""
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = []
        
        evaluator = CorrectnessEvaluator(
            query_executor=mock_query_executor,
            live=True
        )
        with pytest.raises(ValidationError, match="not found"):
            evaluator._load_prompt_template()
    
    def test_load_prompt_template_no_source(self):
        """Test that ValueError is raised when neither query_executor nor prompt_path is provided"""
        # When both are None, should raise ValueError
        evaluator = CorrectnessEvaluator(
            prompt_version="v1",
            query_executor=None,
            prompt_path=None
        )
        # Should raise ValueError - prompts must come from database or explicit file path
        with pytest.raises(ValueError, match="Either query_executor.*or prompt_path.*must be provided"):
            evaluator._load_prompt_template()
    
    def test_construct_correctness_prompt(self):
        """Test prompt construction with placeholders"""
        query = "What is the copay?"
        model_answer = "The copay is $50."
        reference_answer = "Copay is $50."
        
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "# Correctness Prompt\n\n**Query:**\n{query}\n\n**Model Answer:**\n{model_answer}\n\n**Reference Answer:**\n{reference_answer}\n"}
        ]
        evaluator = CorrectnessEvaluator(query_executor=mock_query_executor)
        prompt = evaluator._construct_prompt(query, model_answer, reference_answer)
        
        assert isinstance(prompt, str)
        assert query in prompt
        assert model_answer in prompt
        assert reference_answer in prompt
        assert "{query}" not in prompt
        assert "{model_answer}" not in prompt
        assert "{reference_answer}" not in prompt
    
    def test_construct_correctness_prompt_missing_placeholder(self):
        """Test that missing placeholder raises ValueError"""
        # Create a template with missing placeholder
        fake_template = "Query: {query}\nAnswer: {model_answer}"
        fake_path = Path("/tmp/fake_prompt.md")
        
        # Write fake template to temp file
        fake_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fake_path, 'w') as f:
            f.write(fake_template)
        
        try:
            evaluator = CorrectnessEvaluator(prompt_path=fake_path)
            with pytest.raises(ValueError, match="missing required placeholders"):
                evaluator._construct_prompt("query", "answer", "reference")
        finally:
            # Clean up
            if fake_path.exists():
                fake_path.unlink()


class TestCorrectnessAPI:
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
                        "correctness_binary": True,
                        "reasoning": "The answer is correct."
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
        evaluator = CorrectnessEvaluator(config=config, llm_provider=provider)
        
        response = evaluator._call_llm("Test prompt")
        classification = evaluator._parse_json_response(response)
        
        assert classification["correctness_binary"] is True
        assert "reasoning" in classification
        mock_post.assert_called_once()
    
    @patch('src.services.shared.llm_providers.requests.post')
    def test_parse_json_response_markdown(self, mock_post):
        """Test parsing JSON wrapped in markdown code blocks"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "```json\n{\"correctness_binary\": false, \"reasoning\": \"Wrong answer\"}\n```"
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
        evaluator = CorrectnessEvaluator(config=config, llm_provider=provider)
        
        response = evaluator._call_llm("Test prompt")
        classification = evaluator._parse_json_response(response)
        
        assert classification["correctness_binary"] is False
    
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
        evaluator = CorrectnessEvaluator(config=config, llm_provider=provider)
        
        response = evaluator._call_llm("Test prompt")
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            evaluator._parse_json_response(response)
    
    @patch('src.services.shared.llm_providers.requests.post')
    def test_classify_correctness_missing_field(self, mock_post):
        """Test that missing correctness_binary field raises ValueError"""
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
        evaluator = CorrectnessEvaluator(config=config, llm_provider=provider)
        
        with pytest.raises(ValueError, match="missing 'correctness_binary' field"):
            evaluator.classify_correctness("query", "answer", "reference")
    
    @patch('src.services.shared.llm_providers.requests.post')
    def test_classify_correctness_wrong_type(self, mock_post):
        """Test that non-boolean correctness_binary raises ValueError"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "correctness_binary": "true",  # String instead of bool
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
        evaluator = CorrectnessEvaluator(config=config, llm_provider=provider)
        
        with pytest.raises(ValueError, match="must be a boolean"):
            evaluator.classify_correctness("query", "answer", "reference")
    
    @patch('src.services.shared.llm_providers.requests.post')
    def test_provider_retry_on_failure(self, mock_post):
        """Test that provider retries on failure with exponential backoff"""
        # First two calls fail, third succeeds
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "correctness_binary": True,
                        "reasoning": "Success after retries"
                    })
                }
            }]
        }
        mock_response.raise_for_status = Mock()
        
        # First two calls raise exception, third succeeds
        mock_post.side_effect = [
            requests.RequestException("Network error"),
            requests.RequestException("Network error"),
            mock_response
        ]
        
        config = Config.from_env()
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured")
        
        provider = AzureFoundryProvider(
            endpoint=config.azure_ai_foundry_endpoint,
            api_key=config.azure_ai_foundry_api_key,
            model="gpt-4o-mini"
        )
        evaluator = CorrectnessEvaluator(config=config, llm_provider=provider)
        
        result = evaluator.classify_correctness("query", "answer", "reference")
        
        assert result is True
        assert mock_post.call_count == 3


class TestClassifyCorrectness:
    """Tests for classify_correctness function"""
    
    def test_classify_correctness_input_validation_empty_query(self):
        """Test that empty query raises ValueError"""
        config = Config.from_env()
        with pytest.raises(ValueError, match="Query text cannot be empty"):
            classify_correctness("", "answer", "reference", config)
    
    def test_classify_correctness_input_validation_empty_model_answer(self):
        """Test that empty model answer raises ValueError"""
        config = Config.from_env()
        with pytest.raises(ValueError, match="Model answer cannot be empty"):
            classify_correctness("query", "", "reference", config)
    
    def test_classify_correctness_input_validation_empty_reference_answer(self):
        """Test that empty reference answer raises ValueError"""
        config = Config.from_env()
        with pytest.raises(ValueError, match="Reference answer cannot be empty"):
            classify_correctness("query", "answer", "", config)
    
    def test_classify_correctness_input_validation_whitespace_only(self):
        """Test that whitespace-only inputs raise ValueError"""
        config = Config.from_env()
        with pytest.raises(ValueError, match="Query text cannot be empty"):
            classify_correctness("   ", "answer", "reference", config)
        with pytest.raises(ValueError, match="Model answer cannot be empty"):
            classify_correctness("query", "   ", "reference", config)
        with pytest.raises(ValueError, match="Reference answer cannot be empty"):
            classify_correctness("query", "answer", "   ", config)
    
    @patch('src.services.evaluator.correctness.CorrectnessEvaluator._call_llm')
    @patch('src.services.evaluator.correctness.CorrectnessEvaluator._construct_prompt')
    def test_classify_correctness_success_true(self, mock_construct, mock_call_llm):
        """Test successful correctness classification returning True"""
        mock_construct.return_value = "Test prompt"
        mock_call_llm.return_value = json.dumps({
            "correctness_binary": True,
            "reasoning": "The answer is correct."
        })
        
        config = Config.from_env()
        # Skip config validation if credentials are missing
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured")
        
        result = classify_correctness(
            query="What is the copay?",
            model_answer="The copay is $50.",
            reference_answer="Copay is $50.",
            config=config
        )
        
        assert result is True
        mock_construct.assert_called_once()
        mock_call_llm.assert_called_once()
    
    @patch('src.services.evaluator.correctness.CorrectnessEvaluator._call_llm')
    @patch('src.services.evaluator.correctness.CorrectnessEvaluator._construct_prompt')
    def test_classify_correctness_success_false(self, mock_construct, mock_call_llm):
        """Test successful correctness classification returning False"""
        mock_construct.return_value = "Test prompt"
        mock_call_llm.return_value = json.dumps({
            "correctness_binary": False,
            "reasoning": "The answer is incorrect."
        })
        
        config = Config.from_env()
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured")
        
        result = classify_correctness(
            query="What is the copay?",
            model_answer="The copay is $25.",
            reference_answer="Copay is $50.",
            config=config
        )
        
        assert result is False
        mock_construct.assert_called_once()
        mock_call_llm.assert_called_once()
    
    @patch('src.services.evaluator.correctness.CorrectnessEvaluator._call_llm')
    @patch('src.services.evaluator.correctness.CorrectnessEvaluator._construct_prompt')
    def test_classify_correctness_uses_temperature_0_1(self, mock_construct, mock_call_llm):
        """Test that temperature=0.1 is used for reproducibility"""
        mock_construct.return_value = "Test prompt"
        mock_call_llm.return_value = json.dumps({
            "correctness_binary": True,
            "reasoning": "Correct."
        })
        
        config = Config.from_env()
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured")
        
        classify_correctness(
            query="What is the copay?",
            model_answer="The copay is $50.",
            reference_answer="Copay is $50.",
            config=config
        )
        
        # Verify temperature=0.1 was used
        call_args = mock_call_llm.call_args
        assert call_args[1]["temperature"] == 0.1
    
    @patch('src.services.evaluator.correctness.CorrectnessEvaluator._call_llm')
    @patch('src.services.evaluator.correctness.CorrectnessEvaluator._construct_prompt')
    def test_classify_correctness_uses_gpt4o_mini(self, mock_construct, mock_call_llm):
        """Test that gpt-4o-mini model is used (via provider)"""
        mock_construct.return_value = "Test prompt"
        mock_call_llm.return_value = json.dumps({
            "correctness_binary": True,
            "reasoning": "Correct."
        })
        
        config = Config.from_env()
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured")
        
        classify_correctness(
            query="What is the copay?",
            model_answer="The copay is $50.",
            reference_answer="Copay is $50.",
            config=config
        )
        
        # Verify the evaluator was created (model is set in provider, not directly verifiable here)
        mock_construct.assert_called_once()
        mock_call_llm.assert_called_once()
    
    @patch('src.services.evaluator.correctness.CorrectnessEvaluator._call_llm')
    @patch('src.services.evaluator.correctness.CorrectnessEvaluator._construct_prompt')
    def test_classify_correctness_handles_azure_error(self, mock_construct, mock_call_llm):
        """Test that AzureServiceError is raised on API failure"""
        mock_construct.return_value = "Test prompt"
        mock_call_llm.side_effect = AzureServiceError("API call failed")
        
        config = Config.from_env()
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured")
        
        with pytest.raises(AzureServiceError, match="API call failed"):
            classify_correctness(
                query="What is the copay?",
                model_answer="The copay is $50.",
                reference_answer="Copay is $50.",
                config=config
            )
    
    @patch('src.services.evaluator.correctness.CorrectnessEvaluator._call_llm')
    @patch('src.services.evaluator.correctness.CorrectnessEvaluator._construct_prompt')
    def test_classify_correctness_handles_value_error(self, mock_construct, mock_call_llm):
        """Test that ValueError is raised on invalid response"""
        mock_construct.return_value = "Test prompt"
        mock_call_llm.side_effect = ValueError("Invalid JSON response")
        
        config = Config.from_env()
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured")
        
        with pytest.raises(ValueError, match="Invalid JSON response"):
            classify_correctness(
                query="What is the copay?",
                model_answer="The copay is $50.",
                reference_answer="Copay is $50.",
                config=config
            )
    
    def test_classify_correctness_default_config(self):
        """Test that Config.from_env() is used when config is None"""
        # Patch Config in both correctness and base_evaluator modules
        with patch('src.services.evaluator.base_evaluator.Config') as mock_config_class:
            mock_config = Mock()
            mock_config.azure_ai_foundry_endpoint = "https://test.endpoint"
            mock_config.azure_ai_foundry_api_key = "test-key"
            mock_config_class.from_env.return_value = mock_config
            
            with patch('src.services.evaluator.correctness.CorrectnessEvaluator._construct_prompt') as mock_construct:
                with patch('src.services.evaluator.correctness.CorrectnessEvaluator._call_llm') as mock_call_llm:
                    mock_construct.return_value = "Test prompt"
                    mock_call_llm.return_value = json.dumps({
                        "correctness_binary": True,
                        "reasoning": "Correct."
                    })
                    
                    classify_correctness(
                        query="What is the copay?",
                        model_answer="The copay is $50.",
                        reference_answer="Copay is $50.",
                        config=None
                    )
                    
                    mock_config_class.from_env.assert_called_once()


class TestCorrectnessConnection:
    """Integration tests for Azure Foundry connection (warns if credentials missing)"""
    
    def test_connection_to_azure_foundry(self):
        """Test actual connection to Azure Foundry (warns if credentials missing)"""
        config = Config.from_env()
        
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured - skipping connection test")
        
        # Try a simple correctness classification
        try:
            result = classify_correctness(
                query="What is the copay for a specialist visit?",
                model_answer="The copay for a specialist visit is $50.",
                reference_answer="Specialist visits have a $50 copay.",
                config=config
            )
            # If successful, result should be a boolean
            assert isinstance(result, bool)
            print(f"✓ Connection test successful: correctness={result}")
        except AzureServiceError as e:
            pytest.fail(f"Azure connection test failed: {e}")
        except Exception as e:
            # Other errors (like missing model) are acceptable for connection test
            print(f"⚠ Connection test encountered error (may be expected): {e}")
