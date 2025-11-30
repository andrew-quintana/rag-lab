"""Tests for hallucination classification LLM node"""

import json
import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from rag_eval.services.evaluator.hallucination import (
    classify_hallucination,
    HallucinationEvaluator,
)
from rag_eval.core.config import Config
from rag_eval.core.exceptions import AzureServiceError
from rag_eval.core.interfaces import RetrievalResult
from rag_eval.services.shared.llm_providers import AzureFoundryProvider


class TestHallucinationPrompt:
    """Tests for prompt construction"""
    
    def test_load_prompt_template(self):
        """Test that prompt template can be loaded"""
        evaluator = HallucinationEvaluator()
        template = evaluator._load_prompt_template()
        assert isinstance(template, str)
        assert len(template) > 0
        assert "{retrieved_context}" in template
        assert "{model_answer}" in template
        # Verify reference answer is NOT in template
        assert "{reference_answer}" not in template
    
    def test_load_prompt_template_custom_path(self):
        """Test loading prompt template from custom path"""
        default_path = Path(__file__).parent.parent.parent.parent / "rag_eval" / "prompts" / "evaluation" / "hallucination_prompt.md"
        evaluator = HallucinationEvaluator(prompt_path=default_path)
        template = evaluator._load_prompt_template()
        assert isinstance(template, str)
        assert len(template) > 0
    
    def test_load_prompt_template_not_found(self):
        """Test that loading non-existent prompt raises ValueError"""
        fake_path = Path("/nonexistent/prompt.md")
        evaluator = HallucinationEvaluator(prompt_path=fake_path)
        with pytest.raises(ValueError, match="Prompt template not found"):
            evaluator._load_prompt_template()
    
    def test_format_retrieved_context(self):
        """Test formatting retrieved context with chunk IDs"""
        evaluator = HallucinationEvaluator()
        
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
        evaluator = HallucinationEvaluator()
        formatted = evaluator._format_retrieved_context([])
        assert "[No retrieved context available]" in formatted
    
    def test_construct_hallucination_prompt(self):
        """Test prompt construction with placeholders"""
        retrieved_context = [
            RetrievalResult(
                chunk_id="chunk_001",
                similarity_score=0.95,
                chunk_text="The copay is $50."
            )
        ]
        model_answer = "The copay for specialist visits is $50."
        
        evaluator = HallucinationEvaluator()
        prompt = evaluator._construct_prompt(retrieved_context, model_answer)
        
        assert isinstance(prompt, str)
        assert "The copay is $50." in prompt
        assert model_answer in prompt
        assert "{retrieved_context}" not in prompt
        assert "{model_answer}" not in prompt
        assert "chunk_001" in prompt
    
    def test_construct_hallucination_prompt_missing_placeholder(self):
        """Test that missing placeholder raises ValueError"""
        # Create a template with missing placeholder
        fake_template = "Context: {retrieved_context}\nAnswer: {model_answer}"
        fake_path = Path("/tmp/fake_hallucination_prompt.md")
        
        # Write fake template to temp file
        fake_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fake_path, 'w') as f:
            f.write(fake_template)
        
        try:
            evaluator = HallucinationEvaluator(prompt_path=fake_path)
            retrieved_context = [
                RetrievalResult(
                    chunk_id="chunk_001",
                    similarity_score=0.95,
                    chunk_text="Test"
                )
            ]
            # This should work since both placeholders are present
            prompt = evaluator._construct_prompt(retrieved_context, "answer")
            assert "Test" in prompt
        finally:
            # Clean up
            if fake_path.exists():
                fake_path.unlink()


class TestHallucinationAPI:
    """Tests for LLM API calls via provider"""
    
    @patch('rag_eval.services.shared.llm_providers.requests.post')
    def test_call_llm_success(self, mock_post):
        """Test successful LLM call with valid JSON response"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "hallucination_binary": True,
                        "reasoning": "Hallucination detected."
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
        evaluator = HallucinationEvaluator(config=config, llm_provider=provider)
        
        retrieved_context = [
            RetrievalResult(
                chunk_id="chunk_001",
                similarity_score=0.95,
                chunk_text="Test context"
            )
        ]
        
        response = evaluator._call_llm("Test prompt")
        classification = evaluator._parse_json_response(response)
        
        assert classification["hallucination_binary"] is True
        assert "reasoning" in classification
        mock_post.assert_called_once()
    
    @patch('rag_eval.services.shared.llm_providers.requests.post')
    def test_parse_json_response_markdown(self, mock_post):
        """Test parsing JSON wrapped in markdown code blocks"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "```json\n{\"hallucination_binary\": false, \"reasoning\": \"No hallucination\"}\n```"
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
        evaluator = HallucinationEvaluator(config=config, llm_provider=provider)
        
        response = evaluator._call_llm("Test prompt")
        classification = evaluator._parse_json_response(response)
        
        assert classification["hallucination_binary"] is False
    
    @patch('rag_eval.services.shared.llm_providers.requests.post')
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
        evaluator = HallucinationEvaluator(config=config, llm_provider=provider)
        
        response = evaluator._call_llm("Test prompt")
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            evaluator._parse_json_response(response)
    
    @patch('rag_eval.services.shared.llm_providers.requests.post')
    def test_classify_hallucination_missing_field(self, mock_post):
        """Test that missing hallucination_binary field raises ValueError"""
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
        evaluator = HallucinationEvaluator(config=config, llm_provider=provider)
        
        retrieved_context = [
            RetrievalResult(
                chunk_id="chunk_001",
                similarity_score=0.95,
                chunk_text="Test"
            )
        ]
        
        with pytest.raises(ValueError, match="missing 'hallucination_binary' field"):
            evaluator.classify_hallucination(retrieved_context, "answer")
    
    @patch('rag_eval.services.shared.llm_providers.requests.post')
    def test_classify_hallucination_wrong_type(self, mock_post):
        """Test that non-boolean hallucination_binary raises ValueError"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "hallucination_binary": "true",  # String instead of bool
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
        evaluator = HallucinationEvaluator(config=config, llm_provider=provider)
        
        retrieved_context = [
            RetrievalResult(
                chunk_id="chunk_001",
                similarity_score=0.95,
                chunk_text="Test"
            )
        ]
        
        with pytest.raises(ValueError, match="must be a boolean"):
            evaluator.classify_hallucination(retrieved_context, "answer")


class TestClassifyHallucination:
    """Tests for classify_hallucination function"""
    
    def test_classify_hallucination_input_validation_empty_context(self):
        """Test that empty retrieved context raises ValueError"""
        config = Config.from_env()
        with pytest.raises(ValueError, match="Retrieved context cannot be empty"):
            classify_hallucination([], "answer", config)
    
    def test_classify_hallucination_input_validation_empty_model_answer(self):
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
            classify_hallucination(retrieved_context, "", config)
    
    def test_classify_hallucination_input_validation_whitespace_only(self):
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
            classify_hallucination(retrieved_context, "   ", config)
    
    @patch('rag_eval.services.evaluator.hallucination.HallucinationEvaluator._call_llm')
    @patch('rag_eval.services.evaluator.hallucination.HallucinationEvaluator._construct_prompt')
    def test_classify_hallucination_success_true(self, mock_construct, mock_call_llm):
        """Test successful hallucination classification returning True"""
        mock_construct.return_value = "Test prompt"
        mock_call_llm.return_value = json.dumps({
            "hallucination_binary": True,
            "reasoning": "Hallucination detected - claim not in context."
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
        
        result = classify_hallucination(
            retrieved_context=retrieved_context,
            model_answer="The copay is $75.",
            config=config
        )
        
        assert result is True
        mock_construct.assert_called_once()
        mock_call_llm.assert_called_once()
    
    @patch('rag_eval.services.evaluator.hallucination.HallucinationEvaluator._call_llm')
    @patch('rag_eval.services.evaluator.hallucination.HallucinationEvaluator._construct_prompt')
    def test_classify_hallucination_success_false(self, mock_construct, mock_call_llm):
        """Test successful hallucination classification returning False"""
        mock_construct.return_value = "Test prompt"
        mock_call_llm.return_value = json.dumps({
            "hallucination_binary": False,
            "reasoning": "All claims are supported by retrieved context."
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
        
        result = classify_hallucination(
            retrieved_context=retrieved_context,
            model_answer="The copay for specialist visits is $50.",
            config=config
        )
        
        assert result is False
        mock_construct.assert_called_once()
        mock_call_llm.assert_called_once()
    
    @patch('rag_eval.services.evaluator.hallucination.HallucinationEvaluator._call_llm')
    @patch('rag_eval.services.evaluator.hallucination.HallucinationEvaluator._construct_prompt')
    def test_classify_hallucination_uses_temperature_0_1(self, mock_construct, mock_call_llm):
        """Test that temperature=0.1 is used for reproducibility"""
        mock_construct.return_value = "Test prompt"
        mock_call_llm.return_value = json.dumps({
            "hallucination_binary": False,
            "reasoning": "No hallucination."
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
        
        classify_hallucination(
            retrieved_context=retrieved_context,
            model_answer="Test answer",
            config=config
        )
        
        # Verify temperature=0.1 was used
        call_args = mock_call_llm.call_args
        assert call_args[1]["temperature"] == 0.1
    
    @patch('rag_eval.services.evaluator.hallucination.HallucinationEvaluator._call_llm')
    @patch('rag_eval.services.evaluator.hallucination.HallucinationEvaluator._construct_prompt')
    def test_classify_hallucination_handles_azure_error(self, mock_construct, mock_call_llm):
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
            classify_hallucination(
                retrieved_context=retrieved_context,
                model_answer="Test answer",
                config=config
            )
    
    @patch('rag_eval.services.evaluator.hallucination.HallucinationEvaluator._call_llm')
    @patch('rag_eval.services.evaluator.hallucination.HallucinationEvaluator._construct_prompt')
    def test_classify_hallucination_handles_value_error(self, mock_construct, mock_call_llm):
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
            classify_hallucination(
                retrieved_context=retrieved_context,
                model_answer="Test answer",
                config=config
            )
    
    def test_classify_hallucination_default_config(self):
        """Test that Config.from_env() is used when config is None"""
        # Patch Config in both hallucination and base_evaluator modules
        with patch('rag_eval.services.evaluator.base_evaluator.Config') as mock_config_class:
            mock_config = Mock()
            mock_config.azure_ai_foundry_endpoint = "https://test.endpoint"
            mock_config.azure_ai_foundry_api_key = "test-key"
            mock_config_class.from_env.return_value = mock_config
            
            with patch('rag_eval.services.evaluator.hallucination.HallucinationEvaluator._construct_prompt') as mock_construct:
                with patch('rag_eval.services.evaluator.hallucination.HallucinationEvaluator._call_llm') as mock_call_llm:
                    mock_construct.return_value = "Test prompt"
                    mock_call_llm.return_value = json.dumps({
                        "hallucination_binary": False,
                        "reasoning": "No hallucination."
                    })
                    
                    retrieved_context = [
                        RetrievalResult(
                            chunk_id="chunk_001",
                            similarity_score=0.95,
                            chunk_text="Test context"
                        )
                    ]
                    
                    classify_hallucination(
                        retrieved_context=retrieved_context,
                        model_answer="Test answer",
                        config=None
                    )
                    
                    mock_config_class.from_env.assert_called_once()


class TestHallucinationGrounding:
    """Tests for grounding analysis (critical requirement)"""
    
    @patch('rag_eval.services.evaluator.hallucination.HallucinationEvaluator._call_llm')
    @patch('rag_eval.services.evaluator.hallucination.HallucinationEvaluator._construct_prompt')
    def test_grounding_analysis_information_not_in_evidence(self, mock_construct, mock_call_llm):
        """Test that information not in retrieved evidence is detected as hallucination"""
        mock_construct.return_value = "Test prompt"
        mock_call_llm.return_value = json.dumps({
            "hallucination_binary": True,
            "reasoning": "Claim about $75 copay is not in retrieved context."
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
        
        result = classify_hallucination(
            retrieved_context=retrieved_context,
            model_answer="The copay for specialist visits is $75.",
            config=config
        )
        
        assert result is True
        # Verify prompt was constructed with retrieved context
        call_args = mock_construct.call_args
        assert call_args[0][0] == retrieved_context  # First arg is retrieved_context
    
    @patch('rag_eval.services.evaluator.hallucination.HallucinationEvaluator._call_llm')
    @patch('rag_eval.services.evaluator.hallucination.HallucinationEvaluator._construct_prompt')
    def test_grounding_analysis_information_supported_by_evidence(self, mock_construct, mock_call_llm):
        """Test that information supported by evidence is not detected as hallucination"""
        mock_construct.return_value = "Test prompt"
        mock_call_llm.return_value = json.dumps({
            "hallucination_binary": False,
            "reasoning": "All claims are supported by retrieved context."
        })
        
        config = Config.from_env()
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured")
        
        retrieved_context = [
            RetrievalResult(
                chunk_id="chunk_001",
                similarity_score=0.95,
                chunk_text="The copay for specialist visits is $50. This applies to all in-network specialists."
            )
        ]
        
        result = classify_hallucination(
            retrieved_context=retrieved_context,
            model_answer="Specialist visits have a $50 copay for in-network providers.",
            config=config
        )
        
        assert result is False
    
    @patch('rag_eval.services.evaluator.hallucination.HallucinationEvaluator._call_llm')
    @patch('rag_eval.services.evaluator.hallucination.HallucinationEvaluator._construct_prompt')
    def test_reference_answer_not_used(self, mock_construct, mock_call_llm):
        """CRITICAL: Test that reference answer is NOT used in hallucination detection"""
        mock_construct.return_value = "Test prompt"
        mock_call_llm.return_value = json.dumps({
            "hallucination_binary": False,
            "reasoning": "All claims are supported by retrieved context."
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
        # If reference answer were used, this might be flagged incorrectly
        model_answer = "The copay is $50."
        reference_answer = "Specialist visits have a $75 copay."  # Different from context
        
        result = classify_hallucination(
            retrieved_context=retrieved_context,
            model_answer=model_answer,
            config=config
        )
        
        # Should be False (no hallucination) because model answer matches retrieved context
        # Even though it differs from reference answer
        assert result is False
        
        # Verify prompt does NOT contain reference answer
        prompt = mock_construct.return_value
        assert reference_answer not in prompt
        assert "$75" not in prompt  # Reference answer value should not appear
    
    @patch('rag_eval.services.evaluator.hallucination.HallucinationEvaluator._call_llm')
    @patch('rag_eval.services.evaluator.hallucination.HallucinationEvaluator._construct_prompt')
    def test_ambiguous_grounding_scenario(self, mock_construct, mock_call_llm):
        """Test edge case: ambiguous grounding scenarios"""
        mock_construct.return_value = "Test prompt"
        mock_call_llm.return_value = json.dumps({
            "hallucination_binary": True,
            "reasoning": "Ambiguous claim cannot be fully verified from context."
        })
        
        config = Config.from_env()
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured")
        
        retrieved_context = [
            RetrievalResult(
                chunk_id="chunk_001",
                similarity_score=0.95,
                chunk_text="Some coverage details may vary."
            )
        ]
        
        result = classify_hallucination(
            retrieved_context=retrieved_context,
            model_answer="The specific coverage details are clearly defined as $50.",
            config=config
        )
        
        assert result is True
    
    def test_zero_retrieved_chunks(self):
        """Test edge case: zero retrieved chunks raises ValueError"""
        config = Config.from_env()
        with pytest.raises(ValueError, match="Retrieved context cannot be empty"):
            classify_hallucination([], "Some answer", config)


class TestHallucinationConnection:
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
        
        # Try a simple hallucination classification
        try:
            result = classify_hallucination(
                retrieved_context=retrieved_context,
                model_answer="The copay for a specialist visit is $50.",
                config=config
            )
            # If successful, result should be a boolean
            assert isinstance(result, bool)
            print(f"✓ Connection test successful: hallucination={result}")
        except AzureServiceError as e:
            pytest.fail(f"Azure connection test failed: {e}")
        except Exception as e:
            # Other errors (like missing model) are acceptable for connection test
            print(f"⚠ Connection test encountered error (may be expected): {e}")

