"""Tests for cost extraction LLM node"""

import json
import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from rag_eval.services.evaluator.cost_extraction import (
    extract_costs,
    CostExtractionEvaluator,
)
from rag_eval.core.config import Config
from rag_eval.core.exceptions import AzureServiceError, ValidationError
from rag_eval.services.shared.llm_providers import AzureFoundryProvider
from rag_eval.db.queries import QueryExecutor


class TestCostExtractionPrompt:
    """Tests for prompt construction"""
    
    def setup_method(self):
        """Clear cache before each test"""
        from rag_eval.services.rag.generation import _prompt_cache
        _prompt_cache.clear()
    
    def test_load_prompt_template_from_file(self):
        """Test that prompt template can be loaded from file (for testing)"""
        # Use a mock query_executor that returns empty to force file loading
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = []
        
        # Create a test prompt file path
        test_prompt_path = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "prompts" / "prompt_v1.md"
        
        evaluator = CostExtractionEvaluator(
            query_executor=None,
            prompt_path=test_prompt_path
        )
        template = evaluator._load_prompt_template()
        assert isinstance(template, str)
        assert len(template) > 0
    
    def test_load_prompt_template_from_database(self):
        """Test that prompt template can be loaded from database"""
        # Clear cache to ensure fresh database query
        from rag_eval.services.rag.generation import _prompt_cache
        _prompt_cache.clear()
        
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "# Cost Extraction Prompt\n\nYou are an expert parser...\n\n**Text:**\n{text}\n"}
        ]
        
        evaluator = CostExtractionEvaluator(
            query_executor=mock_query_executor,
            live=True
        )
        template = evaluator._load_prompt_template()
        
        assert isinstance(template, str)
        assert len(template) > 0
        assert "{text}" in template
        
        # Verify database query was called with correct parameters
        mock_query_executor.execute_query.assert_called_once()
        call_args = mock_query_executor.execute_query.call_args
        assert call_args[0][1] == ("evaluation", "cost_extraction_evaluator")
    
    def test_load_prompt_template_custom_path(self):
        """Test loading prompt template from custom path (for testing)"""
        test_prompt_path = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "prompts" / "prompt_v1.md"
        evaluator = CostExtractionEvaluator(prompt_path=test_prompt_path)
        template = evaluator._load_prompt_template()
        assert isinstance(template, str)
        assert len(template) > 0
    
    def test_load_prompt_template_not_found_file(self):
        """Test that loading non-existent prompt file raises ValueError"""
        fake_path = Path("/nonexistent/prompt.md")
        evaluator = CostExtractionEvaluator(prompt_path=fake_path)
        with pytest.raises(ValueError, match="Prompt template not found"):
            evaluator._load_prompt_template()
    
    def test_load_prompt_template_not_found_database(self):
        """Test that loading non-existent prompt from database raises ValidationError"""
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = []
        
        evaluator = CostExtractionEvaluator(
            query_executor=mock_query_executor,
            live=True
        )
        with pytest.raises(ValidationError, match="not found"):
            evaluator._load_prompt_template()
    
    def test_construct_prompt(self):
        """Test prompt construction with text input"""
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "# Cost Extraction Prompt\n\n**Text:**\n{text}\n"}
        ]
        evaluator = CostExtractionEvaluator(query_executor=mock_query_executor)
        text = "The procedure takes 2 hours and costs $500."
        prompt = evaluator._construct_prompt(text)
        
        assert isinstance(prompt, str)
        assert text in prompt
        assert "{text}" not in prompt  # Should be replaced
    
    def test_construct_prompt_missing_placeholder(self):
        """Test that missing placeholder raises ValueError"""
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "No placeholder here"}
        ]
        evaluator = CostExtractionEvaluator(query_executor=mock_query_executor)
        with pytest.raises(ValueError, match="missing required placeholders"):
            evaluator._construct_prompt("test text")


class TestCostExtraction:
    """Tests for cost extraction functionality"""
    
    @patch('rag_eval.services.shared.llm_providers.requests.post')
    def test_extract_time_costs(self, mock_post):
        """Test extraction of time-based costs"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "time": "2 hours",
                        "reasoning": "Found time cost: 2 hours."
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
        evaluator = CostExtractionEvaluator(config=config, llm_provider=provider)
        
        text = "The procedure takes 2 hours to complete."
        result = evaluator.extract_costs(text)
        
        assert "time" in result
        assert result["time"] == "2 hours"
        assert "reasoning" in result
        assert "money" not in result
        assert "steps" not in result
    
    @patch('rag_eval.services.shared.llm_providers.requests.post')
    def test_extract_money_costs(self, mock_post):
        """Test extraction of money-based costs"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "money": "$500",
                        "reasoning": "Found money cost: $500."
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
        evaluator = CostExtractionEvaluator(config=config, llm_provider=provider)
        
        text = "The copay for specialist visits is $500."
        result = evaluator.extract_costs(text)
        
        assert "money" in result
        assert result["money"] == "$500"
        assert "reasoning" in result
        assert "time" not in result
        assert "steps" not in result
    
    @patch('rag_eval.services.shared.llm_providers.requests.post')
    def test_extract_steps_costs(self, mock_post):
        """Test extraction of step-based costs"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "steps": 3,
                        "reasoning": "Found step cost: 3 steps."
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
        evaluator = CostExtractionEvaluator(config=config, llm_provider=provider)
        
        text = "The process requires 3 steps to complete."
        result = evaluator.extract_costs(text)
        
        assert "steps" in result
        assert result["steps"] == 3
        assert "reasoning" in result
        assert "time" not in result
        assert "money" not in result
    
    @patch('rag_eval.services.shared.llm_providers.requests.post')
    def test_extract_mixed_costs(self, mock_post):
        """Test extraction of mixed cost types from same text"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "time": "30 minutes",
                        "money": "$1,500.00",
                        "steps": 5,
                        "reasoning": "Found time: 30 minutes, money: $1,500.00, steps: 5."
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
        evaluator = CostExtractionEvaluator(config=config, llm_provider=provider)
        
        text = "You'll need to complete 5 steps, which takes about 30 minutes, and the total cost is $1,500.00."
        result = evaluator.extract_costs(text)
        
        assert "time" in result
        assert "money" in result
        assert "steps" in result
        assert result["time"] == "30 minutes"
        assert result["money"] == "$1,500.00"
        assert result["steps"] == 5
        assert "reasoning" in result
    
    @patch('rag_eval.services.shared.llm_providers.requests.post')
    def test_extract_no_cost_information(self, mock_post):
        """Test handling of missing cost information (optional fields)"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "reasoning": "No cost information found in the text."
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
        evaluator = CostExtractionEvaluator(config=config, llm_provider=provider)
        
        text = "The insurance plan covers preventive care visits."
        result = evaluator.extract_costs(text)
        
        assert "reasoning" in result
        assert "time" not in result
        assert "money" not in result
        assert "steps" not in result
    
    @patch('rag_eval.services.shared.llm_providers.requests.post')
    def test_extract_ambiguous_cost_expressions(self, mock_post):
        """Test edge case: ambiguous cost expressions"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "time": "24 hours",
                        "reasoning": "Found time reference: within 24 hours (scheduling window)."
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
        evaluator = CostExtractionEvaluator(config=config, llm_provider=provider)
        
        text = "The appointment should be scheduled within 24 hours."
        result = evaluator.extract_costs(text)
        
        assert "reasoning" in result
        # May or may not have time field depending on interpretation
    
    def test_extract_costs_empty_text(self):
        """Test that empty text raises ValueError"""
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "# Cost Extraction Prompt\n\n**Text:**\n{text}\n"}
        ]
        evaluator = CostExtractionEvaluator(query_executor=mock_query_executor)
        
        with pytest.raises(ValueError, match="Text cannot be empty"):
            evaluator.extract_costs("")
        
        with pytest.raises(ValueError, match="Text cannot be empty"):
            evaluator.extract_costs("   ")
    
    @patch('rag_eval.services.shared.llm_providers.requests.post')
    def test_extract_costs_llm_failure(self, mock_post):
        """Test error handling for LLM failures"""
        mock_post.side_effect = Exception("API connection failed")
        
        config = Config.from_env()
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured")
        
        provider = AzureFoundryProvider(
            endpoint=config.azure_ai_foundry_endpoint,
            api_key=config.azure_ai_foundry_api_key,
            model="gpt-4o-mini"
        )
        evaluator = CostExtractionEvaluator(config=config, llm_provider=provider)
        
        with pytest.raises(AzureServiceError):
            evaluator.extract_costs("Test text")
    
    @patch('rag_eval.services.shared.llm_providers.requests.post')
    def test_extract_costs_missing_reasoning(self, mock_post):
        """Test that missing reasoning field raises ValueError"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "time": "2 hours"
                        # Missing reasoning field
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
        evaluator = CostExtractionEvaluator(config=config, llm_provider=provider)
        
        with pytest.raises(ValueError, match="missing 'reasoning' field"):
            evaluator.extract_costs("Test text")
    
    @patch('rag_eval.services.shared.llm_providers.requests.post')
    def test_extract_costs_null_fields_omitted(self, mock_post):
        """Test that null/None fields are omitted from result"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "time": None,
                        "money": "$500",
                        "steps": None,
                        "reasoning": "Found money cost only."
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
        evaluator = CostExtractionEvaluator(config=config, llm_provider=provider)
        
        result = evaluator.extract_costs("The cost is $500.")
        
        assert "money" in result
        assert "reasoning" in result
        assert "time" not in result  # Should be omitted
        assert "steps" not in result  # Should be omitted
    
    @patch('rag_eval.services.shared.llm_providers.requests.post')
    def test_extract_costs_various_money_formats(self, mock_post):
        """Test extraction of money in various formats"""
        test_cases = [
            ("$500", "$500"),
            ("500 dollars", "500 dollars"),
            ("500.00", "500.00"),
            ("$1,500.00", "$1,500.00"),
        ]
        
        for input_format, expected_format in test_cases:
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "money": expected_format,
                            "reasoning": f"Found money cost: {expected_format}."
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
            evaluator = CostExtractionEvaluator(config=config, llm_provider=provider)
            
            text = f"The cost is {input_format}."
            result = evaluator.extract_costs(text)
            
            assert "money" in result
            assert result["money"] == expected_format
    
    @patch('rag_eval.services.shared.llm_providers.requests.post')
    def test_extract_costs_various_time_formats(self, mock_post):
        """Test extraction of time in various formats"""
        test_cases = [
            ("2 hours", "2 hours"),
            ("30 minutes", "30 minutes"),
            ("1 day", "1 day"),
            ("3 weeks", "3 weeks"),
        ]
        
        for input_format, expected_format in test_cases:
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "time": expected_format,
                            "reasoning": f"Found time cost: {expected_format}."
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
            evaluator = CostExtractionEvaluator(config=config, llm_provider=provider)
            
            text = f"The procedure takes {input_format}."
            result = evaluator.extract_costs(text)
            
            assert "time" in result
            assert result["time"] == expected_format


class TestCostExtractionModuleFunction:
    """Tests for module-level extract_costs() function"""
    
    @patch('rag_eval.services.shared.llm_providers.requests.post')
    def test_module_function_extract_costs(self, mock_post):
        """Test module-level extract_costs() function"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "time": "2 hours",
                        "money": "$500",
                        "reasoning": "Found time and money costs."
                    })
                }
            }]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        config = Config.from_env()
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured")
        
        text = "The procedure takes 2 hours and costs $500."
        result = extract_costs(text, config=config)
        
        assert "time" in result
        assert "money" in result
        assert "reasoning" in result
        assert result["time"] == "2 hours"
        assert result["money"] == "$500"
    
    def test_module_function_empty_text(self):
        """Test module-level function with empty text"""
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "# Cost Extraction Prompt\n\n**Text:**\n{text}\n"}
        ]
        with pytest.raises(ValueError, match="Text cannot be empty"):
            extract_costs("", config=Config.from_env(), query_executor=mock_query_executor)


class TestCostExtractionConnection:
    """Connection tests for Azure Foundry API"""
    
    def test_connection_to_azure_foundry(self):
        """Test actual connection to Azure Foundry (warns if credentials missing)"""
        config = Config.from_env()
        
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured - skipping connection test")
        
        provider = AzureFoundryProvider(
            endpoint=config.azure_ai_foundry_endpoint,
            api_key=config.azure_ai_foundry_api_key,
            model="gpt-4o-mini"
        )
        evaluator = CostExtractionEvaluator(config=config, llm_provider=provider)
        
        # Test with a simple text
        text = "The procedure takes 2 hours and costs $500."
        result = evaluator.extract_costs(text)
        
        # Verify we got a result
        assert isinstance(result, dict)
        assert "reasoning" in result
        # May or may not have cost fields depending on LLM interpretation
        print(f"\nConnection test result: {result}")

