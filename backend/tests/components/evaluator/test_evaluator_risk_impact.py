"""Tests for risk impact calculation LLM node"""

import json
import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from src.services.evaluator.risk_impact import (
    calculate_risk_impact,
    RiskImpactEvaluator,
)
from src.core.config import Config
from src.core.exceptions import AzureServiceError, ValidationError
from src.services.shared.llm_providers import AzureFoundryProvider
from src.db.queries import QueryExecutor


class TestRiskImpactPrompt:
    """Tests for prompt construction"""
    
    def setup_method(self):
        """Clear cache before each test"""
        from src.services.rag.generation import _prompt_cache
        _prompt_cache.clear()
    
    def test_load_prompt_template_from_file(self):
        """Test that prompt template can be loaded from file (for testing)"""
        test_prompt_path = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "prompts" / "risk_impact_prompt.md"
        
        evaluator = RiskImpactEvaluator(
            query_executor=None,
            prompt_path=test_prompt_path
        )
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
            {"prompt_text": "# Risk Impact Prompt\n\n**Model Answer Cost:**\n{model_answer_cost}\n\n**Actual Cost:**\n{actual_cost}\n"}
        ]
        
        evaluator = RiskImpactEvaluator(
            query_executor=mock_query_executor,
            live=True
        )
        template = evaluator._load_prompt_template()
        
        assert isinstance(template, str)
        assert len(template) > 0
        assert "{model_answer_cost}" in template
        assert "{actual_cost}" in template
        
        # Verify database query was called with correct parameters
        mock_query_executor.execute_query.assert_called_once()
        call_args = mock_query_executor.execute_query.call_args
        assert call_args[0][1] == ("evaluation", "risk_impact_evaluator")
    
    def test_load_prompt_template_custom_path(self):
        """Test loading prompt template from custom path (for testing)"""
        test_prompt_path = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "prompts" / "risk_impact_prompt.md"
        evaluator = RiskImpactEvaluator(prompt_path=test_prompt_path)
        template = evaluator._load_prompt_template()
        assert isinstance(template, str)
        assert len(template) > 0
    
    def test_load_prompt_template_not_found_file(self):
        """Test that loading non-existent prompt file raises ValueError"""
        fake_path = Path("/nonexistent/prompt.md")
        evaluator = RiskImpactEvaluator(prompt_path=fake_path)
        with pytest.raises(ValueError, match="Prompt template not found"):
            evaluator._load_prompt_template()
    
    def test_load_prompt_template_not_found_database(self):
        """Test that loading non-existent prompt from database raises ValidationError"""
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = []
        
        evaluator = RiskImpactEvaluator(
            query_executor=mock_query_executor,
            live=True
        )
        with pytest.raises(ValidationError, match="not found"):
            evaluator._load_prompt_template()
    
    def test_construct_prompt(self):
        """Test prompt construction with cost dictionaries"""
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "# Risk Impact Prompt\n\n**Model Answer Cost:**\n{model_answer_cost}\n\n**Actual Cost:**\n{actual_cost}\n"}
        ]
        evaluator = RiskImpactEvaluator(query_executor=mock_query_executor)
        
        model_cost = {"money": 500.0, "time": "2 hours"}
        actual_cost = {"money": 50.0, "time": "30 minutes"}
        prompt = evaluator._construct_prompt(model_cost, actual_cost)
        
        assert isinstance(prompt, str)
        assert "500.0" in prompt or "500" in prompt  # JSON formatting may vary
        assert "50.0" in prompt or "50" in prompt
        assert "{model_answer_cost}" not in prompt  # Should be replaced
        assert "{actual_cost}" not in prompt  # Should be replaced
    
    def test_construct_prompt_missing_placeholder(self):
        """Test that missing placeholder raises ValueError"""
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "No placeholder here"}
        ]
        evaluator = RiskImpactEvaluator(query_executor=mock_query_executor)
        with pytest.raises(ValueError, match="missing required placeholders"):
            evaluator._construct_prompt({"money": 100}, {"money": 50})
    
    def test_format_cost_dict(self):
        """Test formatting cost dictionary as JSON"""
        test_prompt_path = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "prompts" / "risk_impact_prompt.md"
        evaluator = RiskImpactEvaluator(prompt_path=test_prompt_path)
        
        cost_dict = {"money": 500.0, "time": "2 hours", "steps": 3}
        formatted = evaluator._format_cost_dict(cost_dict)
        
        assert isinstance(formatted, str)
        # Should be valid JSON
        parsed = json.loads(formatted)
        assert parsed == cost_dict


class TestRiskImpact:
    """Tests for risk impact calculation functionality"""
    
    @patch('src.services.shared.llm_providers.requests.post')
    def test_calculate_impact_time_based_costs(self, mock_post):
        """Test impact calculation for time-based costs"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "risk_impact": 2.0,
                        "reasoning": "The model overestimated time by 1.5 hours, which is a significant deviation."
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
        test_prompt_path = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "prompts" / "risk_impact_prompt.md"
        evaluator = RiskImpactEvaluator(config=config, llm_provider=provider, prompt_path=test_prompt_path)
        
        model_cost = {"time": "2 hours"}
        actual_cost = {"time": "30 minutes"}
        result = evaluator.calculate_risk_impact(model_cost, actual_cost)
        
        assert isinstance(result, float)
        assert 0 <= result <= 3
        assert result == 2.0
    
    @patch('src.services.shared.llm_providers.requests.post')
    def test_calculate_impact_money_based_costs(self, mock_post):
        """Test impact calculation for money-based costs"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "risk_impact": 3.0,
                        "reasoning": "The model overestimated cost by $4,950, which is a massive financial deviation."
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
        test_prompt_path = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "prompts" / "risk_impact_prompt.md"
        evaluator = RiskImpactEvaluator(config=config, llm_provider=provider, prompt_path=test_prompt_path)
        
        model_cost = {"money": 5000.0}
        actual_cost = {"money": 50.0}
        result = evaluator.calculate_risk_impact(model_cost, actual_cost)
        
        assert isinstance(result, float)
        assert 0 <= result <= 3
        assert result == 3.0
    
    @patch('src.services.shared.llm_providers.requests.post')
    def test_calculate_impact_step_based_costs(self, mock_post):
        """Test impact calculation for step-based costs"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "risk_impact": 1.5,
                        "reasoning": "The model overestimated steps by 2, which is a moderate deviation."
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
        test_prompt_path = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "prompts" / "risk_impact_prompt.md"
        evaluator = RiskImpactEvaluator(config=config, llm_provider=provider, prompt_path=test_prompt_path)
        
        model_cost = {"steps": 5}
        actual_cost = {"steps": 3}
        result = evaluator.calculate_risk_impact(model_cost, actual_cost)
        
        assert isinstance(result, float)
        assert 0 <= result <= 3
        assert result == 1.5
    
    @patch('src.services.shared.llm_providers.requests.post')
    def test_calculate_impact_mixed_resource_types(self, mock_post):
        """Test impact calculation for mixed resource types"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "risk_impact": 2.5,
                        "reasoning": "The model overestimated time and steps significantly, while money is accurate. Combined impact is moderate-to-high."
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
        test_prompt_path = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "prompts" / "risk_impact_prompt.md"
        evaluator = RiskImpactEvaluator(config=config, llm_provider=provider, prompt_path=test_prompt_path)
        
        model_cost = {"money": 100.0, "time": "4 hours", "steps": 5}
        actual_cost = {"money": 100.0, "time": "1 hour", "steps": 2}
        result = evaluator.calculate_risk_impact(model_cost, actual_cost)
        
        assert isinstance(result, float)
        assert 0 <= result <= 3
        assert result == 2.5
    
    @patch('src.services.shared.llm_providers.requests.post')
    def test_calculate_impact_zero_impact(self, mock_post):
        """Test edge case: zero impact scenarios"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "risk_impact": 0.0,
                        "reasoning": "The costs are identical, so there is no impact."
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
        test_prompt_path = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "prompts" / "risk_impact_prompt.md"
        evaluator = RiskImpactEvaluator(config=config, llm_provider=provider, prompt_path=test_prompt_path)
        
        model_cost = {"money": 50.0}
        actual_cost = {"money": 50.0}
        result = evaluator.calculate_risk_impact(model_cost, actual_cost)
        
        assert isinstance(result, float)
        assert result == 0.0
    
    @patch('src.services.shared.llm_providers.requests.post')
    def test_calculate_impact_maximum_impact(self, mock_post):
        """Test edge case: maximum impact scenarios"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "risk_impact": 3.0,
                        "reasoning": "The model severely overestimated the cost, which would have severe real-world consequences."
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
        test_prompt_path = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "prompts" / "risk_impact_prompt.md"
        evaluator = RiskImpactEvaluator(config=config, llm_provider=provider, prompt_path=test_prompt_path)
        
        model_cost = {"money": 10000.0}
        actual_cost = {"money": 10.0}
        result = evaluator.calculate_risk_impact(model_cost, actual_cost)
        
        assert isinstance(result, float)
        assert result == 3.0
    
    @patch('src.services.shared.llm_providers.requests.post')
    def test_calculate_impact_range_validation(self, mock_post):
        """Test that impact is validated to be in range [0, 3]"""
        # Test with value below 0
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "risk_impact": -1.0,
                        "reasoning": "Invalid negative impact."
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
        test_prompt_path = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "prompts" / "risk_impact_prompt.md"
        evaluator = RiskImpactEvaluator(config=config, llm_provider=provider, prompt_path=test_prompt_path)
        
        with pytest.raises(ValueError, match="must be in range \\[0, 3\\]"):
            evaluator.calculate_risk_impact({"money": 100}, {"money": 50})
        
        # Test with value above 3
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "risk_impact": 5.0,
                        "reasoning": "Invalid high impact."
                    })
                }
            }]
        }
        
        with pytest.raises(ValueError, match="must be in range \\[0, 3\\]"):
            evaluator.calculate_risk_impact({"money": 100}, {"money": 50})
    
    def test_calculate_impact_invalid_inputs(self):
        """Test error handling for invalid inputs"""
        test_prompt_path = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "prompts" / "risk_impact_prompt.md"
        evaluator = RiskImpactEvaluator(prompt_path=test_prompt_path)
        
        # Test empty dictionary
        with pytest.raises(ValueError, match="must be a non-empty dictionary"):
            evaluator.calculate_risk_impact({}, {"money": 50})
        
        with pytest.raises(ValueError, match="must be a non-empty dictionary"):
            evaluator.calculate_risk_impact({"money": 100}, {})
        
        # Test non-dictionary input
        with pytest.raises(ValueError, match="must be a non-empty dictionary"):
            evaluator.calculate_risk_impact("not a dict", {"money": 50})
        
        with pytest.raises(ValueError, match="must be a non-empty dictionary"):
            evaluator.calculate_risk_impact({"money": 100}, "not a dict")
    
    @patch('src.services.shared.llm_providers.requests.post')
    def test_calculate_impact_llm_failure(self, mock_post):
        """Test error handling for LLM failures"""
        mock_post.side_effect = Exception("LLM API error")
        
        config = Config.from_env()
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured")
        
        provider = AzureFoundryProvider(
            endpoint=config.azure_ai_foundry_endpoint,
            api_key=config.azure_ai_foundry_api_key,
            model="gpt-4o-mini"
        )
        test_prompt_path = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "prompts" / "risk_impact_prompt.md"
        evaluator = RiskImpactEvaluator(config=config, llm_provider=provider, prompt_path=test_prompt_path)
        
        with pytest.raises(AzureServiceError):
            evaluator.calculate_risk_impact({"money": 100}, {"money": 50})
    
    @patch('src.services.shared.llm_providers.requests.post')
    def test_calculate_impact_missing_field(self, mock_post):
        """Test error handling for missing risk_impact field in response"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "reasoning": "No risk_impact field."
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
        test_prompt_path = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "prompts" / "risk_impact_prompt.md"
        evaluator = RiskImpactEvaluator(config=config, llm_provider=provider, prompt_path=test_prompt_path)
        
        with pytest.raises(ValueError, match="missing 'risk_impact' field"):
            evaluator.calculate_risk_impact({"money": 100}, {"money": 50})
    
    @patch('src.services.shared.llm_providers.requests.post')
    def test_calculate_impact_invalid_type(self, mock_post):
        """Test error handling for invalid risk_impact type"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "risk_impact": "not a number",
                        "reasoning": "Invalid type."
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
        test_prompt_path = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "prompts" / "risk_impact_prompt.md"
        evaluator = RiskImpactEvaluator(config=config, llm_provider=provider, prompt_path=test_prompt_path)
        
        with pytest.raises(ValueError, match="must be a number"):
            evaluator.calculate_risk_impact({"money": 100}, {"money": 50})


class TestRiskImpactModuleFunction:
    """Tests for module-level calculate_risk_impact function"""
    
    @patch('src.services.shared.llm_providers.requests.post')
    def test_module_function(self, mock_post):
        """Test module-level calculate_risk_impact function"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "risk_impact": 2.0,
                        "reasoning": "Moderate impact."
                    })
                }
            }]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        config = Config.from_env()
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured")
        
        test_prompt_path = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "prompts" / "risk_impact_prompt.md"
        
        # Note: Module function doesn't support prompt_path, so we need to mock query_executor
        mock_query_executor = Mock(spec=QueryExecutor)
        with open(test_prompt_path, 'r') as f:
            prompt_text = f.read()
        mock_query_executor.execute_query.return_value = [{"prompt_text": prompt_text}]
        
        result = calculate_risk_impact(
            {"money": 200.0},
            {"money": 50.0},
            config=config,
            query_executor=mock_query_executor
        )
        
        assert isinstance(result, float)
        assert 0 <= result <= 3


class TestRiskImpactConnection:
    """Connection tests for Azure Foundry API"""
    
    def test_azure_connection(self):
        """Test actual connection to Azure Foundry (warns if credentials missing, doesn't fail tests)"""
        config = Config.from_env()
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure credentials not configured - skipping connection test")
        
        # Try to create evaluator and make a simple call
        try:
            test_prompt_path = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "prompts" / "risk_impact_prompt.md"
            evaluator = RiskImpactEvaluator(config=config, prompt_path=test_prompt_path)
            
            # Make a simple test call
            model_cost = {"money": 100.0}
            actual_cost = {"money": 50.0}
            result = evaluator.calculate_risk_impact(model_cost, actual_cost)
            
            # If we get here, connection worked
            assert isinstance(result, float)
            assert 0 <= result <= 3
            print(f"✓ Azure Foundry connection successful - impact: {result}")
        except Exception as e:
            pytest.skip(f"Azure Foundry connection failed: {e}")

