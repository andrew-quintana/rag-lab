"""
Integration tests for evaluation prompt database loading.

These tests verify that prompts are correctly stored in and loaded from the database.
They require a real database connection (unlike unit tests which use mocks).
"""

import pytest
from pathlib import Path
from rag_eval.core.config import Config
from rag_eval.db.queries import QueryExecutor
from rag_eval.services.rag.generation import load_prompt_template
from rag_eval.services.evaluator.correctness import CorrectnessEvaluator
from rag_eval.services.evaluator.hallucination import HallucinationEvaluator
from rag_eval.services.evaluator.risk_direction import RiskDirectionEvaluator


@pytest.fixture(scope="module")
def query_executor():
    """Create QueryExecutor for database operations"""
    try:
        from rag_eval.db.connection import DatabaseConnection
        config = Config.from_env()
        db_conn = DatabaseConnection(config)
        return QueryExecutor(db_conn)
    except Exception as e:
        pytest.skip(f"Database not available: {e}")


@pytest.fixture(scope="module")
def prompt_files():
    """Prompt files are now in database only - this fixture is for reference only"""
    # Prompts are stored in database, not in files
    # This fixture is kept for compatibility but returns empty dict
    # Tests should verify database content directly
    return {}


@pytest.mark.integration
class TestPromptDatabaseIntegration:
    """Integration tests for prompt database storage and loading"""
    
    def test_correctness_prompt_in_database(self, query_executor, prompt_files):
        """Test that correctness prompt exists in database and can be loaded"""
        # Query database directly
        query = """
            SELECT prompt_text, id
            FROM prompts
            WHERE prompt_type = 'evaluation' 
              AND name = 'correctness_evaluator' 
              AND version = '0.1'
        """
        results = query_executor.execute_query(query)
        
        assert len(results) > 0, "Correctness prompt not found in database"
        db_prompt = results[0]['prompt_text'].strip()
        
        # Verify database prompt content
        assert len(db_prompt) > 100, "Database prompt seems too short"
        assert "{query}" in db_prompt, "Database prompt missing {query} placeholder"
        assert "{model_answer}" in db_prompt, "Database prompt missing {model_answer} placeholder"
        assert "{reference_answer}" in db_prompt, "Database prompt missing {reference_answer} placeholder"
    
    def test_hallucination_prompt_in_database(self, query_executor, prompt_files):
        """Test that hallucination prompt exists in database and can be loaded"""
        query = """
            SELECT prompt_text, id
            FROM prompts
            WHERE prompt_type = 'evaluation' 
              AND name = 'hallucination_evaluator' 
              AND version = '0.1'
        """
        results = query_executor.execute_query(query)
        
        assert len(results) > 0, "Hallucination prompt not found in database"
        db_prompt = results[0]['prompt_text'].strip()
        
        assert len(db_prompt) > 100, "Database prompt seems too short"
        assert "{retrieved_context}" in db_prompt, "Database prompt missing {retrieved_context} placeholder"
        assert "{model_answer}" in db_prompt, "Database prompt missing {model_answer} placeholder"
    
    def test_risk_direction_prompt_in_database(self, query_executor, prompt_files):
        """Test that risk direction prompt exists in database and can be loaded"""
        query = """
            SELECT prompt_text, id
            FROM prompts
            WHERE prompt_type = 'evaluation' 
              AND name = 'risk_direction_evaluator' 
              AND version = '0.1'
        """
        results = query_executor.execute_query(query)
        
        assert len(results) > 0, "Risk direction prompt not found in database"
        db_prompt = results[0]['prompt_text'].strip()
        
        assert len(db_prompt) > 100, "Database prompt seems too short"
        assert "{model_answer}" in db_prompt, "Database prompt missing {model_answer} placeholder"
        assert "{retrieved_context}" in db_prompt, "Database prompt missing {retrieved_context} placeholder"
    
    def test_load_correctness_prompt_via_function(self, query_executor):
        """Test loading correctness prompt via load_prompt_template()"""
        prompt = load_prompt_template(
            "0.1",
            query_executor,
            prompt_type="evaluation",
            name="correctness_evaluator"
        )
        
        assert prompt is not None, "load_prompt_template() returned None"
        assert len(prompt) > 100, "Loaded prompt seems too short"
        assert "{query}" in prompt, "Loaded prompt missing {query} placeholder"
        assert "{model_answer}" in prompt, "Loaded prompt missing {model_answer} placeholder"
    
    def test_load_hallucination_prompt_via_function(self, query_executor):
        """Test loading hallucination prompt via load_prompt_template()"""
        prompt = load_prompt_template(
            "0.1",
            query_executor,
            prompt_type="evaluation",
            name="hallucination_evaluator"
        )
        
        assert prompt is not None, "load_prompt_template() returned None"
        assert len(prompt) > 100, "Loaded prompt seems too short"
        assert "{retrieved_context}" in prompt, "Loaded prompt missing {retrieved_context} placeholder"
    
    def test_load_risk_direction_prompt_via_function(self, query_executor):
        """Test loading risk direction prompt via load_prompt_template()"""
        prompt = load_prompt_template(
            "0.1",
            query_executor,
            prompt_type="evaluation",
            name="risk_direction_evaluator"
        )
        
        assert prompt is not None, "load_prompt_template() returned None"
        assert len(prompt) > 100, "Loaded prompt seems too short"
        assert "{model_answer}" in prompt, "Loaded prompt missing {model_answer} placeholder"
    
    def test_correctness_evaluator_loads_from_database(self, query_executor):
        """Test that CorrectnessEvaluator loads prompt from database"""
        evaluator = CorrectnessEvaluator(
            prompt_version="0.1",
            query_executor=query_executor
        )
        
        prompt = evaluator._load_prompt_template()
        assert prompt is not None
        assert len(prompt) > 100
        assert "{query}" in prompt
    
    def test_hallucination_evaluator_loads_from_database(self, query_executor):
        """Test that HallucinationEvaluator loads prompt from database"""
        evaluator = HallucinationEvaluator(
            prompt_version="0.1",
            query_executor=query_executor
        )
        
        prompt = evaluator._load_prompt_template()
        assert prompt is not None
        assert len(prompt) > 100
        assert "{retrieved_context}" in prompt
    
    def test_risk_direction_evaluator_loads_from_database(self, query_executor):
        """Test that RiskDirectionEvaluator loads prompt from database"""
        evaluator = RiskDirectionEvaluator(
            prompt_version="0.1",
            query_executor=query_executor
        )
        
        prompt = evaluator._load_prompt_template()
        assert prompt is not None
        assert len(prompt) > 100
        assert "{model_answer}" in prompt

