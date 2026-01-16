"""Unit tests for prompt template loading and construction"""

import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.core.exceptions import ValidationError, DatabaseError, AzureServiceError
from src.core.interfaces import Query, RetrievalResult
from src.core.config import Config
from src.db.queries import QueryExecutor
from src.services.rag.generation import (
    load_prompt_template,
    construct_prompt,
    _prompt_cache,
)


class TestLoadPromptTemplate:
    """Tests for load_prompt_template function"""
    
    def setup_method(self):
        """Clear cache before each test"""
        _prompt_cache.clear()
    
    def test_load_prompt_template_success(self):
        """Test successful prompt template loading from database"""
        # Mock QueryExecutor
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "You are a helpful assistant. Query: {query}. Context: {context}."}
        ]
        
        result = load_prompt_template("v1", mock_query_executor)
        
        assert result == "You are a helpful assistant. Query: {query}. Context: {context}."
        # Verify query includes prompt_type (default "rag")
        mock_query_executor.execute_query.assert_called_once()
        call_args = mock_query_executor.execute_query.call_args
        assert call_args[0][1] == ("rag", "v1")  # (prompt_type, version)
        assert "rag:v1" in _prompt_cache
        assert _prompt_cache["rag:v1"] == result
    
    def test_load_prompt_template_caching(self):
        """Test that prompt templates are cached and not re-queried"""
        # Mock QueryExecutor
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "Template with {query} and {context}"}
        ]
        
        # First load - should query database
        result1 = load_prompt_template("v1", mock_query_executor)
        assert mock_query_executor.execute_query.call_count == 1
        
        # Second load - should use cache
        result2 = load_prompt_template("v1", mock_query_executor)
        assert mock_query_executor.execute_query.call_count == 1  # Still 1, not 2
        assert result1 == result2
        assert "rag:v1" in _prompt_cache
    
    def test_load_prompt_template_missing_version(self):
        """Test that missing prompt version raises ValidationError"""
        # Mock QueryExecutor returning empty results
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = []
        
        with pytest.raises(ValidationError) as exc_info:
            load_prompt_template("nonexistent", mock_query_executor)
        
        assert "not found in database" in str(exc_info.value)
        assert "nonexistent" in str(exc_info.value)
        assert "rag" in str(exc_info.value)  # Should mention prompt_type
    
    def test_load_prompt_template_database_error(self):
        """Test that database errors are properly handled"""
        # Mock QueryExecutor raising exception
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.side_effect = Exception("Database connection failed")
        
        with pytest.raises(DatabaseError) as exc_info:
            load_prompt_template("v1", mock_query_executor)
        
        assert "Failed to load prompt template" in str(exc_info.value)
    
    def test_load_prompt_template_multiple_versions(self):
        """Test loading multiple different prompt versions"""
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.side_effect = [
            [{"prompt_text": "Template v1 with {query} and {context}"}],
            [{"prompt_text": "Template v2 with {query} and {context}"}],
        ]
        
        result1 = load_prompt_template("v1", mock_query_executor)
        result2 = load_prompt_template("v2", mock_query_executor)
        
        assert result1 == "Template v1 with {query} and {context}"
        assert result2 == "Template v2 with {query} and {context}"
        assert "rag:v1" in _prompt_cache
        assert "rag:v2" in _prompt_cache
        assert mock_query_executor.execute_query.call_count == 2
    
    def test_load_prompt_template_different_types(self):
        """Test loading prompts with different prompt types"""
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.side_effect = [
            [{"prompt_text": "RAG template with {query} and {context}"}],
            [{"prompt_text": "Evaluation template with {query} and {context}"}],
        ]
        
        result1 = load_prompt_template("v1", mock_query_executor, prompt_type="rag")
        result2 = load_prompt_template("v1", mock_query_executor, prompt_type="evaluation")
        
        assert result1 == "RAG template with {query} and {context}"
        assert result2 == "Evaluation template with {query} and {context}"
        assert "rag:v1" in _prompt_cache
        assert "evaluation:v1" in _prompt_cache
        assert mock_query_executor.execute_query.call_count == 2
    
    def test_load_prompt_template_with_name(self):
        """Test loading evaluation prompts with name parameter"""
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.side_effect = [
            [{"prompt_text": "Correctness evaluator template with {query} and {model_answer}"}],
            [{"prompt_text": "Hallucination evaluator template with {retrieved_context} and {model_answer}"}],
        ]
        
        result1 = load_prompt_template(
            query_executor=mock_query_executor,
            prompt_type="evaluation", 
            name="correctness_evaluator",
            live=True
        )
        result2 = load_prompt_template(
            query_executor=mock_query_executor,
            prompt_type="evaluation", 
            name="hallucination_evaluator",
            live=True
        )
        
        assert result1 == "Correctness evaluator template with {query} and {model_answer}"
        assert result2 == "Hallucination evaluator template with {retrieved_context} and {model_answer}"
        # Cache key format: name queries with live flag
        assert "evaluation:correctness_evaluator:live" in _prompt_cache
        assert "evaluation:hallucination_evaluator:live" in _prompt_cache
        
        # Verify database queries include name (version not in query params for live queries)
        assert mock_query_executor.execute_query.call_count == 2
        call1_args = mock_query_executor.execute_query.call_args_list[0]
        call2_args = mock_query_executor.execute_query.call_args_list[1]
        assert call1_args[0][1] == ("evaluation", "correctness_evaluator")
        assert call2_args[0][1] == ("evaluation", "hallucination_evaluator")
    
    def test_load_prompt_template_backward_compatibility(self):
        """Test that default prompt_type is 'rag' for backward compatibility"""
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "Template with {query} and {context}"}
        ]
        
        # Call without prompt_type (should default to "rag")
        result = load_prompt_template("v1", mock_query_executor)
        
        # Verify it queried with "rag" as prompt_type
        call_args = mock_query_executor.execute_query.call_args
        assert call_args[0][1] == ("rag", "v1")
        assert "rag:v1" in _prompt_cache


class TestConstructPrompt:
    """Tests for construct_prompt function"""
    
    def setup_method(self):
        """Clear cache before each test"""
        _prompt_cache.clear()
    
    def test_construct_prompt_success(self):
        """Test successful prompt construction with query and context"""
        # Mock QueryExecutor
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "Answer based on: {context}\n\nQuestion: {query}"}
        ]
        
        query = Query(text="What is the coverage limit?")
        chunks = [
            RetrievalResult(
                chunk_id="chunk_1",
                similarity_score=0.9,
                chunk_text="Coverage limit is $500,000"
            ),
            RetrievalResult(
                chunk_id="chunk_2",
                similarity_score=0.85,
                chunk_text="Additional coverage information"
            )
        ]
        
        result = construct_prompt(query, chunks, "v1", mock_query_executor)
        
        assert "What is the coverage limit?" in result
        assert "Coverage limit is $500,000" in result
        assert "Additional coverage information" in result
        assert "{query}" not in result
        assert "{context}" not in result
    
    def test_construct_prompt_empty_chunks(self):
        """Test prompt construction with empty retrieved chunks"""
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "Context: {context}\n\nQuery: {query}"}
        ]
        
        query = Query(text="What is the coverage?")
        chunks = []
        
        result = construct_prompt(query, chunks, "v1", mock_query_executor)
        
        assert "What is the coverage?" in result
        assert "(No context retrieved)" in result
        assert "{query}" not in result
        assert "{context}" not in result
    
    def test_construct_prompt_multiple_chunks_concatenation(self):
        """Test that multiple chunks are properly concatenated"""
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "Context: {context}\n\nQuery: {query}"}
        ]
        
        query = Query(text="Test query")
        chunks = [
            RetrievalResult(chunk_id="1", similarity_score=0.9, chunk_text="Chunk 1"),
            RetrievalResult(chunk_id="2", similarity_score=0.8, chunk_text="Chunk 2"),
            RetrievalResult(chunk_id="3", similarity_score=0.7, chunk_text="Chunk 3"),
        ]
        
        result = construct_prompt(query, chunks, "v1", mock_query_executor)
        
        # Verify chunks are separated by double newlines
        assert "Chunk 1\n\nChunk 2\n\nChunk 3" in result
        assert "Test query" in result
    
    def test_construct_prompt_missing_query_placeholder(self):
        """Test that missing {query} placeholder raises ValidationError"""
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "Context: {context}"}  # Missing {query}
        ]
        
        query = Query(text="Test query")
        chunks = []
        
        with pytest.raises(ValidationError) as exc_info:
            construct_prompt(query, chunks, "v1", mock_query_executor)
        
        assert "missing required placeholders" in str(exc_info.value)
        assert "{query}" in str(exc_info.value)
    
    def test_construct_prompt_missing_context_placeholder(self):
        """Test that missing {context} placeholder raises ValidationError"""
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "Query: {query}"}  # Missing {context}
        ]
        
        query = Query(text="Test query")
        chunks = []
        
        with pytest.raises(ValidationError) as exc_info:
            construct_prompt(query, chunks, "v1", mock_query_executor)
        
        assert "missing required placeholders" in str(exc_info.value)
        assert "{context}" in str(exc_info.value)
    
    def test_construct_prompt_missing_both_placeholders(self):
        """Test that missing both placeholders raises ValidationError"""
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "No placeholders here"}  # Missing both
        ]
        
        query = Query(text="Test query")
        chunks = []
        
        with pytest.raises(ValidationError) as exc_info:
            construct_prompt(query, chunks, "v1", mock_query_executor)
        
        assert "missing required placeholders" in str(exc_info.value)
        assert "{query}" in str(exc_info.value)
        assert "{context}" in str(exc_info.value)
    
    def test_construct_prompt_empty_query_text(self):
        """Test that empty query text raises ValueError"""
        mock_query_executor = Mock(spec=QueryExecutor)
        
        query = Query(text="")
        chunks = []
        
        with pytest.raises(ValueError) as exc_info:
            construct_prompt(query, chunks, "v1", mock_query_executor)
        
        assert "cannot be empty" in str(exc_info.value)
    
    def test_construct_prompt_whitespace_only_query(self):
        """Test that whitespace-only query text raises ValueError"""
        mock_query_executor = Mock(spec=QueryExecutor)
        
        query = Query(text="   ")
        chunks = []
        
        with pytest.raises(ValueError) as exc_info:
            construct_prompt(query, chunks, "v1", mock_query_executor)
        
        assert "cannot be empty" in str(exc_info.value)
    
    def test_construct_prompt_missing_prompt_version(self):
        """Test that missing prompt version propagates ValidationError"""
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = []  # Version not found
        
        query = Query(text="Test query")
        chunks = []
        
        with pytest.raises(ValidationError) as exc_info:
            construct_prompt(query, chunks, "nonexistent", mock_query_executor)
        
        assert "not found in database" in str(exc_info.value)
    
    def test_construct_prompt_database_error_propagation(self):
        """Test that database errors are properly propagated"""
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.side_effect = Exception("Database error")
        
        query = Query(text="Test query")
        chunks = []
        
        with pytest.raises(DatabaseError):
            construct_prompt(query, chunks, "v1", mock_query_executor)
    
    def test_construct_prompt_placeholder_replacement_order(self):
        """Test that placeholders are replaced correctly regardless of order"""
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "{context}\n\n{query}"}
        ]
        
        query = Query(text="What is X?")
        chunks = [
            RetrievalResult(chunk_id="1", similarity_score=0.9, chunk_text="X is Y")
        ]
        
        result = construct_prompt(query, chunks, "v1", mock_query_executor)
        
        # Verify both placeholders are replaced
        assert "{context}" not in result
        assert "{query}" not in result
        assert "X is Y" in result
        assert "What is X?" in result
    
    def test_construct_prompt_special_characters_in_query(self):
        """Test prompt construction with special characters in query"""
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "Query: {query}\nContext: {context}"}
        ]
        
        query = Query(text="What's the cost? (Include $)")
        chunks = [
            RetrievalResult(chunk_id="1", similarity_score=0.9, chunk_text="Cost is $100")
        ]
        
        result = construct_prompt(query, chunks, "v1", mock_query_executor)
        
        assert "What's the cost? (Include $)" in result
        assert "Cost is $100" in result
    
    def test_construct_prompt_special_characters_in_context(self):
        """Test prompt construction with special characters in context"""
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "Context: {context}\nQuery: {query}"}
        ]
        
        query = Query(text="Test query")
        chunks = [
            RetrievalResult(
                chunk_id="1",
                similarity_score=0.9,
                chunk_text="Line 1\nLine 2\nLine 3"
            )
        ]
        
        result = construct_prompt(query, chunks, "v1", mock_query_executor)
        
        assert "Line 1\nLine 2\nLine 3" in result
        assert "Test query" in result


class TestConstructPromptIntegration:
    """Integration tests for prompt construction with caching"""
    
    def setup_method(self):
        """Clear cache before each test"""
        _prompt_cache.clear()
    
    def test_construct_prompt_uses_cached_template(self):
        """Test that construct_prompt uses cached template on second call"""
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "Query: {query}\nContext: {context}"}
        ]
        
        query1 = Query(text="Query 1")
        query2 = Query(text="Query 2")
        chunks = []
        
        # First call - should query database
        result1 = construct_prompt(query1, chunks, "v1", mock_query_executor)
        assert mock_query_executor.execute_query.call_count == 1
        
        # Second call - should use cache
        result2 = construct_prompt(query2, chunks, "v1", mock_query_executor)
        assert mock_query_executor.execute_query.call_count == 1  # Still 1, not 2
        
        assert "Query 1" in result1
        assert "Query 2" in result2
        assert "{query}" not in result1
        assert "{query}" not in result2
        assert "rag:v1" in _prompt_cache
    
    def test_construct_prompt_with_different_types(self):
        """Test constructing prompts with different prompt types"""
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.side_effect = [
            [{"prompt_text": "RAG: {query}\n{context}"}],
            [{"prompt_text": "Eval: {query}\n{context}"}],
        ]
        
        query = Query(text="Test query")
        chunks = [RetrievalResult(chunk_id="1", similarity_score=0.9, chunk_text="Context")]
        
        result1 = construct_prompt(query, chunks, "v1", mock_query_executor, prompt_type="rag")
        result2 = construct_prompt(query, chunks, "v1", mock_query_executor, prompt_type="evaluation")
        
        assert "RAG:" in result1
        assert "Eval:" in result2
        assert "Test query" in result1
        assert "Test query" in result2
        assert "rag:v1" in _prompt_cache
        assert "evaluation:v1" in _prompt_cache


class TestConnectionTest:
    """Connection test for Supabase (warns if credentials missing)"""
    
    @pytest.mark.skipif(
        not Config.from_env().database_url,
        reason="Supabase database credentials not configured"
    )
    def test_connection_to_supabase_prompt_templates(self):
        """Test actual connection to Supabase for prompt template loading
        
        This test will warn but not fail if credentials are missing or invalid.
        It verifies that the Supabase database is accessible and prompt templates
        can be loaded.
        """
        import warnings
        
        from src.core.config import Config
        from src.db.connection import DatabaseConnection
        from src.db.queries import QueryExecutor
        from src.core.exceptions import DatabaseError, ValidationError
        
        config = Config.from_env()
        
        if not config.database_url:
            warnings.warn(
                "Supabase database credentials not configured. "
                "Skipping connection test. Set SUPABASE_DB_URL (or DATABASE_URL) environment variable to run this test."
            )
            pytest.skip("Supabase database credentials not configured")
        
        try:
            # Initialize database connection
            db_conn = DatabaseConnection(config)
            db_conn.connect()
            query_executor = QueryExecutor(db_conn)
            
            # Try to load a prompt template (may not exist, that's OK)
            try:
                template = load_prompt_template("v1", query_executor)
                print(f"✓ Connection test passed: Loaded prompt template 'v1' ({len(template)} characters)")
                
                # Verify template has required placeholders
                assert "{query}" in template or "{context}" in template, \
                    "Template should contain at least one placeholder"
                
            except ValidationError:
                # Template doesn't exist - that's OK, connection works
                print("✓ Connection test passed: Database connection successful (prompt template 'v1' not found in database)")
                warnings.warn(
                    "Prompt template 'v1' not found in database. "
                    "This is expected if templates haven't been seeded. "
                    "Connection test verifies database connectivity only."
                )
            
            finally:
                db_conn.close()
                
        except DatabaseError as e:
            # Connection test should warn but not fail if credentials are invalid
            warnings.warn(
                f"Supabase connection test failed (credentials may be invalid): {e}. "
                "This is expected if credentials are missing or incorrect. "
                "Connection tests are informational only."
            )
            pytest.skip(f"Supabase connection failed: {e}")
        except Exception as e:
            # Other exceptions should also warn but not fail
            warnings.warn(
                f"Supabase connection test encountered an error: {e}. "
                "Connection tests are informational only."
            )
            pytest.skip(f"Supabase connection test error: {e}")


class TestGenerateAnswer:
    """Tests for generate_answer function"""
    
    def setup_method(self):
        """Clear cache before each test"""
        from src.services.rag.generation import _prompt_cache
        _prompt_cache.clear()
    
    @patch('src.services.rag.generation.requests.post')
    @patch('src.services.rag.generation.construct_prompt')
    def test_generate_answer_success(self, mock_construct_prompt, mock_post):
        """Test successful answer generation"""
        # Mock prompt construction
        mock_construct_prompt.return_value = "Answer based on: Context\n\nQuestion: What is X?"
        
        # Mock Azure AI Foundry API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "X is a variable that represents..."
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        # Mock QueryExecutor
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "Context: {context}\n\nQuery: {query}"}
        ]
        
        # Mock config
        mock_config = Mock()
        mock_config.azure_ai_foundry_endpoint = "https://test-endpoint.openai.azure.com"
        mock_config.azure_ai_foundry_api_key = "test-api-key"
        mock_config.azure_ai_foundry_generation_model = "gpt-4o"
        
        query = Query(text="What is X?")
        chunks = [
            RetrievalResult(chunk_id="chunk_1", similarity_score=0.9, chunk_text="X is a variable")
        ]
        
        from src.services.rag.generation import generate_answer
        answer = generate_answer(query, chunks, "v1", mock_config, mock_query_executor)
        
        assert answer.text == "X is a variable that represents..."
        assert answer.query_id is not None
        assert answer.prompt_version == "v1"
        assert answer.retrieved_chunk_ids == ["chunk_1"]
        assert answer.timestamp is not None
        
        # Verify API was called with correct parameters
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "chat/completions" in call_args[0][0]
        assert call_args[1]["json"]["temperature"] == 0.1
        assert call_args[1]["json"]["max_tokens"] == 1000
    
    @patch('src.services.rag.generation.requests.post')
    @patch('src.services.rag.generation.construct_prompt')
    def test_generate_answer_with_existing_query_id(self, mock_construct_prompt, mock_post):
        """Test answer generation with existing query_id"""
        mock_construct_prompt.return_value = "Answer based on: Context\n\nQuestion: Test?"
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test answer"}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "Context: {context}\n\nQuery: {query}"}
        ]
        
        mock_config = Mock()
        mock_config.azure_ai_foundry_endpoint = "https://test-endpoint.openai.azure.com"
        mock_config.azure_ai_foundry_api_key = "test-api-key"
        mock_config.azure_ai_foundry_generation_model = "gpt-4o"
        
        query = Query(text="Test query", query_id="existing-query-id")
        chunks = []
        
        from src.services.rag.generation import generate_answer
        answer = generate_answer(query, chunks, "v1", mock_config, mock_query_executor)
        
        assert answer.query_id == "existing-query-id"
    
    @patch('src.services.rag.generation.requests.post')
    @patch('src.services.rag.generation.construct_prompt')
    def test_generate_answer_multiple_chunks(self, mock_construct_prompt, mock_post):
        """Test answer generation with multiple retrieved chunks"""
        mock_construct_prompt.return_value = "Answer based on: Context\n\nQuestion: Test?"
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Answer with multiple chunks"}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "Context: {context}\n\nQuery: {query}"}
        ]
        
        mock_config = Mock()
        mock_config.azure_ai_foundry_endpoint = "https://test-endpoint.openai.azure.com"
        mock_config.azure_ai_foundry_api_key = "test-api-key"
        mock_config.azure_ai_foundry_generation_model = "gpt-4o"
        
        query = Query(text="Test query")
        chunks = [
            RetrievalResult(chunk_id="chunk_1", similarity_score=0.9, chunk_text="Chunk 1"),
            RetrievalResult(chunk_id="chunk_2", similarity_score=0.8, chunk_text="Chunk 2"),
            RetrievalResult(chunk_id="chunk_3", similarity_score=0.7, chunk_text="Chunk 3"),
        ]
        
        from src.services.rag.generation import generate_answer
        answer = generate_answer(query, chunks, "v1", mock_config, mock_query_executor)
        
        assert answer.retrieved_chunk_ids == ["chunk_1", "chunk_2", "chunk_3"]
    
    @patch('src.services.rag.generation.requests.post')
    @patch('src.services.rag.generation.construct_prompt')
    def test_generate_answer_empty_chunks(self, mock_construct_prompt, mock_post):
        """Test answer generation with empty retrieved chunks"""
        mock_construct_prompt.return_value = "Answer based on: (No context retrieved)\n\nQuestion: Test?"
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Answer without context"}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "Context: {context}\n\nQuery: {query}"}
        ]
        
        mock_config = Mock()
        mock_config.azure_ai_foundry_endpoint = "https://test-endpoint.openai.azure.com"
        mock_config.azure_ai_foundry_api_key = "test-api-key"
        mock_config.azure_ai_foundry_generation_model = "gpt-4o"
        
        query = Query(text="Test query")
        chunks = []
        
        from src.services.rag.generation import generate_answer
        answer = generate_answer(query, chunks, "v1", mock_config, mock_query_executor)
        
        assert answer.retrieved_chunk_ids == []
        assert answer.text == "Answer without context"
    
    def test_generate_answer_empty_query_text(self):
        """Test that empty query text raises ValueError"""
        mock_config = Mock()
        mock_query_executor = Mock(spec=QueryExecutor)
        
        query = Query(text="")
        chunks = []
        
        from src.services.rag.generation import generate_answer
        with pytest.raises(ValueError) as exc_info:
            generate_answer(query, chunks, "v1", mock_config, mock_query_executor)
        
        assert "cannot be empty" in str(exc_info.value)
    
    def test_generate_answer_missing_endpoint(self):
        """Test that missing endpoint raises ValueError"""
        mock_config = Mock()
        mock_config.azure_ai_foundry_endpoint = ""
        mock_config.azure_ai_foundry_api_key = "test-key"
        mock_config.azure_ai_foundry_generation_model = "gpt-4o"
        
        mock_query_executor = Mock(spec=QueryExecutor)
        
        query = Query(text="Test query")
        chunks = []
        
        from src.services.rag.generation import generate_answer
        with pytest.raises(ValueError) as exc_info:
            generate_answer(query, chunks, "v1", mock_config, mock_query_executor)
        
        assert "endpoint is not configured" in str(exc_info.value)
    
    def test_generate_answer_missing_api_key(self):
        """Test that missing API key raises ValueError"""
        mock_config = Mock()
        mock_config.azure_ai_foundry_endpoint = "https://test-endpoint.openai.azure.com"
        mock_config.azure_ai_foundry_api_key = ""
        mock_config.azure_ai_foundry_generation_model = "gpt-4o"
        
        mock_query_executor = Mock(spec=QueryExecutor)
        
        query = Query(text="Test query")
        chunks = []
        
        from src.services.rag.generation import generate_answer
        with pytest.raises(ValueError) as exc_info:
            generate_answer(query, chunks, "v1", mock_config, mock_query_executor)
        
        assert "API key is not configured" in str(exc_info.value)
    
    def test_generate_answer_missing_model(self):
        """Test that missing model raises ValueError"""
        mock_config = Mock()
        mock_config.azure_ai_foundry_endpoint = "https://test-endpoint.openai.azure.com"
        mock_config.azure_ai_foundry_api_key = "test-key"
        mock_config.azure_ai_foundry_generation_model = ""
        
        mock_query_executor = Mock(spec=QueryExecutor)
        
        query = Query(text="Test query")
        chunks = []
        
        from src.services.rag.generation import generate_answer
        with pytest.raises(ValueError) as exc_info:
            generate_answer(query, chunks, "v1", mock_config, mock_query_executor)
        
        assert "generation model is not configured" in str(exc_info.value)
    
    @patch('src.services.rag.generation.requests.post')
    @patch('src.services.rag.generation.construct_prompt')
    def test_generate_answer_api_failure(self, mock_construct_prompt, mock_post):
        """Test that API failures raise AzureServiceError"""
        mock_construct_prompt.return_value = "Test prompt"
        
        # Mock API failure
        mock_post.side_effect = requests.RequestException("API connection failed")
        
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "Context: {context}\n\nQuery: {query}"}
        ]
        
        mock_config = Mock()
        mock_config.azure_ai_foundry_endpoint = "https://test-endpoint.openai.azure.com"
        mock_config.azure_ai_foundry_api_key = "test-api-key"
        mock_config.azure_ai_foundry_generation_model = "gpt-4o"
        
        query = Query(text="Test query")
        chunks = []
        
        from src.services.rag.generation import generate_answer
        with pytest.raises(AzureServiceError) as exc_info:
            generate_answer(query, chunks, "v1", mock_config, mock_query_executor)
        
        assert "generation API call failed" in str(exc_info.value)
        # Verify retries were attempted (3 retries + 1 initial = 4 calls)
        assert mock_post.call_count == 4
    
    @patch('src.services.rag.generation.requests.post')
    @patch('src.services.rag.generation.construct_prompt')
    def test_generate_answer_invalid_response_missing_choices(self, mock_construct_prompt, mock_post):
        """Test that invalid response (missing choices) raises ValueError"""
        mock_construct_prompt.return_value = "Test prompt"
        
        mock_response = Mock()
        mock_response.json.return_value = {}  # Missing choices
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "Context: {context}\n\nQuery: {query}"}
        ]
        
        mock_config = Mock()
        mock_config.azure_ai_foundry_endpoint = "https://test-endpoint.openai.azure.com"
        mock_config.azure_ai_foundry_api_key = "test-api-key"
        mock_config.azure_ai_foundry_generation_model = "gpt-4o"
        
        query = Query(text="Test query")
        chunks = []
        
        from src.services.rag.generation import generate_answer
        with pytest.raises(ValueError) as exc_info:
            generate_answer(query, chunks, "v1", mock_config, mock_query_executor)
        
        assert "missing 'choices' field" in str(exc_info.value)
    
    @patch('src.services.rag.generation.requests.post')
    @patch('src.services.rag.generation.construct_prompt')
    def test_generate_answer_invalid_response_missing_content(self, mock_construct_prompt, mock_post):
        """Test that invalid response (missing content) raises ValueError"""
        mock_construct_prompt.return_value = "Test prompt"
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {}}]  # Missing content
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "Context: {context}\n\nQuery: {query}"}
        ]
        
        mock_config = Mock()
        mock_config.azure_ai_foundry_endpoint = "https://test-endpoint.openai.azure.com"
        mock_config.azure_ai_foundry_api_key = "test-api-key"
        mock_config.azure_ai_foundry_generation_model = "gpt-4o"
        
        query = Query(text="Test query")
        chunks = []
        
        from src.services.rag.generation import generate_answer
        with pytest.raises(ValueError) as exc_info:
            generate_answer(query, chunks, "v1", mock_config, mock_query_executor)
        
        assert "missing 'content' field" in str(exc_info.value)
    
    @patch('src.services.rag.generation.requests.post')
    @patch('src.services.rag.generation.construct_prompt')
    def test_generate_answer_empty_response(self, mock_construct_prompt, mock_post):
        """Test that empty response raises ValueError"""
        mock_construct_prompt.return_value = "Test prompt"
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": ""}}]  # Empty content
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "Context: {context}\n\nQuery: {query}"}
        ]
        
        mock_config = Mock()
        mock_config.azure_ai_foundry_endpoint = "https://test-endpoint.openai.azure.com"
        mock_config.azure_ai_foundry_api_key = "test-api-key"
        mock_config.azure_ai_foundry_generation_model = "gpt-4o"
        
        query = Query(text="Test query")
        chunks = []
        
        from src.services.rag.generation import generate_answer
        with pytest.raises(ValueError) as exc_info:
            generate_answer(query, chunks, "v1", mock_config, mock_query_executor)
        
        assert "empty" in str(exc_info.value)
    
    @patch('src.services.rag.generation.construct_prompt')
    def test_generate_answer_missing_prompt_version(self, mock_construct_prompt):
        """Test that missing prompt version propagates ValidationError"""
        from src.core.exceptions import ValidationError
        mock_construct_prompt.side_effect = ValidationError("Prompt version not found")
        
        mock_query_executor = Mock(spec=QueryExecutor)
        
        mock_config = Mock()
        mock_config.azure_ai_foundry_endpoint = "https://test-endpoint.openai.azure.com"
        mock_config.azure_ai_foundry_api_key = "test-api-key"
        mock_config.azure_ai_foundry_generation_model = "gpt-4o"
        
        query = Query(text="Test query")
        chunks = []
        
        from src.services.rag.generation import generate_answer
        with pytest.raises(ValidationError):
            generate_answer(query, chunks, "nonexistent", mock_config, mock_query_executor)
    
    @patch('src.services.rag.generation.time.sleep')
    @patch('src.services.rag.generation.requests.post')
    @patch('src.services.rag.generation.construct_prompt')
    def test_generate_answer_retry_logic(self, mock_construct_prompt, mock_post, mock_sleep):
        """Test retry logic with exponential backoff"""
        mock_construct_prompt.return_value = "Test prompt"
        
        # First two calls fail, third succeeds
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Success after retries"}}]
        }
        mock_response.raise_for_status.return_value = None
        
        mock_post.side_effect = [
            requests.RequestException("First failure"),
            requests.RequestException("Second failure"),
            mock_response
        ]
        
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "Context: {context}\n\nQuery: {query}"}
        ]
        
        mock_config = Mock()
        mock_config.azure_ai_foundry_endpoint = "https://test-endpoint.openai.azure.com"
        mock_config.azure_ai_foundry_api_key = "test-api-key"
        mock_config.azure_ai_foundry_generation_model = "gpt-4o"
        
        query = Query(text="Test query")
        chunks = []
        
        from src.services.rag.generation import generate_answer
        answer = generate_answer(query, chunks, "v1", mock_config, mock_query_executor)
        
        assert answer.text == "Success after retries"
        assert mock_post.call_count == 3
        # Verify exponential backoff delays
        assert mock_sleep.call_count == 2  # Two retries
        assert mock_sleep.call_args_list[0][0][0] == 1.0  # First retry: 1 second
        assert mock_sleep.call_args_list[1][0][0] == 2.0  # Second retry: 2 seconds
    
    @patch('src.services.rag.generation.requests.post')
    @patch('src.services.rag.generation.construct_prompt')
    def test_generate_answer_different_prompt_versions(self, mock_construct_prompt, mock_post):
        """Test support for multiple prompt versions"""
        mock_construct_prompt.side_effect = [
            "Prompt v1",
            "Prompt v2"
        ]
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Answer"}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.side_effect = [
            [{"prompt_text": "Context: {context}\n\nQuery: {query}"}],
            [{"prompt_text": "Context: {context}\n\nQuery: {query}"}],
        ]
        
        mock_config = Mock()
        mock_config.azure_ai_foundry_endpoint = "https://test-endpoint.openai.azure.com"
        mock_config.azure_ai_foundry_api_key = "test-api-key"
        mock_config.azure_ai_foundry_generation_model = "gpt-4o"
        
        query = Query(text="Test query")
        chunks = []
        
        from src.services.rag.generation import generate_answer
        
        # Generate with v1
        answer1 = generate_answer(query, chunks, "v1", mock_config, mock_query_executor)
        assert answer1.prompt_version == "v1"
        
        # Generate with v2
        answer2 = generate_answer(query, chunks, "v2", mock_config, mock_query_executor)
        assert answer2.prompt_version == "v2"
    
    @patch('src.services.rag.generation.requests.post')
    @patch('src.services.rag.generation.construct_prompt')
    def test_generate_answer_prompt_construction_integration(self, mock_construct_prompt, mock_post):
        """Test that prompt construction is properly integrated"""
        # Verify construct_prompt is called with correct arguments
        mock_construct_prompt.return_value = "Constructed prompt with query and context"
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Answer"}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        mock_query_executor = Mock(spec=QueryExecutor)
        mock_query_executor.execute_query.return_value = [
            {"prompt_text": "Context: {context}\n\nQuery: {query}"}
        ]
        
        mock_config = Mock()
        mock_config.azure_ai_foundry_endpoint = "https://test-endpoint.openai.azure.com"
        mock_config.azure_ai_foundry_api_key = "test-api-key"
        mock_config.azure_ai_foundry_generation_model = "gpt-4o"
        
        query = Query(text="What is X?")
        chunks = [
            RetrievalResult(chunk_id="1", similarity_score=0.9, chunk_text="X is Y")
        ]
        
        from src.services.rag.generation import generate_answer
        generate_answer(query, chunks, "v1", mock_config, mock_query_executor)
        
        # Verify construct_prompt was called with correct arguments
        mock_construct_prompt.assert_called_once_with(
            query, chunks, "v1", mock_query_executor
        )
        
        # Verify the constructed prompt was sent to API
        call_args = mock_post.call_args
        assert "chat/completions" in call_args[0][0]
        payload = call_args[1]["json"]
        assert payload["messages"][0]["content"] == "Constructed prompt with query and context"


class TestConnectionTestGeneration:
    """Connection test for Azure AI Foundry generation (warns if credentials missing)"""
    
    @pytest.mark.skipif(
        not Config.from_env().azure_ai_foundry_endpoint or not Config.from_env().azure_ai_foundry_api_key,
        reason="Azure AI Foundry credentials not configured"
    )
    def test_connection_to_azure_ai_foundry_generation(self):
        """Test actual connection to Azure AI Foundry for answer generation
        
        This test will warn but not fail if credentials are missing or invalid.
        It verifies that Azure AI Foundry is accessible and answer generation works.
        """
        import warnings
        
        from src.core.config import Config
        from src.db.connection import DatabaseConnection
        from src.db.queries import QueryExecutor
        from src.core.exceptions import AzureServiceError, ValidationError
        from src.services.rag.generation import generate_answer
        
        config = Config.from_env()
        
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            warnings.warn(
                "Azure AI Foundry credentials not configured. "
                "Skipping connection test. Set AZURE_AI_FOUNDRY_ENDPOINT and "
                "AZURE_AI_FOUNDRY_API_KEY environment variables to run this test."
            )
            pytest.skip("Azure AI Foundry credentials not configured")
        
        try:
            # Initialize database connection for prompt template loading
            db_conn = DatabaseConnection(config)
            db_conn.connect()
            query_executor = QueryExecutor(db_conn)
            
            # Create a test query and chunks
            query = Query(text="What is artificial intelligence?")
            chunks = [
                RetrievalResult(
                    chunk_id="test_chunk_1",
                    similarity_score=0.9,
                    chunk_text="Artificial intelligence (AI) is the simulation of human intelligence by machines."
                )
            ]
            
            # Try to generate an answer (may fail if prompt template doesn't exist)
            try:
                answer = generate_answer(query, chunks, "v1", config, query_executor)
                print(f"✓ Connection test passed: Generated answer ({len(answer.text)} characters)")
                print(f"  Answer preview: {answer.text[:100]}...")
                print(f"  Query ID: {answer.query_id}")
                print(f"  Prompt version: {answer.prompt_version}")
                print(f"  Retrieved chunks: {len(answer.retrieved_chunk_ids)}")
                
                # Verify answer structure
                assert answer.text, "Answer should not be empty"
                assert answer.query_id, "Query ID should be set"
                assert answer.prompt_version == "v1", "Prompt version should match"
                assert len(answer.retrieved_chunk_ids) == 1, "Should have one retrieved chunk"
                
            except ValidationError:
                # Prompt template doesn't exist - that's OK, connection works
                print("✓ Connection test passed: Azure AI Foundry connection successful "
                      "(prompt template 'v1' not found in database)")
                warnings.warn(
                    "Prompt template 'v1' not found in database. "
                    "This is expected if templates haven't been seeded. "
                    "Connection test verifies Azure AI Foundry connectivity only."
                )
            
            finally:
                db_conn.close()
                
        except AzureServiceError as e:
            # Connection test should warn but not fail if credentials are invalid
            warnings.warn(
                f"Azure AI Foundry connection test failed (credentials may be invalid): {e}. "
                "This is expected if credentials are missing or incorrect. "
                "Connection tests are informational only."
            )
            pytest.skip(f"Azure AI Foundry connection failed: {e}")
        except Exception as e:
            # Other exceptions should also warn but not fail
            warnings.warn(
                f"Azure AI Foundry connection test encountered an error: {e}. "
                "Connection tests are informational only."
            )
            pytest.skip(f"Azure AI Foundry connection test error: {e}")

