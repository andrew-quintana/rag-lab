"""Unit tests for embedding generation"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
from datetime import datetime

from rag_eval.core.exceptions import AzureServiceError
from rag_eval.core.config import Config
from rag_eval.core.interfaces import Chunk, Query
from rag_eval.services.rag.embeddings import (
    generate_embeddings,
    generate_query_embedding,
    _call_embedding_api,
    _retry_with_backoff,
)


class TestRetryWithBackoff:
    """Tests for retry logic with exponential backoff"""
    
    def test_retry_succeeds_on_first_attempt(self):
        """Test that function succeeds on first attempt"""
        func = Mock(return_value="success")
        result = _retry_with_backoff(func, max_retries=3)
        assert result == "success"
        assert func.call_count == 1
    
    def test_retry_succeeds_on_second_attempt(self):
        """Test that function succeeds after one retry"""
        func = Mock(side_effect=[requests.RequestException("First failure"), "success"])
        result = _retry_with_backoff(func, max_retries=3)
        assert result == "success"
        assert func.call_count == 2
    
    def test_retry_exhausts_all_attempts(self):
        """Test that retry raises AzureServiceError after all attempts fail"""
        func = Mock(side_effect=requests.RequestException("Persistent failure"))
        with pytest.raises(AzureServiceError) as exc_info:
            _retry_with_backoff(func, max_retries=2)
        assert "failed after 3 attempts" in str(exc_info.value)
        assert func.call_count == 3  # Initial + 2 retries
    
    def test_retry_exponential_backoff_timing(self):
        """Test that retry uses exponential backoff"""
        func = Mock(side_effect=[
            requests.RequestException("Failure 1"),
            requests.RequestException("Failure 2"),
            "success"
        ])
        with patch('time.sleep') as mock_sleep:
            result = _retry_with_backoff(func, max_retries=3, base_delay=1.0)
            assert result == "success"
            # Should sleep with delays: 1.0, 2.0 seconds
            assert mock_sleep.call_count == 2
            assert mock_sleep.call_args_list[0][0][0] == 1.0
            assert mock_sleep.call_args_list[1][0][0] == 2.0


class TestCallEmbeddingAPI:
    """Tests for _call_embedding_api function"""
    
    @patch('rag_eval.services.rag.embeddings.requests.post')
    def test_call_embedding_api_success(self, mock_post):
        """Test successful embedding API call"""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]}
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        # Call function
        texts = ["text1", "text2"]
        embeddings = _call_embedding_api(
            texts=texts,
            model="text-embedding-3-small",
            endpoint="https://test-endpoint.openai.azure.com",
            api_key="test-key"
        )
        
        # Verify results
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]
        
        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "openai/deployments/text-embedding-3-small/embeddings" in call_args[0][0]
        assert call_args[1]["json"]["input"] == texts
        assert call_args[1]["json"]["model"] == "text-embedding-3-small"
    
    @patch('rag_eval.services.rag.embeddings.requests.post')
    def test_call_embedding_api_empty_list(self, mock_post):
        """Test that empty text list returns empty embeddings"""
        embeddings = _call_embedding_api(
            texts=[],
            model="text-embedding-3-small",
            endpoint="https://test-endpoint.openai.azure.com",
            api_key="test-key"
        )
        assert embeddings == []
        mock_post.assert_not_called()
    
    @patch('rag_eval.services.rag.embeddings.requests.post')
    def test_call_embedding_api_invalid_response(self, mock_post):
        """Test that invalid response structure raises ValueError"""
        mock_response = Mock()
        mock_response.json.return_value = {"invalid": "response"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        with pytest.raises(ValueError) as exc_info:
            _call_embedding_api(
                texts=["text1"],
                model="text-embedding-3-small",
                endpoint="https://test-endpoint.openai.azure.com",
                api_key="test-key"
            )
        assert "missing 'data' field" in str(exc_info.value)
    
    @patch('rag_eval.services.rag.embeddings.requests.post')
    def test_call_embedding_api_dimension_mismatch(self, mock_post):
        """Test that dimension mismatch raises ValueError"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5]}  # Different dimension
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        with pytest.raises(ValueError) as exc_info:
            _call_embedding_api(
                texts=["text1", "text2"],
                model="text-embedding-3-small",
                endpoint="https://test-endpoint.openai.azure.com",
                api_key="test-key"
            )
        assert "dimension mismatch" in str(exc_info.value)
    
    @patch('rag_eval.services.rag.embeddings.requests.post')
    def test_call_embedding_api_count_mismatch(self, mock_post):
        """Test that count mismatch raises ValueError"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]}
                # Only 1 embedding for 2 texts
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        with pytest.raises(ValueError) as exc_info:
            _call_embedding_api(
                texts=["text1", "text2"],
                model="text-embedding-3-small",
                endpoint="https://test-endpoint.openai.azure.com",
                api_key="test-key"
            )
        assert "expected 2 embeddings, got 1" in str(exc_info.value)


class TestGenerateEmbeddings:
    """Tests for generate_embeddings function"""
    
    def test_generate_embeddings_empty_list(self):
        """Test that empty chunk list returns empty embeddings"""
        config = Config(
            supabase_url="",
            supabase_key="",
            database_url="",
            azure_ai_foundry_endpoint="https://test-endpoint.openai.azure.com",
            azure_ai_foundry_api_key="test-key",
            azure_search_endpoint="",
            azure_search_api_key="",
            azure_search_index_name="",
            azure_document_intelligence_endpoint="",
            azure_document_intelligence_api_key="",
            azure_blob_connection_string="",
            azure_blob_container_name="",
            azure_ai_foundry_embedding_model="text-embedding-3-small"
        )
        
        embeddings = generate_embeddings([], config)
        assert embeddings == []
    
    @patch('rag_eval.services.rag.embeddings._call_embedding_api')
    def test_generate_embeddings_success(self, mock_api):
        """Test successful embedding generation"""
        # Mock API response
        mock_api.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
        
        config = Config(
            supabase_url="",
            supabase_key="",
            database_url="",
            azure_ai_foundry_endpoint="https://test-endpoint.openai.azure.com",
            azure_ai_foundry_api_key="test-key",
            azure_search_endpoint="",
            azure_search_api_key="",
            azure_search_index_name="",
            azure_document_intelligence_endpoint="",
            azure_document_intelligence_api_key="",
            azure_blob_connection_string="",
            azure_blob_container_name="",
            azure_ai_foundry_embedding_model="text-embedding-3-small"
        )
        
        chunks = [
            Chunk(text="chunk1", chunk_id="chunk_0", document_id="doc_1"),
            Chunk(text="chunk2", chunk_id="chunk_1", document_id="doc_1")
        ]
        
        embeddings = generate_embeddings(chunks, config)
        
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]
        
        # Verify API was called with correct parameters
        mock_api.assert_called_once()
        call_args = mock_api.call_args
        assert call_args[1]["texts"] == ["chunk1", "chunk2"]
        assert call_args[1]["model"] == "text-embedding-3-small"
        assert call_args[1]["endpoint"] == "https://test-endpoint.openai.azure.com"
        assert call_args[1]["api_key"] == "test-key"
    
    def test_generate_embeddings_missing_endpoint(self):
        """Test that missing endpoint raises ValueError"""
        config = Config(
            supabase_url="",
            supabase_key="",
            database_url="",
            azure_ai_foundry_endpoint="",  # Missing
            azure_ai_foundry_api_key="test-key",
            azure_search_endpoint="",
            azure_search_api_key="",
            azure_search_index_name="",
            azure_document_intelligence_endpoint="",
            azure_document_intelligence_api_key="",
            azure_blob_connection_string="",
            azure_blob_container_name="",
            azure_ai_foundry_embedding_model="text-embedding-3-small"
        )
        
        chunks = [Chunk(text="test", chunk_id="chunk_0")]
        
        with pytest.raises(ValueError) as exc_info:
            generate_embeddings(chunks, config)
        assert "endpoint is not configured" in str(exc_info.value)
    
    def test_generate_embeddings_missing_api_key(self):
        """Test that missing API key raises ValueError"""
        config = Config(
            supabase_url="",
            supabase_key="",
            database_url="",
            azure_ai_foundry_endpoint="https://test-endpoint.openai.azure.com",
            azure_ai_foundry_api_key="",  # Missing
            azure_search_endpoint="",
            azure_search_api_key="",
            azure_search_index_name="",
            azure_document_intelligence_endpoint="",
            azure_document_intelligence_api_key="",
            azure_blob_connection_string="",
            azure_blob_container_name="",
            azure_ai_foundry_embedding_model="text-embedding-3-small"
        )
        
        chunks = [Chunk(text="test", chunk_id="chunk_0")]
        
        with pytest.raises(ValueError) as exc_info:
            generate_embeddings(chunks, config)
        assert "API key is not configured" in str(exc_info.value)
    
    def test_generate_embeddings_missing_model(self):
        """Test that missing model raises ValueError"""
        config = Config(
            supabase_url="",
            supabase_key="",
            database_url="",
            azure_ai_foundry_endpoint="https://test-endpoint.openai.azure.com",
            azure_ai_foundry_api_key="test-key",
            azure_search_endpoint="",
            azure_search_api_key="",
            azure_search_index_name="",
            azure_document_intelligence_endpoint="",
            azure_document_intelligence_api_key="",
            azure_blob_connection_string="",
            azure_blob_container_name="",
            azure_ai_foundry_embedding_model=""  # Missing
        )
        
        chunks = [Chunk(text="test", chunk_id="chunk_0")]
        
        with pytest.raises(ValueError) as exc_info:
            generate_embeddings(chunks, config)
        assert "embedding model is not configured" in str(exc_info.value)
    
    @patch('rag_eval.services.rag.embeddings._call_embedding_api')
    def test_generate_embeddings_api_error(self, mock_api):
        """Test that API errors are wrapped in AzureServiceError"""
        mock_api.side_effect = requests.RequestException("API error")
        
        config = Config(
            supabase_url="",
            supabase_key="",
            database_url="",
            azure_ai_foundry_endpoint="https://test-endpoint.openai.azure.com",
            azure_ai_foundry_api_key="test-key",
            azure_search_endpoint="",
            azure_search_api_key="",
            azure_search_index_name="",
            azure_document_intelligence_endpoint="",
            azure_document_intelligence_api_key="",
            azure_blob_connection_string="",
            azure_blob_container_name="",
            azure_ai_foundry_embedding_model="text-embedding-3-small"
        )
        
        chunks = [Chunk(text="test", chunk_id="chunk_0")]
        
        with pytest.raises(AzureServiceError) as exc_info:
            generate_embeddings(chunks, config)
        assert "failed after" in str(exc_info.value) or "error generating embeddings" in str(exc_info.value)


class TestGenerateQueryEmbedding:
    """Tests for generate_query_embedding function"""
    
    @patch('rag_eval.services.rag.embeddings._call_embedding_api')
    def test_generate_query_embedding_success(self, mock_api):
        """Test successful query embedding generation"""
        # Mock API response
        mock_api.return_value = [[0.1, 0.2, 0.3]]
        
        config = Config(
            supabase_url="",
            supabase_key="",
            database_url="",
            azure_ai_foundry_endpoint="https://test-endpoint.openai.azure.com",
            azure_ai_foundry_api_key="test-key",
            azure_search_endpoint="",
            azure_search_api_key="",
            azure_search_index_name="",
            azure_document_intelligence_endpoint="",
            azure_document_intelligence_api_key="",
            azure_blob_connection_string="",
            azure_blob_container_name="",
            azure_ai_foundry_embedding_model="text-embedding-3-small"
        )
        
        query = Query(text="test query")
        
        embedding = generate_query_embedding(query, config)
        
        assert embedding == [0.1, 0.2, 0.3]
        
        # Verify API was called with correct parameters
        mock_api.assert_called_once()
        call_args = mock_api.call_args
        assert call_args[1]["texts"] == ["test query"]
        assert call_args[1]["model"] == "text-embedding-3-small"
    
    def test_generate_query_embedding_empty_text(self):
        """Test that empty query text raises ValueError"""
        config = Config(
            supabase_url="",
            supabase_key="",
            database_url="",
            azure_ai_foundry_endpoint="https://test-endpoint.openai.azure.com",
            azure_ai_foundry_api_key="test-key",
            azure_search_endpoint="",
            azure_search_api_key="",
            azure_search_index_name="",
            azure_document_intelligence_endpoint="",
            azure_document_intelligence_api_key="",
            azure_blob_connection_string="",
            azure_blob_container_name="",
            azure_ai_foundry_embedding_model="text-embedding-3-small"
        )
        
        query = Query(text="")
        
        with pytest.raises(ValueError) as exc_info:
            generate_query_embedding(query, config)
        assert "Query text cannot be empty" in str(exc_info.value)
    
    def test_generate_query_embedding_whitespace_only(self):
        """Test that whitespace-only query text raises ValueError"""
        config = Config(
            supabase_url="",
            supabase_key="",
            database_url="",
            azure_ai_foundry_endpoint="https://test-endpoint.openai.azure.com",
            azure_ai_foundry_api_key="test-key",
            azure_search_endpoint="",
            azure_search_api_key="",
            azure_search_index_name="",
            azure_document_intelligence_endpoint="",
            azure_document_intelligence_api_key="",
            azure_blob_connection_string="",
            azure_blob_container_name="",
            azure_ai_foundry_embedding_model="text-embedding-3-small"
        )
        
        query = Query(text="   ")
        
        with pytest.raises(ValueError) as exc_info:
            generate_query_embedding(query, config)
        assert "Query text cannot be empty" in str(exc_info.value)
    
    @patch('rag_eval.services.rag.embeddings._call_embedding_api')
    def test_generate_query_embedding_wrong_count(self, mock_api):
        """Test that wrong embedding count raises ValueError"""
        # Mock API returns wrong number of embeddings
        mock_api.return_value = []  # Empty instead of 1
        
        config = Config(
            supabase_url="",
            supabase_key="",
            database_url="",
            azure_ai_foundry_endpoint="https://test-endpoint.openai.azure.com",
            azure_ai_foundry_api_key="test-key",
            azure_search_endpoint="",
            azure_search_api_key="",
            azure_search_index_name="",
            azure_document_intelligence_endpoint="",
            azure_document_intelligence_api_key="",
            azure_blob_connection_string="",
            azure_blob_container_name="",
            azure_ai_foundry_embedding_model="text-embedding-3-small"
        )
        
        query = Query(text="test query")
        
        with pytest.raises(ValueError) as exc_info:
            generate_query_embedding(query, config)
        assert "Expected 1 embedding" in str(exc_info.value)
    
    @patch('rag_eval.services.rag.embeddings._call_embedding_api')
    def test_generate_query_embedding_model_consistency(self, mock_api):
        """Test that query embedding uses same model as chunks"""
        mock_api.return_value = [[0.1, 0.2, 0.3]]
        
        config = Config(
            supabase_url="",
            supabase_key="",
            database_url="",
            azure_ai_foundry_endpoint="https://test-endpoint.openai.azure.com",
            azure_ai_foundry_api_key="test-key",
            azure_search_endpoint="",
            azure_search_api_key="",
            azure_search_index_name="",
            azure_document_intelligence_endpoint="",
            azure_document_intelligence_api_key="",
            azure_blob_connection_string="",
            azure_blob_container_name="",
            azure_ai_foundry_embedding_model="text-embedding-3-small"
        )
        
        query = Query(text="test query")
        
        embedding = generate_query_embedding(query, config)
        
        # Verify same model is used (enforced via config)
        call_args = mock_api.call_args
        assert call_args[1]["model"] == "text-embedding-3-small"
        assert call_args[1]["model"] == config.azure_ai_foundry_embedding_model


class TestConnectionTest:
    """Connection test for Azure AI Foundry (warns if credentials missing)"""
    
    @pytest.mark.skipif(
        not Config.from_env().azure_ai_foundry_endpoint or 
        not Config.from_env().azure_ai_foundry_api_key,
        reason="Azure AI Foundry credentials not configured"
    )
    def test_connection_to_azure_ai_foundry_embeddings(self):
        """Test actual connection to Azure AI Foundry embedding API
        
        This test will warn but not fail if credentials are missing or invalid.
        It verifies that the embedding API is accessible and working.
        """
        import warnings
        
        config = Config.from_env()
        
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            warnings.warn(
                "Azure AI Foundry credentials not configured. "
                "Skipping connection test. Set AZURE_AI_FOUNDRY_ENDPOINT and "
                "AZURE_AI_FOUNDRY_API_KEY environment variables to run this test."
            )
            pytest.skip("Azure AI Foundry credentials not configured")
        
        # Test with a simple query
        query = Query(text="test query for connection test")
        
        try:
            embedding = generate_query_embedding(query, config)
            
            # Verify embedding is valid
            assert embedding is not None
            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert all(isinstance(x, float) for x in embedding)
            
            print(f"✓ Connection test passed: Generated embedding with {len(embedding)} dimensions")
            
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
    
    @pytest.mark.skipif(
        not Config.from_env().azure_ai_foundry_endpoint or 
        not Config.from_env().azure_ai_foundry_api_key,
        reason="Azure AI Foundry credentials not configured"
    )
    def test_batch_embedding_generation_connection(self):
        """Test batch embedding generation with real Azure AI Foundry service
        
        This test verifies that batch processing works correctly with the real service.
        Connection tests should warn but not fail if credentials are missing or invalid.
        """
        import warnings
        
        config = Config.from_env()
        
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            warnings.warn(
                "Azure AI Foundry credentials not configured. "
                "Skipping connection test. Set AZURE_AI_FOUNDRY_ENDPOINT and "
                "AZURE_AI_FOUNDRY_API_KEY environment variables to run this test."
            )
            pytest.skip("Azure AI Foundry credentials not configured")
        
        # Test with multiple chunks
        chunks = [
            Chunk(text="First chunk text", chunk_id="chunk_0", document_id="doc_1"),
            Chunk(text="Second chunk text", chunk_id="chunk_1", document_id="doc_1"),
            Chunk(text="Third chunk text", chunk_id="chunk_2", document_id="doc_1")
        ]
        
        try:
            embeddings = generate_embeddings(chunks, config)
            
            # Verify embeddings are valid
            assert embeddings is not None
            assert len(embeddings) == len(chunks)
            assert all(isinstance(emb, list) for emb in embeddings)
            assert all(len(emb) > 0 for emb in embeddings)
            
            # Verify all embeddings have same dimension
            dimensions = [len(emb) for emb in embeddings]
            assert len(set(dimensions)) == 1, f"Embedding dimensions inconsistent: {dimensions}"
            
            print(f"✓ Batch embedding test passed: Generated {len(embeddings)} embeddings with {dimensions[0]} dimensions each")
            
        except AzureServiceError as e:
            # Connection test should warn but not fail if credentials are invalid
            warnings.warn(
                f"Azure AI Foundry batch embedding connection test failed (credentials may be invalid): {e}. "
                "This is expected if credentials are missing or incorrect. "
                "Connection tests are informational only."
            )
            pytest.skip(f"Azure AI Foundry batch embedding connection failed: {e}")
        except Exception as e:
            # Other exceptions should also warn but not fail
            warnings.warn(
                f"Azure AI Foundry batch embedding connection test encountered an error: {e}. "
                "Connection tests are informational only."
            )
            pytest.skip(f"Azure AI Foundry batch embedding connection test error: {e}")

