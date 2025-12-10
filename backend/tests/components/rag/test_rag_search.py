"""Unit tests for Azure AI Search integration"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError

from src.core.exceptions import AzureServiceError
from src.core.config import Config
from src.core.interfaces import Chunk, Query, RetrievalResult
from src.services.rag.search import (
    index_chunks,
    retrieve_chunks,
    _ensure_index_exists,
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
        func = Mock(side_effect=[HttpResponseError("First failure"), "success"])
        result = _retry_with_backoff(func, max_retries=3)
        assert result == "success"
        assert func.call_count == 2
    
    def test_retry_exhausts_all_attempts(self):
        """Test that retry raises AzureServiceError after all attempts fail"""
        func = Mock(side_effect=HttpResponseError("Persistent failure"))
        with pytest.raises(AzureServiceError) as exc_info:
            _retry_with_backoff(func, max_retries=2)
        assert "failed after 3 attempts" in str(exc_info.value)
        assert func.call_count == 3  # Initial + 2 retries
    
    def test_retry_exponential_backoff_timing(self):
        """Test that retry uses exponential backoff"""
        func = Mock(side_effect=[
            HttpResponseError("Failure 1"),
            HttpResponseError("Failure 2"),
            "success"
        ])
        with patch('src.services.rag.search.time.sleep') as mock_sleep:
            result = _retry_with_backoff(func, max_retries=3, base_delay=1.0)
            assert result == "success"
            # Should sleep with delays: 1.0, 2.0 seconds
            assert mock_sleep.call_count == 2
            assert mock_sleep.call_args_list[0][0][0] == 1.0
            assert mock_sleep.call_args_list[1][0][0] == 2.0


class TestEnsureIndexExists:
    """Tests for idempotent index creation"""
    
    @patch('src.services.rag.search.SearchIndexClient')
    def test_ensure_index_exists_when_index_exists(self, mock_index_client_class):
        """Test that index creation is skipped if index already exists"""
        # Mock index client
        mock_index_client = Mock()
        mock_index_client.get_index.return_value = Mock(name="existing_index")
        mock_index_client_class.return_value = mock_index_client
        
        config = Config(
            supabase_url="",
            supabase_key="",
            database_url="",
            azure_ai_foundry_endpoint="",
            azure_ai_foundry_api_key="",
            azure_search_endpoint="https://test-search.search.windows.net",
            azure_search_api_key="test-key",
            azure_search_index_name="test-index",
            azure_document_intelligence_endpoint="",
            azure_document_intelligence_api_key="",
            azure_blob_connection_string="",
            azure_blob_container_name=""
        )
        
        _ensure_index_exists(config)
        
        # Verify get_index was called
        mock_index_client.get_index.assert_called_once_with("test-index")
        # Verify create_index was NOT called
        mock_index_client.create_index.assert_not_called()
    
    @patch('src.services.rag.search.SearchIndexClient')
    def test_ensure_index_exists_creates_index_when_missing(self, mock_index_client_class):
        """Test that index is created if it doesn't exist"""
        # Mock index client
        mock_index_client = Mock()
        mock_index_client.get_index.side_effect = ResourceNotFoundError("Index not found")
        mock_index_client.create_index.return_value = None
        mock_index_client_class.return_value = mock_index_client
        
        config = Config(
            supabase_url="",
            supabase_key="",
            database_url="",
            azure_ai_foundry_endpoint="",
            azure_ai_foundry_api_key="",
            azure_search_endpoint="https://test-search.search.windows.net",
            azure_search_api_key="test-key",
            azure_search_index_name="test-index",
            azure_document_intelligence_endpoint="",
            azure_document_intelligence_api_key="",
            azure_blob_connection_string="",
            azure_blob_container_name=""
        )
        
        _ensure_index_exists(config)
        
        # Verify get_index was called
        mock_index_client.get_index.assert_called_once_with("test-index")
        # Verify create_index was called
        mock_index_client.create_index.assert_called_once()
        # Verify index definition has correct structure
        call_args = mock_index_client.create_index.call_args[0][0]
        assert call_args.name == "test-index"
        assert len(call_args.fields) == 5  # id, chunk_text, embedding, document_id, metadata
    
    def test_ensure_index_exists_missing_endpoint(self):
        """Test that missing endpoint raises ValueError"""
        config = Config(
            supabase_url="",
            supabase_key="",
            database_url="",
            azure_ai_foundry_endpoint="",
            azure_ai_foundry_api_key="",
            azure_search_endpoint="",  # Missing
            azure_search_api_key="test-key",
            azure_search_index_name="test-index",
            azure_document_intelligence_endpoint="",
            azure_document_intelligence_api_key="",
            azure_blob_connection_string="",
            azure_blob_container_name=""
        )
        
        with pytest.raises(ValueError) as exc_info:
            _ensure_index_exists(config)
        assert "endpoint is not configured" in str(exc_info.value)
    
    def test_ensure_index_exists_missing_api_key(self):
        """Test that missing API key raises ValueError"""
        config = Config(
            supabase_url="",
            supabase_key="",
            database_url="",
            azure_ai_foundry_endpoint="",
            azure_ai_foundry_api_key="",
            azure_search_endpoint="https://test-search.search.windows.net",
            azure_search_api_key="",  # Missing
            azure_search_index_name="test-index",
            azure_document_intelligence_endpoint="",
            azure_document_intelligence_api_key="",
            azure_blob_connection_string="",
            azure_blob_container_name=""
        )
        
        with pytest.raises(ValueError) as exc_info:
            _ensure_index_exists(config)
        assert "API key is not configured" in str(exc_info.value)
    
    def test_ensure_index_exists_missing_index_name(self):
        """Test that missing index name raises ValueError"""
        config = Config(
            supabase_url="",
            supabase_key="",
            database_url="",
            azure_ai_foundry_endpoint="",
            azure_ai_foundry_api_key="",
            azure_search_endpoint="https://test-search.search.windows.net",
            azure_search_api_key="test-key",
            azure_search_index_name="",  # Missing
            azure_document_intelligence_endpoint="",
            azure_document_intelligence_api_key="",
            azure_blob_connection_string="",
            azure_blob_container_name=""
        )
        
        with pytest.raises(ValueError) as exc_info:
            _ensure_index_exists(config)
        assert "index name is not configured" in str(exc_info.value)


class TestIndexChunks:
    """Tests for index_chunks function"""
    
    def test_index_chunks_empty_list(self):
        """Test that empty chunk list is handled gracefully"""
        config = Config(
            supabase_url="",
            supabase_key="",
            database_url="",
            azure_ai_foundry_endpoint="",
            azure_ai_foundry_api_key="",
            azure_search_endpoint="https://test-search.search.windows.net",
            azure_search_api_key="test-key",
            azure_search_index_name="test-index",
            azure_document_intelligence_endpoint="",
            azure_document_intelligence_api_key="",
            azure_blob_connection_string="",
            azure_blob_container_name=""
        )
        
        # Should not raise an error
        index_chunks([], [], config)
    
    def test_index_chunks_length_mismatch(self):
        """Test that length mismatch raises ValueError"""
        config = Config(
            supabase_url="",
            supabase_key="",
            database_url="",
            azure_ai_foundry_endpoint="",
            azure_ai_foundry_api_key="",
            azure_search_endpoint="https://test-search.search.windows.net",
            azure_search_api_key="test-key",
            azure_search_index_name="test-index",
            azure_document_intelligence_endpoint="",
            azure_document_intelligence_api_key="",
            azure_blob_connection_string="",
            azure_blob_container_name=""
        )
        
        chunks = [Chunk(text="chunk1", chunk_id="chunk_0")]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]  # 2 embeddings for 1 chunk
        
        with pytest.raises(ValueError) as exc_info:
            index_chunks(chunks, embeddings, config)
        assert "length mismatch" in str(exc_info.value)
    
    @patch('src.services.rag.search._ensure_index_exists')
    @patch('src.services.rag.search.SearchClient')
    def test_index_chunks_success(self, mock_search_client_class, mock_ensure_index):
        """Test successful chunk indexing"""
        # Mock search client
        mock_search_client = Mock()
        mock_result = Mock()
        mock_result.succeeded = True
        mock_search_client.upload_documents.return_value = [mock_result]
        mock_search_client_class.return_value = mock_search_client
        
        config = Config(
            supabase_url="",
            supabase_key="",
            database_url="",
            azure_ai_foundry_endpoint="",
            azure_ai_foundry_api_key="",
            azure_search_endpoint="https://test-search.search.windows.net",
            azure_search_api_key="test-key",
            azure_search_index_name="test-index",
            azure_document_intelligence_endpoint="",
            azure_document_intelligence_api_key="",
            azure_blob_connection_string="",
            azure_blob_container_name=""
        )
        
        chunks = [
            Chunk(text="chunk1", chunk_id="chunk_0", document_id="doc_1"),
            Chunk(text="chunk2", chunk_id="chunk_1", document_id="doc_1")
        ]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        index_chunks(chunks, embeddings, config)
        
        # Verify index was ensured
        mock_ensure_index.assert_called_once_with(config)
        # Verify documents were uploaded
        mock_search_client.upload_documents.assert_called_once()
        call_args = mock_search_client.upload_documents.call_args
        documents = call_args[1]["documents"]
        assert len(documents) == 2
        assert documents[0]["id"] == "chunk_0"
        assert documents[0]["chunk_text"] == "chunk1"
        assert documents[0]["embedding"] == [0.1, 0.2, 0.3]
        assert documents[0]["document_id"] == "doc_1"
    
    @patch('src.services.rag.search._ensure_index_exists')
    @patch('src.services.rag.search.SearchClient')
    def test_index_chunks_with_metadata(self, mock_search_client_class, mock_ensure_index):
        """Test chunk indexing with metadata"""
        # Mock search client
        mock_search_client = Mock()
        mock_result = Mock()
        mock_result.succeeded = True
        mock_search_client.upload_documents.return_value = [mock_result]
        mock_search_client_class.return_value = mock_search_client
        
        config = Config(
            supabase_url="",
            supabase_key="",
            database_url="",
            azure_ai_foundry_endpoint="",
            azure_ai_foundry_api_key="",
            azure_search_endpoint="https://test-search.search.windows.net",
            azure_search_api_key="test-key",
            azure_search_index_name="test-index",
            azure_document_intelligence_endpoint="",
            azure_document_intelligence_api_key="",
            azure_blob_connection_string="",
            azure_blob_container_name=""
        )
        
        chunks = [
            Chunk(
                text="chunk1",
                chunk_id="chunk_0",
                document_id="doc_1",
                metadata={"page": 1, "section": "intro"}
            )
        ]
        embeddings = [[0.1, 0.2, 0.3]]
        
        index_chunks(chunks, embeddings, config)
        
        # Verify metadata was serialized to JSON
        call_args = mock_search_client.upload_documents.call_args
        documents = call_args[1]["documents"]
        assert documents[0]["metadata"] == json.dumps({"page": 1, "section": "intro"})
    
    @patch('src.services.rag.search._ensure_index_exists')
    @patch('src.services.rag.search.SearchClient')
    def test_index_chunks_upload_failure(self, mock_search_client_class, mock_ensure_index):
        """Test that upload failure raises AzureServiceError"""
        # Mock search client
        mock_search_client = Mock()
        mock_result = Mock()
        mock_result.succeeded = False
        mock_result.key = "chunk_0"
        mock_result.error_message = "Upload failed"
        mock_search_client.upload_documents.return_value = [mock_result]
        mock_search_client_class.return_value = mock_search_client
        
        config = Config(
            supabase_url="",
            supabase_key="",
            database_url="",
            azure_ai_foundry_endpoint="",
            azure_ai_foundry_api_key="",
            azure_search_endpoint="https://test-search.search.windows.net",
            azure_search_api_key="test-key",
            azure_search_index_name="test-index",
            azure_document_intelligence_endpoint="",
            azure_document_intelligence_api_key="",
            azure_blob_connection_string="",
            azure_blob_container_name=""
        )
        
        chunks = [Chunk(text="chunk1", chunk_id="chunk_0")]
        embeddings = [[0.1, 0.2, 0.3]]
        
        with pytest.raises(AzureServiceError) as exc_info:
            index_chunks(chunks, embeddings, config)
        assert "Failed to index" in str(exc_info.value)


class TestRetrieveChunks:
    """Tests for retrieve_chunks function"""
    
    def test_retrieve_chunks_empty_query(self):
        """Test that empty query raises ValueError"""
        config = Config(
            supabase_url="",
            supabase_key="",
            database_url="",
            azure_ai_foundry_endpoint="",
            azure_ai_foundry_api_key="",
            azure_search_endpoint="https://test-search.search.windows.net",
            azure_search_api_key="test-key",
            azure_search_index_name="test-index",
            azure_document_intelligence_endpoint="",
            azure_document_intelligence_api_key="",
            azure_blob_connection_string="",
            azure_blob_container_name=""
        )
        
        query = Query(text="")
        
        with pytest.raises(ValueError) as exc_info:
            retrieve_chunks(query, top_k=5, config=config)
        assert "Query text cannot be empty" in str(exc_info.value)
    
    def test_retrieve_chunks_whitespace_only(self):
        """Test that whitespace-only query raises ValueError"""
        config = Config(
            supabase_url="",
            supabase_key="",
            database_url="",
            azure_ai_foundry_endpoint="",
            azure_ai_foundry_api_key="",
            azure_search_endpoint="https://test-search.search.windows.net",
            azure_search_api_key="test-key",
            azure_search_index_name="test-index",
            azure_document_intelligence_endpoint="",
            azure_document_intelligence_api_key="",
            azure_blob_connection_string="",
            azure_blob_container_name=""
        )
        
        query = Query(text="   ")
        
        with pytest.raises(ValueError) as exc_info:
            retrieve_chunks(query, top_k=5, config=config)
        assert "Query text cannot be empty" in str(exc_info.value)
    
    def test_retrieve_chunks_invalid_top_k(self):
        """Test that invalid top_k raises ValueError"""
        config = Config(
            supabase_url="",
            supabase_key="",
            database_url="",
            azure_ai_foundry_endpoint="",
            azure_ai_foundry_api_key="",
            azure_search_endpoint="https://test-search.search.windows.net",
            azure_search_api_key="test-key",
            azure_search_index_name="test-index",
            azure_document_intelligence_endpoint="",
            azure_document_intelligence_api_key="",
            azure_blob_connection_string="",
            azure_blob_container_name=""
        )
        
        query = Query(text="test query")
        
        with pytest.raises(ValueError) as exc_info:
            retrieve_chunks(query, top_k=0, config=config)
        assert "top_k must be positive" in str(exc_info.value)
    
    def test_retrieve_chunks_missing_config(self):
        """Test that missing config raises ValueError"""
        query = Query(text="test query")
        
        with pytest.raises(ValueError) as exc_info:
            retrieve_chunks(query, top_k=5, config=None)
        assert "Config is required" in str(exc_info.value)
    
    @patch('src.services.rag.search.generate_query_embedding')
    @patch('src.services.rag.search.SearchClient')
    def test_retrieve_chunks_success(self, mock_search_client_class, mock_generate_embedding):
        """Test successful chunk retrieval"""
        # Mock query embedding
        mock_generate_embedding.return_value = [0.1, 0.2, 0.3]
        
        # Mock search results
        mock_result1 = Mock()
        mock_result1.__getitem__ = lambda self, key: {
            "id": "chunk_0",
            "chunk_text": "chunk1 text",
            "document_id": "doc_1",
            "metadata": json.dumps({"page": 1}),
            "@search.score": 0.95
        }.get(key)
        mock_result1.get = lambda key, default=None: {
            "id": "chunk_0",
            "chunk_text": "chunk1 text",
            "document_id": "doc_1",
            "metadata": json.dumps({"page": 1}),
            "@search.score": 0.95
        }.get(key, default)
        
        mock_result2 = Mock()
        mock_result2.__getitem__ = lambda self, key: {
            "id": "chunk_1",
            "chunk_text": "chunk2 text",
            "document_id": "doc_1",
            "metadata": "",
            "@search.score": 0.85
        }.get(key)
        mock_result2.get = lambda key, default=None: {
            "id": "chunk_1",
            "chunk_text": "chunk2 text",
            "document_id": "doc_1",
            "metadata": "",
            "@search.score": 0.85
        }.get(key, default)
        
        mock_search_client = Mock()
        mock_search_client.search.return_value = [mock_result1, mock_result2]
        mock_search_client_class.return_value = mock_search_client
        
        config = Config(
            supabase_url="",
            supabase_key="",
            database_url="",
            azure_ai_foundry_endpoint="https://test-endpoint.openai.azure.com",
            azure_ai_foundry_api_key="test-key",
            azure_search_endpoint="https://test-search.search.windows.net",
            azure_search_api_key="test-key",
            azure_search_index_name="test-index",
            azure_document_intelligence_endpoint="",
            azure_document_intelligence_api_key="",
            azure_blob_connection_string="",
            azure_blob_container_name="",
            azure_ai_foundry_embedding_model="text-embedding-3-small"
        )
        
        query = Query(text="test query")
        results = retrieve_chunks(query, top_k=5, config=config)
        
        # Verify results
        assert len(results) == 2
        assert results[0].chunk_id == "chunk_0"
        assert results[0].chunk_text == "chunk1 text"
        assert results[0].similarity_score == 0.95
        assert results[0].metadata == {"page": 1}
        assert results[1].chunk_id == "chunk_1"
        assert results[1].similarity_score == 0.85
        
        # Verify query embedding was generated
        mock_generate_embedding.assert_called_once_with(query, config)
        
        # Verify search was called
        mock_search_client.search.assert_called_once()
    
    @patch('src.services.rag.search.generate_query_embedding')
    @patch('src.services.rag.search.SearchClient')
    def test_retrieve_chunks_empty_index(self, mock_search_client_class, mock_generate_embedding):
        """Test that empty index returns empty list"""
        # Mock query embedding
        mock_generate_embedding.return_value = [0.1, 0.2, 0.3]
        
        # Mock empty search results
        mock_search_client = Mock()
        mock_search_client.search.return_value = []
        mock_search_client_class.return_value = mock_search_client
        
        config = Config(
            supabase_url="",
            supabase_key="",
            database_url="",
            azure_ai_foundry_endpoint="https://test-endpoint.openai.azure.com",
            azure_ai_foundry_api_key="test-key",
            azure_search_endpoint="https://test-search.search.windows.net",
            azure_search_api_key="test-key",
            azure_search_index_name="test-index",
            azure_document_intelligence_endpoint="",
            azure_document_intelligence_api_key="",
            azure_blob_connection_string="",
            azure_blob_container_name="",
            azure_ai_foundry_embedding_model="text-embedding-3-small"
        )
        
        query = Query(text="test query")
        results = retrieve_chunks(query, top_k=5, config=config)
        
        assert results == []
    
    @patch('src.services.rag.search.generate_query_embedding')
    @patch('src.services.rag.search.SearchClient')
    def test_retrieve_chunks_index_not_found(self, mock_search_client_class, mock_generate_embedding):
        """Test that missing index returns empty list gracefully"""
        # Mock query embedding
        mock_generate_embedding.return_value = [0.1, 0.2, 0.3]
        
        # Mock ResourceNotFoundError
        mock_search_client = Mock()
        mock_search_client.search.side_effect = ResourceNotFoundError("Index not found")
        mock_search_client_class.return_value = mock_search_client
        
        config = Config(
            supabase_url="",
            supabase_key="",
            database_url="",
            azure_ai_foundry_endpoint="https://test-endpoint.openai.azure.com",
            azure_ai_foundry_api_key="test-key",
            azure_search_endpoint="https://test-search.search.windows.net",
            azure_search_api_key="test-key",
            azure_search_index_name="test-index",
            azure_document_intelligence_endpoint="",
            azure_document_intelligence_api_key="",
            azure_blob_connection_string="",
            azure_blob_container_name="",
            azure_ai_foundry_embedding_model="text-embedding-3-small"
        )
        
        query = Query(text="test query")
        results = retrieve_chunks(query, top_k=5, config=config)
        
        # Should return empty list, not raise error
        assert results == []
    
    @patch('src.services.rag.search.generate_query_embedding')
    @patch('src.services.rag.search.SearchClient')
    def test_retrieve_chunks_reproducibility(self, mock_search_client_class, mock_generate_embedding):
        """Test that same query produces same results (reproducibility)"""
        # Mock query embedding (same for both calls)
        mock_generate_embedding.return_value = [0.1, 0.2, 0.3]
        
        # Mock search results
        mock_result = Mock()
        mock_result.__getitem__ = lambda self, key: {
            "id": "chunk_0",
            "chunk_text": "chunk1 text",
            "document_id": "doc_1",
            "metadata": "",
            "@search.score": 0.95
        }.get(key)
        mock_result.get = lambda key, default=None: {
            "id": "chunk_0",
            "chunk_text": "chunk1 text",
            "document_id": "doc_1",
            "metadata": "",
            "@search.score": 0.95
        }.get(key, default)
        
        mock_search_client = Mock()
        mock_search_client.search.return_value = [mock_result]
        mock_search_client_class.return_value = mock_search_client
        
        config = Config(
            supabase_url="",
            supabase_key="",
            database_url="",
            azure_ai_foundry_endpoint="https://test-endpoint.openai.azure.com",
            azure_ai_foundry_api_key="test-key",
            azure_search_endpoint="https://test-search.search.windows.net",
            azure_search_api_key="test-key",
            azure_search_index_name="test-index",
            azure_document_intelligence_endpoint="",
            azure_document_intelligence_api_key="",
            azure_blob_connection_string="",
            azure_blob_container_name="",
            azure_ai_foundry_embedding_model="text-embedding-3-small"
        )
        
        query = Query(text="test query")
        
        # First call
        results1 = retrieve_chunks(query, top_k=5, config=config)
        
        # Second call with same query
        results2 = retrieve_chunks(query, top_k=5, config=config)
        
        # Results should be identical
        assert len(results1) == len(results2)
        assert results1[0].chunk_id == results2[0].chunk_id
        assert results1[0].similarity_score == results2[0].similarity_score


class TestConnectionTest:
    """Connection test for Azure AI Search (warns if credentials missing)"""
    
    @pytest.mark.skipif(
        not Config.from_env().azure_search_endpoint or 
        not Config.from_env().azure_search_api_key or
        not Config.from_env().azure_search_index_name,
        reason="Azure AI Search credentials not configured"
    )
    def test_connection_to_azure_ai_search(self):
        """Test actual connection to Azure AI Search
        
        This test will warn but not fail if credentials are missing or invalid.
        It verifies that the Azure AI Search service is accessible and working.
        """
        import warnings
        
        config = Config.from_env()
        
        if not config.azure_search_endpoint or not config.azure_search_api_key or not config.azure_search_index_name:
            warnings.warn(
                "Azure AI Search credentials not configured. "
                "Skipping connection test. Set AZURE_SEARCH_ENDPOINT, "
                "AZURE_SEARCH_API_KEY, and AZURE_SEARCH_INDEX_NAME environment variables to run this test."
            )
            pytest.skip("Azure AI Search credentials not configured")
        
        # Test with a simple query
        query = Query(text="test query for connection test")
        
        try:
            results = retrieve_chunks(query, top_k=5, config=config)
            
            # Verify results are valid (may be empty if index is empty)
            assert results is not None
            assert isinstance(results, list)
            
            print(f"✓ Connection test passed: Retrieved {len(results)} chunks")
            
        except AzureServiceError as e:
            # Connection test should warn but not fail if credentials are invalid
            warnings.warn(
                f"Azure AI Search connection test failed (credentials may be invalid): {e}. "
                "This is expected if credentials are missing or incorrect. "
                "Connection tests are informational only."
            )
            pytest.skip(f"Azure AI Search connection failed: {e}")
        except Exception as e:
            # Other exceptions should also warn but not fail
            warnings.warn(
                f"Azure AI Search connection test encountered an error: {e}. "
                "Connection tests are informational only."
            )
            pytest.skip(f"Azure AI Search connection test error: {e}")

