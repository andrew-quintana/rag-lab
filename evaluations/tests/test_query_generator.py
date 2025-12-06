"""Unit and integration tests for query generator"""

import pytest
import json
import tempfile
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError

# Add backend directory to path for rag_eval imports
backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Add evaluations directory to path for in_corpus_eval imports
evaluations_path = Path(__file__).parent.parent
sys.path.insert(0, str(evaluations_path))

from rag_eval.core.config import Config
from rag_eval.core.interfaces import Chunk
from rag_eval.core.exceptions import AzureServiceError

from _shared.scripts.query_generator import (
    sample_chunks_from_index,
    generate_query_from_chunk,
    generate_queries_from_index,
    save_queries_to_json,
    _retry_with_backoff
)


@pytest.fixture
def mock_config():
    """Create a mock Config object"""
    config = Mock(spec=Config)
    config.azure_search_endpoint = "https://test-search.search.windows.net"
    config.azure_search_api_key = "test-search-key"
    config.azure_search_index_name = "test-index"
    config.azure_ai_foundry_endpoint = "https://test-foundry.openai.azure.com"
    config.azure_ai_foundry_api_key = "test-foundry-key"
    config.azure_ai_foundry_embedding_model = "text-embedding-3-small"
    config.azure_ai_foundry_generation_model = "gpt-4o-mini"
    return config


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing"""
    return [
        Chunk(
            text="The coverage limit for this plan is $500,000 per year. This includes all medical expenses.",
            chunk_id="chunk_1",
            document_id="doc_123",
            metadata={"page": 1}
        ),
        Chunk(
            text="Deductibles vary by plan type. Standard plans have a $1,000 deductible.",
            chunk_id="chunk_2",
            document_id="doc_123",
            metadata={"page": 2}
        ),
        Chunk(
            text="Preventive care services are covered at 100% with no deductible required.",
            chunk_id="chunk_3",
            document_id="doc_456",
            metadata={"page": 1}
        ),
    ]


@pytest.fixture
def sample_search_results():
    """Create sample Azure AI Search results"""
    return [
        {
            "id": "chunk_1",
            "chunk_text": "The coverage limit for this plan is $500,000 per year.",
            "document_id": "doc_123",
            "metadata": json.dumps({"page": 1})
        },
        {
            "id": "chunk_2",
            "chunk_text": "Deductibles vary by plan type.",
            "document_id": "doc_123",
            "metadata": json.dumps({"page": 2})
        },
        {
            "id": "chunk_3",
            "chunk_text": "Preventive care services are covered at 100%.",
            "document_id": "doc_456",
            "metadata": json.dumps({"page": 1})
        },
    ]


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider"""
    provider = Mock()
    provider.call_completion.return_value = "What is the coverage limit for this plan?"
    return provider


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
        assert func.call_count == 3
    
    def test_retry_does_not_retry_resource_not_found(self):
        """Test that ResourceNotFoundError is not retried"""
        func = Mock(side_effect=ResourceNotFoundError("Index not found"))
        with pytest.raises(ResourceNotFoundError):
            _retry_with_backoff(func, max_retries=3)
        assert func.call_count == 1


class TestSampleChunksFromIndex:
    """Tests for sampling chunks from Azure AI Search index"""
    
    @patch('_shared.scripts.query_generator.SearchClient')
    def test_sample_chunks_success(self, mock_search_client_class, mock_config, sample_search_results):
        """Test successful chunk sampling"""
        # Setup mock
        mock_search_client = Mock()
        mock_search_client_class.return_value = mock_search_client
        
        mock_results = Mock()
        mock_results.__iter__ = Mock(return_value=iter(sample_search_results))
        mock_search_client.search.return_value = mock_results
        
        # Sample chunks
        chunks = sample_chunks_from_index(mock_config, num_chunks=2)
        
        # Verify
        assert len(chunks) == 2
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert mock_search_client.search.called
    
    @patch('_shared.scripts.query_generator.SearchClient')
    def test_sample_chunks_empty_index(self, mock_search_client_class, mock_config):
        """Test sampling from empty index"""
        # Setup mock
        mock_search_client = Mock()
        mock_search_client_class.return_value = mock_search_client
        
        mock_results = Mock()
        mock_results.__iter__ = Mock(return_value=iter([]))
        mock_search_client.search.return_value = mock_results
        
        # Sample chunks
        chunks = sample_chunks_from_index(mock_config, num_chunks=10)
        
        # Verify
        assert chunks == []
    
    @patch('_shared.scripts.query_generator.SearchClient')
    def test_sample_chunks_index_not_found(self, mock_search_client_class, mock_config):
        """Test sampling when index doesn't exist"""
        # Setup mock
        mock_search_client = Mock()
        mock_search_client_class.return_value = mock_search_client
        mock_search_client.search.side_effect = ResourceNotFoundError("Index not found")
        
        # Sample chunks
        chunks = sample_chunks_from_index(mock_config, num_chunks=10)
        
        # Verify
        assert chunks == []
    
    def test_sample_chunks_invalid_config(self, mock_config):
        """Test sampling with invalid config"""
        mock_config.azure_search_endpoint = None
        with pytest.raises(ValueError, match="endpoint is not configured"):
            sample_chunks_from_index(mock_config, num_chunks=10)
    
    def test_sample_chunks_invalid_num_chunks(self, mock_config):
        """Test sampling with invalid num_chunks"""
        with pytest.raises(ValueError, match="num_chunks must be positive"):
            sample_chunks_from_index(mock_config, num_chunks=0)


class TestGenerateQueryFromChunk:
    """Tests for generating queries from chunks"""
    
    def test_generate_query_success(self, sample_chunks, mock_llm_provider):
        """Test successful query generation"""
        chunk = sample_chunks[0]
        query = generate_query_from_chunk(chunk, mock_llm_provider)
        
        # Verify
        assert isinstance(query, str)
        assert len(query) > 0
        assert mock_llm_provider.call_completion.called
        
        # Verify prompt contains chunk text
        call_args = mock_llm_provider.call_completion.call_args
        assert chunk.text[:100] in call_args[1]["prompt"] or chunk.text[:100] in call_args[0][0]
    
    def test_generate_query_empty_chunk(self, mock_llm_provider):
        """Test query generation with empty chunk"""
        chunk = Chunk(text="", chunk_id="chunk_empty")
        with pytest.raises(ValueError, match="Chunk text cannot be empty"):
            generate_query_from_chunk(chunk, mock_llm_provider)
    
    def test_generate_query_llm_failure(self, sample_chunks, mock_llm_provider):
        """Test query generation when LLM fails"""
        chunk = sample_chunks[0]
        mock_llm_provider.call_completion.side_effect = Exception("LLM error")
        
        with pytest.raises(AzureServiceError, match="Failed to generate query"):
            generate_query_from_chunk(chunk, mock_llm_provider)
    
    def test_generate_query_removes_quotes(self, sample_chunks, mock_llm_provider):
        """Test that generated query removes surrounding quotes"""
        chunk = sample_chunks[0]
        mock_llm_provider.call_completion.return_value = '"What is the coverage limit?"'
        
        query = generate_query_from_chunk(chunk, mock_llm_provider)
        
        # Verify quotes are removed
        assert not query.startswith('"')
        assert not query.endswith('"')


class TestGenerateQueriesFromIndex:
    """Tests for generating queries from index"""
    
    @patch('_shared.scripts.query_generator.get_llm_provider')
    @patch('_shared.scripts.query_generator.sample_chunks_from_index')
    def test_generate_queries_success(
        self, mock_sample_chunks, mock_get_llm_provider,
        mock_config, sample_chunks, mock_llm_provider
    ):
        """Test successful query generation"""
        # Setup mocks
        mock_sample_chunks.return_value = sample_chunks
        mock_get_llm_provider.return_value = mock_llm_provider
        
        # Generate queries
        queries = generate_queries_from_index(mock_config, num_queries=2)
        
        # Verify
        assert len(queries) == 2
        assert all("input" in q for q in queries)
        assert all("metadata" in q for q in queries)
        assert all("source_chunk_ids" in q["metadata"] for q in queries)
        assert all("document_id" in q["metadata"] for q in queries)
        # generation_method field removed - metadata only contains source_chunk_ids and document_id
    
    def test_generate_queries_with_provided_chunks(self, mock_config, sample_chunks, mock_llm_provider):
        """Test query generation with provided chunks"""
        with patch('_shared.scripts.query_generator.get_llm_provider') as mock_get_llm_provider:
            mock_get_llm_provider.return_value = mock_llm_provider
            
            # Generate queries
            queries = generate_queries_from_index(
                mock_config,
                num_queries=2,
                sample_chunks=sample_chunks
            )
            
            # Verify
            assert len(queries) == 2
    
    def test_generate_queries_invalid_num_queries(self, mock_config):
        """Test query generation with invalid num_queries"""
        with pytest.raises(ValueError, match="num_queries must be positive"):
            generate_queries_from_index(mock_config, num_queries=0)
    
    def test_generate_queries_invalid_chunks_per_query(self, mock_config, sample_chunks):
        """Test query generation with invalid chunks_per_query"""
        with pytest.raises(ValueError, match="chunks_per_query must be positive"):
            generate_queries_from_index(
                mock_config,
                num_queries=2,
                sample_chunks=sample_chunks,
                chunks_per_query=0
            )
    
    @patch('_shared.scripts.query_generator.sample_chunks_from_index')
    def test_generate_queries_no_chunks_found(self, mock_sample_chunks, mock_config):
        """Test query generation when no chunks are found"""
        mock_sample_chunks.return_value = []
        
        with pytest.raises(AzureServiceError, match="No chunks found"):
            generate_queries_from_index(mock_config, num_queries=2)
    
    @patch('_shared.scripts.query_generator.get_llm_provider')
    @patch('_shared.scripts.query_generator.sample_chunks_from_index')
    def test_generate_queries_llm_provider_failure(
        self, mock_sample_chunks, mock_get_llm_provider,
        mock_config, sample_chunks
    ):
        """Test query generation when LLM provider fails"""
        mock_sample_chunks.return_value = sample_chunks
        mock_get_llm_provider.side_effect = Exception("LLM provider error")
        
        with pytest.raises(AzureServiceError, match="Failed to initialize LLM provider"):
            generate_queries_from_index(mock_config, num_queries=2)
    
    @patch('_shared.scripts.query_generator.get_llm_provider')
    def test_generate_queries_handles_individual_failures(
        self, mock_get_llm_provider, mock_config, sample_chunks, mock_llm_provider
    ):
        """Test that individual query generation failures don't stop the process"""
        # Setup mock to fail on first call, succeed on second
        mock_llm_provider.call_completion.side_effect = [
            Exception("First failure"),
            "What is the coverage limit?"
        ]
        mock_get_llm_provider.return_value = mock_llm_provider
        
        # Generate queries - should continue after first failure
        queries = generate_queries_from_index(
            mock_config,
            num_queries=2,
            sample_chunks=sample_chunks
        )
        
        # Should have one successful query
        assert len(queries) == 1


class TestSaveQueriesToJson:
    """Tests for saving queries to JSON"""
    
    def test_save_queries_success(self):
        """Test successful saving of queries to JSON"""
        queries = [
            {
                "input": "What is the coverage limit?",
                "metadata": {
                    "source_chunk_ids": ["chunk_1"],
                    "document_id": "doc_123",
                }
            },
            {
                "input": "What is the deductible?",
                "metadata": {
                    "source_chunk_ids": ["chunk_2"],
                    "document_id": "doc_123",
                }
            }
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "eval_inputs.json"
            save_queries_to_json(queries, output_path)
            
            # Verify file exists
            assert output_path.exists()
            
            # Verify content
            with open(output_path, 'r', encoding='utf-8') as f:
                loaded_queries = json.load(f)
            
            assert len(loaded_queries) == 2
            assert loaded_queries[0]["input"] == "What is the coverage limit?"
            assert loaded_queries[1]["input"] == "What is the deductible?"
    
    def test_save_queries_creates_directory(self):
        """Test that save_queries_to_json creates directory if it doesn't exist"""
        queries = [
            {
                "input": "Test query",
                "metadata": {
                    "source_chunk_ids": ["chunk_1"],
                    "document_id": "doc_123",
                }
            }
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "eval_inputs.json"
            save_queries_to_json(queries, output_path)
            
            # Verify directory and file exist
            assert output_path.parent.exists()
            assert output_path.exists()
    
    def test_save_queries_empty_list(self):
        """Test saving empty queries list raises error"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "eval_inputs.json"
            with pytest.raises(ValueError, match="Cannot save empty queries list"):
                save_queries_to_json([], output_path)


class TestIntegration:
    """Integration tests with mocked Azure AI Search and LLM"""
    
    @patch('_shared.scripts.query_generator.get_llm_provider')
    @patch('_shared.scripts.query_generator.SearchClient')
    def test_end_to_end_query_generation(
        self, mock_search_client_class, mock_get_llm_provider,
        mock_config, sample_search_results, mock_llm_provider
    ):
        """Test end-to-end query generation with mocked services"""
        # Setup Azure AI Search mock
        mock_search_client = Mock()
        mock_search_client_class.return_value = mock_search_client
        
        mock_results = Mock()
        mock_results.__iter__ = Mock(return_value=iter(sample_search_results))
        mock_search_client.search.return_value = mock_results
        
        # Setup LLM provider mock
        mock_get_llm_provider.return_value = mock_llm_provider
        
        # Generate queries
        queries = generate_queries_from_index(mock_config, num_queries=2)
        
        # Verify structure
        assert len(queries) == 2
        assert all("input" in q for q in queries)
        assert all("metadata" in q for q in queries)
        assert all("source_chunk_ids" in q["metadata"] for q in queries)
        
        # Verify Azure AI Search was called
        assert mock_search_client.search.called
        
        # Verify LLM was called
        assert mock_llm_provider.call_completion.called
    
    @patch('_shared.scripts.query_generator.get_llm_provider')
    @patch('_shared.scripts.query_generator.SearchClient')
    def test_end_to_end_with_json_output(
        self, mock_search_client_class, mock_get_llm_provider,
        mock_config, sample_search_results, mock_llm_provider
    ):
        """Test end-to-end query generation and JSON output"""
        # Setup mocks
        mock_search_client = Mock()
        mock_search_client_class.return_value = mock_search_client
        
        mock_results = Mock()
        mock_results.__iter__ = Mock(return_value=iter(sample_search_results))
        mock_search_client.search.return_value = mock_results
        
        mock_get_llm_provider.return_value = mock_llm_provider
        
        # Generate queries
        queries = generate_queries_from_index(mock_config, num_queries=2)
        
        # Save to JSON
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "eval_inputs.json"
            save_queries_to_json(queries, output_path)
            
            # Verify JSON structure matches specification
            with open(output_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            
            assert isinstance(loaded, list)
            assert len(loaded) == 2
            for item in loaded:
                assert "input" in item
                assert "metadata" in item
                assert "source_chunk_ids" in item["metadata"]
                assert "document_id" in item["metadata"]
                # generation_method field removed


class TestMainFunction:
    """Tests for main function"""
    
    @patch('_shared.scripts.query_generator.save_queries_to_json')
    @patch('_shared.scripts.query_generator.generate_queries_from_index')
    @patch('_shared.scripts.query_generator.Config')
    def test_main_with_defaults(self, mock_config_class, mock_generate, mock_save):
        """Test main function with default parameters"""
        from _shared.scripts.query_generator import main
        
        # Setup mocks
        mock_config = Mock()
        mock_config_class.from_env.return_value = mock_config
        mock_generate.return_value = [
            {
                "input": "Test query",
                "metadata": {"source_chunk_ids": ["chunk_1"], "document_id": "doc_123"}
            }
        ]
        
        # Call main
        main()
        
        # Verify
        mock_config_class.from_env.assert_called_once()
        mock_generate.assert_called_once_with(mock_config, num_queries=10)
        mock_save.assert_called_once()
    
    @patch('_shared.scripts.query_generator.save_queries_to_json')
    @patch('_shared.scripts.query_generator.generate_queries_from_index')
    def test_main_with_custom_config(self, mock_generate, mock_save):
        """Test main function with custom config"""
        from _shared.scripts.query_generator import main
        
        # Setup mocks
        mock_config = Mock()
        mock_generate.return_value = [
            {
                "input": "Test query",
                "metadata": {"source_chunk_ids": ["chunk_1"], "document_id": "doc_123"}
            }
        ]
        
        # Call main with custom config
        main(config=mock_config, num_queries=5)
        
        # Verify
        mock_generate.assert_called_once_with(mock_config, num_queries=5)
        mock_save.assert_called_once()
    
    @patch('_shared.scripts.query_generator.generate_queries_from_index')
    def test_main_handles_generation_error(self, mock_generate):
        """Test main function handles generation errors"""
        from _shared.scripts.query_generator import main
        
        # Setup mock to raise error
        mock_generate.side_effect = AzureServiceError("Generation failed")
        
        # Call main - should raise error
        with pytest.raises(AzureServiceError, match="Generation failed"):
            main()
    
    @patch('_shared.scripts.query_generator.save_queries_to_json')
    @patch('_shared.scripts.query_generator.generate_queries_from_index')
    def test_main_handles_save_error(self, mock_generate, mock_save):
        """Test main function handles save errors"""
        from _shared.scripts.query_generator import main
        
        # Setup mocks
        mock_generate.return_value = [
            {
                "input": "Test query",
                "metadata": {"source_chunk_ids": ["chunk_1"], "document_id": "doc_123"}
            }
        ]
        mock_save.side_effect = IOError("Save failed")
        
        # Call main - should raise error
        with pytest.raises(IOError, match="Save failed"):
            main()


class TestErrorHandling:
    """Tests for error handling edge cases"""
    
    @patch('_shared.scripts.query_generator.SearchClient')
    def test_sample_chunks_handles_unexpected_error(self, mock_search_client_class, mock_config):
        """Test sample_chunks handles unexpected errors"""
        # Setup mock to raise unexpected error (not ResourceNotFoundError)
        mock_search_client = Mock()
        mock_search_client_class.return_value = mock_search_client
        # Use a different exception type that will be caught by the generic Exception handler
        mock_search_client.search.side_effect = ValueError("Unexpected error")
        
        with pytest.raises(AzureServiceError, match="Operation failed after"):
            sample_chunks_from_index(mock_config, num_chunks=10)
    
    def test_generate_query_from_chunk_handles_unexpected_error(self, sample_chunks, mock_llm_provider):
        """Test generate_query_from_chunk handles unexpected errors"""
        chunk = sample_chunks[0]
        mock_llm_provider.call_completion.side_effect = Exception("Unexpected LLM error")
        
        with pytest.raises(AzureServiceError, match="Failed to generate query"):
            generate_query_from_chunk(chunk, mock_llm_provider)
    
    @patch('_shared.scripts.query_generator.get_llm_provider')
    @patch('_shared.scripts.query_generator.sample_chunks_from_index')
    def test_generate_queries_handles_unexpected_error(
        self, mock_sample_chunks, mock_get_llm_provider, mock_config
    ):
        """Test generate_queries handles unexpected errors"""
        # Test error handling when sample_chunks raises an error
        # We'll provide sample_chunks=None to trigger sampling, but mock it to fail
        mock_sample_chunks.side_effect = AzureServiceError("Unexpected sampling error")
        
        with pytest.raises(AzureServiceError, match="Unexpected sampling error"):
            generate_queries_from_index(mock_config, num_queries=2)
    
    def test_save_queries_handles_io_error(self):
        """Test save_queries handles IO errors"""
        queries = [
            {
                "input": "Test query",
                "metadata": {"source_chunk_ids": ["chunk_1"], "document_id": "doc_123"}
            }
        ]
        
        # Use a path that will cause an IO error (invalid directory)
        invalid_path = Path("/invalid/path/that/does/not/exist/eval_inputs.json")
        
        with pytest.raises(IOError):
            save_queries_to_json(queries, invalid_path)

