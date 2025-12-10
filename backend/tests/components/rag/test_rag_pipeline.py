"""Unit tests for RAG pipeline orchestration"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from src.core.exceptions import AzureServiceError, DatabaseError, ValidationError
from src.core.config import Config
from src.core.interfaces import Query, ModelAnswer, RetrievalResult
from src.services.rag.pipeline import run_rag


@pytest.fixture
def mock_config():
    """Create a mock config for testing"""
    return Config(
        supabase_url="https://test.supabase.co",
        supabase_key="test-key",
        database_url="postgresql://test:test@localhost/test",
        azure_ai_foundry_endpoint="https://test-foundry.openai.azure.com",
        azure_ai_foundry_api_key="test-foundry-key",
        azure_ai_foundry_embedding_model="text-embedding-3-small",
        azure_ai_foundry_generation_model="gpt-4o",
        azure_search_endpoint="https://test-search.search.windows.net",
        azure_search_api_key="test-search-key",
        azure_search_index_name="test-index",
        azure_document_intelligence_endpoint="https://test-docint.cognitiveservices.azure.com",
        azure_document_intelligence_api_key="test-docint-key",
        azure_blob_connection_string="",
        azure_blob_container_name=""
    )


@pytest.fixture
def sample_query():
    """Create a sample query for testing"""
    return Query(
        text="What is the coverage limit?",
        query_id="test_query_123",
        timestamp=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_retrieval_results():
    """Create sample retrieval results for testing"""
    return [
        RetrievalResult(
            chunk_id="chunk_1",
            similarity_score=0.92,
            chunk_text="The coverage limit is $500,000.",
            metadata={"document_id": "doc_1"}
        ),
        RetrievalResult(
            chunk_id="chunk_2",
            similarity_score=0.85,
            chunk_text="Coverage applies to all medical expenses.",
            metadata={"document_id": "doc_1"}
        )
    ]


@pytest.fixture
def sample_model_answer():
    """Create a sample model answer for testing"""
    return ModelAnswer(
        text="The coverage limit is $500,000 based on the policy documents.",
        query_id="test_query_123",
        prompt_version="v1",
        retrieved_chunk_ids=["chunk_1", "chunk_2"],
        timestamp=datetime.now(timezone.utc)
    )


class TestRunRAGEndToEnd:
    """Tests for end-to-end pipeline flow with mocked components"""
    
    @patch('src.services.rag.pipeline.generate_answer')
    @patch('src.services.rag.pipeline.retrieve_chunks')
    @patch('src.services.rag.pipeline.generate_query_embedding')
    @patch('src.services.rag.pipeline.DatabaseConnection')
    def test_run_rag_success(
        self,
        mock_db_conn_class,
        mock_generate_query_embedding,
        mock_retrieve_chunks,
        mock_generate_answer,
        mock_config,
        sample_query,
        sample_retrieval_results,
        sample_model_answer
    ):
        """Test successful end-to-end pipeline execution"""
        # Setup mocks
        mock_query_embedding = [0.1] * 1536  # Mock embedding vector
        mock_generate_query_embedding.return_value = mock_query_embedding
        mock_retrieve_chunks.return_value = sample_retrieval_results
        mock_generate_answer.return_value = sample_model_answer
        
        # Mock database connection
        mock_db_conn = Mock()
        mock_db_conn_class.return_value = mock_db_conn
        
        # Run pipeline
        result = run_rag(sample_query, prompt_version="v1", config=mock_config)
        
        # Verify result
        assert result == sample_model_answer
        assert result.query_id == "test_query_123"
        assert result.prompt_version == "v1"
        assert len(result.retrieved_chunk_ids) == 2
        
        # Verify component calls
        mock_generate_query_embedding.assert_called_once_with(sample_query, mock_config)
        mock_retrieve_chunks.assert_called_once_with(sample_query, top_k=5, config=mock_config)
        mock_generate_answer.assert_called_once()
        
        # Verify database connection was created and closed
        mock_db_conn.connect.assert_called_once()
        mock_db_conn.close.assert_called_once()
    
    @patch('src.services.rag.pipeline.generate_answer')
    @patch('src.services.rag.pipeline.retrieve_chunks')
    @patch('src.services.rag.pipeline.generate_query_embedding')
    @patch('src.services.rag.pipeline.generate_id')
    @patch('src.services.rag.pipeline.DatabaseConnection')
    def test_run_rag_generates_query_id_if_missing(
        self,
        mock_db_conn_class,
        mock_generate_id,
        mock_generate_query_embedding,
        mock_retrieve_chunks,
        mock_generate_answer,
        mock_config,
        sample_retrieval_results,
        sample_model_answer
    ):
        """Test that query ID is generated if missing"""
        # Create query without ID
        query = Query(text="What is the coverage limit?")
        mock_generate_id.return_value = "generated_query_456"
        
        # Setup mocks
        mock_query_embedding = [0.1] * 1536
        mock_generate_query_embedding.return_value = mock_query_embedding
        mock_retrieve_chunks.return_value = sample_retrieval_results
        mock_generate_answer.return_value = sample_model_answer
        
        # Mock database connection
        mock_db_conn = Mock()
        mock_db_conn_class.return_value = mock_db_conn
        
        # Run pipeline
        result = run_rag(query, prompt_version="v1", config=mock_config)
        
        # Verify query ID was generated
        mock_generate_id.assert_called_once_with("query")
        assert result.query_id == sample_model_answer.query_id
    
    @patch('src.services.rag.pipeline.generate_answer')
    @patch('src.services.rag.pipeline.retrieve_chunks')
    @patch('src.services.rag.pipeline.generate_query_embedding')
    @patch('src.services.rag.pipeline.DatabaseConnection')
    def test_run_rag_generates_timestamp_if_missing(
        self,
        mock_db_conn_class,
        mock_generate_query_embedding,
        mock_retrieve_chunks,
        mock_generate_answer,
        mock_config,
        sample_query,
        sample_retrieval_results,
        sample_model_answer
    ):
        """Test that timestamp is generated if missing"""
        # Create query without timestamp
        query = Query(text="What is the coverage limit?", query_id="test_query_123")
        
        # Setup mocks
        mock_query_embedding = [0.1] * 1536
        mock_generate_query_embedding.return_value = mock_query_embedding
        mock_retrieve_chunks.return_value = sample_retrieval_results
        mock_generate_answer.return_value = sample_model_answer
        
        # Mock database connection
        mock_db_conn = Mock()
        mock_db_conn_class.return_value = mock_db_conn
        
        # Run pipeline
        with patch('src.services.rag.pipeline.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            result = run_rag(query, prompt_version="v1", config=mock_config)
        
        # Verify pipeline completed
        assert result == sample_model_answer


class TestRunRAGComponentIntegration:
    """Tests for component integration and data flow"""
    
    @patch('src.services.rag.pipeline.generate_answer')
    @patch('src.services.rag.pipeline.retrieve_chunks')
    @patch('src.services.rag.pipeline.generate_query_embedding')
    @patch('src.services.rag.pipeline.DatabaseConnection')
    def test_pipeline_passes_data_correctly(
        self,
        mock_db_conn_class,
        mock_generate_query_embedding,
        mock_retrieve_chunks,
        mock_generate_answer,
        mock_config,
        sample_query,
        sample_retrieval_results,
        sample_model_answer
    ):
        """Test that data flows correctly between components"""
        # Setup mocks
        mock_query_embedding = [0.2] * 1536
        mock_generate_query_embedding.return_value = mock_query_embedding
        mock_retrieve_chunks.return_value = sample_retrieval_results
        mock_generate_answer.return_value = sample_model_answer
        
        # Mock database connection
        mock_db_conn = Mock()
        mock_db_conn_class.return_value = mock_db_conn
        
        # Run pipeline
        result = run_rag(sample_query, prompt_version="v2", config=mock_config)
        
        # Verify data flow
        # 1. Query embedding should be generated with correct query
        call_args = mock_generate_query_embedding.call_args
        assert call_args[0][0] == sample_query
        assert call_args[0][1] == mock_config
        
        # 2. Retrieval should use the same query
        call_args = mock_retrieve_chunks.call_args
        assert call_args[0][0] == sample_query
        assert call_args[1]['top_k'] == 5
        assert call_args[1]['config'] == mock_config
        
        # 3. Generation should receive query, retrieval results, and prompt version
        call_args = mock_generate_answer.call_args
        assert call_args[1]['query'] == sample_query
        assert call_args[1]['retrieved_chunks'] == sample_retrieval_results
        assert call_args[1]['prompt_version'] == "v2"
        assert call_args[1]['config'] == mock_config
    
    @patch('src.services.rag.pipeline.generate_answer')
    @patch('src.services.rag.pipeline.retrieve_chunks')
    @patch('src.services.rag.pipeline.generate_query_embedding')
    @patch('src.services.rag.pipeline.DatabaseConnection')
    def test_pipeline_uses_default_config(
        self,
        mock_db_conn_class,
        mock_generate_query_embedding,
        mock_retrieve_chunks,
        mock_generate_answer,
        sample_query,
        sample_retrieval_results,
        sample_model_answer
    ):
        """Test that pipeline uses default config if not provided"""
        # Setup mocks
        mock_query_embedding = [0.1] * 1536
        mock_generate_query_embedding.return_value = mock_query_embedding
        mock_retrieve_chunks.return_value = sample_retrieval_results
        mock_generate_answer.return_value = sample_model_answer
        
        # Mock database connection
        mock_db_conn = Mock()
        mock_db_conn_class.return_value = mock_db_conn
        
        # Mock Config.from_env()
        with patch('src.services.rag.pipeline.Config') as mock_config_class:
            mock_config = Mock()
            mock_config_class.from_env.return_value = mock_config
            
            # Run pipeline without config
            result = run_rag(sample_query, prompt_version="v1")
            
            # Verify Config.from_env() was called
            mock_config_class.from_env.assert_called_once()
            
            # Verify components received the config
            mock_generate_query_embedding.assert_called_once()
            assert mock_generate_query_embedding.call_args[0][1] == mock_config


class TestRunRAGErrorHandling:
    """Tests for error propagation and handling"""
    
    @patch('src.services.rag.pipeline.generate_query_embedding')
    def test_run_rag_handles_embedding_error(
        self,
        mock_generate_query_embedding,
        mock_config,
        sample_query
    ):
        """Test that embedding errors are properly propagated"""
        # Setup mock to raise error
        mock_generate_query_embedding.side_effect = AzureServiceError("Embedding API failed")
        
        # Run pipeline and expect error
        with pytest.raises(AzureServiceError) as exc_info:
            run_rag(sample_query, prompt_version="v1", config=mock_config)
        
        assert "Embedding API failed" in str(exc_info.value)
        assert "Query embedding generation failed" in str(exc_info.value)
    
    @patch('src.services.rag.pipeline.generate_answer')
    @patch('src.services.rag.pipeline.retrieve_chunks')
    @patch('src.services.rag.pipeline.generate_query_embedding')
    def test_run_rag_handles_retrieval_error(
        self,
        mock_generate_query_embedding,
        mock_retrieve_chunks,
        mock_generate_answer,
        mock_config,
        sample_query
    ):
        """Test that retrieval errors are properly propagated"""
        # Setup mocks
        mock_query_embedding = [0.1] * 1536
        mock_generate_query_embedding.return_value = mock_query_embedding
        mock_retrieve_chunks.side_effect = AzureServiceError("Search API failed")
        
        # Run pipeline and expect error
        with pytest.raises(AzureServiceError) as exc_info:
            run_rag(sample_query, prompt_version="v1", config=mock_config)
        
        assert "Search API failed" in str(exc_info.value)
        assert "Chunk retrieval failed" in str(exc_info.value)
    
    @patch('src.services.rag.pipeline.generate_answer')
    @patch('src.services.rag.pipeline.retrieve_chunks')
    @patch('src.services.rag.pipeline.generate_query_embedding')
    @patch('src.services.rag.pipeline.DatabaseConnection')
    def test_run_rag_handles_generation_error(
        self,
        mock_db_conn_class,
        mock_generate_query_embedding,
        mock_retrieve_chunks,
        mock_generate_answer,
        mock_config,
        sample_query,
        sample_retrieval_results
    ):
        """Test that generation errors are properly propagated"""
        # Setup mocks
        mock_query_embedding = [0.1] * 1536
        mock_generate_query_embedding.return_value = mock_query_embedding
        mock_retrieve_chunks.return_value = sample_retrieval_results
        mock_generate_answer.side_effect = AzureServiceError("Generation API failed")
        
        # Mock database connection
        mock_db_conn = Mock()
        mock_db_conn_class.return_value = mock_db_conn
        
        # Run pipeline and expect error
        with pytest.raises(AzureServiceError) as exc_info:
            run_rag(sample_query, prompt_version="v1", config=mock_config)
        
        assert "Generation API failed" in str(exc_info.value)
        assert "Answer generation failed" in str(exc_info.value)
    
    @patch('src.services.rag.pipeline.generate_answer')
    @patch('src.services.rag.pipeline.retrieve_chunks')
    @patch('src.services.rag.pipeline.generate_query_embedding')
    @patch('src.services.rag.pipeline.DatabaseConnection')
    def test_run_rag_handles_validation_error(
        self,
        mock_db_conn_class,
        mock_generate_query_embedding,
        mock_retrieve_chunks,
        mock_generate_answer,
        mock_config,
        sample_query,
        sample_retrieval_results
    ):
        """Test that validation errors are properly propagated"""
        # Setup mocks
        mock_query_embedding = [0.1] * 1536
        mock_generate_query_embedding.return_value = mock_query_embedding
        mock_retrieve_chunks.return_value = sample_retrieval_results
        mock_generate_answer.side_effect = ValidationError("Prompt version not found")
        
        # Mock database connection
        mock_db_conn = Mock()
        mock_db_conn_class.return_value = mock_db_conn
        
        # Run pipeline and expect error
        with pytest.raises(ValidationError) as exc_info:
            run_rag(sample_query, prompt_version="v1", config=mock_config)
        
        assert "Prompt version not found" in str(exc_info.value)
    
    def test_run_rag_handles_empty_query_text(self, mock_config):
        """Test that empty query text raises ValueError"""
        query = Query(text="")
        
        with pytest.raises(ValueError) as exc_info:
            run_rag(query, prompt_version="v1", config=mock_config)
        
        assert "Query text cannot be empty" in str(exc_info.value)
    
    def test_run_rag_handles_whitespace_only_query(self, mock_config):
        """Test that whitespace-only query raises ValueError"""
        query = Query(text="   ")
        
        with pytest.raises(ValueError) as exc_info:
            run_rag(query, prompt_version="v1", config=mock_config)
        
        assert "Query text cannot be empty" in str(exc_info.value)


class TestRunRAGLatencyMeasurement:
    """Tests for latency measurement"""
    
    @patch('src.services.rag.pipeline.generate_answer')
    @patch('src.services.rag.pipeline.retrieve_chunks')
    @patch('src.services.rag.pipeline.generate_query_embedding')
    @patch('src.services.rag.pipeline.DatabaseConnection')
    @patch('src.services.rag.pipeline.time')
    def test_pipeline_measures_latency(
        self,
        mock_time,
        mock_db_conn_class,
        mock_generate_query_embedding,
        mock_retrieve_chunks,
        mock_generate_answer,
        mock_config,
        sample_query,
        sample_retrieval_results,
        sample_model_answer
    ):
        """Test that pipeline measures and logs latency"""
        # Setup time mock to simulate elapsed time
        # Pipeline calls time.time() 10 times:
        # 1. pipeline_start_time
        # 2. embedding_start_time
        # 3. embedding_latency calculation
        # 4. retrieval_start_time
        # 5. retrieval_latency calculation
        # 6. generation_start_time
        # 7. generation_latency calculation
        # 8. logging_start_time
        # 9. logging_latency calculation
        # 10. total_latency calculation
        mock_time.time.side_effect = [
            0.0,   # pipeline_start_time
            0.1,   # embedding_start_time
            0.5,   # embedding_latency calculation (0.4s latency)
            0.6,   # retrieval_start_time
            1.0,   # retrieval_latency calculation (0.4s latency)
            1.1,   # generation_start_time
            2.0,   # generation_latency calculation (0.9s latency)
            2.1,   # logging_start_time
            2.15,  # logging_latency calculation (0.05s latency)
            2.2    # total_latency calculation (2.2s total)
        ]
        
        # Setup mocks
        mock_query_embedding = [0.1] * 1536
        mock_generate_query_embedding.return_value = mock_query_embedding
        mock_retrieve_chunks.return_value = sample_retrieval_results
        mock_generate_answer.return_value = sample_model_answer
        
        # Mock database connection
        mock_db_conn = Mock()
        mock_db_conn_class.return_value = mock_db_conn
        
        # Run pipeline
        result = run_rag(sample_query, prompt_version="v1", config=mock_config)
        
        # Verify pipeline completed
        assert result == sample_model_answer
        
        # Verify time.time() was called multiple times (for latency measurement)
        assert mock_time.time.call_count == 10


class TestRunRAGResponseAssembly:
    """Tests for response assembly and formatting"""
    
    @patch('src.services.rag.pipeline.generate_answer')
    @patch('src.services.rag.pipeline.retrieve_chunks')
    @patch('src.services.rag.pipeline.generate_query_embedding')
    @patch('src.services.rag.pipeline.DatabaseConnection')
    def test_pipeline_assembles_complete_model_answer(
        self,
        mock_db_conn_class,
        mock_generate_query_embedding,
        mock_retrieve_chunks,
        mock_generate_answer,
        mock_config,
        sample_query,
        sample_retrieval_results,
        sample_model_answer
    ):
        """Test that pipeline returns complete ModelAnswer object"""
        # Setup mocks
        mock_query_embedding = [0.1] * 1536
        mock_generate_query_embedding.return_value = mock_query_embedding
        mock_retrieve_chunks.return_value = sample_retrieval_results
        mock_generate_answer.return_value = sample_model_answer
        
        # Mock database connection
        mock_db_conn = Mock()
        mock_db_conn_class.return_value = mock_db_conn
        
        # Run pipeline
        result = run_rag(sample_query, prompt_version="v1", config=mock_config)
        
        # Verify ModelAnswer structure
        assert isinstance(result, ModelAnswer)
        assert result.text == sample_model_answer.text
        assert result.query_id == sample_query.query_id
        assert result.prompt_version == "v1"
        assert result.retrieved_chunk_ids == ["chunk_1", "chunk_2"]
        assert result.timestamp is not None
    
    @patch('src.services.rag.pipeline.generate_answer')
    @patch('src.services.rag.pipeline.retrieve_chunks')
    @patch('src.services.rag.pipeline.generate_query_embedding')
    @patch('src.services.rag.pipeline.DatabaseConnection')
    def test_pipeline_preserves_query_id(
        self,
        mock_db_conn_class,
        mock_generate_query_embedding,
        mock_retrieve_chunks,
        mock_generate_answer,
        mock_config,
        sample_query,
        sample_retrieval_results
    ):
        """Test that pipeline preserves query_id in the answer"""
        # Create model answer with different query_id
        model_answer = ModelAnswer(
            text="Answer text",
            query_id="different_id",  # Different from sample_query.query_id
            prompt_version="v1",
            retrieved_chunk_ids=["chunk_1"],
            timestamp=datetime.now(timezone.utc)
        )
        
        # Setup mocks
        mock_query_embedding = [0.1] * 1536
        mock_generate_query_embedding.return_value = mock_query_embedding
        mock_retrieve_chunks.return_value = sample_retrieval_results
        mock_generate_answer.return_value = model_answer
        
        # Mock database connection
        mock_db_conn = Mock()
        mock_db_conn_class.return_value = mock_db_conn
        
        # Run pipeline
        result = run_rag(sample_query, prompt_version="v1", config=mock_config)
        
        # Verify query_id is preserved from original query
        assert result.query_id == sample_query.query_id  # Should be corrected to match query


class TestRunRAGPipelineStateManagement:
    """Tests for pipeline state management"""
    
    @patch('src.services.rag.pipeline.generate_answer')
    @patch('src.services.rag.pipeline.retrieve_chunks')
    @patch('src.services.rag.pipeline.generate_query_embedding')
    @patch('src.services.rag.pipeline.DatabaseConnection')
    def test_pipeline_closes_database_connection_on_success(
        self,
        mock_db_conn_class,
        mock_generate_query_embedding,
        mock_retrieve_chunks,
        mock_generate_answer,
        mock_config,
        sample_query,
        sample_retrieval_results,
        sample_model_answer
    ):
        """Test that database connection is closed after successful execution"""
        # Setup mocks
        mock_query_embedding = [0.1] * 1536
        mock_generate_query_embedding.return_value = mock_query_embedding
        mock_retrieve_chunks.return_value = sample_retrieval_results
        mock_generate_answer.return_value = sample_model_answer
        
        # Mock database connection
        mock_db_conn = Mock()
        mock_db_conn_class.return_value = mock_db_conn
        
        # Run pipeline
        run_rag(sample_query, prompt_version="v1", config=mock_config)
        
        # Verify database connection was closed
        mock_db_conn.close.assert_called_once()
    
    @patch('src.services.rag.pipeline.generate_answer')
    @patch('src.services.rag.pipeline.retrieve_chunks')
    @patch('src.services.rag.pipeline.generate_query_embedding')
    @patch('src.services.rag.pipeline.DatabaseConnection')
    def test_pipeline_closes_database_connection_on_error(
        self,
        mock_db_conn_class,
        mock_generate_query_embedding,
        mock_retrieve_chunks,
        mock_generate_answer,
        mock_config,
        sample_query,
        sample_retrieval_results
    ):
        """Test that database connection is closed even on error"""
        # Setup mocks
        mock_query_embedding = [0.1] * 1536
        mock_generate_query_embedding.return_value = mock_query_embedding
        mock_retrieve_chunks.return_value = sample_retrieval_results
        mock_generate_answer.side_effect = AzureServiceError("Generation failed")
        
        # Mock database connection
        mock_db_conn = Mock()
        mock_db_conn_class.return_value = mock_db_conn
        
        # Run pipeline and expect error
        with pytest.raises(AzureServiceError):
            run_rag(sample_query, prompt_version="v1", config=mock_config)
        
        # Verify database connection was still closed
        mock_db_conn.close.assert_called_once()
