"""Unit and integration tests for query endpoint"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from fastapi import HTTPException
from datetime import datetime, timezone

# Disable logging during tests
logging.disable(logging.CRITICAL)

from src.core.exceptions import AzureServiceError, DatabaseError, ValidationError
from src.core.config import Config
from src.core.interfaces import ModelAnswer
from src.api.routes.query import router, handle_query, QueryRequest, QueryResponse


@pytest.fixture
def mock_config():
    """Create a mock config with all required Azure credentials"""
    config = Mock(spec=Config)
    config.supabase_url = "https://test.supabase.co"
    config.supabase_key = "test-key"
    config.database_url = "postgresql://test:test@localhost/test"
    config.azure_ai_foundry_endpoint = "https://test-foundry.openai.azure.com"
    config.azure_ai_foundry_api_key = "test-foundry-key"
    config.azure_ai_foundry_embedding_model = "text-embedding-3-small"
    config.azure_ai_foundry_generation_model = "gpt-4o"
    config.azure_search_endpoint = "https://test-search.search.windows.net"
    config.azure_search_api_key = "test-search-key"
    config.azure_search_index_name = "test-index"
    config.azure_document_intelligence_endpoint = "https://test-docint.cognitiveservices.azure.com"
    config.azure_document_intelligence_api_key = "test-docint-key"
    config.azure_blob_connection_string = ""
    config.azure_blob_container_name = ""
    return config


@pytest.fixture
def sample_query_request():
    """Sample query request for testing"""
    return QueryRequest(
        text="What is the coverage limit?",
        prompt_version="v1"
    )


@pytest.fixture
def sample_model_answer():
    """Sample model answer from pipeline"""
    return ModelAnswer(
        text="The coverage limit is $500,000 based on the policy documents.",
        query_id="test_query_123",
        prompt_version="v1",
        retrieved_chunk_ids=["chunk_1", "chunk_2"],
        timestamp=datetime.now(timezone.utc)
    )


class TestQueryEndpointUnit:
    """Unit tests for query endpoint handler"""
    
    @patch('src.api.routes.query.run_rag')
    @patch('src.api.routes.query.config')
    @pytest.mark.asyncio
    async def test_query_success(
        self,
        mock_config_obj,
        mock_run_rag,
        mock_config,
        sample_query_request,
        sample_model_answer
    ):
        """Test successful query processing"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_run_rag.return_value = sample_model_answer
        
        # Execute
        result = await handle_query(sample_query_request)
        
        # Verify response
        assert isinstance(result, QueryResponse)
        assert result.answer == sample_model_answer.text
        assert result.query_id == sample_model_answer.query_id
        assert result.prompt_version == sample_model_answer.prompt_version
        
        # Verify pipeline was called
        mock_run_rag.assert_called_once()
        call_args = mock_run_rag.call_args
        assert call_args[0][0].text == sample_query_request.text
        assert call_args[1]['prompt_version'] == sample_query_request.prompt_version
    
    @patch('src.api.routes.query.run_rag')
    @patch('src.api.routes.query.config')
    @pytest.mark.asyncio
    async def test_query_pipeline_error(
        self,
        mock_config_obj,
        mock_run_rag,
        mock_config,
        sample_query_request
    ):
        """Test query handles pipeline errors"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_run_rag.side_effect = AzureServiceError("Pipeline failed")
        
        # Execute and verify exception
        with pytest.raises(HTTPException) as exc_info:
            await handle_query(sample_query_request)
        
        assert exc_info.value.status_code == 500
        assert "Pipeline failed" in exc_info.value.detail
    
    @patch('src.api.routes.query.run_rag')
    @patch('src.api.routes.query.config')
    @pytest.mark.asyncio
    async def test_query_not_implemented_error(
        self,
        mock_config_obj,
        mock_run_rag,
        mock_config,
        sample_query_request
    ):
        """Test query handles NotImplementedError"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_run_rag.side_effect = NotImplementedError("Not implemented")
        
        # Execute and verify exception
        with pytest.raises(HTTPException) as exc_info:
            await handle_query(sample_query_request)
        
        assert exc_info.value.status_code == 501
        assert "RAG pipeline not yet implemented" in exc_info.value.detail
    
    @patch('src.api.routes.query.run_rag')
    @patch('src.api.routes.query.config')
    @pytest.mark.asyncio
    async def test_query_validation_error(
        self,
        mock_config_obj,
        mock_run_rag,
        mock_config,
        sample_query_request
    ):
        """Test query handles validation errors"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_run_rag.side_effect = ValidationError("Invalid prompt version")
        
        # Execute and verify exception
        with pytest.raises(HTTPException) as exc_info:
            await handle_query(sample_query_request)
        
        assert exc_info.value.status_code == 500
        assert "Invalid prompt version" in exc_info.value.detail
    
    @patch('src.api.routes.query.run_rag')
    @patch('src.api.routes.query.config')
    @pytest.mark.asyncio
    async def test_query_database_error(
        self,
        mock_config_obj,
        mock_run_rag,
        mock_config,
        sample_query_request
    ):
        """Test query handles database errors"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_run_rag.side_effect = DatabaseError("Database connection failed")
        
        # Execute and verify exception
        with pytest.raises(HTTPException) as exc_info:
            await handle_query(sample_query_request)
        
        assert exc_info.value.status_code == 500
        assert "Database connection failed" in exc_info.value.detail
    
    @patch('src.api.routes.query.run_rag')
    @patch('src.api.routes.query.config')
    @pytest.mark.asyncio
    async def test_query_generic_error(
        self,
        mock_config_obj,
        mock_run_rag,
        mock_config,
        sample_query_request
    ):
        """Test query handles generic errors"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_run_rag.side_effect = ValueError("Invalid input")
        
        # Execute and verify exception
        with pytest.raises(HTTPException) as exc_info:
            await handle_query(sample_query_request)
        
        assert exc_info.value.status_code == 500
        assert "Invalid input" in exc_info.value.detail
    
    @patch('src.api.routes.query.run_rag')
    @patch('src.api.routes.query.config')
    @pytest.mark.asyncio
    async def test_query_different_prompt_version(
        self,
        mock_config_obj,
        mock_run_rag,
        mock_config,
        sample_model_answer
    ):
        """Test query with different prompt version"""
        # Setup
        mock_config_obj.return_value = mock_config
        request = QueryRequest(text="What is the coverage?", prompt_version="v2")
        answer_v2 = ModelAnswer(
            text="Answer with v2 prompt",
            query_id="query_v2",
            prompt_version="v2",
            retrieved_chunk_ids=["chunk_1"],
            timestamp=datetime.now(timezone.utc)
        )
        mock_run_rag.return_value = answer_v2
        
        # Execute
        result = await handle_query(request)
        
        # Verify
        assert result.prompt_version == "v2"
        assert result.answer == answer_v2.text
        mock_run_rag.assert_called_once()
        call_args = mock_run_rag.call_args
        assert call_args[1]['prompt_version'] == "v2"


class TestQueryEndpointIntegration:
    """Integration tests for query endpoint with mocked pipeline"""
    
    @patch('src.api.routes.query.run_rag')
    @patch('src.api.routes.query.config')
    @pytest.mark.asyncio
    async def test_query_endpoint_integration(
        self,
        mock_config_obj,
        mock_run_rag,
        mock_config,
        sample_query_request,
        sample_model_answer
    ):
        """Test end-to-end query pipeline with mocked services"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_run_rag.return_value = sample_model_answer
        
        # Execute
        result = await handle_query(sample_query_request)
        
        # Verify response
        assert isinstance(result, QueryResponse)
        assert result.answer == sample_model_answer.text
        assert result.query_id == sample_model_answer.query_id
        assert result.prompt_version == sample_model_answer.prompt_version
        
        # Verify pipeline was called
        mock_run_rag.assert_called_once()
    
    @patch('src.api.routes.query.run_rag')
    @patch('src.api.routes.query.config')
    @pytest.mark.asyncio
    async def test_query_endpoint_empty_answer(
        self,
        mock_config_obj,
        mock_run_rag,
        mock_config,
        sample_query_request
    ):
        """Test query endpoint handles empty answer"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        empty_answer = ModelAnswer(
            text="",
            query_id="query_empty",
            prompt_version="v1",
            retrieved_chunk_ids=[],
            timestamp=datetime.now(timezone.utc)
        )
        mock_run_rag.return_value = empty_answer
        
        # Execute
        result = await handle_query(sample_query_request)
        
        # Verify response (empty answer is valid)
        assert result.answer == ""
        assert result.query_id == "query_empty"


class TestQueryEndpointResponseFormat:
    """Tests for query endpoint response format and validation"""
    
    @patch('src.api.routes.query.run_rag')
    @patch('src.api.routes.query.config')
    @pytest.mark.asyncio
    async def test_query_response_format(
        self,
        mock_config_obj,
        mock_run_rag,
        mock_config,
        sample_query_request,
        sample_model_answer
    ):
        """Test query response has correct format"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_run_rag.return_value = sample_model_answer
        
        # Execute
        result = await handle_query(sample_query_request)
        
        # Verify response format
        assert isinstance(result, QueryResponse)
        
        # Verify all required fields
        assert hasattr(result, "answer")
        assert hasattr(result, "query_id")
        assert hasattr(result, "prompt_version")
        
        # Verify field types
        assert isinstance(result.answer, str)
        assert isinstance(result.query_id, str)
        assert isinstance(result.prompt_version, str)
        
        # Verify field values
        assert len(result.answer) > 0
        assert len(result.query_id) > 0
        assert len(result.prompt_version) > 0

