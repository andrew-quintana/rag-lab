"""Unit and integration tests for upload endpoint"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from fastapi import UploadFile, HTTPException
from io import BytesIO

# Disable logging during tests to avoid potential hangs
logging.disable(logging.CRITICAL)

from src.core.exceptions import AzureServiceError
from src.core.config import Config
from src.core.interfaces import Chunk
from src.api.routes.upload import router, handle_upload, UploadResponse
from src.api.main import app


@pytest.fixture
def mock_config():
    """Create a mock config with all required Azure credentials"""
    config = Mock(spec=Config)
    config.azure_document_intelligence_endpoint = "https://test-docint.cognitiveservices.azure.com"
    config.azure_document_intelligence_api_key = "test-docint-key"
    config.azure_ai_foundry_endpoint = "https://test-foundry.openai.azure.com"
    config.azure_ai_foundry_api_key = "test-foundry-key"
    config.azure_ai_foundry_embedding_model = "text-embedding-3-small"
    config.azure_search_endpoint = "https://test-search.search.windows.net"
    config.azure_search_api_key = "test-search-key"
    config.azure_search_index_name = "test-index"
    config.supabase_url = "https://test.supabase.co"
    config.supabase_key = "test-supabase-key"
    config.database_url = "postgresql://test:test@localhost/test"
    return config


@pytest.fixture
def sample_file_content():
    """Sample PDF file content for testing"""
    return b"Sample PDF content for testing"


@pytest.fixture
def sample_extracted_text():
    """Sample extracted text from document"""
    return "This is extracted text from a document. It contains multiple sentences. " * 20


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing"""
    return [
        Chunk(
            text="This is chunk 1",
            chunk_id="chunk_0",
            document_id="doc_123",
            metadata={"start": 0, "end": 15, "chunking_method": "fixed_size"}
        ),
        Chunk(
            text="This is chunk 2",
            chunk_id="chunk_1",
            document_id="doc_123",
            metadata={"start": 15, "end": 30, "chunking_method": "fixed_size"}
        )
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing"""
    return [
        [0.1] * 1536,  # Mock embedding for chunk 1
        [0.2] * 1536   # Mock embedding for chunk 2
    ]


@pytest.fixture
def mock_upload_file(sample_file_content):
    """Create a mock UploadFile for testing"""
    file = Mock(spec=UploadFile)
    file.filename = "test_document.pdf"
    file.read = AsyncMock(return_value=sample_file_content)
    return file


class TestUploadEndpointUnit:
    """Unit tests for upload endpoint handler"""
    
    @patch('src.api.routes.upload.doc_service')
    @patch('src.api.routes.upload.generate_image_preview')
    @patch('src.api.routes.upload.upload_document_to_storage')
    @patch('src.api.routes.upload.index_chunks')
    @patch('src.api.routes.upload.generate_embeddings')
    @patch('src.api.routes.upload.chunk_text')
    @patch('src.api.routes.upload.ingest_document')
    @patch('src.api.routes.upload.generate_id')
    @patch('src.api.routes.upload.config')
    @pytest.mark.asyncio
    async def test_upload_success(
        self,
        mock_config_obj,
        mock_generate_id,
        mock_ingest_document,
        mock_chunk_text,
        mock_generate_embeddings,
        mock_index_chunks,
        mock_upload_storage,
        mock_generate_preview,
        mock_doc_service,
        mock_config,
        mock_upload_file,
        sample_file_content,
        sample_extracted_text,
        sample_chunks,
        sample_embeddings
    ):
        """Test successful document upload and processing"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_generate_id.return_value = "doc_123"
        mock_upload_storage.return_value = "doc_123"
        mock_generate_preview.return_value = None
        mock_doc_service.insert_document.return_value = None
        mock_doc_service.update_document_status.return_value = None
        mock_doc_service.update_document_chunks.return_value = None
        mock_ingest_document.return_value = sample_extracted_text
        mock_chunk_text.return_value = sample_chunks
        mock_generate_embeddings.return_value = sample_embeddings
        mock_index_chunks.return_value = None
        
        # Execute
        result = await handle_upload(mock_upload_file)
        
        # Verify response
        assert isinstance(result, UploadResponse)
        assert result.document_id == "doc_123"
        assert result.status == "success"
        assert result.chunks_created == 2
        assert "successfully" in result.message.lower()
        
        # Verify pipeline steps were called
        # Note: config is the patched module-level config, not the fixture
        mock_ingest_document.assert_called_once()
        assert mock_ingest_document.call_args[0][0] == sample_file_content
        mock_chunk_text.assert_called_once()
        assert mock_chunk_text.call_args[0][0] == sample_extracted_text
        assert mock_chunk_text.call_args[1]['document_id'] == "doc_123"
        mock_generate_embeddings.assert_called_once()
        assert mock_generate_embeddings.call_args[0][0] == sample_chunks
        mock_index_chunks.assert_called_once()
        assert mock_index_chunks.call_args[0][0] == sample_chunks
        assert mock_index_chunks.call_args[0][1] == sample_embeddings
    
    @patch('src.api.routes.upload.doc_service')
    @patch('src.api.routes.upload.generate_image_preview')
    @patch('src.api.routes.upload.upload_document_to_storage')
    @patch('src.api.routes.upload.ingest_document')
    @patch('src.api.routes.upload.generate_id')
    @patch('src.api.routes.upload.config')
    @pytest.mark.asyncio
    async def test_upload_empty_text_extraction(
        self,
        mock_config_obj,
        mock_generate_id,
        mock_ingest_document,
        mock_upload_storage,
        mock_generate_preview,
        mock_doc_service,
        mock_config,
        mock_upload_file
    ):
        """Test upload fails when no text can be extracted"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_generate_id.return_value = "doc_123"
        mock_upload_storage.return_value = "doc_123"
        mock_generate_preview.return_value = None
        mock_doc_service.insert_document.return_value = None
        mock_doc_service.update_document_status.return_value = None
        mock_ingest_document.return_value = ""  # Empty text
        
        # Execute and verify exception
        with pytest.raises(HTTPException) as exc_info:
            await handle_upload(mock_upload_file)
        
        assert exc_info.value.status_code == 400
        assert "No text could be extracted" in exc_info.value.detail
    
    @patch('src.api.routes.upload.doc_service')
    @patch('src.api.routes.upload.generate_image_preview')
    @patch('src.api.routes.upload.upload_document_to_storage')
    @patch('src.api.routes.upload.ingest_document')
    @patch('src.api.routes.upload.generate_id')
    @patch('src.api.routes.upload.config')
    @pytest.mark.asyncio
    async def test_upload_whitespace_only_text(
        self,
        mock_config_obj,
        mock_generate_id,
        mock_ingest_document,
        mock_upload_storage,
        mock_generate_preview,
        mock_doc_service,
        mock_config,
        mock_upload_file
    ):
        """Test upload fails when only whitespace is extracted"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_generate_id.return_value = "doc_123"
        mock_upload_storage.return_value = "doc_123"
        mock_generate_preview.return_value = None
        mock_doc_service.insert_document.return_value = None
        mock_doc_service.update_document_status.return_value = None
        mock_ingest_document.return_value = "   \n\t  "  # Whitespace only
        
        # Execute and verify exception
        with pytest.raises(HTTPException) as exc_info:
            await handle_upload(mock_upload_file)
        
        assert exc_info.value.status_code == 400
        assert "No text could be extracted" in exc_info.value.detail
    
    @patch('src.api.routes.upload.doc_service')
    @patch('src.api.routes.upload.generate_image_preview')
    @patch('src.api.routes.upload.upload_document_to_storage')
    @patch('src.api.routes.upload.chunk_text')
    @patch('src.api.routes.upload.ingest_document')
    @patch('src.api.routes.upload.generate_id')
    @patch('src.api.routes.upload.config')
    @pytest.mark.asyncio
    async def test_upload_no_chunks_created(
        self,
        mock_config_obj,
        mock_generate_id,
        mock_ingest_document,
        mock_chunk_text,
        mock_upload_storage,
        mock_generate_preview,
        mock_doc_service,
        mock_config,
        mock_upload_file,
        sample_extracted_text
    ):
        """Test upload fails when no chunks are created"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_generate_id.return_value = "doc_123"
        mock_upload_storage.return_value = "doc_123"
        mock_generate_preview.return_value = None
        mock_doc_service.insert_document.return_value = None
        mock_doc_service.update_document_status.return_value = None
        mock_ingest_document.return_value = sample_extracted_text
        mock_chunk_text.return_value = []  # No chunks
        
        # Execute and verify exception
        with pytest.raises(HTTPException) as exc_info:
            await handle_upload(mock_upload_file)
        
        assert exc_info.value.status_code == 500
        assert "Failed to create chunks" in exc_info.value.detail
    
    @patch('src.api.routes.upload.doc_service')
    @patch('src.api.routes.upload.generate_image_preview')
    @patch('src.api.routes.upload.upload_document_to_storage')
    @patch('src.api.routes.upload.generate_embeddings')
    @patch('src.api.routes.upload.chunk_text')
    @patch('src.api.routes.upload.ingest_document')
    @patch('src.api.routes.upload.generate_id')
    @patch('src.api.routes.upload.config')
    @pytest.mark.asyncio
    async def test_upload_embedding_mismatch(
        self,
        mock_config_obj,
        mock_generate_id,
        mock_ingest_document,
        mock_chunk_text,
        mock_generate_embeddings,
        mock_upload_storage,
        mock_generate_preview,
        mock_doc_service,
        mock_config,
        mock_upload_file,
        sample_extracted_text,
        sample_chunks
    ):
        """Test upload fails when embedding count doesn't match chunk count"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_generate_id.return_value = "doc_123"
        mock_upload_storage.return_value = "doc_123"
        mock_generate_preview.return_value = None
        mock_doc_service.insert_document.return_value = None
        mock_doc_service.update_document_status.return_value = None
        mock_ingest_document.return_value = sample_extracted_text
        mock_chunk_text.return_value = sample_chunks  # 2 chunks
        mock_generate_embeddings.return_value = [[0.1] * 1536]  # Only 1 embedding
        
        # Execute and verify exception
        with pytest.raises(HTTPException) as exc_info:
            await handle_upload(mock_upload_file)
        
        assert exc_info.value.status_code == 500
        assert "Embedding generation failed" in exc_info.value.detail
        assert "expected 2 embeddings" in exc_info.value.detail
    
    @patch('src.api.routes.upload.doc_service')
    @patch('src.api.routes.upload.generate_image_preview')
    @patch('src.api.routes.upload.upload_document_to_storage')
    @patch('src.api.routes.upload.ingest_document')
    @patch('src.api.routes.upload.generate_id')
    @patch('src.api.routes.upload.config')
    @pytest.mark.asyncio
    async def test_upload_ingestion_error(
        self,
        mock_config_obj,
        mock_generate_id,
        mock_ingest_document,
        mock_upload_storage,
        mock_generate_preview,
        mock_doc_service,
        mock_config,
        mock_upload_file
    ):
        """Test upload handles ingestion errors"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_generate_id.return_value = "doc_123"
        mock_upload_storage.return_value = "doc_123"
        mock_generate_preview.return_value = None
        mock_doc_service.insert_document.return_value = None
        mock_doc_service.update_document_status.return_value = None
        mock_ingest_document.side_effect = AzureServiceError("Document Intelligence failed")
        
        # Execute and verify exception
        with pytest.raises(HTTPException) as exc_info:
            await handle_upload(mock_upload_file)
        
        assert exc_info.value.status_code == 500
        assert "Upload processing failed" in exc_info.value.detail
    
    @patch('src.api.routes.upload.doc_service')
    @patch('src.api.routes.upload.generate_image_preview')
    @patch('src.api.routes.upload.upload_document_to_storage')
    @patch('src.api.routes.upload.index_chunks')
    @patch('src.api.routes.upload.generate_embeddings')
    @patch('src.api.routes.upload.chunk_text')
    @patch('src.api.routes.upload.ingest_document')
    @patch('src.api.routes.upload.generate_id')
    @patch('src.api.routes.upload.config')
    @pytest.mark.asyncio
    async def test_upload_indexing_error(
        self,
        mock_config_obj,
        mock_generate_id,
        mock_ingest_document,
        mock_chunk_text,
        mock_generate_embeddings,
        mock_index_chunks,
        mock_upload_storage,
        mock_generate_preview,
        mock_doc_service,
        mock_config,
        mock_upload_file,
        sample_extracted_text,
        sample_chunks,
        sample_embeddings
    ):
        """Test upload handles indexing errors"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_generate_id.return_value = "doc_123"
        mock_upload_storage.return_value = "doc_123"
        mock_generate_preview.return_value = None
        mock_doc_service.insert_document.return_value = None
        mock_doc_service.update_document_status.return_value = None
        mock_ingest_document.return_value = sample_extracted_text
        mock_chunk_text.return_value = sample_chunks
        mock_generate_embeddings.return_value = sample_embeddings
        mock_index_chunks.side_effect = AzureServiceError("Indexing failed")
        
        # Execute and verify exception
        with pytest.raises(HTTPException) as exc_info:
            await handle_upload(mock_upload_file)
        
        assert exc_info.value.status_code == 500
        assert "Upload processing failed" in exc_info.value.detail
    
    @patch('src.api.routes.upload.generate_id')
    @patch('src.api.routes.upload.config')
    @pytest.mark.asyncio
    async def test_upload_file_read_error(
        self,
        mock_config_obj,
        mock_generate_id,
        mock_config,
        mock_upload_file
    ):
        """Test upload handles file read errors"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_generate_id.return_value = "doc_123"
        mock_upload_file.read.side_effect = IOError("File read failed")
        
        # Execute and verify exception
        with pytest.raises(HTTPException) as exc_info:
            await handle_upload(mock_upload_file)
        
        assert exc_info.value.status_code == 500
        assert "Upload processing failed" in exc_info.value.detail
    
    @patch('src.api.routes.upload.doc_service')
    @patch('src.api.routes.upload.generate_image_preview')
    @patch('src.api.routes.upload.upload_document_to_storage')
    @patch('src.api.routes.upload.index_chunks')
    @patch('src.api.routes.upload.generate_embeddings')
    @patch('src.api.routes.upload.chunk_text')
    @patch('src.api.routes.upload.ingest_document')
    @patch('src.api.routes.upload.generate_id')
    @patch('src.api.routes.upload.config')
    @pytest.mark.asyncio
    async def test_upload_not_implemented_error(
        self,
        mock_config_obj,
        mock_generate_id,
        mock_ingest_document,
        mock_chunk_text,
        mock_generate_embeddings,
        mock_index_chunks,
        mock_upload_storage,
        mock_generate_preview,
        mock_doc_service,
        mock_config,
        mock_upload_file,
        sample_extracted_text,
        sample_chunks,
        sample_embeddings
    ):
        """Test upload handles NotImplementedError"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_generate_id.return_value = "doc_123"
        mock_upload_storage.return_value = "doc_123"
        mock_generate_preview.return_value = None
        mock_doc_service.insert_document.return_value = None
        mock_doc_service.update_document_status.return_value = None
        mock_ingest_document.return_value = sample_extracted_text
        mock_chunk_text.side_effect = NotImplementedError("Feature not implemented")
        
        # Execute and verify exception
        with pytest.raises(HTTPException) as exc_info:
            await handle_upload(mock_upload_file)
        
        assert exc_info.value.status_code == 501
        assert "Feature not yet implemented" in exc_info.value.detail


class TestUploadEndpointIntegration:
    """Integration tests for upload endpoint with mocked services"""
    
    @patch('src.api.routes.upload.doc_service')
    @patch('src.api.routes.upload.generate_image_preview')
    @patch('src.api.routes.upload.upload_document_to_storage')
    @patch('src.api.routes.upload.index_chunks')
    @patch('src.api.routes.upload.generate_embeddings')
    @patch('src.api.routes.upload.chunk_text')
    @patch('src.api.routes.upload.ingest_document')
    @patch('src.api.routes.upload.config')
    @pytest.mark.asyncio
    async def test_upload_endpoint_integration(
        self,
        mock_config_obj,
        mock_ingest_document,
        mock_chunk_text,
        mock_generate_embeddings,
        mock_index_chunks,
        mock_upload_storage,
        mock_generate_preview,
        mock_doc_service,
        mock_config,
        sample_file_content,
        sample_extracted_text,
        sample_chunks,
        sample_embeddings
    ):
        """Test end-to-end upload pipeline with mocked services"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_upload_storage.return_value = "doc_integration_123"
        mock_generate_preview.return_value = None
        mock_doc_service.insert_document.return_value = None
        mock_doc_service.update_document_status.return_value = None
        mock_doc_service.update_document_chunks.return_value = None
        mock_ingest_document.return_value = sample_extracted_text
        mock_chunk_text.return_value = sample_chunks
        mock_generate_embeddings.return_value = sample_embeddings
        mock_index_chunks.return_value = None
        
        # Create mock upload file
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "test_document.pdf"
        mock_file.read = AsyncMock(return_value=sample_file_content)
        
        # Execute request
        result = await handle_upload(mock_file)
        
        # Verify response
        assert isinstance(result, UploadResponse)
        assert result.status == "success"
        assert len(result.document_id) > 0  # document_id is a UUID string
        assert result.chunks_created == 2
        assert len(result.message) > 0
        
        # Verify pipeline steps were called
        mock_ingest_document.assert_called_once()
        mock_chunk_text.assert_called_once()
        mock_generate_embeddings.assert_called_once()
        mock_index_chunks.assert_called_once()
    
    @patch('src.api.routes.upload.doc_service')
    @patch('src.api.routes.upload.generate_image_preview')
    @patch('src.api.routes.upload.upload_document_to_storage')
    @patch('src.api.routes.upload.ingest_document')
    @patch('src.api.routes.upload.config')
    @pytest.mark.asyncio
    async def test_upload_endpoint_empty_file(
        self,
        mock_config_obj,
        mock_ingest_document,
        mock_upload_storage,
        mock_generate_preview,
        mock_doc_service,
        mock_config
    ):
        """Test upload endpoint handles empty file"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        # Empty file should fail at storage upload, not ingestion
        mock_upload_storage.side_effect = ValueError("file_content cannot be empty")
        mock_ingest_document.return_value = ""  # Empty text
        
        # Create mock upload file with empty content
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "empty.pdf"
        mock_file.read = AsyncMock(return_value=b"")
        
        # Execute request
        with pytest.raises(HTTPException) as exc_info:
            await handle_upload(mock_file)
        
        # Verify error response - empty file fails at storage upload
        assert exc_info.value.status_code == 500
        assert "file_content cannot be empty" in exc_info.value.detail


class TestUploadEndpointResponseFormat:
    """Tests for upload endpoint response format and validation"""
    
    @patch('src.api.routes.upload.doc_service')
    @patch('src.api.routes.upload.generate_image_preview')
    @patch('src.api.routes.upload.upload_document_to_storage')
    @patch('src.api.routes.upload.index_chunks')
    @patch('src.api.routes.upload.generate_embeddings')
    @patch('src.api.routes.upload.chunk_text')
    @patch('src.api.routes.upload.ingest_document')
    @patch('src.api.routes.upload.config')
    @pytest.mark.asyncio
    async def test_upload_response_format(
        self,
        mock_config_obj,
        mock_ingest_document,
        mock_chunk_text,
        mock_generate_embeddings,
        mock_index_chunks,
        mock_upload_storage,
        mock_generate_preview,
        mock_doc_service,
        mock_config,
        sample_file_content,
        sample_extracted_text,
        sample_chunks,
        sample_embeddings
    ):
        """Test upload response has correct format"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_upload_storage.return_value = "doc_format_123"
        mock_generate_preview.return_value = None
        mock_doc_service.insert_document.return_value = None
        mock_doc_service.update_document_status.return_value = None
        mock_doc_service.update_document_chunks.return_value = None
        mock_ingest_document.return_value = sample_extracted_text
        mock_chunk_text.return_value = sample_chunks
        mock_generate_embeddings.return_value = sample_embeddings
        mock_index_chunks.return_value = None
        
        # Create mock upload file
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "test_document.pdf"
        mock_file.read = AsyncMock(return_value=sample_file_content)
        
        # Execute request
        result = await handle_upload(mock_file)
        
        # Verify response format
        assert isinstance(result, UploadResponse)
        
        # Verify all required fields
        assert hasattr(result, "document_id")
        assert hasattr(result, "status")
        assert hasattr(result, "message")
        assert hasattr(result, "chunks_created")
        
        # Verify field types
        assert isinstance(result.document_id, str)
        assert isinstance(result.status, str)
        assert isinstance(result.message, str)
        assert isinstance(result.chunks_created, int)
        
        # Verify field values
        assert result.status == "success"
        assert result.chunks_created == 2
        assert len(result.document_id) > 0

