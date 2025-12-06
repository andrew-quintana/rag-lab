"""Unit and integration tests for upload endpoint (Phase 4: API Integration)"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from fastapi import UploadFile, HTTPException
from io import BytesIO
from datetime import datetime

# Disable logging during tests to avoid potential hangs
logging.disable(logging.CRITICAL)

from rag_eval.core.exceptions import AzureServiceError, ValidationError
from rag_eval.core.config import Config
from rag_eval.api.routes.upload import router, handle_upload, UploadResponse
from rag_eval.api.routes.documents import (
    get_document_status,
    delete_document,
    DocumentStatusResponse,
    DeleteDocumentResponse
)
from rag_eval.services.workers.queue_client import QueueMessage, SourceStorage, ProcessingStage


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
    config.azure_blob_connection_string = "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=test;EndpointSuffix=core.windows.net"
    return config


@pytest.fixture
def sample_file_content():
    """Sample PDF file content for testing"""
    return b"Sample PDF content for testing"


@pytest.fixture
def mock_upload_file(sample_file_content):
    """Create a mock UploadFile for testing"""
    file = Mock(spec=UploadFile)
    file.filename = "test_document.pdf"
    file.content_type = "application/pdf"
    file.read = AsyncMock(return_value=sample_file_content)
    return file


class TestUploadEndpointUnit:
    """Unit tests for upload endpoint handler (asynchronous)"""
    
    @patch('rag_eval.api.routes.upload.enqueue_message')
    @patch('rag_eval.api.routes.upload.doc_service')
    @patch('rag_eval.api.routes.upload.generate_image_preview')
    @patch('rag_eval.api.routes.upload.upload_document_to_storage')
    @patch('rag_eval.api.routes.upload.generate_id')
    @patch('rag_eval.api.routes.upload.config')
    @pytest.mark.asyncio
    async def test_upload_enqueues_message(
        self,
        mock_config_obj,
        mock_generate_id,
        mock_upload_storage,
        mock_generate_preview,
        mock_doc_service,
        mock_enqueue_message,
        mock_config,
        mock_upload_file,
        sample_file_content
    ):
        """Test upload endpoint enqueues message correctly"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_generate_id.return_value = "doc_123"
        mock_upload_storage.return_value = "doc_123"
        mock_generate_preview.return_value = None
        mock_doc_service.insert_document.return_value = None
        mock_enqueue_message.return_value = None
        
        # Execute
        result = await handle_upload(mock_upload_file)
        
        # Verify response
        assert isinstance(result, UploadResponse)
        assert result.document_id == "doc_123"
        assert result.status == "uploaded"
        assert "enqueued" in result.message.lower()
        
        # Verify message was enqueued
        mock_enqueue_message.assert_called_once()
        call_args = mock_enqueue_message.call_args
        assert call_args[0][0] == "ingestion-uploads"  # queue name
        assert isinstance(call_args[0][1], QueueMessage)  # message
        # Config is passed from module-level, so we just verify it's a config object
        assert call_args[0][2] is not None  # config
        
        # Verify message content
        message = call_args[0][1]
        assert message.document_id == "doc_123"
        assert message.source_storage == SourceStorage.SUPABASE
        assert message.filename == "test_document.pdf"
        assert message.attempt == 1
        assert message.stage == ProcessingStage.UPLOADED
        assert message.metadata is not None
        assert message.metadata["mime_type"] == "application/pdf"
    
    @patch('rag_eval.api.routes.upload.enqueue_message')
    @patch('rag_eval.api.routes.upload.doc_service')
    @patch('rag_eval.api.routes.upload.generate_image_preview')
    @patch('rag_eval.api.routes.upload.upload_document_to_storage')
    @patch('rag_eval.api.routes.upload.generate_id')
    @patch('rag_eval.api.routes.upload.config')
    @pytest.mark.asyncio
    async def test_upload_returns_immediately(
        self,
        mock_config_obj,
        mock_generate_id,
        mock_upload_storage,
        mock_generate_preview,
        mock_doc_service,
        mock_enqueue_message,
        mock_config,
        mock_upload_file
    ):
        """Test upload endpoint returns immediately with document_id"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_generate_id.return_value = "doc_456"
        mock_upload_storage.return_value = "doc_456"
        mock_generate_preview.return_value = None
        mock_doc_service.insert_document.return_value = None
        mock_enqueue_message.return_value = None
        
        # Execute
        result = await handle_upload(mock_upload_file)
        
        # Verify response is immediate (no processing wait)
        assert result.document_id == "doc_456"
        assert result.status == "uploaded"
        
        # Verify message was enqueued (asynchronous processing)
        mock_enqueue_message.assert_called_once()
        # Verify no synchronous processing happened - message was enqueued instead
    
    @patch('rag_eval.api.routes.upload.enqueue_message')
    @patch('rag_eval.api.routes.upload.doc_service')
    @patch('rag_eval.api.routes.upload.generate_image_preview')
    @patch('rag_eval.api.routes.upload.upload_document_to_storage')
    @patch('rag_eval.api.routes.upload.generate_id')
    @patch('rag_eval.api.routes.upload.config')
    @pytest.mark.asyncio
    async def test_upload_creates_document_with_uploaded_status(
        self,
        mock_config_obj,
        mock_generate_id,
        mock_upload_storage,
        mock_generate_preview,
        mock_doc_service,
        mock_enqueue_message,
        mock_config,
        mock_upload_file
    ):
        """Test upload creates document record with status='uploaded'"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_generate_id.return_value = "doc_789"
        mock_upload_storage.return_value = "doc_789"
        mock_generate_preview.return_value = None
        mock_doc_service.insert_document.return_value = None
        mock_enqueue_message.return_value = None
        
        # Execute
        await handle_upload(mock_upload_file)
        
        # Verify document was inserted with status='uploaded'
        mock_doc_service.insert_document.assert_called_once()
        call_args = mock_doc_service.insert_document.call_args
        document = call_args[0][0]
        assert document.status == "uploaded"
        assert document.id == "doc_789"
    
    @patch('rag_eval.api.routes.upload.enqueue_message')
    @patch('rag_eval.api.routes.upload.doc_service')
    @patch('rag_eval.api.routes.upload.generate_image_preview')
    @patch('rag_eval.api.routes.upload.upload_document_to_storage')
    @patch('rag_eval.api.routes.upload.generate_id')
    @patch('rag_eval.api.routes.upload.config')
    @pytest.mark.asyncio
    async def test_upload_handles_storage_error(
        self,
        mock_config_obj,
        mock_generate_id,
        mock_upload_storage,
        mock_generate_preview,
        mock_doc_service,
        mock_enqueue_message,
        mock_config,
        mock_upload_file
    ):
        """Test upload handles storage upload errors"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_generate_id.return_value = "doc_error"
        mock_upload_storage.side_effect = AzureServiceError("Storage upload failed")
        
        # Execute and verify exception
        with pytest.raises(HTTPException) as exc_info:
            await handle_upload(mock_upload_file)
        
        assert exc_info.value.status_code == 500
        assert "Upload processing failed" in exc_info.value.detail
    
    @patch('rag_eval.api.routes.upload.enqueue_message')
    @patch('rag_eval.api.routes.upload.doc_service')
    @patch('rag_eval.api.routes.upload.generate_image_preview')
    @patch('rag_eval.api.routes.upload.upload_document_to_storage')
    @patch('rag_eval.api.routes.upload.generate_id')
    @patch('rag_eval.api.routes.upload.config')
    @pytest.mark.asyncio
    async def test_upload_handles_enqueue_error(
        self,
        mock_config_obj,
        mock_generate_id,
        mock_upload_storage,
        mock_generate_preview,
        mock_doc_service,
        mock_enqueue_message,
        mock_config,
        mock_upload_file
    ):
        """Test upload handles queue enqueue errors"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_generate_id.return_value = "doc_enqueue_error"
        mock_upload_storage.return_value = "doc_enqueue_error"
        mock_generate_preview.return_value = None
        mock_doc_service.insert_document.return_value = None
        mock_enqueue_message.side_effect = AzureServiceError("Queue enqueue failed")
        
        # Execute and verify exception
        with pytest.raises(HTTPException) as exc_info:
            await handle_upload(mock_upload_file)
        
        assert exc_info.value.status_code == 500
        assert "Upload processing failed" in exc_info.value.detail


class TestStatusEndpoint:
    """Tests for document status query endpoint"""
    
    @patch('rag_eval.api.routes.documents.QueryExecutor')
    @patch('rag_eval.api.routes.documents.doc_service')
    @pytest.mark.asyncio
    async def test_get_status_returns_correct_status(
        self,
        mock_doc_service,
        mock_query_executor_class,
    ):
        """Test status query endpoint returns correct status"""
        # Setup mocks
        from rag_eval.db.models import Document
        document = Document(
            id="doc_123",
            filename="test.pdf",
            file_size=1024,
            status="parsed",
            storage_path="doc_123"
        )
        mock_doc_service.get_document.return_value = document
        
        # Mock QueryExecutor instance
        mock_executor = Mock()
        mock_executor.execute_query.return_value = [{
            "status": "parsed",
            "parsed_at": datetime(2024, 1, 1, 12, 0, 0),
            "chunked_at": None,
            "embedded_at": None,
            "indexed_at": None,
            "metadata": None
        }]
        mock_query_executor_class.return_value = mock_executor
        
        # Execute
        result = await get_document_status("doc_123")
        
        # Verify response
        assert isinstance(result, DocumentStatusResponse)
        assert result.document_id == "doc_123"
        assert result.status == "parsed"
        assert result.parsed_at is not None
        assert result.chunked_at is None
    
    @patch('rag_eval.api.routes.documents.QueryExecutor')
    @patch('rag_eval.api.routes.documents.doc_service')
    @pytest.mark.asyncio
    async def test_get_status_returns_error_details_for_failed_status(
        self,
        mock_doc_service,
        mock_query_executor_class,
    ):
        """Test status endpoint returns error details for failed status"""
        # Setup mocks
        from rag_eval.db.models import Document
        document = Document(
            id="doc_123",
            filename="test.pdf",
            file_size=1024,
            status="failed_parsing",
            storage_path="doc_123"
        )
        mock_doc_service.get_document.return_value = document
        
        # Mock QueryExecutor with error metadata
        mock_executor = Mock()
        mock_executor.execute_query.return_value = [{
            "status": "failed_parsing",
            "parsed_at": None,
            "chunked_at": None,
            "embedded_at": None,
            "indexed_at": None,
            "metadata": {"error_details": "Document Intelligence API error"}
        }]
        mock_query_executor_class.return_value = mock_executor
        
        # Execute
        result = await get_document_status("doc_123")
        
        # Verify response
        assert result.status == "failed_parsing"
        assert result.error_details == "Document Intelligence API error"
    
    @patch('rag_eval.api.routes.documents.doc_service')
    @pytest.mark.asyncio
    async def test_get_status_returns_404_for_missing_document(
        self,
        mock_doc_service,
    ):
        """Test status endpoint returns 404 for missing document"""
        # Setup mocks
        mock_doc_service.get_document.return_value = None
        
        # Execute and verify exception
        with pytest.raises(HTTPException) as exc_info:
            await get_document_status("non_existent")
        
        assert exc_info.value.status_code == 404
        assert "Document not found" in exc_info.value.detail


class TestDeleteEndpoint:
    """Tests for document deletion endpoint"""
    
    @patch('rag_eval.api.routes.documents.delete_chunks_from_db')
    @patch('rag_eval.api.routes.documents.delete_chunks_from_ai_search')
    @patch('rag_eval.api.routes.documents.delete_document_from_storage')
    @patch('rag_eval.api.routes.documents.doc_service')
    @pytest.mark.asyncio
    async def test_delete_removes_chunks_from_both_systems(
        self,
        mock_doc_service,
        mock_delete_storage,
        mock_delete_ai_search,
        mock_delete_db,
    ):
        """Test delete endpoint removes chunks from both chunks table and Azure AI Search"""
        # Setup mocks
        from rag_eval.db.models import Document
        document = Document(
            id="doc_123",
            filename="test.pdf",
            file_size=1024,
            status="indexed",
            storage_path="doc_123"
        )
        mock_doc_service.get_document.return_value = document
        mock_delete_ai_search.return_value = 5  # 5 chunks deleted from AI Search
        mock_delete_db.return_value = 5  # 5 chunks deleted from DB
        mock_delete_storage.return_value = None
        mock_doc_service.delete_document.return_value = None
        
        # Execute
        result = await delete_document("doc_123")
        
        # Verify response
        assert isinstance(result, DeleteDocumentResponse)
        assert result.document_id == "doc_123"
        assert result.chunks_deleted_db == 5
        assert result.chunks_deleted_ai_search == 5
        
        # Verify all deletion functions were called
        # Note: config is passed from the module-level config, not from doc_service
        mock_delete_ai_search.assert_called_once()
        mock_delete_db.assert_called_once()
        mock_delete_storage.assert_called_once()
        mock_doc_service.delete_document.assert_called_once_with("doc_123")
    
    @patch('rag_eval.api.routes.documents.delete_chunks_from_db')
    @patch('rag_eval.api.routes.documents.delete_chunks_from_ai_search')
    @patch('rag_eval.api.routes.documents.delete_document_from_storage')
    @patch('rag_eval.api.routes.documents.doc_service')
    @pytest.mark.asyncio
    async def test_delete_graceful_degradation(
        self,
        mock_doc_service,
        mock_delete_storage,
        mock_delete_ai_search,
        mock_delete_db,
    ):
        """Test delete endpoint continues with other deletions if one fails"""
        # Setup mocks
        from rag_eval.db.models import Document
        document = Document(
            id="doc_123",
            filename="test.pdf",
            file_size=1024,
            status="indexed",
            storage_path="doc_123"
        )
        mock_doc_service.get_document.return_value = document
        mock_delete_ai_search.side_effect = AzureServiceError("AI Search deletion failed")
        mock_delete_db.return_value = 3  # DB deletion succeeds
        mock_delete_storage.return_value = None
        mock_doc_service.delete_document.return_value = None
        
        # Execute
        result = await delete_document("doc_123")
        
        # Verify response - should still succeed with partial deletion
        assert result.chunks_deleted_db == 3
        assert result.chunks_deleted_ai_search == 0  # Failed, so 0
        
        # Verify other deletions still happened
        mock_delete_db.assert_called_once()
        mock_delete_storage.assert_called_once()
        mock_doc_service.delete_document.assert_called_once()
    
    @patch('rag_eval.api.routes.documents.doc_service')
    @pytest.mark.asyncio
    async def test_delete_returns_404_for_missing_document(
        self,
        mock_doc_service,
    ):
        """Test delete endpoint returns 404 for missing document"""
        # Setup mocks
        mock_doc_service.get_document.return_value = None
        
        # Execute and verify exception
        with pytest.raises(HTTPException) as exc_info:
            await delete_document("non_existent")
        
        assert exc_info.value.status_code == 404
        assert "Document not found" in exc_info.value.detail


class TestResponseModels:
    """Tests for response model validation"""
    
    def test_upload_response_model(self):
        """Test UploadResponse model validation"""
        response = UploadResponse(
            document_id="doc_123",
            status="uploaded",
            message="Document uploaded and enqueued for processing"
        )
        
        assert response.document_id == "doc_123"
        assert response.status == "uploaded"
        assert response.message is not None
    
    def test_document_status_response_model(self):
        """Test DocumentStatusResponse model validation"""
        response = DocumentStatusResponse(
            document_id="doc_123",
            status="parsed",
            parsed_at=datetime(2024, 1, 1, 12, 0, 0)
        )
        
        assert response.document_id == "doc_123"
        assert response.status == "parsed"
        assert response.parsed_at is not None
    
    def test_delete_document_response_model(self):
        """Test DeleteDocumentResponse model validation"""
        response = DeleteDocumentResponse(
            message="Document deleted successfully",
            document_id="doc_123",
            chunks_deleted_db=5,
            chunks_deleted_ai_search=5
        )
        
        assert response.document_id == "doc_123"
        assert response.chunks_deleted_db == 5
        assert response.chunks_deleted_ai_search == 5
