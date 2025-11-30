"""Tests for document management API endpoints"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime

logging.disable(logging.CRITICAL)

from rag_eval.api.routes.documents import (
    list_documents,
    get_document,
    download_document,
    get_preview,
    delete_document,
    DocumentResponse,
    DocumentListResponse
)
from rag_eval.db.models import Document


@pytest.fixture
def sample_document():
    """Sample document for testing"""
    return Document(
        document_id="test-doc-123",
        filename="test.pdf",
        file_size=1024,
        mime_type="application/pdf",
        upload_timestamp=datetime.utcnow(),
        status="uploaded",
        chunks_created=5,
        storage_path="test-doc-123",
        preview_image_path="test-doc-123_preview.jpg",
        metadata={"original_filename": "test.pdf"}
    )


class TestDocumentEndpoints:
    """Tests for document API endpoints"""
    
    @patch('rag_eval.api.routes.documents.doc_service')
    @pytest.mark.asyncio
    async def test_list_documents(self, mock_doc_service, sample_document):
        """Test listing documents"""
        mock_doc_service.list_documents.return_value = [sample_document]
        mock_doc_service.count_documents.return_value = 1
        
        response = await list_documents(
            search=None,
            file_type=None,
            status=None,
            date_from=None,
            date_to=None,
            size_min=None,
            size_max=None,
            sort_by="upload_timestamp",
            sort_order="desc",
            limit=100,
            offset=0
        )
        
        assert isinstance(response, DocumentListResponse)
        assert len(response.documents) == 1
        assert response.total == 1
        assert response.documents[0].document_id == "test-doc-123"
    
    @patch('rag_eval.api.routes.documents.doc_service')
    @pytest.mark.asyncio
    async def test_list_documents_with_filters(self, mock_doc_service, sample_document):
        """Test listing documents with filters"""
        mock_doc_service.list_documents.return_value = [sample_document]
        mock_doc_service.count_documents.return_value = 1
        
        response = await list_documents(
            search="test",
            file_type="application/pdf",
            status="uploaded",
            date_from=None,
            date_to=None,
            size_min=None,
            size_max=None,
            sort_by="upload_timestamp",
            sort_order="desc",
            limit=100,
            offset=0
        )
        
        assert isinstance(response, DocumentListResponse)
        mock_doc_service.list_documents.assert_called_once()
        call_kwargs = mock_doc_service.list_documents.call_args[1]
        assert call_kwargs["search"] == "test"
        assert call_kwargs["file_type"] == "application/pdf"
        assert call_kwargs["status"] == "uploaded"
    
    @patch('rag_eval.api.routes.documents.doc_service')
    @pytest.mark.asyncio
    async def test_get_document(self, mock_doc_service, sample_document):
        """Test getting a document by ID"""
        mock_doc_service.get_document.return_value = sample_document
        
        response = await get_document("test-doc-123")
        
        assert isinstance(response, DocumentResponse)
        assert response.document_id == "test-doc-123"
        assert response.filename == "test.pdf"
    
    @patch('rag_eval.api.routes.documents.doc_service')
    @pytest.mark.asyncio
    async def test_get_document_not_found(self, mock_doc_service):
        """Test getting a non-existent document"""
        from fastapi import HTTPException
        mock_doc_service.get_document.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            await get_document("non-existent")
        
        assert exc_info.value.status_code == 404
    
    @patch('rag_eval.api.routes.documents.download_document_from_storage')
    @patch('rag_eval.api.routes.documents.config')
    @patch('rag_eval.api.routes.documents.doc_service')
    @pytest.mark.asyncio
    async def test_download_document(self, mock_doc_service, mock_config, mock_download, sample_document):
        """Test downloading a document"""
        from fastapi.responses import StreamingResponse
        mock_doc_service.get_document.return_value = sample_document
        mock_download.return_value = b"file content"
        
        response = await download_document("test-doc-123")
        
        assert isinstance(response, StreamingResponse)
        mock_doc_service.get_document.assert_called_once_with("test-doc-123")
        mock_download.assert_called_once_with("test-doc-123", mock_config)
    
    @patch('rag_eval.api.routes.documents.download_document_from_storage')
    @patch('rag_eval.api.routes.documents.doc_service')
    @pytest.mark.asyncio
    async def test_get_preview(self, mock_doc_service, mock_download, sample_document):
        """Test getting preview image"""
        from fastapi import Response
        mock_doc_service.get_document.return_value = sample_document
        mock_download.return_value = b"image content"
        
        response = await get_preview("test-doc-123")
        
        assert isinstance(response, Response)
        mock_doc_service.get_document.assert_called_once_with("test-doc-123")
    
    @patch('rag_eval.api.routes.documents.doc_service')
    @pytest.mark.asyncio
    async def test_get_preview_not_available(self, mock_doc_service):
        """Test getting preview when not available"""
        from fastapi import HTTPException
        doc = Document(
            document_id="test-doc-123",
            filename="test.pdf",
            file_size=1024,
            status="uploaded",
            storage_path="test-doc-123",
            preview_image_path=None
        )
        mock_doc_service.get_document.return_value = doc
        
        with pytest.raises(HTTPException) as exc_info:
            await get_preview("test-doc-123")
        
        assert exc_info.value.status_code == 404
    
    @patch('rag_eval.api.routes.documents.delete_document_from_storage')
    @patch('rag_eval.api.routes.documents.doc_service')
    @pytest.mark.asyncio
    async def test_delete_document(self, mock_doc_service, mock_delete_storage, sample_document):
        """Test deleting a document"""
        mock_doc_service.get_document.return_value = sample_document
        mock_doc_service.delete_document.return_value = None
        
        response = await delete_document("test-doc-123")
        
        assert response["message"] == "Document deleted successfully"
        assert response["document_id"] == "test-doc-123"
        mock_doc_service.delete_document.assert_called_once_with("test-doc-123")
    
    @patch('rag_eval.api.routes.documents.doc_service')
    @pytest.mark.asyncio
    async def test_delete_document_not_found(self, mock_doc_service):
        """Test deleting a non-existent document"""
        from fastapi import HTTPException
        mock_doc_service.get_document.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            await delete_document("non-existent")
        
        assert exc_info.value.status_code == 404
    
    @patch('rag_eval.api.routes.documents.doc_service')
    @pytest.mark.asyncio
    async def test_list_documents_pagination(self, mock_doc_service, sample_document):
        """Test document list pagination"""
        mock_doc_service.list_documents.return_value = [sample_document]
        mock_doc_service.count_documents.return_value = 50
        
        response = await list_documents(
            search=None,
            file_type=None,
            status=None,
            date_from=None,
            date_to=None,
            size_min=None,
            size_max=None,
            sort_by="upload_timestamp",
            sort_order="desc",
            limit=10,
            offset=20
        )
        
        call_kwargs = mock_doc_service.list_documents.call_args[1]
        assert call_kwargs["limit"] == 10
        assert call_kwargs["offset"] == 20
    
    @patch('rag_eval.api.routes.documents.doc_service')
    @pytest.mark.asyncio
    async def test_list_documents_sorting(self, mock_doc_service, sample_document):
        """Test document list sorting"""
        mock_doc_service.list_documents.return_value = [sample_document]
        mock_doc_service.count_documents.return_value = 1
        
        response = await list_documents(
            search=None,
            file_type=None,
            status=None,
            date_from=None,
            date_to=None,
            size_min=None,
            size_max=None,
            sort_by="filename",
            sort_order="asc",
            limit=100,
            offset=0
        )
        
        call_kwargs = mock_doc_service.list_documents.call_args[1]
        assert call_kwargs["sort_by"] == "filename"
        assert call_kwargs["sort_order"] == "asc"
