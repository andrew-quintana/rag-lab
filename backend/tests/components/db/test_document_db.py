"""Tests for document database service"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import json

logging.disable(logging.CRITICAL)

from rag_eval.core.exceptions import DatabaseError
from rag_eval.db.connection import DatabaseConnection
from rag_eval.db.documents import DocumentService
from rag_eval.db.models import Document


@pytest.fixture
def mock_db_conn():
    """Create a mock database connection"""
    conn = Mock(spec=DatabaseConnection)
    return conn


@pytest.fixture
def mock_query_executor():
    """Create a mock query executor"""
    executor = MagicMock()
    return executor


@pytest.fixture
def document_service(mock_db_conn, mock_query_executor):
    """Create a document service with mocked dependencies"""
    with patch('rag_eval.db.documents.QueryExecutor', return_value=mock_query_executor):
        service = DocumentService(mock_db_conn)
        service.executor = mock_query_executor
        return service


@pytest.fixture
def sample_document():
    """Sample document for testing"""
    return Document(
        id="test-doc-123",
        filename="test.pdf",
        file_size=1024,
        mime_type="application/pdf",
        upload_timestamp=datetime.utcnow(),
        status="uploaded",
        chunks_created=None,
        storage_path="test-doc-123",
        preview_image_path=None,
        metadata=None
    )


class TestDocumentService:
    """Tests for DocumentService"""
    
    def test_insert_document(self, document_service, sample_document, mock_query_executor):
        """Test inserting a document"""
        mock_query_executor.execute_insert.return_value = None
        
        document_service.insert_document(sample_document)
        
        mock_query_executor.execute_insert.assert_called_once()
        call_args = mock_query_executor.execute_insert.call_args
        assert "INSERT INTO documents" in call_args[0][0]
    
    def test_update_document_status(self, document_service, mock_query_executor):
        """Test updating document status"""
        mock_query_executor.execute_insert.return_value = None
        
        document_service.update_document_status("test-doc-123", "processed")
        
        mock_query_executor.execute_insert.assert_called_once()
        call_args = mock_query_executor.execute_insert.call_args
        assert "UPDATE documents SET status" in call_args[0][0]
        assert call_args[0][1] == ("processed", "test-doc-123")
    
    def test_update_document_chunks(self, document_service, mock_query_executor):
        """Test updating document chunks count"""
        mock_query_executor.execute_insert.return_value = None
        
        document_service.update_document_chunks("test-doc-123", 5)
        
        mock_query_executor.execute_insert.assert_called_once()
        call_args = mock_query_executor.execute_insert.call_args
        assert "UPDATE documents SET chunks_created" in call_args[0][0]
        assert call_args[0][1] == (5, "test-doc-123")
    
    def test_get_document(self, document_service, mock_query_executor, sample_document):
        """Test getting a document by ID"""
        mock_query_executor.execute_query.return_value = [{
            "id": sample_document.id,
            "filename": sample_document.filename,
            "file_size": sample_document.file_size,
            "mime_type": sample_document.mime_type,
            "upload_timestamp": sample_document.upload_timestamp,
            "status": sample_document.status,
            "chunks_created": sample_document.chunks_created,
            "storage_path": sample_document.storage_path,
            "preview_image_path": sample_document.preview_image_path,
            "metadata": None
        }]
        
        result = document_service.get_document("test-doc-123")
        
        assert result is not None
        assert result.id == sample_document.id
        assert result.filename == sample_document.filename
    
    def test_get_document_not_found(self, document_service, mock_query_executor):
        """Test getting a non-existent document"""
        mock_query_executor.execute_query.return_value = []
        
        result = document_service.get_document("non-existent")
        
        assert result is None
    
    def test_list_documents(self, document_service, mock_query_executor, sample_document):
        """Test listing documents with filters"""
        mock_query_executor.execute_query.return_value = [{
            "id": sample_document.id,
            "filename": sample_document.filename,
            "file_size": sample_document.file_size,
            "mime_type": sample_document.mime_type,
            "upload_timestamp": sample_document.upload_timestamp,
            "status": sample_document.status,
            "chunks_created": sample_document.chunks_created,
            "storage_path": sample_document.storage_path,
            "preview_image_path": sample_document.preview_image_path,
            "metadata": None
        }]
        
        results = document_service.list_documents(
            search="test",
            file_type="application/pdf",
            status="uploaded",
            limit=10,
            offset=0
        )
        
        assert len(results) == 1
        assert results[0].id == sample_document.id
        mock_query_executor.execute_query.assert_called_once()
    
    def test_count_documents(self, document_service, mock_query_executor):
        """Test counting documents"""
        mock_query_executor.execute_query.return_value = [{"count": 5}]
        
        count = document_service.count_documents(search="test")
        
        assert count == 5
        mock_query_executor.execute_query.assert_called_once()
        call_args = mock_query_executor.execute_query.call_args
        assert "SELECT COUNT(*)" in call_args[0][0]
    
    def test_delete_document(self, document_service, mock_query_executor):
        """Test deleting a document"""
        mock_query_executor.execute_insert.return_value = None
        
        document_service.delete_document("test-doc-123")
        
        mock_query_executor.execute_insert.assert_called_once()
        call_args = mock_query_executor.execute_insert.call_args
        assert "DELETE FROM documents" in call_args[0][0]
        assert call_args[0][1] == ("test-doc-123",)
    
    def test_list_documents_with_date_range(self, document_service, mock_query_executor):
        """Test listing documents with date range filter"""
        mock_query_executor.execute_query.return_value = []
        
        date_from = datetime(2024, 1, 1)
        date_to = datetime(2024, 12, 31)
        
        document_service.list_documents(
            date_from=date_from,
            date_to=date_to
        )
        
        call_args = mock_query_executor.execute_query.call_args
        assert date_from in call_args[0][1]
        assert date_to in call_args[0][1]
    
    def test_list_documents_with_size_range(self, document_service, mock_query_executor):
        """Test listing documents with size range filter"""
        mock_query_executor.execute_query.return_value = []
        
        document_service.list_documents(
            size_min=1000,
            size_max=10000
        )
        
        call_args = mock_query_executor.execute_query.call_args
        assert 1000 in call_args[0][1]
        assert 10000 in call_args[0][1]

