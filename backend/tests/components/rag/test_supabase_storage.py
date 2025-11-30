"""Tests for Supabase Storage service"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO
from PIL import Image

logging.disable(logging.CRITICAL)

from rag_eval.core.exceptions import DatabaseError
from rag_eval.core.config import Config
from rag_eval.services.rag.supabase_storage import (
    upload_document_to_storage,
    download_document_from_storage,
    delete_document_from_storage,
    get_public_url,
    generate_image_preview,
    _get_supabase_client
)


@pytest.fixture
def mock_config():
    """Create a mock config with Supabase credentials"""
    config = Mock(spec=Config)
    config.supabase_url = "https://test.supabase.co"
    config.supabase_key = "test-key"
    return config


@pytest.fixture
def mock_supabase_client():
    """Create a mock Supabase client"""
    client = MagicMock()
    storage = MagicMock()
    bucket = MagicMock()
    
    bucket.upload.return_value = None
    bucket.download.return_value = b"file content"
    bucket.remove.return_value = None
    bucket.create_signed_url.return_value = "https://test.supabase.co/storage/v1/object/sign/documents/test-id"
    
    storage.from_.return_value = bucket
    client.storage = storage
    
    return client


class TestSupabaseStorage:
    """Tests for Supabase Storage operations"""
    
    @patch('rag_eval.services.rag.supabase_storage._get_supabase_client')
    def test_upload_document_success(self, mock_get_client, mock_config, mock_supabase_client):
        """Test successful document upload"""
        mock_get_client.return_value = mock_supabase_client
        
        result = upload_document_to_storage(
            b"test content",
            "test-id",
            "test.pdf",
            mock_config,
            content_type="application/pdf"
        )
        
        assert result == "test-id"
        mock_supabase_client.storage.from_.assert_called_once_with("documents")
        mock_supabase_client.storage.from_().upload.assert_called_once()
    
    @patch('rag_eval.services.rag.supabase_storage._get_supabase_client')
    def test_upload_document_empty_content(self, mock_get_client, mock_config):
        """Test upload with empty content raises ValueError"""
        with pytest.raises(ValueError, match="file_content cannot be empty"):
            upload_document_to_storage(
                b"",
                "test-id",
                "test.pdf",
                mock_config
            )
    
    @patch('rag_eval.services.rag.supabase_storage._get_supabase_client')
    def test_upload_document_empty_id(self, mock_get_client, mock_config):
        """Test upload with empty document_id raises ValueError"""
        with pytest.raises(ValueError, match="document_id cannot be empty"):
            upload_document_to_storage(
                b"content",
                "",
                "test.pdf",
                mock_config
            )
    
    @patch('rag_eval.services.rag.supabase_storage._get_supabase_client')
    def test_download_document_success(self, mock_get_client, mock_config, mock_supabase_client):
        """Test successful document download"""
        mock_get_client.return_value = mock_supabase_client
        
        result = download_document_from_storage("test-id", mock_config)
        
        assert result == b"file content"
        mock_supabase_client.storage.from_().download.assert_called_once_with("test-id")
    
    @patch('rag_eval.services.rag.supabase_storage._get_supabase_client')
    def test_delete_document_success(self, mock_get_client, mock_config, mock_supabase_client):
        """Test successful document deletion"""
        mock_get_client.return_value = mock_supabase_client
        
        delete_document_from_storage("test-id", mock_config)
        
        mock_supabase_client.storage.from_().remove.assert_called_once_with(["test-id"])
    
    @patch('rag_eval.services.rag.supabase_storage._get_supabase_client')
    def test_get_public_url_success(self, mock_get_client, mock_config, mock_supabase_client):
        """Test getting public URL"""
        mock_get_client.return_value = mock_supabase_client
        
        result = get_public_url("test-id", mock_config)
        
        assert result == "https://test.supabase.co/storage/v1/object/sign/documents/test-id"
        mock_supabase_client.storage.from_().create_signed_url.assert_called_once()
    
    @patch('rag_eval.services.rag.supabase_storage.upload_document_to_storage')
    @patch('rag_eval.services.rag.supabase_storage._get_supabase_client')
    def test_generate_image_preview_jpeg(self, mock_get_client, mock_upload, mock_config):
        """Test generating preview for JPEG image"""
        # Create a test image
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        content = img_bytes.read()
        
        mock_upload.return_value = "test-id_preview.jpg"
        
        result = generate_image_preview(
            content,
            "test-id",
            "image/jpeg",
            mock_config
        )
        
        assert result == "test-id_preview.jpg"
        mock_upload.assert_called_once()
    
    @patch('rag_eval.services.rag.supabase_storage.upload_document_to_storage')
    @patch('rag_eval.services.rag.supabase_storage._get_supabase_client')
    def test_generate_image_preview_png(self, mock_get_client, mock_upload, mock_config):
        """Test generating preview for PNG image"""
        # Create a test PNG image
        img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        content = img_bytes.read()
        
        mock_upload.return_value = "test-id_preview.jpg"
        
        result = generate_image_preview(
            content,
            "test-id",
            "image/png",
            mock_config
        )
        
        assert result == "test-id_preview.jpg"
        mock_upload.assert_called_once()
    
    @patch('rag_eval.services.rag.supabase_storage._get_supabase_client')
    def test_generate_image_preview_unsupported(self, mock_get_client, mock_config):
        """Test preview generation for unsupported file type"""
        result = generate_image_preview(
            b"not an image",
            "test-id",
            "application/pdf",
            mock_config
        )
        
        assert result is None
    
    def test_get_supabase_client_missing_url(self):
        """Test client creation with missing URL"""
        config = Mock(spec=Config)
        config.supabase_url = ""
        config.supabase_key = "test-key"
        
        with pytest.raises(DatabaseError, match="Supabase URL is not configured"):
            _get_supabase_client(config)
    
    def test_get_supabase_client_missing_key(self):
        """Test client creation with missing key"""
        config = Mock(spec=Config)
        config.supabase_url = "https://test.supabase.co"
        config.supabase_key = ""
        
        with pytest.raises(DatabaseError, match="Supabase key is not configured"):
            _get_supabase_client(config)

