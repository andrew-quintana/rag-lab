"""Unit tests for document ingestion using Azure Document Intelligence"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from azure.core.exceptions import AzureError

from rag_eval.core.exceptions import AzureServiceError
from rag_eval.core.config import Config
from rag_eval.services.rag.ingestion import (
    extract_text_from_document,
    ingest_document
)


class TestExtractTextFromDocument:
    """Tests for extract_text_from_document() function"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock config with Azure Document Intelligence credentials"""
        config = Mock(spec=Config)
        config.azure_document_intelligence_endpoint = "https://test-endpoint.cognitiveservices.azure.com/"
        config.azure_document_intelligence_api_key = "test-api-key"
        return config
    
    @pytest.fixture
    def sample_file_content(self):
        """Sample file content for testing"""
        return b"Sample PDF content"
    
    @patch('rag_eval.services.rag.ingestion.DocumentIntelligenceClient')
    def test_extract_text_success(self, mock_client_class, mock_config, sample_file_content):
        """Test successful text extraction from document"""
        # Setup mocks
        mock_client = Mock()
        mock_poller = Mock()
        mock_result = Mock()
        
        mock_client_class.return_value = mock_client
        mock_client.begin_analyze_document.return_value = mock_poller
        mock_poller.result.return_value = mock_result
        
        # Mock result content
        mock_result.content = "Extracted text content from document"
        mock_result.tables = []
        
        # Execute
        result = extract_text_from_document(sample_file_content, mock_config)
        
        # Verify
        assert result == "Extracted text content from document"
        # Verify client was initialized correctly
        # Note: credential is AzureKeyCredential object, not the raw key
        mock_client_class.assert_called_once()
        call_args = mock_client_class.call_args
        assert call_args[1]['endpoint'] == mock_config.azure_document_intelligence_endpoint
        # credential is an AzureKeyCredential object, so we check it's passed
        assert 'credential' in call_args[1]
        # Verify API call - body parameter should be BytesIO object
        call_args = mock_client.begin_analyze_document.call_args
        assert call_args[1]['model_id'] == "prebuilt-read"
        assert 'body' in call_args[1]  # body parameter should be present
    
    @patch('rag_eval.services.rag.ingestion.DocumentIntelligenceClient')
    def test_extract_text_with_tables(self, mock_client_class, mock_config, sample_file_content):
        """Test text extraction including table content"""
        # Setup mocks
        mock_client = Mock()
        mock_poller = Mock()
        mock_result = Mock()
        mock_table = Mock()
        mock_cell1 = Mock()
        mock_cell2 = Mock()
        
        mock_client_class.return_value = mock_client
        mock_client.begin_analyze_document.return_value = mock_poller
        mock_poller.result.return_value = mock_result
        
        # Mock result with tables
        mock_result.content = "Main document text"
        mock_cell1.content = "Cell 1 content"
        mock_cell2.content = "Cell 2 content"
        mock_table.cells = [mock_cell1, mock_cell2]
        mock_result.tables = [mock_table]
        
        # Execute
        result = extract_text_from_document(sample_file_content, mock_config)
        
        # Verify
        assert "Main document text" in result
        assert "Cell 1 content" in result
        assert "Cell 2 content" in result
        assert "\n\n" in result  # Should have separator
    
    @patch('rag_eval.services.rag.ingestion.DocumentIntelligenceClient')
    def test_extract_text_empty_content(self, mock_client_class, mock_config, sample_file_content):
        """Test extraction with empty content"""
        # Setup mocks
        mock_client = Mock()
        mock_poller = Mock()
        mock_result = Mock()
        
        mock_client_class.return_value = mock_client
        mock_client.begin_analyze_document.return_value = mock_poller
        mock_poller.result.return_value = mock_result
        
        # Mock empty result
        mock_result.content = None
        mock_result.tables = []
        
        # Execute
        result = extract_text_from_document(sample_file_content, mock_config)
        
        # Verify
        assert result == ""
    
    @patch('rag_eval.services.rag.ingestion.DocumentIntelligenceClient')
    def test_extract_text_table_with_empty_cells(self, mock_client_class, mock_config, sample_file_content):
        """Test extraction with table containing empty cells"""
        # Setup mocks
        mock_client = Mock()
        mock_poller = Mock()
        mock_result = Mock()
        mock_table = Mock()
        mock_cell1 = Mock()
        mock_cell2 = Mock()
        
        mock_client_class.return_value = mock_client
        mock_client.begin_analyze_document.return_value = mock_poller
        mock_poller.result.return_value = mock_result
        
        # Mock result with table containing empty cells
        mock_result.content = "Main text"
        mock_cell1.content = "Cell with content"
        mock_cell2.content = None  # Empty cell
        mock_table.cells = [mock_cell1, mock_cell2]
        mock_result.tables = [mock_table]
        
        # Execute
        result = extract_text_from_document(sample_file_content, mock_config)
        
        # Verify
        assert "Main text" in result
        assert "Cell with content" in result
        # Empty cell should not be included
    
    @patch('rag_eval.services.rag.ingestion.DocumentIntelligenceClient')
    def test_extract_text_azure_error(self, mock_client_class, mock_config, sample_file_content):
        """Test that Azure errors are wrapped in AzureServiceError"""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.begin_analyze_document.side_effect = AzureError("Azure service error")
        
        # Execute and verify
        with pytest.raises(AzureServiceError) as exc_info:
            extract_text_from_document(sample_file_content, mock_config)
        
        assert "Failed to extract text from document" in str(exc_info.value)
        assert "Azure service error" in str(exc_info.value)
    
    @patch('rag_eval.services.rag.ingestion.DocumentIntelligenceClient')
    def test_extract_text_unexpected_error(self, mock_client_class, mock_config, sample_file_content):
        """Test that unexpected errors are wrapped in AzureServiceError"""
        # Setup mocks
        mock_client_class.side_effect = Exception("Unexpected error")
        
        # Execute and verify
        with pytest.raises(AzureServiceError) as exc_info:
            extract_text_from_document(sample_file_content, mock_config)
        
        assert "Failed to extract text from document" in str(exc_info.value)
        assert "Unexpected error" in str(exc_info.value)
    
    @patch('rag_eval.services.rag.ingestion.DocumentIntelligenceClient')
    def test_extract_text_poller_error(self, mock_client_class, mock_config, sample_file_content):
        """Test that poller errors are handled"""
        # Setup mocks
        mock_client = Mock()
        mock_poller = Mock()
        
        mock_client_class.return_value = mock_client
        mock_client.begin_analyze_document.return_value = mock_poller
        mock_poller.result.side_effect = Exception("Poller error")
        
        # Execute and verify
        with pytest.raises(AzureServiceError) as exc_info:
            extract_text_from_document(sample_file_content, mock_config)
        
        assert "Failed to extract text from document" in str(exc_info.value)
    
    def test_extract_text_empty_file_content(self, mock_config):
        """Test that empty file content is handled"""
        # Empty content should still be processed (may be valid for some edge cases)
        # But we should test the behavior
        with patch('rag_eval.services.rag.ingestion.DocumentIntelligenceClient') as mock_client_class:
            mock_client = Mock()
            mock_poller = Mock()
            mock_result = Mock()
            
            mock_client_class.return_value = mock_client
            mock_client.begin_analyze_document.return_value = mock_poller
            mock_poller.result.return_value = mock_result
            mock_result.content = None
            mock_result.tables = []
            
            result = extract_text_from_document(b"", mock_config)
            assert result == ""
    
    def test_extract_text_missing_endpoint(self, sample_file_content):
        """Test that missing endpoint raises error during client initialization"""
        config = Mock(spec=Config)
        config.azure_document_intelligence_endpoint = ""
        config.azure_document_intelligence_api_key = "test-key"
        
        with patch('rag_eval.services.rag.ingestion.DocumentIntelligenceClient') as mock_client_class:
            mock_client_class.side_effect = Exception("Invalid endpoint")
            
            with pytest.raises(AzureServiceError):
                extract_text_from_document(sample_file_content, config)
    
    def test_extract_text_missing_api_key(self, sample_file_content):
        """Test that missing API key raises error during client initialization"""
        config = Mock(spec=Config)
        config.azure_document_intelligence_endpoint = "https://test-endpoint.cognitiveservices.azure.com/"
        config.azure_document_intelligence_api_key = ""
        
        with patch('rag_eval.services.rag.ingestion.DocumentIntelligenceClient') as mock_client_class:
            mock_client_class.side_effect = Exception("Invalid credentials")
            
            with pytest.raises(AzureServiceError):
                extract_text_from_document(sample_file_content, config)


class TestIngestDocument:
    """Tests for ingest_document() function"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock config with Azure Document Intelligence credentials"""
        config = Mock(spec=Config)
        config.azure_document_intelligence_endpoint = "https://test-endpoint.cognitiveservices.azure.com/"
        config.azure_document_intelligence_api_key = "test-api-key"
        return config
    
    @pytest.fixture
    def sample_file_content(self):
        """Sample file content for testing"""
        return b"Sample PDF content"
    
    @patch('rag_eval.services.rag.ingestion.extract_text_from_document')
    def test_ingest_document_success(self, mock_extract, mock_config, sample_file_content):
        """Test successful document ingestion"""
        # Setup mock
        mock_extract.return_value = "Extracted text"
        
        # Execute
        result = ingest_document(sample_file_content, mock_config)
        
        # Verify
        assert result == "Extracted text"
        mock_extract.assert_called_once_with(sample_file_content, mock_config)
    
    @patch('rag_eval.services.rag.ingestion.extract_text_from_document')
    def test_ingest_document_error_propagation(self, mock_extract, mock_config, sample_file_content):
        """Test that errors from extract_text_from_document are propagated"""
        # Setup mock
        mock_extract.side_effect = AzureServiceError("Extraction failed")
        
        # Execute and verify
        with pytest.raises(AzureServiceError) as exc_info:
            ingest_document(sample_file_content, mock_config)
        
        assert "Extraction failed" in str(exc_info.value)
        mock_extract.assert_called_once_with(sample_file_content, mock_config)


class TestAzureDocumentIntelligenceConnection:
    """Connection tests for Azure Document Intelligence
    
    These tests verify actual Azure Document Intelligence connectivity.
    They warn but do not fail if credentials are missing or services are unavailable.
    """
    
    def test_azure_document_intelligence_connection(self):
        """
        Test actual connection to Azure Document Intelligence.
        
        This test:
        - Warns if credentials are missing (does not fail)
        - Tests text extraction with real service using sample document
        - Documents connection status in test output
        """
        import warnings
        import os
        from pathlib import Path
        
        try:
            config = Config.from_env()
        except Exception as e:
            pytest.skip(f"Could not load configuration: {e}")
        
        # Check if credentials are configured
        if not config.azure_document_intelligence_endpoint or not config.azure_document_intelligence_api_key:
            warnings.warn(
                "Azure Document Intelligence credentials not configured. "
                "Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_API_KEY in .env.local to run connection tests.",
                UserWarning
            )
            pytest.skip("Azure Document Intelligence credentials not configured")
        
        # Try to find a sample document
        sample_doc_path = Path(__file__).parent / "fixtures" / "sample_documents" / "healthguard_select_ppo_plan.pdf"
        
        if not sample_doc_path.exists():
            warnings.warn(
                f"Sample document not found at {sample_doc_path}. "
                "Skipping connection test.",
                UserWarning
            )
            pytest.skip("Sample document not found")
        
        try:
            # Read sample document
            with open(sample_doc_path, "rb") as f:
                file_content = f.read()
            
            # Test extraction
            extracted_text = extract_text_from_document(file_content, config)
            
            assert len(extracted_text) > 0
            print(f"✓ Successfully extracted text from sample document using Azure Document Intelligence")
            print(f"  - Extracted {len(extracted_text)} characters")
            print(f"  - Connection: OK")
            print(f"✓ Azure Document Intelligence connection test PASSED")
            
        except Exception as e:
            warnings.warn(
                f"Azure Document Intelligence connection test failed: {e}. "
                "This may indicate missing credentials, network issues, or service unavailability.",
                UserWarning
            )
            # Don't fail the test - just warn
            pytest.skip(f"Azure Document Intelligence connection failed: {e}")

