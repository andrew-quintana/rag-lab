"""Unit tests for text chunking - fixed-size chunking only (non-LLM)"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock

# Disable logging during tests to avoid potential hangs
logging.disable(logging.CRITICAL)

from src.core.exceptions import AzureServiceError
from src.core.config import Config
from src.core.interfaces import Chunk
from src.services.rag.chunking import (
    chunk_text_fixed_size,
    chunk_text
)


class TestChunkTextFixedSize:
    """Tests for chunk_text_fixed_size() function - deterministic chunking"""
    
    def test_chunk_text_basic(self):
        """Test basic fixed-size chunking"""
        text = "This is a test document. " * 50  # ~1200 characters
        chunks = chunk_text_fixed_size(text, document_id="doc_123", chunk_size=100, overlap=20)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.document_id == "doc_123" for chunk in chunks)
        assert all(chunk.metadata["chunking_method"] == "fixed_size" for chunk in chunks)
    
    def test_chunk_text_deterministic(self):
        """Test that chunking is deterministic (same input = same output)"""
        text = "This is a test document. " * 50
        document_id = "doc_123"
        
        # Run chunking twice
        chunks1 = chunk_text_fixed_size(text, document_id=document_id, chunk_size=100, overlap=20)
        chunks2 = chunk_text_fixed_size(text, document_id=document_id, chunk_size=100, overlap=20)
        
        # Verify identical results
        assert len(chunks1) == len(chunks2)
        for i, (chunk1, chunk2) in enumerate(zip(chunks1, chunks2)):
            assert chunk1.text == chunk2.text
            assert chunk1.chunk_id == chunk2.chunk_id
            assert chunk1.document_id == chunk2.document_id
            assert chunk1.metadata == chunk2.metadata
    
    def test_chunk_text_with_overlap(self):
        """Test that chunks have proper overlap"""
        text = "A" * 1000  # 1000 characters
        chunks = chunk_text_fixed_size(text, chunk_size=100, overlap=20)
        
        assert len(chunks) > 1
        
        # Check that chunks overlap correctly
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # Verify overlap by checking metadata
            current_end = current_chunk.metadata["end"]
            next_start = next_chunk.metadata["start"]
            
            # Overlap should be: current_end - next_start
            # Since we move back by overlap amount: start = end - overlap
            # So next_start should be current_end - overlap
            expected_overlap = current_end - next_start
            assert expected_overlap == 20
    
    def test_chunk_text_empty_document(self):
        """Test chunking empty document"""
        chunks = chunk_text_fixed_size("", document_id="doc_123")
        
        # Empty document should produce no chunks or one empty chunk
        # Based on implementation, it should produce no chunks
        assert len(chunks) == 0
    
    def test_chunk_text_smaller_than_chunk_size(self):
        """Test chunking text smaller than chunk size"""
        text = "Short text"
        chunks = chunk_text_fixed_size(text, chunk_size=1000, overlap=200)
        
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].metadata["start"] == 0
        assert chunks[0].metadata["end"] == len(text)
    
    def test_chunk_text_exact_chunk_size(self):
        """Test chunking text exactly matching chunk size"""
        text = "A" * 1000
        chunks = chunk_text_fixed_size(text, chunk_size=1000, overlap=200)
        
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].metadata["start"] == 0
        assert chunks[0].metadata["end"] == 1000
    
    def test_chunk_text_very_large_document(self):
        """Test chunking very large document"""
        text = "A" * 100000  # 100k characters
        chunks = chunk_text_fixed_size(text, chunk_size=1000, overlap=200)
        
        # Should produce many chunks
        assert len(chunks) > 50
        
        # Verify all chunks have correct size (except last)
        for i, chunk in enumerate(chunks[:-1]):
            assert len(chunk.text) == 1000
            assert chunk.metadata["start"] == i * (1000 - 200)
            assert chunk.metadata["end"] == chunk.metadata["start"] + 1000
    
    def test_chunk_text_metadata_preservation(self):
        """Test that metadata is preserved correctly"""
        text = "Test document " * 100
        chunks = chunk_text_fixed_size(text, document_id="doc_456", chunk_size=50, overlap=10)
        
        for chunk in chunks:
            assert chunk.document_id == "doc_456"
            assert "start" in chunk.metadata
            assert "end" in chunk.metadata
            assert "chunking_method" in chunk.metadata
            assert chunk.metadata["chunking_method"] == "fixed_size"
            assert chunk.metadata["start"] < chunk.metadata["end"]
    
    def test_chunk_text_chunk_ids(self):
        """Test that chunk IDs are generated correctly"""
        text = "A" * 500
        chunks = chunk_text_fixed_size(text, chunk_size=100, overlap=20)
        
        # Verify chunk IDs are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == f"chunk_{i}"
    
    def test_chunk_text_no_document_id(self):
        """Test chunking without document_id"""
        text = "Test document " * 50
        chunks = chunk_text_fixed_size(text, document_id=None, chunk_size=100, overlap=20)
        
        assert len(chunks) > 0
        assert all(chunk.document_id is None for chunk in chunks)
    
    def test_chunk_text_custom_parameters(self):
        """Test chunking with custom chunk_size and overlap"""
        text = "A" * 2000
        chunks = chunk_text_fixed_size(text, chunk_size=500, overlap=100)
        
        assert len(chunks) > 0
        
        # Verify chunk size (except last)
        for chunk in chunks[:-1]:
            assert len(chunk.text) == 500
        
        # Verify overlap
        for i in range(len(chunks) - 1):
            current_end = chunks[i].metadata["end"]
            next_start = chunks[i + 1].metadata["start"]
            assert current_end - next_start == 100
    
    def test_chunk_text_overlap_validation(self):
        """Test that overlap >= chunk_size raises ValueError"""
        text = "Test text"
        
        # Test overlap equals chunk_size
        with pytest.raises(ValueError) as exc_info:
            chunk_text_fixed_size(text, chunk_size=100, overlap=100)
        assert "overlap" in str(exc_info.value).lower()
        assert "chunk_size" in str(exc_info.value).lower()
        
        # Test overlap greater than chunk_size
        with pytest.raises(ValueError) as exc_info:
            chunk_text_fixed_size(text, chunk_size=100, overlap=150)
        assert "overlap" in str(exc_info.value).lower()
        assert "chunk_size" in str(exc_info.value).lower()


# LLM chunking tests removed - focusing only on fixed-size (deterministic) chunking
# LLM chunking requires Azure AI Foundry credentials and is not deterministic
# For Phase 2, we only test fixed-size chunking which is deterministic and has no external dependencies


class TestChunkText:
    """Tests for chunk_text() function - main entry point"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock config"""
        config = Mock(spec=Config)
        config.azure_ai_foundry_endpoint = "https://test-endpoint.openai.azure.com/"
        config.azure_ai_foundry_api_key = "test-api-key"
        return config
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing"""
        return "This is a sample document. " * 100
    
    @patch('src.services.rag.chunking.chunk_text_fixed_size')
    def test_chunk_text_default_behavior(self, mock_fixed_size, mock_config, sample_text):
        """Test that default behavior uses fixed-size chunking"""
        mock_fixed_size.return_value = [
            Chunk(text="Chunk 1", chunk_id="chunk_0", document_id="doc_123", metadata={})
        ]
        
        # Execute with default use_llm=False
        chunks = chunk_text(sample_text, mock_config, document_id="doc_123")
        
        # Verify fixed-size chunking was called
        mock_fixed_size.assert_called_once_with(sample_text, "doc_123", 1000, 200)
        assert len(chunks) == 1
    
    # LLM chunking option test removed - focusing only on fixed-size chunking
    
    @patch('src.services.rag.chunking.chunk_text_fixed_size')
    def test_chunk_text_custom_parameters(self, mock_fixed_size, mock_config, sample_text):
        """Test that custom chunk_size and overlap are passed correctly"""
        mock_fixed_size.return_value = []
        
        # Execute with custom parameters
        chunk_text(sample_text, mock_config, document_id="doc_123", chunk_size=500, overlap=100)
        
        # Verify parameters were passed
        mock_fixed_size.assert_called_once_with(sample_text, "doc_123", 500, 100)
    
    @patch('src.services.rag.chunking.chunk_text_fixed_size')
    def test_chunk_text_no_document_id(self, mock_fixed_size, mock_config, sample_text):
        """Test chunking without document_id"""
        mock_fixed_size.return_value = []
        
        # Execute without document_id
        chunk_text(sample_text, mock_config, document_id=None)
        
        # Verify None was passed
        mock_fixed_size.assert_called_once_with(sample_text, None, 1000, 200)
    
    # LLM chunking tests removed - focusing only on fixed-size chunking
    # Connection test for Azure AI Foundry removed - not needed for Phase 2

