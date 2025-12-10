"""Comprehensive unit tests for indexing worker

Tests queue message parsing, chunk and embedding loading, indexing, status updates,
retry logic, and idempotency.

These tests use actual files to ensure realistic test data.
"""

import pytest
import logging
from unittest.mock import Mock, patch

logging.disable(logging.CRITICAL)

from src.core.exceptions import AzureServiceError, DatabaseError, ValidationError
from src.services.workers.indexing_worker import indexing_worker
from src.services.workers.queue_client import QueueMessage, ProcessingStage
from src.core.interfaces import Chunk


@pytest.fixture
def mock_config():
    """Create a mock config"""
    config = Mock()
    config.azure_search_endpoint = "https://test.search.windows.net"
    config.azure_search_api_key = "test-key"
    config.azure_search_index_name = "test-index"
    config.database_url = "postgresql://test:test@localhost/test"
    return config


@pytest.fixture
def actual_chunks():
    """Create actual chunks for testing"""
    from src.services.rag.chunking import chunk_text_fixed_size
    text = """HEALTHGUARD SELECT PPO PLAN
2025 Medicare Evidence of Coverage

This document provides information about your health insurance coverage.
It includes details about benefits, copayments, and coverage limits."""
    return chunk_text_fixed_size(
        text=text,
        document_id="doc_123",
        chunk_size=500,
        overlap=100
    )


@pytest.fixture
def actual_embeddings(actual_chunks):
    """Create realistic embeddings for actual chunks"""
    import random
    random.seed(42)
    
    embeddings = []
    for i, chunk in enumerate(actual_chunks):
        random.seed(42 + i)
        embedding = [random.uniform(-1.0, 1.0) for _ in range(1536)]
        embeddings.append(embedding)
    
    return embeddings


@pytest.fixture
def valid_message_dict():
    """Valid message dictionary for testing"""
    return {
        "document_id": "123e4567-e89b-12d3-a456-426614174000",
        "source_storage": "supabase",
        "filename": "test_document.pdf",
        "attempt": 1,
        "stage": "embedded",
        "metadata": {}
    }


@pytest.fixture
def context_with_config(mock_config):
    """Context object with config"""
    return {"config": mock_config}


class TestIndexingWorkerMessageParsing:
    """Tests for queue message parsing"""
    
    def test_valid_message(self, valid_message_dict, context_with_config, mock_config, actual_chunks, actual_embeddings):
        """Test indexing worker with valid message"""
        with patch('src.services.workers.indexing_worker.should_process_document', return_value=True), \
             patch('src.services.workers.indexing_worker.load_chunks', return_value=actual_chunks), \
             patch('src.services.workers.indexing_worker.load_embeddings', return_value=actual_embeddings), \
             patch('src.services.workers.indexing_worker.index_chunks'), \
             patch('src.services.workers.indexing_worker.update_document_status'):
            
            indexing_worker(valid_message_dict, context_with_config)
            
            # Verify chunks and embeddings were loaded
            from src.services.workers.indexing_worker import load_chunks, load_embeddings
            load_chunks.assert_called_once_with(
                valid_message_dict["document_id"],
                mock_config
            )
            load_embeddings.assert_called_once_with(
                valid_message_dict["document_id"],
                mock_config
            )
            
            # Verify indexing was called
            from src.services.workers.indexing_worker import index_chunks
            index_chunks.assert_called_once_with(actual_chunks, actual_embeddings, mock_config)
    
    def test_invalid_message(self, context_with_config):
        """Test indexing worker with invalid message"""
        invalid_message = {"document_id": "123"}  # Missing required fields
        
        with pytest.raises(ValidationError):
            indexing_worker(invalid_message, context_with_config)


class TestIndexingWorkerIdempotency:
    """Tests for idempotency checks"""
    
    def test_skip_if_already_indexed(self, valid_message_dict, context_with_config):
        """Test that worker skips processing if document is already indexed"""
        with patch('src.services.workers.indexing_worker.should_process_document', return_value=False), \
             patch('src.services.workers.indexing_worker.load_chunks') as mock_load_chunks, \
             patch('src.services.workers.indexing_worker.load_embeddings') as mock_load_embeddings:
            
            indexing_worker(valid_message_dict, context_with_config)
            
            # Verify load was not called
            mock_load_chunks.assert_not_called()
            mock_load_embeddings.assert_not_called()


class TestIndexingWorkerLoad:
    """Tests for chunk and embedding loading"""
    
    def test_load_chunks_and_embeddings_success(self, valid_message_dict, context_with_config, mock_config, actual_chunks, actual_embeddings):
        """Test successful loading of chunks and embeddings"""
        with patch('src.services.workers.indexing_worker.should_process_document', return_value=True), \
             patch('src.services.workers.indexing_worker.load_chunks', return_value=actual_chunks) as mock_load_chunks, \
             patch('src.services.workers.indexing_worker.load_embeddings', return_value=actual_embeddings) as mock_load_embeddings, \
             patch('src.services.workers.indexing_worker.index_chunks'), \
             patch('src.services.workers.indexing_worker.update_document_status'):
            
            indexing_worker(valid_message_dict, context_with_config)
            
            mock_load_chunks.assert_called_once_with(
                valid_message_dict["document_id"],
                mock_config
            )
            mock_load_embeddings.assert_called_once_with(
                valid_message_dict["document_id"],
                mock_config
            )
    
    def test_load_no_chunks(self, valid_message_dict, context_with_config):
        """Test handling of no chunks found"""
        with patch('src.services.workers.indexing_worker.should_process_document', return_value=True), \
             patch('src.services.workers.indexing_worker.load_chunks', return_value=[]), \
             patch('src.services.workers.indexing_worker.load_embeddings'), \
             patch('src.services.workers.indexing_worker.index_chunks'), \
             patch('src.services.workers.indexing_worker.update_document_status'):
            
            # The worker raises ValueError which gets wrapped in AzureServiceError
            with pytest.raises((ValueError, AzureServiceError)):
                indexing_worker(valid_message_dict, context_with_config)
    
    def test_load_no_embeddings(self, valid_message_dict, context_with_config, actual_chunks):
        """Test handling of no embeddings found"""
        with patch('src.services.workers.indexing_worker.should_process_document', return_value=True), \
             patch('src.services.workers.indexing_worker.load_chunks', return_value=actual_chunks), \
             patch('src.services.workers.indexing_worker.load_embeddings', return_value=[]), \
             patch('src.services.workers.indexing_worker.index_chunks'), \
             patch('src.services.workers.indexing_worker.update_document_status'):
            
            # The worker raises ValueError which gets wrapped in AzureServiceError
            with pytest.raises((ValueError, AzureServiceError)):
                indexing_worker(valid_message_dict, context_with_config)
    
    def test_load_count_mismatch(self, valid_message_dict, context_with_config, actual_chunks):
        """Test handling of chunks/embeddings count mismatch"""
        # Create a mismatch: return 2 chunks but only 1 embedding
        chunks_with_mismatch = actual_chunks + [actual_chunks[0]] if len(actual_chunks) > 0 else actual_chunks
        
        with patch('src.services.workers.indexing_worker.should_process_document', return_value=True), \
             patch('src.services.workers.indexing_worker.load_chunks', return_value=chunks_with_mismatch), \
             patch('src.services.workers.indexing_worker.load_embeddings', return_value=[[0.1] * 1536]), \
             patch('src.services.workers.indexing_worker.index_chunks'), \
             patch('src.services.workers.indexing_worker.update_document_status'):
            
            # The worker raises ValueError when count mismatch
            # The worker catches ValueError and re-raises as AzureServiceError in the outer handler
            with pytest.raises((ValueError, AzureServiceError)):
                indexing_worker(valid_message_dict, context_with_config)
    
    def test_load_failure(self, valid_message_dict, context_with_config):
        """Test handling of load failure"""
        with patch('src.services.workers.indexing_worker.should_process_document', return_value=True), \
             patch('src.services.workers.indexing_worker.load_chunks') as mock_load_chunks, \
             patch('src.services.workers.indexing_worker.update_document_status'):
            
            mock_load_chunks.side_effect = DatabaseError("Database connection failed")
            
            with pytest.raises(DatabaseError):
                indexing_worker(valid_message_dict, context_with_config)


class TestIndexingWorkerProcess:
    """Tests for indexing operations"""
    
    def test_indexing_success(self, valid_message_dict, context_with_config, mock_config, actual_chunks, actual_embeddings):
        """Test successful indexing"""
        with patch('src.services.workers.indexing_worker.should_process_document', return_value=True), \
             patch('src.services.workers.indexing_worker.load_chunks', return_value=actual_chunks), \
             patch('src.services.workers.indexing_worker.load_embeddings', return_value=actual_embeddings), \
             patch('src.services.workers.indexing_worker.index_chunks') as mock_index, \
             patch('src.services.workers.indexing_worker.update_document_status'):
            
            indexing_worker(valid_message_dict, context_with_config)
            
            # Verify indexing was called
            mock_index.assert_called_once_with(actual_chunks, actual_embeddings, mock_config)
    
    def test_indexing_retry(self, valid_message_dict, context_with_config, mock_config, actual_chunks, actual_embeddings):
        """Test retry logic on indexing failure"""
        with patch('src.services.workers.indexing_worker.should_process_document', return_value=True), \
             patch('src.services.workers.indexing_worker.load_chunks', return_value=actual_chunks), \
             patch('src.services.workers.indexing_worker.load_embeddings', return_value=actual_embeddings), \
             patch('src.services.workers.indexing_worker.index_chunks') as mock_index:
            
            # First two calls fail, third succeeds
            mock_index.side_effect = [
                AzureServiceError("Service unavailable"),
                AzureServiceError("Service unavailable"),
                None
            ]
            
            with patch('src.services.workers.indexing_worker.update_document_status'):
                indexing_worker(valid_message_dict, context_with_config)
                
                # Verify retry was attempted
                assert mock_index.call_count == 3
    
    def test_indexing_failure(self, valid_message_dict, context_with_config, mock_config, actual_chunks, actual_embeddings):
        """Test handling of indexing failure"""
        with patch('src.services.workers.indexing_worker.should_process_document', return_value=True), \
             patch('src.services.workers.indexing_worker.load_chunks', return_value=actual_chunks), \
             patch('src.services.workers.indexing_worker.load_embeddings', return_value=actual_embeddings), \
             patch('src.services.workers.indexing_worker.index_chunks') as mock_index, \
             patch('src.services.workers.indexing_worker.update_document_status'):
            
            mock_index.side_effect = AzureServiceError("Index not found")
            
            with pytest.raises(AzureServiceError):
                indexing_worker(valid_message_dict, context_with_config)


class TestIndexingWorkerPersistence:
    """Tests for status update operations"""
    
    def test_update_status_success(self, valid_message_dict, context_with_config, mock_config, actual_chunks, actual_embeddings):
        """Test successful status update"""
        with patch('src.services.workers.indexing_worker.should_process_document', return_value=True), \
             patch('src.services.workers.indexing_worker.load_chunks', return_value=actual_chunks), \
             patch('src.services.workers.indexing_worker.load_embeddings', return_value=actual_embeddings), \
             patch('src.services.workers.indexing_worker.index_chunks'), \
             patch('src.services.workers.indexing_worker.update_document_status') as mock_update:
            
            indexing_worker(valid_message_dict, context_with_config)
            
            # Verify status update was called
            mock_update.assert_called_once_with(
                valid_message_dict["document_id"],
                "indexed",
                timestamp_field="indexed_at",
                config=mock_config
            )
    
    def test_update_status_failure(self, valid_message_dict, context_with_config, mock_config, actual_chunks, actual_embeddings):
        """Test status update failure handling"""
        with patch('src.services.workers.indexing_worker.should_process_document', return_value=True), \
             patch('src.services.workers.indexing_worker.load_chunks', return_value=actual_chunks), \
             patch('src.services.workers.indexing_worker.load_embeddings', return_value=actual_embeddings), \
             patch('src.services.workers.indexing_worker.index_chunks'), \
             patch('src.services.workers.indexing_worker.update_document_status') as mock_update:
            
            mock_update.side_effect = DatabaseError("Database connection failed")
            
            with pytest.raises(DatabaseError):
                indexing_worker(valid_message_dict, context_with_config)

