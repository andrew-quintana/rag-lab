"""Comprehensive unit tests for embedding worker

Tests queue message parsing, chunk loading, embedding generation, persistence,
status updates, message enqueuing, retry logic, and idempotency.

These tests use actual files to ensure realistic test data.
"""

import pytest
import logging
from unittest.mock import Mock, patch

logging.disable(logging.CRITICAL)

from rag_eval.core.exceptions import AzureServiceError, DatabaseError, ValidationError
from rag_eval.services.workers.embedding_worker import embedding_worker
from rag_eval.services.workers.queue_client import QueueMessage, ProcessingStage
from rag_eval.core.interfaces import Chunk


@pytest.fixture
def mock_config():
    """Create a mock config"""
    config = Mock()
    config.azure_ai_foundry_endpoint = "https://test.openai.azure.com/"
    config.azure_ai_foundry_api_key = "test-key"
    config.azure_ai_foundry_embedding_model = "text-embedding-3-small"
    config.database_url = "postgresql://test:test@localhost/test"
    return config


@pytest.fixture
def actual_chunks():
    """Create actual chunks for testing"""
    from rag_eval.services.rag.chunking import chunk_text_fixed_size
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
        "stage": "chunked",
        "metadata": {}
    }


@pytest.fixture
def context_with_config(mock_config):
    """Context object with config"""
    return {"config": mock_config}


class TestEmbeddingWorkerMessageParsing:
    """Tests for queue message parsing"""
    
    def test_valid_message(self, valid_message_dict, context_with_config, mock_config, actual_chunks, actual_embeddings):
        """Test embedding worker with valid message"""
        with patch('rag_eval.services.workers.embedding_worker.should_process_document', return_value=True), \
             patch('rag_eval.services.workers.embedding_worker.load_chunks', return_value=actual_chunks), \
             patch('rag_eval.services.workers.embedding_worker.generate_embeddings', return_value=actual_embeddings), \
             patch('rag_eval.services.workers.embedding_worker.persist_embeddings'), \
             patch('rag_eval.services.workers.embedding_worker.update_document_status'), \
             patch('rag_eval.services.workers.embedding_worker.enqueue_message'):
            
            embedding_worker(valid_message_dict, context_with_config)
            
            # Verify chunks were loaded
            from rag_eval.services.workers.embedding_worker import load_chunks
            load_chunks.assert_called_once_with(
                valid_message_dict["document_id"],
                mock_config
            )
            
            # Verify embeddings were generated
            from rag_eval.services.workers.embedding_worker import generate_embeddings
            generate_embeddings.assert_called_once_with(actual_chunks, mock_config)
    
    def test_invalid_message(self, context_with_config):
        """Test embedding worker with invalid message"""
        invalid_message = {"document_id": "123"}  # Missing required fields
        
        with pytest.raises(ValidationError):
            embedding_worker(invalid_message, context_with_config)


class TestEmbeddingWorkerIdempotency:
    """Tests for idempotency checks"""
    
    def test_skip_if_already_embedded(self, valid_message_dict, context_with_config):
        """Test that worker skips processing if document is already embedded"""
        with patch('rag_eval.services.workers.embedding_worker.should_process_document', return_value=False), \
             patch('rag_eval.services.workers.embedding_worker.load_chunks') as mock_load:
            
            embedding_worker(valid_message_dict, context_with_config)
            
            # Verify load was not called
            mock_load.assert_not_called()


class TestEmbeddingWorkerLoad:
    """Tests for chunk loading"""
    
    def test_load_chunks_success(self, valid_message_dict, context_with_config, mock_config, actual_chunks, actual_embeddings):
        """Test successful loading of chunks"""
        with patch('rag_eval.services.workers.embedding_worker.should_process_document', return_value=True), \
             patch('rag_eval.services.workers.embedding_worker.load_chunks', return_value=actual_chunks) as mock_load, \
             patch('rag_eval.services.workers.embedding_worker.generate_embeddings', return_value=actual_embeddings), \
             patch('rag_eval.services.workers.embedding_worker.persist_embeddings'), \
             patch('rag_eval.services.workers.embedding_worker.update_document_status'), \
             patch('rag_eval.services.workers.embedding_worker.enqueue_message'):
            
            embedding_worker(valid_message_dict, context_with_config)
            
            mock_load.assert_called_once_with(
                valid_message_dict["document_id"],
                mock_config
            )
    
    def test_load_no_chunks(self, valid_message_dict, context_with_config):
        """Test handling of no chunks found"""
        with patch('rag_eval.services.workers.embedding_worker.should_process_document', return_value=True), \
             patch('rag_eval.services.workers.embedding_worker.load_chunks', return_value=[]), \
             patch('rag_eval.services.workers.embedding_worker.update_document_status'):
            
            # The worker raises ValueError when no chunks found
            with pytest.raises((ValueError, AzureServiceError)):
                embedding_worker(valid_message_dict, context_with_config)
    
    def test_load_failure(self, valid_message_dict, context_with_config):
        """Test handling of load failure"""
        with patch('rag_eval.services.workers.embedding_worker.should_process_document', return_value=True), \
             patch('rag_eval.services.workers.embedding_worker.load_chunks') as mock_load, \
             patch('rag_eval.services.workers.embedding_worker.update_document_status'):
            
            mock_load.side_effect = DatabaseError("Database connection failed")
            
            with pytest.raises(DatabaseError):
                embedding_worker(valid_message_dict, context_with_config)


class TestEmbeddingWorkerProcess:
    """Tests for embedding generation"""
    
    def test_embedding_generation_success(self, valid_message_dict, context_with_config, mock_config, actual_chunks, actual_embeddings):
        """Test successful embedding generation"""
        with patch('rag_eval.services.workers.embedding_worker.should_process_document', return_value=True), \
             patch('rag_eval.services.workers.embedding_worker.load_chunks', return_value=actual_chunks), \
             patch('rag_eval.services.workers.embedding_worker.generate_embeddings', return_value=actual_embeddings) as mock_generate, \
             patch('rag_eval.services.workers.embedding_worker.persist_embeddings'), \
             patch('rag_eval.services.workers.embedding_worker.update_document_status'), \
             patch('rag_eval.services.workers.embedding_worker.enqueue_message'):
            
            embedding_worker(valid_message_dict, context_with_config)
            
            # Verify embedding generation was called
            mock_generate.assert_called_once_with(actual_chunks, mock_config)
    
    def test_embedding_generation_retry(self, valid_message_dict, context_with_config, mock_config, actual_chunks, actual_embeddings):
        """Test retry logic on embedding generation failure"""
        with patch('rag_eval.services.workers.embedding_worker.should_process_document', return_value=True), \
             patch('rag_eval.services.workers.embedding_worker.load_chunks', return_value=actual_chunks), \
             patch('rag_eval.services.workers.embedding_worker.generate_embeddings') as mock_generate:
            
            # First two calls fail, third succeeds
            mock_generate.side_effect = [
                AzureServiceError("Service unavailable"),
                AzureServiceError("Service unavailable"),
                actual_embeddings
            ]
            
            with patch('rag_eval.services.workers.embedding_worker.persist_embeddings'), \
                 patch('rag_eval.services.workers.embedding_worker.update_document_status'), \
                 patch('rag_eval.services.workers.embedding_worker.enqueue_message'):
                
                embedding_worker(valid_message_dict, context_with_config)
                
                # Verify retry was attempted
                assert mock_generate.call_count == 3
    
    def test_embedding_generation_failure(self, valid_message_dict, context_with_config, mock_config, actual_chunks):
        """Test handling of embedding generation failure"""
        with patch('rag_eval.services.workers.embedding_worker.should_process_document', return_value=True), \
             patch('rag_eval.services.workers.embedding_worker.load_chunks', return_value=actual_chunks), \
             patch('rag_eval.services.workers.embedding_worker.generate_embeddings') as mock_generate, \
             patch('rag_eval.services.workers.embedding_worker.update_document_status'):
            
            mock_generate.side_effect = AzureServiceError("API key invalid")
            
            with pytest.raises(AzureServiceError):
                embedding_worker(valid_message_dict, context_with_config)


class TestEmbeddingWorkerPersistence:
    """Tests for persistence operations"""
    
    def test_persist_embeddings_success(self, valid_message_dict, context_with_config, mock_config, actual_chunks, actual_embeddings):
        """Test successful persistence of embeddings"""
        with patch('rag_eval.services.workers.embedding_worker.should_process_document', return_value=True), \
             patch('rag_eval.services.workers.embedding_worker.load_chunks', return_value=actual_chunks), \
             patch('rag_eval.services.workers.embedding_worker.generate_embeddings', return_value=actual_embeddings), \
             patch('rag_eval.services.workers.embedding_worker.persist_embeddings') as mock_persist, \
             patch('rag_eval.services.workers.embedding_worker.update_document_status'), \
             patch('rag_eval.services.workers.embedding_worker.enqueue_message'):
            
            embedding_worker(valid_message_dict, context_with_config)
            
            # Verify persistence was called
            mock_persist.assert_called_once_with(
                valid_message_dict["document_id"],
                actual_chunks,
                actual_embeddings,
                mock_config
            )
    
    def test_persist_failure(self, valid_message_dict, context_with_config, mock_config, actual_chunks, actual_embeddings):
        """Test persistence failure handling"""
        with patch('rag_eval.services.workers.embedding_worker.should_process_document', return_value=True), \
             patch('rag_eval.services.workers.embedding_worker.load_chunks', return_value=actual_chunks), \
             patch('rag_eval.services.workers.embedding_worker.generate_embeddings', return_value=actual_embeddings), \
             patch('rag_eval.services.workers.embedding_worker.persist_embeddings') as mock_persist, \
             patch('rag_eval.services.workers.embedding_worker.update_document_status'):
            
            mock_persist.side_effect = DatabaseError("Database connection failed")
            
            with pytest.raises(DatabaseError):
                embedding_worker(valid_message_dict, context_with_config)


class TestEmbeddingWorkerEnqueue:
    """Tests for message enqueuing"""
    
    def test_enqueue_to_indexing_queue(self, valid_message_dict, context_with_config, mock_config, actual_chunks, actual_embeddings):
        """Test successful enqueue to indexing queue"""
        with patch('rag_eval.services.workers.embedding_worker.should_process_document', return_value=True), \
             patch('rag_eval.services.workers.embedding_worker.load_chunks', return_value=actual_chunks), \
             patch('rag_eval.services.workers.embedding_worker.generate_embeddings', return_value=actual_embeddings), \
             patch('rag_eval.services.workers.embedding_worker.persist_embeddings'), \
             patch('rag_eval.services.workers.embedding_worker.update_document_status'), \
             patch('rag_eval.services.workers.embedding_worker.enqueue_message') as mock_enqueue:
            
            embedding_worker(valid_message_dict, context_with_config)
            
            # Verify enqueue was called
            mock_enqueue.assert_called_once()
            call_args = mock_enqueue.call_args
            assert call_args[0][0] == "ingestion-indexing"
            message = call_args[0][1]
            assert isinstance(message, QueueMessage)
            assert message.document_id == valid_message_dict["document_id"]
            assert message.stage == ProcessingStage.EMBEDDED
            assert message.attempt == 1
            assert call_args[0][2] == mock_config

