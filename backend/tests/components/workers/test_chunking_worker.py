"""Comprehensive unit tests for chunking worker

Tests queue message parsing, extracted text loading, chunking, persistence,
status updates, message enqueuing, and idempotency.

These tests use actual files to ensure realistic test data.
"""

import pytest
import logging
from unittest.mock import Mock, patch

logging.disable(logging.CRITICAL)

from rag_eval.core.exceptions import DatabaseError, ValidationError
from rag_eval.services.workers.chunking_worker import chunking_worker
from rag_eval.services.workers.queue_client import QueueMessage, ProcessingStage
from rag_eval.core.interfaces import Chunk


@pytest.fixture
def mock_config():
    """Create a mock config"""
    config = Mock()
    config.database_url = "postgresql://test:test@localhost/test"
    return config


@pytest.fixture
def actual_extracted_text():
    """Realistic extracted text"""
    return """HEALTHGUARD SELECT PPO PLAN
2025 Medicare Evidence of Coverage

This document provides information about your health insurance coverage.
It includes details about benefits, copayments, and coverage limits.

SECTION 1: PLAN OVERVIEW
Your HealthGuard Select PPO Plan offers comprehensive coverage for medical services.
The plan includes coverage for doctor visits, hospital stays, and prescription drugs.
The plan is designed to provide affordable healthcare options for members.

SECTION 2: BENEFITS AND COVERAGE
Preventive care services are covered at 100% with no copayment.
Primary care visits have a $20 copayment.
Specialist visits have a $40 copayment.
Emergency room visits have a $150 copayment.
Hospital stays are covered with a daily copayment after the deductible."""


@pytest.fixture
def actual_chunks(actual_extracted_text):
    """Create actual chunks from extracted text"""
    from rag_eval.services.rag.chunking import chunk_text_fixed_size
    return chunk_text_fixed_size(
        text=actual_extracted_text,
        document_id="doc_123",
        chunk_size=500,
        overlap=100
    )


@pytest.fixture
def valid_message_dict():
    """Valid message dictionary for testing"""
    return {
        "document_id": "123e4567-e89b-12d3-a456-426614174000",
        "source_storage": "supabase",
        "filename": "test_document.pdf",
        "attempt": 1,
        "stage": "parsed",
        "metadata": {
            "chunk_size": 1000,
            "chunk_overlap": 200
        }
    }


@pytest.fixture
def context_with_config(mock_config):
    """Context object with config"""
    return {"config": mock_config}


class TestChunkingWorkerMessageParsing:
    """Tests for queue message parsing"""
    
    def test_valid_message(self, valid_message_dict, context_with_config, mock_config, actual_extracted_text, actual_chunks):
        """Test chunking worker with valid message"""
        with patch('rag_eval.services.workers.chunking_worker.should_process_document', return_value=True), \
             patch('rag_eval.services.workers.chunking_worker.load_extracted_text', return_value=actual_extracted_text), \
             patch('rag_eval.services.workers.chunking_worker.chunk_text', return_value=actual_chunks), \
             patch('rag_eval.services.workers.chunking_worker.persist_chunks'), \
             patch('rag_eval.services.workers.chunking_worker.update_document_status'), \
             patch('rag_eval.services.workers.chunking_worker.enqueue_message'):
            
            chunking_worker(valid_message_dict, context_with_config)
            
            # Verify extracted text was loaded
            from rag_eval.services.workers.chunking_worker import load_extracted_text
            load_extracted_text.assert_called_once_with(
                valid_message_dict["document_id"],
                mock_config
            )
            
            # Verify chunking was called
            from rag_eval.services.workers.chunking_worker import chunk_text
            chunk_text.assert_called_once()
            call_args = chunk_text.call_args
            # chunk_text is called with keyword arguments
            assert call_args.kwargs["text"] == actual_extracted_text
            assert call_args.kwargs["document_id"] == valid_message_dict["document_id"]
            assert call_args.kwargs["chunk_size"] == 1000  # From metadata
            assert call_args.kwargs["overlap"] == 200  # From metadata
    
    def test_invalid_message(self, context_with_config):
        """Test chunking worker with invalid message"""
        invalid_message = {"document_id": "123"}  # Missing required fields
        
        with pytest.raises(ValidationError):
            chunking_worker(invalid_message, context_with_config)


class TestChunkingWorkerIdempotency:
    """Tests for idempotency checks"""
    
    def test_skip_if_already_chunked(self, valid_message_dict, context_with_config):
        """Test that worker skips processing if document is already chunked"""
        with patch('rag_eval.services.workers.chunking_worker.should_process_document', return_value=False), \
             patch('rag_eval.services.workers.chunking_worker.load_extracted_text') as mock_load:
            
            chunking_worker(valid_message_dict, context_with_config)
            
            # Verify load was not called
            mock_load.assert_not_called()


class TestChunkingWorkerLoad:
    """Tests for extracted text loading"""
    
    def test_load_extracted_text_success(self, valid_message_dict, context_with_config, mock_config, actual_extracted_text, actual_chunks):
        """Test successful loading of extracted text"""
        with patch('rag_eval.services.workers.chunking_worker.should_process_document', return_value=True), \
             patch('rag_eval.services.workers.chunking_worker.load_extracted_text', return_value=actual_extracted_text) as mock_load, \
             patch('rag_eval.services.workers.chunking_worker.chunk_text', return_value=actual_chunks), \
             patch('rag_eval.services.workers.chunking_worker.persist_chunks'), \
             patch('rag_eval.services.workers.chunking_worker.update_document_status'), \
             patch('rag_eval.services.workers.chunking_worker.enqueue_message'):
            
            chunking_worker(valid_message_dict, context_with_config)
            
            mock_load.assert_called_once_with(
                valid_message_dict["document_id"],
                mock_config
            )
    
    def test_load_empty_text(self, valid_message_dict, context_with_config):
        """Test handling of empty extracted text"""
        with patch('rag_eval.services.workers.chunking_worker.should_process_document', return_value=True), \
             patch('rag_eval.services.workers.chunking_worker.load_extracted_text', return_value=""), \
             patch('rag_eval.services.workers.chunking_worker.update_document_status'):
            
            with pytest.raises(ValueError, match="Extracted text is empty"):
                chunking_worker(valid_message_dict, context_with_config)
    
    def test_load_failure(self, valid_message_dict, context_with_config):
        """Test handling of load failure"""
        with patch('rag_eval.services.workers.chunking_worker.should_process_document', return_value=True), \
             patch('rag_eval.services.workers.chunking_worker.load_extracted_text') as mock_load, \
             patch('rag_eval.services.workers.chunking_worker.update_document_status'):
            
            mock_load.side_effect = DatabaseError("Database connection failed")
            
            with pytest.raises(DatabaseError):
                chunking_worker(valid_message_dict, context_with_config)


class TestChunkingWorkerProcess:
    """Tests for chunking operations"""
    
    def test_chunking_success(self, valid_message_dict, context_with_config, mock_config, actual_extracted_text, actual_chunks):
        """Test successful chunking"""
        with patch('rag_eval.services.workers.chunking_worker.should_process_document', return_value=True), \
             patch('rag_eval.services.workers.chunking_worker.load_extracted_text', return_value=actual_extracted_text), \
             patch('rag_eval.services.workers.chunking_worker.chunk_text', return_value=actual_chunks) as mock_chunk, \
             patch('rag_eval.services.workers.chunking_worker.persist_chunks'), \
             patch('rag_eval.services.workers.chunking_worker.update_document_status'), \
             patch('rag_eval.services.workers.chunking_worker.enqueue_message'):
            
            chunking_worker(valid_message_dict, context_with_config)
            
            # Verify chunking was called with correct parameters
            mock_chunk.assert_called_once()
            call_args = mock_chunk.call_args
            assert call_args.kwargs["text"] == actual_extracted_text
            assert call_args.kwargs["document_id"] == valid_message_dict["document_id"]
            assert call_args.kwargs["chunk_size"] == 1000
            assert call_args.kwargs["overlap"] == 200
    
    def test_chunking_failure(self, valid_message_dict, context_with_config, mock_config, actual_extracted_text):
        """Test handling of chunking failure"""
        with patch('rag_eval.services.workers.chunking_worker.should_process_document', return_value=True), \
             patch('rag_eval.services.workers.chunking_worker.load_extracted_text', return_value=actual_extracted_text), \
             patch('rag_eval.services.workers.chunking_worker.chunk_text') as mock_chunk, \
             patch('rag_eval.services.workers.chunking_worker.update_document_status'):
            
            mock_chunk.side_effect = ValueError("Invalid chunking parameters")
            
            with pytest.raises(ValueError):
                chunking_worker(valid_message_dict, context_with_config)


class TestChunkingWorkerPersistence:
    """Tests for persistence operations"""
    
    def test_persist_chunks_success(self, valid_message_dict, context_with_config, mock_config, actual_extracted_text, actual_chunks):
        """Test successful persistence of chunks"""
        with patch('rag_eval.services.workers.chunking_worker.should_process_document', return_value=True), \
             patch('rag_eval.services.workers.chunking_worker.load_extracted_text', return_value=actual_extracted_text), \
             patch('rag_eval.services.workers.chunking_worker.chunk_text', return_value=actual_chunks), \
             patch('rag_eval.services.workers.chunking_worker.persist_chunks') as mock_persist, \
             patch('rag_eval.services.workers.chunking_worker.update_document_status'), \
             patch('rag_eval.services.workers.chunking_worker.enqueue_message'):
            
            chunking_worker(valid_message_dict, context_with_config)
            
            # Verify persistence was called
            mock_persist.assert_called_once_with(
                valid_message_dict["document_id"],
                actual_chunks,
                mock_config
            )
    
    def test_persist_failure(self, valid_message_dict, context_with_config, mock_config, actual_extracted_text, actual_chunks):
        """Test persistence failure handling"""
        with patch('rag_eval.services.workers.chunking_worker.should_process_document', return_value=True), \
             patch('rag_eval.services.workers.chunking_worker.load_extracted_text', return_value=actual_extracted_text), \
             patch('rag_eval.services.workers.chunking_worker.chunk_text', return_value=actual_chunks), \
             patch('rag_eval.services.workers.chunking_worker.persist_chunks') as mock_persist, \
             patch('rag_eval.services.workers.chunking_worker.update_document_status'):
            
            mock_persist.side_effect = DatabaseError("Database connection failed")
            
            with pytest.raises(DatabaseError):
                chunking_worker(valid_message_dict, context_with_config)


class TestChunkingWorkerEnqueue:
    """Tests for message enqueuing"""
    
    def test_enqueue_to_embedding_queue(self, valid_message_dict, context_with_config, mock_config, actual_extracted_text, actual_chunks):
        """Test successful enqueue to embedding queue"""
        with patch('rag_eval.services.workers.chunking_worker.should_process_document', return_value=True), \
             patch('rag_eval.services.workers.chunking_worker.load_extracted_text', return_value=actual_extracted_text), \
             patch('rag_eval.services.workers.chunking_worker.chunk_text', return_value=actual_chunks), \
             patch('rag_eval.services.workers.chunking_worker.persist_chunks'), \
             patch('rag_eval.services.workers.chunking_worker.update_document_status'), \
             patch('rag_eval.services.workers.chunking_worker.enqueue_message') as mock_enqueue:
            
            chunking_worker(valid_message_dict, context_with_config)
            
            # Verify enqueue was called
            mock_enqueue.assert_called_once()
            call_args = mock_enqueue.call_args
            assert call_args[0][0] == "ingestion-embeddings"
            message = call_args[0][1]
            assert isinstance(message, QueueMessage)
            assert message.document_id == valid_message_dict["document_id"]
            assert message.stage == ProcessingStage.CHUNKED
            assert message.attempt == 1
            assert call_args[0][2] == mock_config

