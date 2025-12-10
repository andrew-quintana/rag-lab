"""Comprehensive unit tests for ingestion worker

Tests queue message parsing, file download, batch processing, text extraction, persistence,
status updates, message enqueuing, retry logic, dead-letter handling, and idempotency.

These tests use actual files to ensure realistic test data.
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

logging.disable(logging.CRITICAL)

from src.core.exceptions import (
    AzureServiceError,
    DatabaseError,
    ValidationError,
)
from src.services.workers.ingestion_worker import ingestion_worker
from src.services.workers.queue_client import QueueMessage, SourceStorage, ProcessingStage


@pytest.fixture
def mock_config():
    """Create a mock config with all required credentials"""
    config = Mock()
    config.azure_blob_connection_string = "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=testkey;EndpointSuffix=core.windows.net"
    config.azure_blob_container_name = "test-container"
    config.supabase_url = "https://test.supabase.co"
    config.supabase_key = "test-key"
    config.azure_document_intelligence_endpoint = "https://test.cognitiveservices.azure.com/"
    config.azure_document_intelligence_api_key = "test-key"
    config.database_url = "postgresql://test:test@localhost/test"
    return config


@pytest.fixture
def sample_pdf_path():
    """Path to actual sample PDF file for testing"""
    pdf_path = Path(__file__).parent.parent.parent / "fixtures" / "sample_documents" / "healthguard_select_ppo_plan.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Sample PDF not found at {pdf_path}")
    return pdf_path


@pytest.fixture
def sample_pdf_content(sample_pdf_path):
    """Actual PDF file content for testing"""
    with open(sample_pdf_path, "rb") as f:
        return f.read()


@pytest.fixture
def actual_extracted_text():
    """Realistic extracted text that represents actual PDF content"""
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
def valid_message_dict():
    """Valid message dictionary for testing"""
    return {
        "document_id": "123e4567-e89b-12d3-a456-426614174000",
        "source_storage": "supabase",
        "filename": "test_document.pdf",
        "attempt": 1,
        "stage": "uploaded",
        "metadata": {
            "tenant_id": "tenant_abc",
            "user_id": "user_123",
            "mime_type": "application/pdf"
        }
    }


@pytest.fixture
def context_with_config(mock_config):
    """Context object with config for testing"""
    return {"config": mock_config}


class TestIngestionWorkerMessageParsing:
    """Tests for queue message parsing and validation"""
    
    def test_valid_message_supabase(self, valid_message_dict, context_with_config, mock_config, sample_pdf_content, actual_extracted_text):
        """Test ingestion worker with valid Supabase message (batch processing)"""
        # Mock batch processing functions
        with patch('src.services.workers.ingestion_worker.should_process_document', return_value=True), \
             patch('src.services.workers.ingestion_worker.download_document_from_storage', return_value=sample_pdf_content), \
             patch('src.services.workers.ingestion_worker.get_pdf_page_count', return_value=2), \
             patch('src.services.workers.ingestion_worker.generate_page_batches', return_value=[(1, 3)]), \
             patch('src.services.workers.ingestion_worker.get_ingestion_metadata', return_value={
                 "num_pages": 0,
                 "num_batches_total": 0,
                 "last_successful_page": 0,
                 "next_unparsed_batch_index": 0,
                 "parsing_status": "pending",
                 "batch_size": 2,
                 "batches_completed": {},
                 "parsing_started_at": None,
                 "parsing_completed_at": None,
                 "errors": []
             }), \
             patch('src.services.workers.ingestion_worker.get_completed_batches', return_value=set()), \
             patch('src.services.workers.ingestion_worker.slice_pdf_to_batch', return_value=sample_pdf_content), \
             patch('src.services.workers.ingestion_worker.extract_text_from_batch', return_value=actual_extracted_text), \
             patch('src.services.workers.ingestion_worker.persist_batch_result'), \
             patch('src.services.workers.ingestion_worker.delete_batch_chunk'), \
             patch('src.services.workers.ingestion_worker.update_ingestion_metadata'), \
             patch('src.services.workers.ingestion_worker.persist_extracted_text'), \
             patch('src.services.workers.ingestion_worker.update_document_status'), \
             patch('src.services.workers.ingestion_worker.enqueue_message'):
            
            ingestion_worker(valid_message_dict, context_with_config)
            
            # Verify file download was called
            from src.services.workers.ingestion_worker import download_document_from_storage
            download_document_from_storage.assert_called_once_with(
                valid_message_dict["document_id"],
                mock_config
            )
            
            # Verify page count was detected
            from src.services.workers.ingestion_worker import get_pdf_page_count
            get_pdf_page_count.assert_called_once_with(sample_pdf_content)
            
            # Verify batch extraction was called
            from src.services.workers.ingestion_worker import extract_text_from_batch
            extract_text_from_batch.assert_called_once()
            
            # Verify persistence was called
            from src.services.workers.ingestion_worker import persist_extracted_text
            persist_extracted_text.assert_called_once_with(
                valid_message_dict["document_id"],
                actual_extracted_text,
                mock_config
            )
            
            # Verify status update was called
            from src.services.workers.ingestion_worker import update_document_status
            update_document_status.assert_called_once_with(
                valid_message_dict["document_id"],
                "parsed",
                timestamp_field="parsed_at",
                config=mock_config
            )
            
            # Verify enqueue was called
            from src.services.workers.ingestion_worker import enqueue_message
            enqueue_message.assert_called_once()
            call_args = enqueue_message.call_args
            assert call_args[0][0] == "ingestion-chunking"
            assert call_args[0][1].document_id == valid_message_dict["document_id"]
            assert call_args[0][1].stage == ProcessingStage.PARSED
    
    def test_valid_message_azure_blob(self, mock_config, sample_pdf_content, actual_extracted_text):
        """Test ingestion worker with valid Azure Blob message (batch processing)"""
        message_dict = {
            "document_id": "123e4567-e89b-12d3-a456-426614174000",
            "source_storage": "azure_blob",
            "filename": "test_document.pdf",
            "attempt": 1,
            "stage": "uploaded",
        }
        context = {"config": mock_config}
        
        with patch('src.services.workers.ingestion_worker.should_process_document', return_value=True), \
             patch('src.services.workers.ingestion_worker.download_document_from_blob', return_value=sample_pdf_content), \
             patch('src.services.workers.ingestion_worker.get_pdf_page_count', return_value=2), \
             patch('src.services.workers.ingestion_worker.generate_page_batches', return_value=[(1, 3)]), \
             patch('src.services.workers.ingestion_worker.get_ingestion_metadata', return_value={
                 "num_pages": 0,
                 "num_batches_total": 0,
                 "last_successful_page": 0,
                 "next_unparsed_batch_index": 0,
                 "parsing_status": "pending",
                 "batch_size": 2,
                 "batches_completed": {},
                 "parsing_started_at": None,
                 "parsing_completed_at": None,
                 "errors": []
             }), \
             patch('src.services.workers.ingestion_worker.get_completed_batches', return_value=set()), \
             patch('src.services.workers.ingestion_worker.slice_pdf_to_batch', return_value=sample_pdf_content), \
             patch('src.services.workers.ingestion_worker.extract_text_from_batch', return_value=actual_extracted_text), \
             patch('src.services.workers.ingestion_worker.persist_batch_result'), \
             patch('src.services.workers.ingestion_worker.delete_batch_chunk'), \
             patch('src.services.workers.ingestion_worker.update_ingestion_metadata'), \
             patch('src.services.workers.ingestion_worker.persist_extracted_text'), \
             patch('src.services.workers.ingestion_worker.update_document_status'), \
             patch('src.services.workers.ingestion_worker.enqueue_message'):
            
            ingestion_worker(message_dict, context)
            
            # Verify Azure Blob download was called
            from src.services.workers.ingestion_worker import download_document_from_blob
            download_document_from_blob.assert_called_once_with(
                message_dict["document_id"],
                mock_config
            )
    
    def test_invalid_message_missing_fields(self, context_with_config):
        """Test ingestion worker with invalid message (missing fields)"""
        invalid_message = {
            "document_id": "123e4567-e89b-12d3-a456-426614174000",
            # Missing required fields
        }
        
        with pytest.raises(ValidationError):
            ingestion_worker(invalid_message, context_with_config)
    
    def test_invalid_message_invalid_storage(self, context_with_config):
        """Test ingestion worker with invalid source_storage"""
        invalid_message = {
            "document_id": "123e4567-e89b-12d3-a456-426614174000",
            "source_storage": "invalid_storage",
            "filename": "test.pdf",
            "attempt": 1,
            "stage": "uploaded",
        }
        
        with pytest.raises(ValidationError):
            ingestion_worker(invalid_message, context_with_config)


class TestIngestionWorkerIdempotency:
    """Tests for idempotency checks"""
    
    def test_skip_if_already_parsed(self, valid_message_dict, context_with_config):
        """Test that worker skips processing if document is already parsed"""
        with patch('src.services.workers.ingestion_worker.should_process_document', return_value=False), \
             patch('src.services.workers.ingestion_worker.download_document_from_storage') as mock_download:
            
            ingestion_worker(valid_message_dict, context_with_config)
            
            # Verify download was not called (idempotency check)
            mock_download.assert_not_called()


class TestIngestionWorkerFileDownload:
    """Tests for file download operations"""
    
    def test_download_from_supabase_success(self, valid_message_dict, context_with_config, mock_config, sample_pdf_content, actual_extracted_text):
        """Test successful file download from Supabase (batch processing)"""
        with patch('src.services.workers.ingestion_worker.should_process_document', return_value=True), \
             patch('src.services.workers.ingestion_worker.download_document_from_storage', return_value=sample_pdf_content), \
             patch('src.services.workers.ingestion_worker.get_pdf_page_count', return_value=2), \
             patch('src.services.workers.ingestion_worker.generate_page_batches', return_value=[(1, 3)]), \
             patch('src.services.workers.ingestion_worker.get_ingestion_metadata', return_value={
                 "num_pages": 0,
                 "num_batches_total": 0,
                 "last_successful_page": 0,
                 "next_unparsed_batch_index": 0,
                 "parsing_status": "pending",
                 "batch_size": 2,
                 "batches_completed": {},
                 "parsing_started_at": None,
                 "parsing_completed_at": None,
                 "errors": []
             }), \
             patch('src.services.workers.ingestion_worker.get_completed_batches', return_value=set()), \
             patch('src.services.workers.ingestion_worker.slice_pdf_to_batch', return_value=sample_pdf_content), \
             patch('src.services.workers.ingestion_worker.extract_text_from_batch', return_value=actual_extracted_text), \
             patch('src.services.workers.ingestion_worker.persist_batch_result'), \
             patch('src.services.workers.ingestion_worker.delete_batch_chunk'), \
             patch('src.services.workers.ingestion_worker.update_ingestion_metadata'), \
             patch('src.services.workers.ingestion_worker.persist_extracted_text'), \
             patch('src.services.workers.ingestion_worker.update_document_status'), \
             patch('src.services.workers.ingestion_worker.enqueue_message'):
            
            ingestion_worker(valid_message_dict, context_with_config)
            
            # Verify download was called with correct parameters
            from src.services.workers.ingestion_worker import download_document_from_storage
            download_document_from_storage.assert_called_once_with(
                valid_message_dict["document_id"],
                mock_config
            )
    
    def test_download_failure_retry(self, valid_message_dict, context_with_config, mock_config, sample_pdf_content, actual_extracted_text):
        """Test retry logic on download failure (batch processing)"""
        with patch('src.services.workers.ingestion_worker.should_process_document', return_value=True), \
             patch('src.services.workers.ingestion_worker.download_document_from_storage') as mock_download:
            
            # First two calls fail, third succeeds
            mock_download.side_effect = [
                AzureServiceError("Network error"),
                AzureServiceError("Network error"),
                sample_pdf_content
            ]
            
            with patch('src.services.workers.ingestion_worker.get_pdf_page_count', return_value=2), \
                 patch('src.services.workers.ingestion_worker.generate_page_batches', return_value=[(1, 3)]), \
                 patch('src.services.workers.ingestion_worker.get_ingestion_metadata', return_value={
                     "num_pages": 0,
                     "num_batches_total": 0,
                     "last_successful_page": 0,
                     "next_unparsed_batch_index": 0,
                     "parsing_status": "pending",
                     "batch_size": 2,
                     "batches_completed": {},
                     "parsing_started_at": None,
                     "parsing_completed_at": None,
                     "errors": []
                 }), \
                 patch('src.services.workers.ingestion_worker.get_completed_batches', return_value=set()), \
                 patch('src.services.workers.ingestion_worker.slice_pdf_to_batch', return_value=sample_pdf_content), \
                 patch('src.services.workers.ingestion_worker.extract_text_from_batch', return_value=actual_extracted_text), \
                 patch('src.services.workers.ingestion_worker.persist_batch_result'), \
                 patch('src.services.workers.ingestion_worker.delete_batch_chunk'), \
                 patch('src.services.workers.ingestion_worker.update_ingestion_metadata'), \
                 patch('src.services.workers.ingestion_worker.persist_extracted_text'), \
                 patch('src.services.workers.ingestion_worker.update_document_status'), \
                 patch('src.services.workers.ingestion_worker.enqueue_message'):
                
                ingestion_worker(valid_message_dict, context_with_config)
                
                # Verify retry was attempted
                assert mock_download.call_count == 3
    
    def test_download_failure_max_retries(self, valid_message_dict, context_with_config):
        """Test dead-letter handling after max retries"""
        with patch('src.services.workers.ingestion_worker.should_process_document', return_value=True), \
             patch('src.services.workers.ingestion_worker.download_document_from_storage') as mock_download, \
             patch('src.services.workers.ingestion_worker.send_to_dead_letter') as mock_dead_letter, \
             patch('src.services.workers.ingestion_worker.update_document_status'):
            
            # All retries fail
            mock_download.side_effect = AzureServiceError("File not found")
            
            # Set attempt to max retries
            valid_message_dict["attempt"] = 3
            
            with pytest.raises(AzureServiceError):
                ingestion_worker(valid_message_dict, context_with_config)
            
            # Verify dead-letter was called
            mock_dead_letter.assert_called_once()


class TestIngestionWorkerTextExtraction:
    """Tests for text extraction operations (batch processing)"""
    
    def test_extraction_success(self, valid_message_dict, context_with_config, mock_config, sample_pdf_content, actual_extracted_text):
        """Test successful batch text extraction"""
        with patch('src.services.workers.ingestion_worker.should_process_document', return_value=True), \
             patch('src.services.workers.ingestion_worker.download_document_from_storage', return_value=sample_pdf_content), \
             patch('src.services.workers.ingestion_worker.get_pdf_page_count', return_value=2), \
             patch('src.services.workers.ingestion_worker.generate_page_batches', return_value=[(1, 3)]), \
             patch('src.services.workers.ingestion_worker.get_ingestion_metadata', return_value={
                 "num_pages": 0,
                 "num_batches_total": 0,
                 "last_successful_page": 0,
                 "next_unparsed_batch_index": 0,
                 "parsing_status": "pending",
                 "batch_size": 2,
                 "batches_completed": {},
                 "parsing_started_at": None,
                 "parsing_completed_at": None,
                 "errors": []
             }), \
             patch('src.services.workers.ingestion_worker.get_completed_batches', return_value=set()), \
             patch('src.services.workers.ingestion_worker.slice_pdf_to_batch', return_value=sample_pdf_content), \
             patch('src.services.workers.ingestion_worker.extract_text_from_batch', return_value=actual_extracted_text) as mock_extract, \
             patch('src.services.workers.ingestion_worker.persist_batch_result'), \
             patch('src.services.workers.ingestion_worker.delete_batch_chunk'), \
             patch('src.services.workers.ingestion_worker.update_ingestion_metadata'), \
             patch('src.services.workers.ingestion_worker.persist_extracted_text'), \
             patch('src.services.workers.ingestion_worker.update_document_status'), \
             patch('src.services.workers.ingestion_worker.enqueue_message'):
            
            ingestion_worker(valid_message_dict, context_with_config)
            
            # Verify batch extraction was called with correct parameters
            mock_extract.assert_called_once()
            call_args = mock_extract.call_args
            assert call_args[0][0] == sample_pdf_content  # Sliced PDF
            assert call_args[0][1] == "1-3"  # Page range
            assert call_args[0][2] == mock_config
    
    def test_extraction_failure_retry(self, valid_message_dict, context_with_config, mock_config, sample_pdf_content, actual_extracted_text):
        """Test retry logic on batch extraction failure"""
        with patch('src.services.workers.ingestion_worker.should_process_document', return_value=True), \
             patch('src.services.workers.ingestion_worker.download_document_from_storage', return_value=sample_pdf_content), \
             patch('src.services.workers.ingestion_worker.get_pdf_page_count', return_value=2), \
             patch('src.services.workers.ingestion_worker.generate_page_batches', return_value=[(1, 3)]), \
             patch('src.services.workers.ingestion_worker.get_ingestion_metadata', return_value={
                 "num_pages": 0,
                 "num_batches_total": 0,
                 "last_successful_page": 0,
                 "next_unparsed_batch_index": 0,
                 "parsing_status": "pending",
                 "batch_size": 2,
                 "batches_completed": {},
                 "parsing_started_at": None,
                 "parsing_completed_at": None,
                 "errors": []
             }), \
             patch('src.services.workers.ingestion_worker.get_completed_batches', return_value=set()), \
             patch('src.services.workers.ingestion_worker.slice_pdf_to_batch', return_value=sample_pdf_content), \
             patch('src.services.workers.ingestion_worker.extract_text_from_batch') as mock_extract:
            
            # First two calls fail, third succeeds
            mock_extract.side_effect = [
                AzureServiceError("Service unavailable"),
                AzureServiceError("Service unavailable"),
                actual_extracted_text
            ]
            
            with patch('src.services.workers.ingestion_worker.persist_batch_result'), \
                 patch('src.services.workers.ingestion_worker.delete_batch_chunk'), \
                 patch('src.services.workers.ingestion_worker.update_ingestion_metadata'), \
                 patch('src.services.workers.ingestion_worker.persist_extracted_text'), \
                 patch('src.services.workers.ingestion_worker.update_document_status'), \
                 patch('src.services.workers.ingestion_worker.enqueue_message'):
                
                ingestion_worker(valid_message_dict, context_with_config)
                
                # Verify retry was attempted
                assert mock_extract.call_count == 3


class TestIngestionWorkerPersistence:
    """Tests for persistence operations"""
    
    def test_persist_extracted_text_success(self, valid_message_dict, context_with_config, mock_config, sample_pdf_content, actual_extracted_text):
        """Test successful persistence of extracted text (batch processing)"""
        with patch('src.services.workers.ingestion_worker.should_process_document', return_value=True), \
             patch('src.services.workers.ingestion_worker.download_document_from_storage', return_value=sample_pdf_content), \
             patch('src.services.workers.ingestion_worker.get_pdf_page_count', return_value=2), \
             patch('src.services.workers.ingestion_worker.generate_page_batches', return_value=[(1, 3)]), \
             patch('src.services.workers.ingestion_worker.get_ingestion_metadata', return_value={
                 "num_pages": 0,
                 "num_batches_total": 0,
                 "last_successful_page": 0,
                 "next_unparsed_batch_index": 0,
                 "parsing_status": "pending",
                 "batch_size": 2,
                 "batches_completed": {},
                 "parsing_started_at": None,
                 "parsing_completed_at": None,
                 "errors": []
             }), \
             patch('src.services.workers.ingestion_worker.get_completed_batches', return_value=set()), \
             patch('src.services.workers.ingestion_worker.slice_pdf_to_batch', return_value=sample_pdf_content), \
             patch('src.services.workers.ingestion_worker.extract_text_from_batch', return_value=actual_extracted_text), \
             patch('src.services.workers.ingestion_worker.persist_batch_result'), \
             patch('src.services.workers.ingestion_worker.delete_batch_chunk'), \
             patch('src.services.workers.ingestion_worker.update_ingestion_metadata'), \
             patch('src.services.workers.ingestion_worker.persist_extracted_text') as mock_persist, \
             patch('src.services.workers.ingestion_worker.update_document_status'), \
             patch('src.services.workers.ingestion_worker.enqueue_message'):
            
            ingestion_worker(valid_message_dict, context_with_config)
            
            # Verify persistence was called with correct parameters (merged text)
            mock_persist.assert_called_once_with(
                valid_message_dict["document_id"],
                actual_extracted_text,
                mock_config
            )
    
    def test_persist_failure(self, valid_message_dict, context_with_config, mock_config, sample_pdf_content, actual_extracted_text):
        """Test persistence failure handling (batch processing)"""
        with patch('src.services.workers.ingestion_worker.should_process_document', return_value=True), \
             patch('src.services.workers.ingestion_worker.download_document_from_storage', return_value=sample_pdf_content), \
             patch('src.services.workers.ingestion_worker.get_pdf_page_count', return_value=2), \
             patch('src.services.workers.ingestion_worker.generate_page_batches', return_value=[(1, 3)]), \
             patch('src.services.workers.ingestion_worker.get_ingestion_metadata', return_value={
                 "num_pages": 0,
                 "num_batches_total": 0,
                 "last_successful_page": 0,
                 "next_unparsed_batch_index": 0,
                 "parsing_status": "pending",
                 "batch_size": 2,
                 "batches_completed": {},
                 "parsing_started_at": None,
                 "parsing_completed_at": None,
                 "errors": []
             }), \
             patch('src.services.workers.ingestion_worker.get_completed_batches', return_value=set()), \
             patch('src.services.workers.ingestion_worker.slice_pdf_to_batch', return_value=sample_pdf_content), \
             patch('src.services.workers.ingestion_worker.extract_text_from_batch', return_value=actual_extracted_text), \
             patch('src.services.workers.ingestion_worker.persist_batch_result'), \
             patch('src.services.workers.ingestion_worker.delete_batch_chunk'), \
             patch('src.services.workers.ingestion_worker.update_ingestion_metadata'), \
             patch('src.services.workers.ingestion_worker.persist_extracted_text') as mock_persist:
            
            mock_persist.side_effect = DatabaseError("Database connection failed")
            
            with pytest.raises(DatabaseError):
                ingestion_worker(valid_message_dict, context_with_config)


class TestIngestionWorkerEnqueue:
    """Tests for message enqueuing"""
    
    def test_enqueue_to_chunking_queue(self, valid_message_dict, context_with_config, mock_config, sample_pdf_content, actual_extracted_text):
        """Test successful enqueue to chunking queue (batch processing)"""
        with patch('src.services.workers.ingestion_worker.should_process_document', return_value=True), \
             patch('src.services.workers.ingestion_worker.download_document_from_storage', return_value=sample_pdf_content), \
             patch('src.services.workers.ingestion_worker.get_pdf_page_count', return_value=2), \
             patch('src.services.workers.ingestion_worker.generate_page_batches', return_value=[(1, 3)]), \
             patch('src.services.workers.ingestion_worker.get_ingestion_metadata', return_value={
                 "num_pages": 0,
                 "num_batches_total": 0,
                 "last_successful_page": 0,
                 "next_unparsed_batch_index": 0,
                 "parsing_status": "pending",
                 "batch_size": 2,
                 "batches_completed": {},
                 "parsing_started_at": None,
                 "parsing_completed_at": None,
                 "errors": []
             }), \
             patch('src.services.workers.ingestion_worker.get_completed_batches', return_value=set()), \
             patch('src.services.workers.ingestion_worker.slice_pdf_to_batch', return_value=sample_pdf_content), \
             patch('src.services.workers.ingestion_worker.extract_text_from_batch', return_value=actual_extracted_text), \
             patch('src.services.workers.ingestion_worker.persist_batch_result'), \
             patch('src.services.workers.ingestion_worker.delete_batch_chunk'), \
             patch('src.services.workers.ingestion_worker.update_ingestion_metadata'), \
             patch('src.services.workers.ingestion_worker.persist_extracted_text'), \
             patch('src.services.workers.ingestion_worker.update_document_status'), \
             patch('src.services.workers.ingestion_worker.enqueue_message') as mock_enqueue:
            
            ingestion_worker(valid_message_dict, context_with_config)
            
            # Verify enqueue was called with correct parameters
            mock_enqueue.assert_called_once()
            call_args = mock_enqueue.call_args
            assert call_args[0][0] == "ingestion-chunking"
            message = call_args[0][1]
            assert isinstance(message, QueueMessage)
            assert message.document_id == valid_message_dict["document_id"]
            assert message.stage == ProcessingStage.PARSED
            assert message.attempt == 1  # Reset attempt counter
            assert call_args[0][2] == mock_config
    
    def test_enqueue_failure(self, valid_message_dict, context_with_config, mock_config, sample_pdf_content, actual_extracted_text):
        """Test enqueue failure handling (batch processing)"""
        with patch('src.services.workers.ingestion_worker.should_process_document', return_value=True), \
             patch('src.services.workers.ingestion_worker.download_document_from_storage', return_value=sample_pdf_content), \
             patch('src.services.workers.ingestion_worker.get_pdf_page_count', return_value=2), \
             patch('src.services.workers.ingestion_worker.generate_page_batches', return_value=[(1, 3)]), \
             patch('src.services.workers.ingestion_worker.get_ingestion_metadata', return_value={
                 "num_pages": 0,
                 "num_batches_total": 0,
                 "last_successful_page": 0,
                 "next_unparsed_batch_index": 0,
                 "parsing_status": "pending",
                 "batch_size": 2,
                 "batches_completed": {},
                 "parsing_started_at": None,
                 "parsing_completed_at": None,
                 "errors": []
             }), \
             patch('src.services.workers.ingestion_worker.get_completed_batches', return_value=set()), \
             patch('src.services.workers.ingestion_worker.slice_pdf_to_batch', return_value=sample_pdf_content), \
             patch('src.services.workers.ingestion_worker.extract_text_from_batch', return_value=actual_extracted_text), \
             patch('src.services.workers.ingestion_worker.persist_batch_result'), \
             patch('src.services.workers.ingestion_worker.delete_batch_chunk'), \
             patch('src.services.workers.ingestion_worker.update_ingestion_metadata'), \
             patch('src.services.workers.ingestion_worker.persist_extracted_text'), \
             patch('src.services.workers.ingestion_worker.update_document_status'), \
             patch('src.services.workers.ingestion_worker.enqueue_message') as mock_enqueue:
            
            mock_enqueue.side_effect = AzureServiceError("Queue connection failed")
            
            with pytest.raises(AzureServiceError):
                ingestion_worker(valid_message_dict, context_with_config)
            
            # Verify status was updated to indicate partial failure
            from src.services.workers.ingestion_worker import update_document_status
            update_document_status.assert_called()
            # Check that status was updated to failed_chunking_enqueue
            status_calls = [call for call in update_document_status.call_args_list if call[0][1] == "failed_chunking_enqueue"]
            assert len(status_calls) > 0


class TestIngestionWorkerBatchProcessing:
    """Tests for batch processing features"""
    
    def test_memory_limit_enforcement(self, valid_message_dict, context_with_config, mock_config):
        """Test that PDFs exceeding 50MB are rejected"""
        # Create a mock PDF that exceeds 50MB
        large_pdf = b"x" * (51 * 1024 * 1024)  # 51MB
        
        with patch('src.services.workers.ingestion_worker.should_process_document', return_value=True), \
             patch('src.services.workers.ingestion_worker.download_document_from_storage', return_value=large_pdf), \
             patch('src.services.workers.ingestion_worker.send_to_dead_letter') as mock_dlq, \
             patch('src.services.workers.ingestion_worker.update_document_status'), \
             patch('src.services.workers.ingestion_worker.update_ingestion_metadata'):
            
            with pytest.raises(AzureServiceError, match="exceeds 50MB"):
                ingestion_worker(valid_message_dict, context_with_config)
            
            # Verify DLQ was called
            mock_dlq.assert_called_once()
    
    def test_batch_resumption(self, valid_message_dict, context_with_config, mock_config, sample_pdf_content, actual_extracted_text):
        """Test that worker resumes from completed batches"""
        # Simulate that batch 0 is already completed
        completed_batches = {0}
        
        with patch('src.services.workers.ingestion_worker.should_process_document', return_value=True), \
             patch('src.services.workers.ingestion_worker.download_document_from_storage', return_value=sample_pdf_content), \
             patch('src.services.workers.ingestion_worker.get_pdf_page_count', return_value=4), \
             patch('src.services.workers.ingestion_worker.generate_page_batches', return_value=[(1, 3), (3, 5)]), \
             patch('src.services.workers.ingestion_worker.get_ingestion_metadata', return_value={
                 "num_pages": 4,
                 "num_batches_total": 2,
                 "last_successful_page": 2,
                 "next_unparsed_batch_index": 1,
                 "parsing_status": "in_progress",
                 "batch_size": 2,
                 "batches_completed": {"0": True},
                 "parsing_started_at": "2024-01-01T00:00:00Z",
                 "parsing_completed_at": None,
                 "errors": []
             }), \
             patch('src.services.workers.ingestion_worker.get_completed_batches', return_value=completed_batches), \
             patch('src.services.workers.ingestion_worker.load_batch_result', return_value=actual_extracted_text), \
             patch('src.services.workers.ingestion_worker.slice_pdf_to_batch', return_value=sample_pdf_content), \
             patch('src.services.workers.ingestion_worker.extract_text_from_batch', return_value=actual_extracted_text) as mock_extract, \
             patch('src.services.workers.ingestion_worker.persist_batch_result'), \
             patch('src.services.workers.ingestion_worker.delete_batch_chunk'), \
             patch('src.services.workers.ingestion_worker.update_ingestion_metadata'), \
             patch('src.services.workers.ingestion_worker.persist_extracted_text'), \
             patch('src.services.workers.ingestion_worker.update_document_status'), \
             patch('src.services.workers.ingestion_worker.enqueue_message'):
            
            ingestion_worker(valid_message_dict, context_with_config)
            
            # Verify batch 0 was skipped (not extracted again)
            # Only batch 1 should be extracted
            assert mock_extract.call_count == 1
    
    def test_configurable_batch_size(self, valid_message_dict, context_with_config, mock_config, sample_pdf_content, actual_extracted_text):
        """Test that batch size can be configured via metadata"""
        valid_message_dict["metadata"] = {"batch_size": 5}
        
        with patch('src.services.workers.ingestion_worker.should_process_document', return_value=True), \
             patch('src.services.workers.ingestion_worker.download_document_from_storage', return_value=sample_pdf_content), \
             patch('src.services.workers.ingestion_worker.get_pdf_page_count', return_value=10), \
             patch('src.services.workers.ingestion_worker.generate_page_batches') as mock_generate, \
             patch('src.services.workers.ingestion_worker.get_ingestion_metadata', return_value={
                 "num_pages": 0,
                 "num_batches_total": 0,
                 "last_successful_page": 0,
                 "next_unparsed_batch_index": 0,
                 "parsing_status": "pending",
                 "batch_size": 5,
                 "batches_completed": {},
                 "parsing_started_at": None,
                 "parsing_completed_at": None,
                 "errors": []
             }), \
             patch('src.services.workers.ingestion_worker.get_completed_batches', return_value=set()), \
             patch('src.services.workers.ingestion_worker.slice_pdf_to_batch', return_value=sample_pdf_content), \
             patch('src.services.workers.ingestion_worker.extract_text_from_batch', return_value=actual_extracted_text), \
             patch('src.services.workers.ingestion_worker.persist_batch_result'), \
             patch('src.services.workers.ingestion_worker.delete_batch_chunk'), \
             patch('src.services.workers.ingestion_worker.update_ingestion_metadata'), \
             patch('src.services.workers.ingestion_worker.persist_extracted_text'), \
             patch('src.services.workers.ingestion_worker.update_document_status'), \
             patch('src.services.workers.ingestion_worker.enqueue_message'):
            
            ingestion_worker(valid_message_dict, context_with_config)
            
            # Verify batch generation was called with batch_size=5
            mock_generate.assert_called_once_with(10, 5)

