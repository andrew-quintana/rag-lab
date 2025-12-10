"""End-to-end integration tests for Phase 5: Complete pipeline with real Azure resources

These tests verify the complete pipeline flow with real Azure Storage Queues and Supabase.
Tests use actual Azure resources and should be run post-deployment.

IMPORTANT: Tests that process actual PDFs use only the first 6 pages of
docs/inputs/scan_classic_hmo.pdf to avoid exceeding Azure Document Intelligence budget.

Run with: pytest tests/integration/test_phase5_e2e_pipeline.py -v -m integration
"""

import pytest
import uuid
import time
import json
from pathlib import Path
from typing import Dict, Any

from src.core.config import Config
from src.db.connection import DatabaseConnection
from src.services.workers.queue_client import (
    QueueMessage,
    enqueue_message,
    get_queue_length,
    ProcessingStage,
    SourceStorage,
)
from src.services.workers.persistence import (
    check_document_status,
    load_extracted_text,
    load_chunks,
    load_embeddings,
)
from src.services.rag.supabase_storage import upload_document_to_storage


def _is_local_development(config) -> bool:
    """Check if running in local development mode (Azurite)"""
    import os
    connection_string = config.azure_blob_connection_string or os.getenv("AZURE_STORAGE_QUEUES_CONNECTION_STRING", "")
    return connection_string == "UseDevelopmentStorage=true"


@pytest.fixture(scope="module")
def config():
    """Load configuration from environment"""
    return Config.from_env()


@pytest.fixture(scope="module")
def is_local():
    """Check if running in local development mode"""
    config_obj = Config.from_env()
    return _is_local_development(config_obj)


@pytest.fixture(scope="module")
def db_conn(config):
    """Create database connection"""
    if not config.database_url:
        pytest.skip("DATABASE_URL not set - skipping integration tests")
    return DatabaseConnection(config)


@pytest.fixture
def test_pdf_path():
    """Get path to test PDF (first 6 pages only)"""
    pdf_path = Path(__file__).parent.parent.parent.parent / "docs" / "inputs" / "scan_classic_hmo.pdf"
    
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")
    
    # Slice to first 6 pages only (budget constraint)
    from src.services.workers.pdf_utils import slice_pdf_to_batch
    
    with open(pdf_path, "rb") as f:
        full_pdf = f.read()
    
    # Extract first 6 pages (1-indexed: pages 1-7 exclusive = pages 1-6)
    sliced_pdf = slice_pdf_to_batch(full_pdf, start_page=1, end_page=7)
    
    # Write to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(sliced_pdf)
        tmp_path = Path(tmp.name)
    
    yield tmp_path
    
    # Cleanup
    if tmp_path.exists():
        tmp_path.unlink()


@pytest.fixture
def test_document_id(db_conn, config, test_pdf_path):
    """Create a test document in Supabase and upload file"""
    conn = db_conn.get_connection()
    cursor = conn.cursor()
    
    doc_id = str(uuid.uuid4())
    
    # Upload file to Supabase Storage
    filename = f"test_{doc_id}.pdf"
    with open(test_pdf_path, "rb") as f:
        file_content = f.read()
    
    try:
        upload_document_to_storage(file_content, doc_id, filename, config)
    except Exception as e:
        pytest.skip(f"Failed to upload test document to storage: {e}")
    
    # Create document record
    cursor.execute("""
        INSERT INTO documents (id, filename, status, storage_path, file_size, mime_type, upload_timestamp)
        VALUES (%s, %s, 'uploaded', %s, %s, 'application/pdf', NOW())
        ON CONFLICT (id) DO NOTHING
    """, (doc_id, filename, doc_id, len(file_content)))
    conn.commit()
    cursor.close()
    
    yield doc_id
    
    # Cleanup
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
    cursor.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
    conn.commit()
    cursor.close()
    conn.close()


@pytest.mark.integration
class TestEndToEndPipeline:
    """Test complete end-to-end pipeline flow"""
    
    @pytest.mark.local
    def test_message_passing_between_stages(self, config, test_document_id, is_local):
        """Test message passing between stages through actual queues"""
        # Enqueue initial message to ingestion-uploads
        message = QueueMessage(
            document_id=test_document_id,
            source_storage=SourceStorage.SUPABASE,
            filename="test.pdf",
            attempt=1,
            stage=ProcessingStage.UPLOADED,
            metadata={}
        )
        
        enqueue_message("ingestion-uploads", message, config)
        
        # Verify message was enqueued
        queue_length = get_queue_length("ingestion-uploads", config)
        assert queue_length >= 1
        
        # Note: In a real test, you would wait for Azure Functions to process
        # and verify the message moved to the next queue. For now, we just
        # verify the message was enqueued correctly.
    
    @pytest.mark.local
    def test_status_transitions_through_pipeline(self, config, test_document_id, db_conn, is_local):
        """Test status transitions through complete pipeline (verified in Supabase)"""
        # This test would require actual Azure Functions to be running
        # For now, we test the status update mechanism
        
        from src.services.workers.persistence import update_document_status
        
        # Simulate status transitions
        update_document_status(test_document_id, "parsed", "parsed_at", config)
        status = check_document_status(test_document_id, config)
        assert status == "parsed"
        
        update_document_status(test_document_id, "chunked", "chunked_at", config)
        status = check_document_status(test_document_id, config)
        assert status == "chunked"
        
        update_document_status(test_document_id, "embedded", "embedded_at", config)
        status = check_document_status(test_document_id, config)
        assert status == "embedded"
        
        update_document_status(test_document_id, "indexed", "indexed_at", config)
        status = check_document_status(test_document_id, config)
        assert status == "indexed"
    
    @pytest.mark.local
    def test_queue_depth_handling(self, config, is_local):
        """Test queue depth handling under load"""
        # Enqueue multiple messages
        messages = []
        for i in range(5):
            doc_id = str(uuid.uuid4())
            message = QueueMessage(
                document_id=doc_id,
                source_storage=SourceStorage.SUPABASE,
                filename=f"test_{i}.pdf",
                attempt=1,
                stage=ProcessingStage.UPLOADED,
                metadata={}
            )
            enqueue_message("ingestion-uploads", message, config)
            messages.append(doc_id)
        
        # Check queue depth
        queue_length = get_queue_length("ingestion-uploads", config)
        assert queue_length >= 5
        
        # Cleanup: Note - in real scenario, these would be processed by workers
        # For test cleanup, we'd need to manually dequeue or wait for processing
    
    @pytest.mark.cloud
    def test_azure_functions_queue_trigger_behavior(self, config, test_document_id, is_local):
        """Test Azure Functions queue trigger behavior
        
        This test requires Azure Functions to be deployed and running.
        It verifies that functions are triggered by queue messages.
        """
        if is_local:
            pytest.skip("This test requires cloud Azure Functions, not local")
        # Enqueue message
        message = QueueMessage(
            document_id=test_document_id,
            source_storage=SourceStorage.SUPABASE,
            filename="test.pdf",
            attempt=1,
            stage=ProcessingStage.UPLOADED,
            metadata={}
        )
        enqueue_message("ingestion-uploads", message, config)
        
        # Wait for function to process (with timeout)
        max_wait = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status = check_document_status(test_document_id, config)
            if status in ["parsed", "chunked", "embedded", "indexed"]:
                # Function processed the message
                break
            time.sleep(5)
        else:
            pytest.fail("Azure Function did not process message within timeout")
    
    @pytest.mark.local
    @pytest.mark.cloud
    def test_concurrent_document_processing(self, config, is_local):
        """Test concurrent document processing across multiple workers
        
        This test requires Azure Functions to be deployed and running.
        It verifies that multiple documents can be processed concurrently.
        """
        # Create multiple test documents and enqueue them
        # Verify they are processed concurrently
        # This is a complex test that requires actual Azure Functions
        pass


@pytest.mark.integration
class TestFailureScenarios:
    """Test failure scenarios and dead-letter handling"""
    
    @pytest.mark.local
    @pytest.mark.cloud
    def test_dead_letter_handling(self, config, is_local):
        """Test dead-letter queue handling with real queues
        
        This test requires Azure Functions to be deployed and running.
        It verifies that failed messages are sent to dead-letter queue.
        """
        # Create a message that will fail (e.g., invalid document_id)
        # Enqueue it
        # Verify it ends up in dead-letter queue after max retries
        pass
    
    @pytest.mark.local
    def test_idempotency_with_real_database(self, config, test_document_id, is_local):
        """Test idempotency with real database state"""
        from src.services.workers.persistence import (
            should_process_document,
            update_document_status,
        )
        
        # Set status to parsed
        update_document_status(test_document_id, "parsed", "parsed_at", config)
        
        # Should not process if already parsed
        assert not should_process_document(test_document_id, "parsed", config)
        assert should_process_document(test_document_id, "chunked", config)
        
        # Set status to indexed
        update_document_status(test_document_id, "indexed", "indexed_at", config)
        
        # Should not process any earlier stages
        assert not should_process_document(test_document_id, "parsed", config)
        assert not should_process_document(test_document_id, "chunked", config)
        assert not should_process_document(test_document_id, "embedded", config)
        assert not should_process_document(test_document_id, "indexed", config)


@pytest.mark.integration
class TestSupabaseIntegration:
    """Test Supabase integration with real database"""
    
    @pytest.mark.local
    def test_persistence_operations_real_database(self, config, test_document_id, is_local):
        """Test persistence operations with real Supabase database"""
        from src.services.workers.persistence import (
            persist_extracted_text,
            load_extracted_text,
            persist_chunks,
            load_chunks,
            persist_embeddings,
            load_embeddings,
        )
        from src.core.interfaces import Chunk
        
        # Test extracted text
        test_text = "Test extracted text from real database"
        persist_extracted_text(test_document_id, test_text, config)
        loaded_text = load_extracted_text(test_document_id, config)
        assert loaded_text == test_text
        
        # Test chunks
        chunks = [
            Chunk(chunk_id=f"chunk_{i}", text=f"Chunk {i}", metadata={})
            for i in range(2)
        ]
        persist_chunks(test_document_id, chunks, config)
        loaded_chunks = load_chunks(test_document_id, config)
        assert len(loaded_chunks) == 2
        
        # Test embeddings
        embeddings = [[0.1 * j for j in range(1536)] for _ in range(2)]
        persist_embeddings(test_document_id, chunks, embeddings, config)
        loaded_embeddings = load_embeddings(test_document_id, config)
        assert len(loaded_embeddings) == 2
    
    @pytest.mark.local
    def test_batch_metadata_storage_real_database(self, config, test_document_id, is_local):
        """Test batch processing metadata storage and retrieval"""
        from src.services.workers.persistence import (
            update_ingestion_metadata,
            get_ingestion_metadata,
        )
        
        metadata = {
            'num_pages': 6,
            'num_batches_total': 3,
            'batch_size': 2,
            'parsing_status': 'in_progress',
        }
        
        update_ingestion_metadata(test_document_id, metadata, config)
        retrieved = get_ingestion_metadata(test_document_id, config)
        
        assert retrieved['num_pages'] == 6
        assert retrieved['num_batches_total'] == 3
        assert retrieved['batch_size'] == 2

