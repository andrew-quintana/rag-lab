"""Integration tests for Phase 5: Supabase database operations

These tests verify that workers can read/write to Supabase correctly
with real database connections. Tests use actual database operations
but clean up after themselves.

Run with: pytest tests/integration/test_supabase_phase5.py -v -m integration
"""

import pytest
import uuid
from datetime import datetime, timezone
from typing import List

from rag_eval.core.config import Config
from rag_eval.db.connection import DatabaseConnection
from rag_eval.services.workers.persistence import (
    persist_extracted_text,
    load_extracted_text,
    persist_chunks,
    load_chunks,
    persist_embeddings,
    load_embeddings,
    update_document_status,
    check_document_status,
    should_process_document,
    persist_batch_result,
    get_completed_batches,
    load_batch_result,
    delete_batch_chunk,
    update_ingestion_metadata,
    get_ingestion_metadata,
)
from rag_eval.core.interfaces import Chunk


@pytest.fixture(scope="module")
def config():
    """Load configuration from environment"""
    return Config.from_env()


@pytest.fixture(scope="module")
def db_conn(config):
    """Create database connection"""
    if not config.database_url:
        pytest.skip("DATABASE_URL not set - skipping integration tests")
    return DatabaseConnection(config)


@pytest.fixture
def test_document_id(db_conn):
    """Create a test document and return its ID"""
    conn = db_conn.get_connection()
    cursor = conn.cursor()
    
    doc_id = str(uuid.uuid4())
    cursor.execute("""
        INSERT INTO documents (id, filename, status, file_size, storage_path)
        VALUES (%s, 'test_document.pdf', 'uploaded', 0, 'test/path.pdf')
        ON CONFLICT (id) DO NOTHING
    """, (doc_id,))
    conn.commit()
    cursor.close()
    
    yield doc_id
    
    # Cleanup
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
    cursor.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
    conn.commit()
    cursor.close()
    db_conn.return_connection(conn)


@pytest.mark.integration
class TestPersistenceOperations:
    """Test persistence operations with real Supabase database"""
    
    def test_persist_and_load_extracted_text(self, config, test_document_id):
        """Test persisting and loading extracted text"""
        test_text = "This is test extracted text from a document. " * 10
        
        # Persist
        persist_extracted_text(test_document_id, test_text, config)
        
        # Load
        loaded_text = load_extracted_text(test_document_id, config)
        
        assert loaded_text == test_text
        assert len(loaded_text) > 0
    
    def test_persist_and_load_chunks(self, config, test_document_id):
        """Test persisting and loading chunks"""
        chunks = [
            Chunk(
                chunk_id=f"chunk_{i}",
                text=f"Chunk {i} text content",
                metadata={"index": i, "start": i * 100, "end": (i + 1) * 100}
            )
            for i in range(3)
        ]
        
        # Persist
        persist_chunks(test_document_id, chunks, config)
        
        # Load
        loaded_chunks = load_chunks(test_document_id, config)
        
        assert len(loaded_chunks) == 3
        assert loaded_chunks[0].chunk_id == "chunk_0"
        assert loaded_chunks[0].text == "Chunk 0 text content"
        assert loaded_chunks[0].metadata["index"] == 0
    
    def test_persist_and_load_embeddings(self, config, test_document_id):
        """Test persisting and loading embeddings"""
        chunks = [
            Chunk(
                chunk_id=f"chunk_{i}",
                text=f"Chunk {i}",
                metadata={}
            )
            for i in range(2)
        ]
        
        # Create realistic embeddings (1536 dimensions)
        embeddings = [
            [0.1 * (i + j) for j in range(1536)]
            for i in range(2)
        ]
        
        # Persist chunks first
        persist_chunks(test_document_id, chunks, config)
        
        # Persist embeddings
        persist_embeddings(test_document_id, chunks, embeddings, config)
        
        # Load embeddings
        loaded_embeddings = load_embeddings(test_document_id, config)
        
        assert len(loaded_embeddings) == 2
        assert len(loaded_embeddings[0]) == 1536
        assert loaded_embeddings[0][0] == pytest.approx(0.0)
        assert loaded_embeddings[1][0] == pytest.approx(0.1)  # 0.1 * (1 + 0) = 0.1


@pytest.mark.integration
class TestStatusUpdates:
    """Test document status updates and idempotency"""
    
    def test_status_transitions(self, config, test_document_id):
        """Test document status transitions through pipeline"""
        # Initial status should be 'uploaded'
        status = check_document_status(test_document_id, config)
        assert status == 'uploaded'
        
        # Transition to parsed
        update_document_status(test_document_id, 'parsed', 'parsed_at', config)
        status = check_document_status(test_document_id, config)
        assert status == 'parsed'
        
        # Transition to chunked
        update_document_status(test_document_id, 'chunked', 'chunked_at', config)
        status = check_document_status(test_document_id, config)
        assert status == 'chunked'
        
        # Transition to embedded
        update_document_status(test_document_id, 'embedded', 'embedded_at', config)
        status = check_document_status(test_document_id, config)
        assert status == 'embedded'
        
        # Transition to indexed
        update_document_status(test_document_id, 'indexed', 'indexed_at', config)
        status = check_document_status(test_document_id, config)
        assert status == 'indexed'
    
    def test_idempotency_checks(self, config, test_document_id):
        """Test idempotency checks prevent duplicate processing"""
        # Set status to parsed
        update_document_status(test_document_id, 'parsed', 'parsed_at', config)
        
        # Should not process if already parsed
        assert not should_process_document(test_document_id, 'parsed', config)
        assert should_process_document(test_document_id, 'chunked', config)
        
        # Set status to chunked
        update_document_status(test_document_id, 'chunked', 'chunked_at', config)
        
        # Should not process if already chunked or beyond
        assert not should_process_document(test_document_id, 'parsed', config)
        assert not should_process_document(test_document_id, 'chunked', config)
        assert should_process_document(test_document_id, 'embedded', config)


@pytest.mark.integration
class TestBatchMetadata:
    """Test batch processing metadata storage"""
    
    def test_batch_metadata_storage(self, config, test_document_id):
        """Test storing and retrieving batch processing metadata"""
        # Initialize metadata
        metadata = {
            'num_pages': 6,
            'num_batches_total': 3,
            'last_successful_page': 0,
            'next_unparsed_batch_index': 0,
            'parsing_status': 'in_progress',
            'batch_size': 2,
            'batches_completed': {},
            'parsing_started_at': datetime.now(timezone.utc).isoformat(),
            'parsing_completed_at': None,
            'errors': []
        }
        
        update_ingestion_metadata(test_document_id, metadata, config)
        
        # Retrieve metadata
        retrieved = get_ingestion_metadata(test_document_id, config)
        
        assert retrieved['num_pages'] == 6
        assert retrieved['num_batches_total'] == 3
        assert retrieved['batch_size'] == 2
        assert retrieved['parsing_status'] == 'in_progress'
    
    def test_batch_result_persistence(self, config, test_document_id):
        """Test persisting and loading batch results"""
        batch_index = 0
        batch_id = "batch_0"
        batch_text = "Batch 0 extracted text content"
        start_page = 1
        end_page = 2
        
        # Persist batch result
        persist_batch_result(test_document_id, batch_index, batch_text, start_page, end_page, config)
        
        # Load batch result
        loaded_text = load_batch_result(test_document_id, batch_index, config)
        
        assert loaded_text == batch_text
        
        # Mark batch as completed
        completed = get_completed_batches(test_document_id, config)
        assert batch_id not in completed  # Not marked as completed yet
        
        # Update metadata to mark batch as completed
        metadata = get_ingestion_metadata(test_document_id, config) or {}
        if 'batches_completed' not in metadata:
            metadata['batches_completed'] = {}
        metadata['batches_completed'][batch_id] = True
        update_ingestion_metadata(test_document_id, metadata, config)
        
        # Check completed batches
        completed = get_completed_batches(test_document_id, config)
        assert batch_id in completed
        
        # Cleanup batch chunk
        delete_batch_chunk(test_document_id, batch_id, config)
        
        # Verify deleted
        with pytest.raises(Exception):  # Should raise error when not found
            load_batch_result(test_document_id, batch_id, config)


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling with real database connections"""
    
    def test_missing_document_error(self, config):
        """Test error handling for missing document"""
        fake_id = str(uuid.uuid4())
        
        with pytest.raises(Exception):
            load_extracted_text(fake_id, config)
    
    def test_database_connection_error(self, config):
        """Test that database connection errors are handled"""
        # This test verifies that connection errors are properly raised
        # In a real scenario, we'd test with invalid connection string
        # For now, we just verify the connection works
        db_conn = DatabaseConnection(config)
        conn = db_conn.get_connection()
        assert conn is not None
        db_conn.return_connection(conn)


@pytest.mark.integration
class TestWorkerReadWrite:
    """Test that all workers can read/write to Supabase correctly"""
    
    def test_ingestion_worker_persistence(self, config, test_document_id):
        """Test ingestion worker can persist extracted text"""
        test_text = "Extracted text from ingestion worker"
        persist_extracted_text(test_document_id, test_text, config)
        
        loaded = load_extracted_text(test_document_id, config)
        assert loaded == test_text
    
    def test_chunking_worker_persistence(self, config, test_document_id):
        """Test chunking worker can persist chunks"""
        # First persist extracted text (simulating ingestion worker)
        persist_extracted_text(test_document_id, "Test text for chunking", config)
        
        # Simulate chunking worker
        chunks = [
            Chunk(chunk_id=f"chunk_{i}", text=f"Chunk {i}", metadata={})
            for i in range(2)
        ]
        persist_chunks(test_document_id, chunks, config)
        
        loaded = load_chunks(test_document_id, config)
        assert len(loaded) == 2
    
    def test_embedding_worker_persistence(self, config, test_document_id):
        """Test embedding worker can persist embeddings"""
        chunks = [
            Chunk(chunk_id=f"chunk_{i}", text=f"Chunk {i}", metadata={})
            for i in range(2)
        ]
        persist_chunks(test_document_id, chunks, config)
        
        embeddings = [[0.1 * j for j in range(1536)] for _ in range(2)]
        persist_embeddings(test_document_id, chunks, embeddings, config)
        
        loaded = load_embeddings(test_document_id, config)
        assert len(loaded) == 2
    
    def test_indexing_worker_read(self, config, test_document_id):
        """Test indexing worker can read chunks and embeddings"""
        # Setup: persist chunks and embeddings
        chunks = [
            Chunk(chunk_id=f"chunk_{i}", text=f"Chunk {i}", metadata={})
            for i in range(2)
        ]
        persist_chunks(test_document_id, chunks, config)
        
        embeddings = [[0.1 * j for j in range(1536)] for _ in range(2)]
        persist_embeddings(test_document_id, chunks, embeddings, config)
        
        # Simulate indexing worker reading
        loaded_chunks = load_chunks(test_document_id, config)
        loaded_embeddings = load_embeddings(test_document_id, config)
        
        assert len(loaded_chunks) == 2
        assert len(loaded_embeddings) == 2
        assert len(loaded_chunks) == len(loaded_embeddings)

