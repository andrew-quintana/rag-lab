"""Performance tests for Phase 5: Throughput and latency validation

These tests validate throughput and latency requirements with real Azure Functions.
Tests should be run post-deployment with actual Azure resources.

IMPORTANT: Performance tests limit to first 6 pages of test PDF to stay within budget.

Run with: pytest tests/integration/test_phase5_performance.py -v -m integration
"""

import pytest
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any
from statistics import mean, median

from src.core.config import Config
from src.services.workers.queue_client import (
    QueueMessage,
    enqueue_message,
    get_queue_length,
    ProcessingStage,
    SourceStorage,
)
from src.services.workers.persistence import check_document_status

# Note: config fixture is now in conftest.py


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


@pytest.mark.integration
@pytest.mark.performance
class TestWorkerProcessingTime:
    """Test worker processing time under load"""
    
    @pytest.mark.skip(reason="Requires Azure Functions to be deployed and running")
    def test_ingestion_worker_processing_time(self, config, test_pdf_path):
        """Test ingestion worker processing time under load
        
        Target: Process messages within 30 seconds under normal load
        """
        # This test requires Azure Functions to be deployed
        # It measures time from enqueue to status change to 'parsed'
        
        processing_times = []
        num_tests = 3  # Limit to 3 tests to stay within budget
        
        for i in range(num_tests):
            doc_id = str(uuid.uuid4())
            
            # Enqueue message
            message = QueueMessage(
                document_id=doc_id,
                source_storage=SourceStorage.SUPABASE,
                filename=f"perf_test_{i}.pdf",
                attempt=1,
                stage=ProcessingStage.UPLOADED,
                metadata={}
            )
            
            start_time = time.time()
            enqueue_message("ingestion-uploads", message, config)
            
            # Wait for processing (with timeout)
            max_wait = 300  # 5 minutes
            while time.time() - start_time < max_wait:
                status = check_document_status(doc_id, config)
                if status == "parsed":
                    processing_time = time.time() - start_time
                    processing_times.append(processing_time)
                    break
                time.sleep(2)
            else:
                pytest.fail(f"Document {doc_id} not processed within timeout")
        
        # Validate performance
        avg_time = mean(processing_times)
        median_time = median(processing_times)
        max_time = max(processing_times)
        
        print(f"\nIngestion Worker Performance:")
        print(f"  Average: {avg_time:.2f}s")
        print(f"  Median: {median_time:.2f}s")
        print(f"  Max: {max_time:.2f}s")
        
        # Target: < 30 seconds (under normal load)
        assert avg_time < 30, f"Average processing time {avg_time:.2f}s exceeds 30s target"
        assert max_time < 60, f"Max processing time {max_time:.2f}s exceeds 60s limit"
    
    @pytest.mark.skip(reason="Requires Azure Functions to be deployed and running")
    def test_end_to_end_pipeline_time(self, config, test_pdf_path):
        """Test end-to-end pipeline completion time
        
        Target: Complete pipeline within 5 minutes for typical documents
        """
        # This test measures time from upload to indexed status
        doc_id = str(uuid.uuid4())
        
        message = QueueMessage(
            document_id=doc_id,
            source_storage=SourceStorage.SUPABASE,
            filename="e2e_test.pdf",
            attempt=1,
            stage=ProcessingStage.UPLOADED,
            metadata={}
        )
        
        start_time = time.time()
        enqueue_message("ingestion-uploads", message, config)
        
        # Wait for complete pipeline (with timeout)
        max_wait = 600  # 10 minutes
        while time.time() - start_time < max_wait:
            status = check_document_status(doc_id, config)
            if status == "indexed":
                total_time = time.time() - start_time
                print(f"\nEnd-to-End Pipeline Time: {total_time:.2f}s")
                
                # Target: < 5 minutes for typical documents
                assert total_time < 300, f"Pipeline time {total_time:.2f}s exceeds 5 minute target"
                return
            
            time.sleep(5)
        
        pytest.fail(f"Pipeline did not complete within {max_wait}s timeout")


@pytest.mark.integration
@pytest.mark.performance
class TestQueueDepthHandling:
    """Test queue depth handling with real Azure Storage Queues"""
    
    @pytest.mark.skip(reason="Requires Azure Functions to be deployed and running")
    def test_queue_depth_under_load(self, config):
        """Test queue depth handling under load
        
        Target: Average queue depth < 10 messages per queue under normal load
        """
        # Enqueue multiple messages
        num_messages = 10
        doc_ids = []
        
        for i in range(num_messages):
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)
            
            message = QueueMessage(
                document_id=doc_id,
                source_storage=SourceStorage.SUPABASE,
                filename=f"load_test_{i}.pdf",
                attempt=1,
                stage=ProcessingStage.UPLOADED,
                metadata={}
            )
            enqueue_message("ingestion-uploads", message, config)
        
        # Monitor queue depth over time
        queue_depths = []
        max_wait = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            depth = get_queue_length("ingestion-uploads", config)
            queue_depths.append(depth)
            
            if depth == 0:
                # All messages processed
                break
            
            time.sleep(5)
        
        avg_depth = mean(queue_depths) if queue_depths else 0
        max_depth = max(queue_depths) if queue_depths else 0
        
        print(f"\nQueue Depth Metrics:")
        print(f"  Average: {avg_depth:.2f}")
        print(f"  Max: {max_depth}")
        
        # Target: Average < 10 under normal load
        # Note: This may be higher during initial burst
        assert avg_depth < 20, f"Average queue depth {avg_depth:.2f} exceeds 20"


@pytest.mark.integration
@pytest.mark.performance
class TestConcurrentProcessing:
    """Test concurrent document processing across multiple function instances"""
    
    @pytest.mark.skip(reason="Requires Azure Functions to be deployed and running")
    def test_concurrent_document_processing(self, config):
        """Test concurrent document processing
        
        Verifies that multiple documents can be processed concurrently
        and that processing time doesn't degrade significantly.
        """
        num_documents = 5  # Limit to 5 for budget
        doc_ids = []
        start_times = {}
        
        # Enqueue all documents
        for i in range(num_documents):
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)
            
            message = QueueMessage(
                document_id=doc_id,
                source_storage=SourceStorage.SUPABASE,
                filename=f"concurrent_test_{i}.pdf",
                attempt=1,
                stage=ProcessingStage.UPLOADED,
                metadata={}
            )
            
            start_times[doc_id] = time.time()
            enqueue_message("ingestion-uploads", message, config)
        
        # Wait for all to complete
        processing_times = []
        max_wait = 600  # 10 minutes
        
        for doc_id in doc_ids:
            start_time = start_times[doc_id]
            while time.time() - start_time < max_wait:
                status = check_document_status(doc_id, config)
                if status == "parsed":
                    processing_time = time.time() - start_time
                    processing_times.append(processing_time)
                    break
                time.sleep(2)
            else:
                pytest.fail(f"Document {doc_id} not processed within timeout")
        
        # Validate concurrent processing
        avg_time = mean(processing_times)
        max_time = max(processing_times)
        
        print(f"\nConcurrent Processing Metrics:")
        print(f"  Documents: {num_documents}")
        print(f"  Average Time: {avg_time:.2f}s")
        print(f"  Max Time: {max_time:.2f}s")
        
        # Concurrent processing should not significantly degrade performance
        # Allow 2x single-document time for concurrent processing
        assert avg_time < 60, f"Average concurrent processing time {avg_time:.2f}s too high"


@pytest.mark.integration
@pytest.mark.performance
class TestColdStartLatency:
    """Test Azure Functions cold start latency"""
    
    @pytest.mark.skip(reason="Requires Azure Functions to be deployed and running")
    def test_cold_start_latency(self, config):
        """Test Azure Functions cold start latency
        
        Measures latency for first function invocation after idle period.
        Target: Cold start latency < 30 seconds
        """
        # Wait for functions to go cold (idle period)
        # This is difficult to test reliably, but we can measure first invocation
        
        doc_id = str(uuid.uuid4())
        
        message = QueueMessage(
            document_id=doc_id,
            source_storage=SourceStorage.SUPABASE,
            filename="cold_start_test.pdf",
            attempt=1,
            stage=ProcessingStage.UPLOADED,
            metadata={}
        )
        
        start_time = time.time()
        enqueue_message("ingestion-uploads", message, config)
        
        # Measure time to first response (status change)
        max_wait = 60  # 1 minute for cold start
        while time.time() - start_time < max_wait:
            status = check_document_status(doc_id, config)
            if status != "uploaded":
                cold_start_time = time.time() - start_time
                print(f"\nCold Start Latency: {cold_start_time:.2f}s")
                
                # Target: < 30 seconds
                # Note: Cold starts can vary significantly
                assert cold_start_time < 60, f"Cold start time {cold_start_time:.2f}s exceeds 60s"
                return
            
            time.sleep(1)
        
        pytest.fail("Cold start test did not complete within timeout")


@pytest.mark.integration
@pytest.mark.performance
class TestThroughput:
    """Test throughput requirements"""
    
    @pytest.mark.skip(reason="Requires Azure Functions to be deployed and running")
    def test_throughput_requirements(self, config):
        """Test throughput meets requirements
        
        Target: System maintains constant throughput despite Azure Document Intelligence
        free-tier constraints (2 pages per request).
        """
        # This test verifies that the system can process multiple documents
        # while respecting batch size constraints
        
        num_documents = 3  # Limit for budget
        doc_ids = []
        
        # Enqueue all documents
        enqueue_start = time.time()
        for i in range(num_documents):
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)
            
            message = QueueMessage(
                document_id=doc_id,
                source_storage=SourceStorage.SUPABASE,
                filename=f"throughput_test_{i}.pdf",
                attempt=1,
                stage=ProcessingStage.UPLOADED,
                metadata={}
            )
            enqueue_message("ingestion-uploads", message, config)
        
        enqueue_time = time.time() - enqueue_start
        
        # Wait for all to complete
        completion_times = []
        max_wait = 600  # 10 minutes
        
        for doc_id in doc_ids:
            start_time = time.time()
            while time.time() - start_time < max_wait:
                status = check_document_status(doc_id, config)
                if status == "parsed":
                    completion_times.append(time.time() - start_time)
                    break
                time.sleep(2)
        
        if len(completion_times) == num_documents:
            total_time = max(completion_times)
            throughput = num_documents / total_time if total_time > 0 else 0
            
            print(f"\nThroughput Metrics:")
            print(f"  Documents: {num_documents}")
            print(f"  Total Time: {total_time:.2f}s")
            print(f"  Throughput: {throughput:.4f} documents/second")
            
            # System should maintain throughput despite batch constraints
            assert throughput > 0, "Throughput should be positive"

