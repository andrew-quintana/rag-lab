"""Unit tests for BEIR Metrics Evaluator"""

import pytest
import math
from typing import List

from rag_eval.core.interfaces import RetrievalResult, BEIRMetricsResult
from rag_eval.services.evaluator.beir_metrics import (
    compute_beir_metrics,
    _compute_recall_at_k,
    _compute_precision_at_k,
    _compute_ndcg_at_k
)


@pytest.fixture
def sample_retrieved_chunks():
    """Create sample retrieved chunks with varying similarity scores"""
    return [
        RetrievalResult(
            chunk_id="chunk_1",
            similarity_score=0.95,
            chunk_text="First chunk"
        ),
        RetrievalResult(
            chunk_id="chunk_2",
            similarity_score=0.90,
            chunk_text="Second chunk"
        ),
        RetrievalResult(
            chunk_id="chunk_3",
            similarity_score=0.85,
            chunk_text="Third chunk"
        ),
        RetrievalResult(
            chunk_id="chunk_4",
            similarity_score=0.80,
            chunk_text="Fourth chunk"
        ),
        RetrievalResult(
            chunk_id="chunk_5",
            similarity_score=0.75,
            chunk_text="Fifth chunk"
        ),
    ]


@pytest.fixture
def sample_ground_truth():
    """Create sample ground truth chunk IDs"""
    return ["chunk_1", "chunk_3", "chunk_5"]


class TestComputeBeirMetrics:
    """Test the main compute_beir_metrics function"""
    
    def test_basic_metrics_calculation(self, sample_retrieved_chunks, sample_ground_truth):
        """Test basic metrics calculation with standard inputs"""
        metrics = compute_beir_metrics(sample_retrieved_chunks, sample_ground_truth, k=5)
        
        assert isinstance(metrics, BEIRMetricsResult)
        assert 0.0 <= metrics.recall_at_k <= 1.0
        assert 0.0 <= metrics.precision_at_k <= 1.0
        assert 0.0 <= metrics.ndcg_at_k <= 1.0
        
        # With k=5, we have chunks 1, 2, 3, 4, 5
        # Ground truth: chunk_1, chunk_3, chunk_5
        # Relevant in top-5: chunk_1, chunk_3, chunk_5 (3 out of 3)
        # Recall@5 = 3/3 = 1.0
        # Precision@5 = 3/5 = 0.6
        assert metrics.recall_at_k == 1.0
        assert metrics.precision_at_k == 0.6
    
    def test_recall_at_k_calculation(self, sample_retrieved_chunks, sample_ground_truth):
        """Test recall@k calculation"""
        # Test with k=2: only chunk_1 and chunk_2 in top-2
        # Ground truth: chunk_1, chunk_3, chunk_5
        # Relevant in top-2: chunk_1 (1 out of 3)
        # Recall@2 = 1/3 ≈ 0.333
        metrics = compute_beir_metrics(sample_retrieved_chunks, sample_ground_truth, k=2)
        assert abs(metrics.recall_at_k - 1.0/3.0) < 1e-6
    
    def test_precision_at_k_calculation(self, sample_retrieved_chunks, sample_ground_truth):
        """Test precision@k calculation"""
        # Test with k=2: only chunk_1 and chunk_2 in top-2
        # Relevant in top-2: chunk_1 (1 out of 2)
        # Precision@2 = 1/2 = 0.5
        metrics = compute_beir_metrics(sample_retrieved_chunks, sample_ground_truth, k=2)
        assert metrics.precision_at_k == 0.5
    
    def test_ndcg_at_k_calculation(self, sample_retrieved_chunks, sample_ground_truth):
        """Test nDCG@k calculation"""
        metrics = compute_beir_metrics(sample_retrieved_chunks, sample_ground_truth, k=5)
        
        # Verify nDCG is between 0 and 1
        assert 0.0 <= metrics.ndcg_at_k <= 1.0
        
        # With ideal ranking (all relevant first), nDCG should be 1.0
        # In our case: chunk_1 (relevant, rank 1), chunk_2 (irrelevant, rank 2),
        #              chunk_3 (relevant, rank 3), chunk_4 (irrelevant, rank 4),
        #              chunk_5 (relevant, rank 5)
        # DCG = 1/log2(2) + 0/log2(3) + 1/log2(4) + 0/log2(5) + 1/log2(6)
        #     = 1/1 + 0 + 1/2 + 0 + 1/2.585 ≈ 1.886
        # IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1 + 0.631 + 0.5 ≈ 2.131
        # nDCG ≈ 1.886 / 2.131 ≈ 0.885
        assert metrics.ndcg_at_k > 0.8  # Should be high since relevant chunks are well-ranked
    
    def test_edge_case_zero_relevant_retrieved(self, sample_retrieved_chunks):
        """Test edge case: zero relevant passages retrieved"""
        # Ground truth: chunks that don't exist in retrieved
        ground_truth = ["chunk_999", "chunk_888"]
        metrics = compute_beir_metrics(sample_retrieved_chunks, ground_truth, k=5)
        
        # Recall@5 = 0/2 = 0.0
        # Precision@5 = 0/5 = 0.0
        # nDCG@5 = 0.0 (no relevant chunks retrieved)
        assert metrics.recall_at_k == 0.0
        assert metrics.precision_at_k == 0.0
        assert metrics.ndcg_at_k == 0.0
    
    def test_edge_case_all_relevant_retrieved(self, sample_retrieved_chunks):
        """Test edge case: all relevant passages retrieved"""
        # Ground truth: all chunks in retrieved list
        ground_truth = ["chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5"]
        metrics = compute_beir_metrics(sample_retrieved_chunks, ground_truth, k=5)
        
        # Recall@5 = 5/5 = 1.0
        # Precision@5 = 5/5 = 1.0
        # nDCG@5 should be 1.0 (perfect ranking)
        assert metrics.recall_at_k == 1.0
        assert metrics.precision_at_k == 1.0
        assert metrics.ndcg_at_k == 1.0
    
    def test_edge_case_k_larger_than_retrieved(self, sample_retrieved_chunks, sample_ground_truth):
        """Test edge case: k larger than number of retrieved chunks"""
        # Request k=10, but only 5 chunks available
        metrics = compute_beir_metrics(sample_retrieved_chunks, sample_ground_truth, k=10)
        
        # Should use actual number of retrieved chunks (5)
        # Recall@10 = 3/3 = 1.0 (all relevant retrieved)
        # Precision@10 = 3/5 = 0.6 (3 relevant out of 5 retrieved)
        assert metrics.recall_at_k == 1.0
        assert metrics.precision_at_k == 0.6
    
    def test_edge_case_empty_retrieved_chunks(self, sample_ground_truth):
        """Test edge case: empty retrieved chunks list"""
        with pytest.raises(ValueError, match="retrieved_chunks cannot be empty"):
            compute_beir_metrics([], sample_ground_truth, k=5)
    
    def test_edge_case_empty_ground_truth(self, sample_retrieved_chunks):
        """Test edge case: empty ground-truth chunk IDs list"""
        with pytest.raises(ValueError, match="ground_truth_chunk_ids cannot be empty"):
            compute_beir_metrics(sample_retrieved_chunks, [], k=5)
    
    def test_edge_case_invalid_k(self, sample_retrieved_chunks, sample_ground_truth):
        """Test edge case: invalid k value"""
        with pytest.raises(ValueError, match="k must be positive"):
            compute_beir_metrics(sample_retrieved_chunks, sample_ground_truth, k=0)
        
        with pytest.raises(ValueError, match="k must be positive"):
            compute_beir_metrics(sample_retrieved_chunks, sample_ground_truth, k=-1)
    
    def test_metrics_with_single_retrieved_chunk(self, sample_ground_truth):
        """Test metrics with single retrieved chunk"""
        retrieved = [
            RetrievalResult(chunk_id="chunk_1", similarity_score=0.9, chunk_text="Single chunk")
        ]
        metrics = compute_beir_metrics(retrieved, sample_ground_truth, k=1)
        
        # chunk_1 is in ground truth, so:
        # Recall@1 = 1/3 ≈ 0.333
        # Precision@1 = 1/1 = 1.0
        assert abs(metrics.recall_at_k - 1.0/3.0) < 1e-6
        assert metrics.precision_at_k == 1.0
    
    def test_metrics_with_k_equals_one(self, sample_retrieved_chunks, sample_ground_truth):
        """Test metrics with k=1"""
        metrics = compute_beir_metrics(sample_retrieved_chunks, sample_ground_truth, k=1)
        
        # Top-1: chunk_1 (relevant)
        # Recall@1 = 1/3 ≈ 0.333
        # Precision@1 = 1/1 = 1.0
        assert abs(metrics.recall_at_k - 1.0/3.0) < 1e-6
        assert metrics.precision_at_k == 1.0


class TestComputeRecallAtK:
    """Test the _compute_recall_at_k helper function"""
    
    def test_basic_recall(self):
        """Test basic recall calculation"""
        retrieved = ["chunk_1", "chunk_2", "chunk_3"]
        ground_truth = {"chunk_1", "chunk_3", "chunk_5"}
        
        # 2 relevant out of 3 total relevant
        recall = _compute_recall_at_k(retrieved, ground_truth)
        assert abs(recall - 2.0/3.0) < 1e-6
    
    def test_perfect_recall(self):
        """Test perfect recall (all relevant retrieved)"""
        retrieved = ["chunk_1", "chunk_2", "chunk_3"]
        ground_truth = {"chunk_1", "chunk_2", "chunk_3"}
        
        recall = _compute_recall_at_k(retrieved, ground_truth)
        assert recall == 1.0
    
    def test_zero_recall(self):
        """Test zero recall (no relevant retrieved)"""
        retrieved = ["chunk_1", "chunk_2", "chunk_3"]
        ground_truth = {"chunk_4", "chunk_5"}
        
        recall = _compute_recall_at_k(retrieved, ground_truth)
        assert recall == 0.0
    
    def test_empty_ground_truth(self):
        """Test edge case: empty ground truth"""
        retrieved = ["chunk_1", "chunk_2"]
        ground_truth = set()
        
        recall = _compute_recall_at_k(retrieved, ground_truth)
        assert recall == 0.0


class TestComputePrecisionAtK:
    """Test the _compute_precision_at_k helper function"""
    
    def test_basic_precision(self):
        """Test basic precision calculation"""
        retrieved = ["chunk_1", "chunk_2", "chunk_3"]
        ground_truth = {"chunk_1", "chunk_3"}
        k = 3
        
        # 2 relevant out of 3 retrieved
        precision = _compute_precision_at_k(retrieved, ground_truth, k)
        assert abs(precision - 2.0/3.0) < 1e-6
    
    def test_perfect_precision(self):
        """Test perfect precision (all retrieved are relevant)"""
        retrieved = ["chunk_1", "chunk_2", "chunk_3"]
        ground_truth = {"chunk_1", "chunk_2", "chunk_3"}
        k = 3
        
        precision = _compute_precision_at_k(retrieved, ground_truth, k)
        assert precision == 1.0
    
    def test_zero_precision(self):
        """Test zero precision (no relevant retrieved)"""
        retrieved = ["chunk_1", "chunk_2", "chunk_3"]
        ground_truth = {"chunk_4", "chunk_5"}
        k = 3
        
        precision = _compute_precision_at_k(retrieved, ground_truth, k)
        assert precision == 0.0
    
    def test_k_larger_than_retrieved(self):
        """Test k larger than number of retrieved chunks"""
        retrieved = ["chunk_1", "chunk_2"]
        ground_truth = {"chunk_1"}
        k = 5
        
        # Should use actual number of retrieved (2)
        # 1 relevant out of 2 retrieved
        precision = _compute_precision_at_k(retrieved, ground_truth, k)
        assert precision == 0.5


class TestComputeNdcgAtK:
    """Test the _compute_ndcg_at_k helper function"""
    
    def test_basic_ndcg(self, sample_retrieved_chunks, sample_ground_truth):
        """Test basic nDCG calculation"""
        ndcg = _compute_ndcg_at_k(sample_retrieved_chunks[:3], sample_ground_truth, k=3)
        
        # Verify nDCG is between 0 and 1
        assert 0.0 <= ndcg <= 1.0
    
    def test_perfect_ndcg(self):
        """Test perfect nDCG (ideal ranking: all relevant first)"""
        # Create chunks with all relevant first
        chunks = [
            RetrievalResult(chunk_id="chunk_1", similarity_score=0.9, chunk_text="Relevant 1"),
            RetrievalResult(chunk_id="chunk_2", similarity_score=0.8, chunk_text="Relevant 2"),
            RetrievalResult(chunk_id="chunk_3", similarity_score=0.7, chunk_text="Relevant 3"),
        ]
        ground_truth = {"chunk_1", "chunk_2", "chunk_3"}
        
        ndcg = _compute_ndcg_at_k(chunks, ground_truth, k=3)
        assert abs(ndcg - 1.0) < 1e-6  # Perfect ranking
    
    def test_zero_ndcg(self):
        """Test zero nDCG (no relevant chunks)"""
        chunks = [
            RetrievalResult(chunk_id="chunk_1", similarity_score=0.9, chunk_text="Irrelevant 1"),
            RetrievalResult(chunk_id="chunk_2", similarity_score=0.8, chunk_text="Irrelevant 2"),
        ]
        ground_truth = {"chunk_999"}
        
        ndcg = _compute_ndcg_at_k(chunks, ground_truth, k=2)
        assert ndcg == 0.0
    
    def test_empty_ground_truth(self):
        """Test edge case: empty ground truth"""
        chunks = [
            RetrievalResult(chunk_id="chunk_1", similarity_score=0.9, chunk_text="Chunk 1"),
        ]
        ground_truth = set()
        
        ndcg = _compute_ndcg_at_k(chunks, ground_truth, k=1)
        assert ndcg == 0.0
    
    def test_ndcg_with_partial_relevance(self):
        """Test nDCG with partial relevance (some relevant, some not)"""
        chunks = [
            RetrievalResult(chunk_id="chunk_1", similarity_score=0.9, chunk_text="Relevant"),
            RetrievalResult(chunk_id="chunk_2", similarity_score=0.8, chunk_text="Irrelevant"),
            RetrievalResult(chunk_id="chunk_3", similarity_score=0.7, chunk_text="Relevant"),
        ]
        ground_truth = {"chunk_1", "chunk_3"}
        
        ndcg = _compute_ndcg_at_k(chunks, ground_truth, k=3)
        
        # DCG = 1/log2(2) + 0/log2(3) + 1/log2(4) = 1 + 0 + 0.5 = 1.5
        # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.631 = 1.631
        # nDCG = 1.5 / 1.631 ≈ 0.919
        assert 0.9 <= ndcg <= 1.0
    
    def test_ndcg_ranking_importance(self):
        """Test that nDCG rewards relevant items at higher ranks"""
        # Scenario 1: Relevant item at rank 1
        chunks1 = [
            RetrievalResult(chunk_id="chunk_1", similarity_score=0.9, chunk_text="Relevant"),
            RetrievalResult(chunk_id="chunk_2", similarity_score=0.8, chunk_text="Irrelevant"),
        ]
        
        # Scenario 2: Relevant item at rank 2
        chunks2 = [
            RetrievalResult(chunk_id="chunk_2", similarity_score=0.8, chunk_text="Irrelevant"),
            RetrievalResult(chunk_id="chunk_1", similarity_score=0.9, chunk_text="Relevant"),
        ]
        
        ground_truth = {"chunk_1"}
        
        ndcg1 = _compute_ndcg_at_k(chunks1, ground_truth, k=2)
        ndcg2 = _compute_ndcg_at_k(chunks2, ground_truth, k=2)
        
        # Scenario 1 should have higher nDCG (relevant at rank 1)
        assert ndcg1 > ndcg2

