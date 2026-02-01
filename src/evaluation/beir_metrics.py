"""BEIR-style retrieval metrics computation

This module implements BEIR-style retrieval metrics (recall@k, precision@k, nDCG@k)
following the BEIR benchmark framework for information retrieval evaluation.

BEIR Reference: https://github.com/beir-cellar/beir
"""

import math
from typing import List

from ..core.interfaces import RetrievalResult, BEIRMetricsResult


def compute_beir_metrics(
    retrieved_chunks: List[RetrievalResult],
    ground_truth_chunk_ids: List[str],
    k: int = 5
) -> BEIRMetricsResult:
    """
    Compute BEIR-style retrieval metrics (recall@k, precision@k, nDCG@k).
    
    This function computes standard information retrieval metrics following the
    BEIR benchmark framework. All metrics are pure Python implementations with
    no LLM calls.
    
    Args:
        retrieved_chunks: List of retrieved chunks, ordered by relevance (highest first).
                         Should already be sorted by similarity_score descending.
        ground_truth_chunk_ids: List of chunk IDs that are considered relevant (ground truth).
        k: Number of top results to consider for metrics (default: 5).
        
    Returns:
        BEIRMetricsResult object containing:
        - recall_at_k: Recall@k metric (0.0 to 1.0)
        - precision_at_k: Precision@k metric (0.0 to 1.0)
        - ndcg_at_k: Normalized discounted cumulative gain@k (0.0 to 1.0)
        
    Raises:
        ValueError: If inputs are invalid (empty lists, k <= 0)
        
    Examples:
        >>> retrieved = [
        ...     RetrievalResult(chunk_id="chunk_1", similarity_score=0.9, chunk_text="..."),
        ...     RetrievalResult(chunk_id="chunk_2", similarity_score=0.8, chunk_text="..."),
        ... ]
        >>> ground_truth = ["chunk_1", "chunk_3"]
        >>> metrics = compute_beir_metrics(retrieved, ground_truth, k=2)
        >>> metrics.recall_at_k  # 0.5 (1 relevant in top-2, 2 total relevant)
        >>> metrics.precision_at_k  # 0.5 (1 relevant in top-2, k=2)
    """
    # Validate inputs
    if not retrieved_chunks:
        raise ValueError("retrieved_chunks cannot be empty")
    
    if not ground_truth_chunk_ids:
        raise ValueError("ground_truth_chunk_ids cannot be empty")
    
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    
    # Extract chunk IDs from top-k retrieved chunks
    top_k_chunk_ids = [chunk.chunk_id for chunk in retrieved_chunks[:k]]
    
    # Convert ground truth to set for efficient lookup
    ground_truth_set = set(ground_truth_chunk_ids)
    
    # Compute metrics
    recall = _compute_recall_at_k(top_k_chunk_ids, ground_truth_set)
    precision = _compute_precision_at_k(top_k_chunk_ids, ground_truth_set, k)
    ndcg = _compute_ndcg_at_k(retrieved_chunks[:k], ground_truth_set, k)
    
    return BEIRMetricsResult(
        recall_at_k=recall,
        precision_at_k=precision,
        ndcg_at_k=ndcg
    )


def _compute_recall_at_k(
    retrieved_chunk_ids: List[str],
    ground_truth_chunk_ids: set
) -> float:
    """
    Compute recall@k metric.
    
    Formula: (Number of relevant chunks in top-k) / (Total number of relevant chunks)
    
    Edge cases:
    - If no relevant chunks exist (ground_truth is empty), returns 0.0
    - If all relevant chunks are retrieved, returns 1.0
    
    Args:
        retrieved_chunk_ids: List of chunk IDs in top-k results
        ground_truth_chunk_ids: Set of relevant chunk IDs
        
    Returns:
        Recall@k value (0.0 to 1.0)
    """
    if not ground_truth_chunk_ids:
        # No relevant chunks - recall is undefined, return 0.0
        return 0.0
    
    # Count how many relevant chunks are in top-k
    relevant_in_top_k = sum(1 for chunk_id in retrieved_chunk_ids if chunk_id in ground_truth_chunk_ids)
    
    # Recall = relevant retrieved / total relevant
    recall = relevant_in_top_k / len(ground_truth_chunk_ids)
    
    return recall


def _compute_precision_at_k(
    retrieved_chunk_ids: List[str],
    ground_truth_chunk_ids: set,
    k: int
) -> float:
    """
    Compute precision@k metric.
    
    Formula: (Number of relevant chunks in top-k) / k
    
    Edge cases:
    - If k is larger than number of retrieved chunks, use actual number of retrieved chunks
    - If no relevant chunks in top-k, returns 0.0
    
    Args:
        retrieved_chunk_ids: List of chunk IDs in top-k results
        ground_truth_chunk_ids: Set of relevant chunk IDs
        k: Number of top results considered
        
    Returns:
        Precision@k value (0.0 to 1.0)
    """
    # Count how many relevant chunks are in top-k
    relevant_in_top_k = sum(1 for chunk_id in retrieved_chunk_ids if chunk_id in ground_truth_chunk_ids)
    
    # Precision = relevant retrieved / k
    # Use actual number of retrieved chunks if k > len(retrieved_chunk_ids)
    actual_k = min(k, len(retrieved_chunk_ids))
    
    if actual_k == 0:
        return 0.0
    
    precision = relevant_in_top_k / actual_k
    
    return precision


def _compute_ndcg_at_k(
    retrieved_chunks: List[RetrievalResult],
    ground_truth_chunk_ids: set,
    k: int
) -> float:
    """
    Compute normalized discounted cumulative gain@k (nDCG@k).
    
    Formula:
    - DCG@k = sum(relevance_i / log2(i+1)) for i in 1..k
    - IDCG@k = DCG@k for ideal ranking (all relevant items first)
    - nDCG@k = DCG@k / IDCG@k
    
    Relevance scores: 1 for relevant chunks, 0 for irrelevant chunks.
    
    Edge cases:
    - If no relevant chunks exist, returns 0.0
    - If IDCG@k is 0 (no relevant chunks), returns 0.0
    - If k is larger than number of retrieved chunks, use actual number
    
    Args:
        retrieved_chunks: List of RetrievalResult objects in top-k (ordered by relevance)
        ground_truth_chunk_ids: Set of relevant chunk IDs
        k: Number of top results considered
        
    Returns:
        nDCG@k value (0.0 to 1.0)
    """
    if not ground_truth_chunk_ids:
        # No relevant chunks - nDCG is undefined, return 0.0
        return 0.0
    
    # Compute DCG@k
    dcg = 0.0
    for i, chunk in enumerate(retrieved_chunks[:k], start=1):
        # Relevance: 1 if chunk is relevant, 0 otherwise
        relevance = 1.0 if chunk.chunk_id in ground_truth_chunk_ids else 0.0
        # DCG formula: relevance / log2(rank)
        dcg += relevance / math.log2(i + 1)
    
    # Compute IDCG@k (ideal DCG: all relevant chunks ranked first)
    num_relevant = len(ground_truth_chunk_ids)
    actual_k = min(k, len(retrieved_chunks))
    num_relevant_in_k = min(num_relevant, actual_k)
    
    idcg = 0.0
    for i in range(1, num_relevant_in_k + 1):
        # In ideal ranking, all relevant items are at positions 1..num_relevant_in_k
        idcg += 1.0 / math.log2(i + 1)
    
    # Normalize: nDCG = DCG / IDCG
    if idcg == 0.0:
        return 0.0
    
    ndcg = dcg / idcg
    
    return ndcg