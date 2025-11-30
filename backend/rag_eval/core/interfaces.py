"""Common interface definitions"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Query:
    """Represents a user query"""
    text: str
    query_id: Optional[str] = None
    timestamp: Optional[datetime] = None


@dataclass
class Chunk:
    """Represents a text chunk"""
    text: str
    chunk_id: str
    document_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RetrievalResult:
    """Represents a retrieval result from Azure AI Search"""
    chunk_id: str
    similarity_score: float
    chunk_text: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ModelAnswer:
    """Represents a model-generated answer"""
    text: str
    query_id: str
    prompt_version: str
    retrieved_chunk_ids: List[str]
    timestamp: Optional[datetime] = None


@dataclass
class EvaluationScore:
    """Represents an evaluation score from LLM-as-judge"""
    grounding: float
    relevance: float
    hallucination_risk: float
    query_id: str
    prompt_version: str
    judge_reasoning: Optional[str] = None


@dataclass
class JudgeEvaluationResult:
    """Output from LLM-as-Judge evaluation."""
    correctness_binary: bool
    hallucination_binary: bool
    risk_direction: Optional[int]  # -1, 0, or 1, None if no deviation
    risk_impact: Optional[int]  # 0, 1, 2, or 3, None if no deviation
    reasoning: str
    failure_mode: Optional[str] = None


@dataclass
class MetaEvaluationResult:
    """Output from Meta-Evaluator."""
    judge_correct: bool
    explanation: Optional[str] = None
    # Ground truth values for metrics calculation (optional)
    ground_truth_correctness: Optional[bool] = None
    ground_truth_hallucination: Optional[bool] = None
    ground_truth_risk_direction: Optional[int] = None
    ground_truth_risk_impact: Optional[int] = None


@dataclass
class JudgeMetricScores:
    """Performance metrics for a single judge output metric."""
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    total_samples: int


@dataclass
class JudgePerformanceMetrics:
    """Comprehensive performance metrics for LLM-as-Judge across all output metrics."""
    correctness: JudgeMetricScores
    hallucination: JudgeMetricScores
    risk_direction: Optional[JudgeMetricScores] = None  # None if insufficient data
    risk_impact: Optional[JudgeMetricScores] = None  # None if insufficient data

