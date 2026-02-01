"""Common interface definitions for RAG evaluation"""

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
    """Represents a retrieval result from FAISS or other retrieval system"""
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


@dataclass
class BEIRMetricsResult:
    """BEIR-style retrieval metrics result."""
    recall_at_k: float
    precision_at_k: float
    ndcg_at_k: float


@dataclass
class EvaluationExample:
    """Single evaluation example with question, reference answer, and ground truth."""
    example_id: str
    question: str
    reference_answer: str
    ground_truth_chunk_ids: List[str]
    beir_failure_scale_factor: float  # Preserved for dataset completeness, not used by pipeline


@dataclass
class BiasCorrectedAccuracy:
    """Bias-corrected accuracy estimate with confidence interval."""
    raw_accuracy: float
    corrected_accuracy: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    confidence_level: float
    q0: float  # Judge specificity
    q1: float  # Judge sensitivity
    n: int  # Number of test examples
    m0: int  # Number of calibration examples (ground truth False)
    m1: int  # Number of calibration examples (ground truth True)


@dataclass
class EvaluationResult:
    """Complete evaluation result for a single example."""
    example_id: str
    judge_output: JudgeEvaluationResult
    meta_eval_output: MetaEvaluationResult
    beir_metrics: BEIRMetricsResult
    timestamp: datetime