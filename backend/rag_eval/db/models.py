"""Lightweight dataclasses for database records"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime


@dataclass
class QueryRecord:
    """Database record for a query"""
    query_id: str
    query_text: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RetrievalLog:
    """Database record for retrieval operation"""
    log_id: str
    query_id: str
    chunk_id: str
    similarity_score: float
    timestamp: datetime


@dataclass
class ModelAnswerRecord:
    """Database record for model answer"""
    answer_id: str
    query_id: str
    answer_text: str
    prompt_version: str
    retrieved_chunk_ids: List[str]
    timestamp: datetime


@dataclass
class EvalJudgment:
    """Database record for evaluation judgment
    
    Note: The `hallucination_risk` field is a FLOAT (0-1) representing an aggregated
    risk score, which is different from the `risk_direction` field (INT -1/+1) used
    in the evaluation state schema. The `risk_direction` field classifies the direction
    of system-level deviations, while `hallucination_risk` represents an overall risk
    magnitude score.
    """
    judgment_id: str
    query_id: str
    prompt_version: str
    grounding_score: float
    relevance_score: float
    hallucination_risk: float  # Aggregated risk score (0-1), distinct from risk_direction (-1/+1)
    judge_reasoning: Optional[str] = None
    timestamp: Optional[datetime] = None


@dataclass
class PromptVersion:
    """Database record for prompt version"""
    version_id: str
    version_name: str
    prompt_text: str
    created_at: datetime
    prompt_type: str = "rag"  # Default to "rag" for backward compatibility


@dataclass
class MetaEvalSummary:
    """Database record for meta-evaluation summary"""
    summary_id: str
    version_1: str
    version_2: str
    delta_grounding: float
    delta_relevance: float
    delta_hallucination: float
    judge_consistency: float
    timestamp: datetime


@dataclass
class Document:
    """Database record for uploaded document"""
    document_id: str
    filename: str
    file_size: int
    mime_type: Optional[str] = None
    upload_timestamp: Optional[datetime] = None
    status: str = "uploaded"
    chunks_created: Optional[int] = None
    storage_path: str = ""
    preview_image_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

