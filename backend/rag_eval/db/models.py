"""Lightweight dataclasses for database records"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime


@dataclass
class QueryRecord:
    """Database record for a query"""
    id: str  # UUID
    query_text: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RetrievalLog:
    """Database record for retrieval operation"""
    id: str  # UUID
    query_id: str  # UUID reference
    chunk_id: str
    similarity_score: float
    timestamp: datetime


@dataclass
class ModelAnswerRecord:
    """Database record for model answer"""
    id: str  # UUID
    query_id: str  # UUID reference
    answer_text: str
    prompt_version: str
    retrieved_chunk_ids: List[str]
    timestamp: datetime


@dataclass
class EvalJudgment:
    """Database record for evaluation judgment
    
    Stores full JudgeEvaluationResult as JSONB in judge_output field.
    """
    id: str  # UUID
    query_id: str  # UUID reference
    prompt_version_id: str
    judge_output: Dict[str, Any]  # JSONB storage of JudgeEvaluationResult
    created_at: Optional[datetime] = None


@dataclass
class PromptVersion:
    """Database record for prompt version"""
    id: str  # UUID
    version: str  # Semantic version (e.g., v0.1)
    name: Optional[str]  # Name for evaluation prompts (renamed from evaluator_type)
    live: bool  # Live/active version flag
    prompt_text: str
    created_at: datetime
    prompt_type: str = "rag"  # Default to "rag" for backward compatibility


@dataclass
class MetaEvalSummary:
    """Database record for meta-evaluation summary"""
    id: str  # UUID
    prompt_version_id: str
    dataset_id: str  # UUID reference
    meta_eval: Dict[str, Any]  # JSONB storage of MetaEvaluationResult


@dataclass
class Document:
    """Database record for uploaded document"""
    id: str  # UUID
    filename: str
    file_size: int
    mime_type: Optional[str] = None
    upload_timestamp: Optional[datetime] = None
    status: str = "uploaded"
    chunks_created: Optional[int] = None
    storage_path: str = ""
    preview_image_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

