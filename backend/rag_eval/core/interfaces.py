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

