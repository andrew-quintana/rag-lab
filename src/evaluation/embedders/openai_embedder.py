"""OpenAI embedding implementation"""

from typing import List, Dict, Any, Optional, Callable
import numpy as np
from ..base.base_embedder import BaseEmbedder
from ...core.registry import register_embedder


@register_embedder(
    name="openai_embedder",
    description="OpenAI text-embedding models",
    provider="OpenAI"
)
class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedding implementation"""
    
    def __init__(
        self, 
        embedding_function: Callable[[List[str]], np.ndarray],
        model_name: str = "text-embedding-ada-002",
        embedding_dimension: int = 1536,
        **config
    ):
        """
        Initialize OpenAI embedder
        
        Args:
            embedding_function: Function that takes texts and returns embeddings
            model_name: Name of the OpenAI model
            embedding_dimension: Dimension of embeddings
            **config: Additional configuration
        """
        super().__init__(
            model_name=model_name,
            embedding_dimension=embedding_dimension,
            **config
        )
        self.embedding_function = embedding_function
        self.model_name = model_name
        self.embedding_dimension = embedding_dimension
    
    def get_name(self) -> str:
        return "openai_embedder"
    
    def get_description(self) -> str:
        return f"OpenAI embedding model: {self.model_name} (dim={self.embedding_dimension})"
    
    def embed_texts(self, texts: List[str], **kwargs) -> np.ndarray:
        """Generate embeddings using OpenAI API"""
        if not texts:
            return np.array([])
        
        return self.embedding_function(texts)
    
    def get_embedding_dimension(self) -> int:
        return self.embedding_dimension


@register_embedder(
    name="sentence_transformer_embedder", 
    description="Sentence Transformers embedding models",
    provider="Hugging Face"
)
class SentenceTransformerEmbedder(BaseEmbedder):
    """Sentence Transformers embedding implementation"""
    
    def __init__(
        self,
        embedding_function: Callable[[List[str]], np.ndarray],
        model_name: str = "all-MiniLM-L6-v2",
        embedding_dimension: int = 384,
        **config
    ):
        """
        Initialize Sentence Transformer embedder
        
        Args:
            embedding_function: Function that takes texts and returns embeddings
            model_name: Name of the sentence transformer model
            embedding_dimension: Dimension of embeddings
            **config: Additional configuration
        """
        super().__init__(
            model_name=model_name,
            embedding_dimension=embedding_dimension,
            **config
        )
        self.embedding_function = embedding_function
        self.model_name = model_name
        self.embedding_dimension = embedding_dimension
    
    def get_name(self) -> str:
        return "sentence_transformer_embedder"
    
    def get_description(self) -> str:
        return f"Sentence Transformer model: {self.model_name} (dim={self.embedding_dimension})"
    
    def embed_texts(self, texts: List[str], **kwargs) -> np.ndarray:
        """Generate embeddings using Sentence Transformers"""
        if not texts:
            return np.array([])
        
        return self.embedding_function(texts)
    
    def get_embedding_dimension(self) -> int:
        return self.embedding_dimension