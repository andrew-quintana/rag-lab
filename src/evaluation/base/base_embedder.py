"""Abstract base class for embedding implementations"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for text embedding implementations"""
    
    def __init__(self, **config):
        """
        Initialize embedder with configuration
        
        Args:
            **config: Embedder-specific configuration parameters
        """
        self.config = config
    
    @abstractmethod
    def embed_texts(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of texts to embed
            **kwargs: Additional embedding parameters
            
        Returns:
            Array of embeddings with shape (len(texts), embedding_dim)
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimensionality of embeddings produced by this model"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name/identifier of this embedder implementation"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get a human-readable description of this embedder"""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration used by this embedder"""
        return self.config.copy()
    
    def embed_single_text(self, text: str, **kwargs) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            **kwargs: Additional embedding parameters
            
        Returns:
            Array with shape (embedding_dim,)
        """
        embeddings = self.embed_texts([text], **kwargs)
        return embeddings[0]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about this embedder
        
        Returns:
            Dictionary with model information
        """
        return {
            'name': self.get_name(),
            'description': self.get_description(),
            'embedding_dimension': self.get_embedding_dimension(),
            'config': self.get_config()
        }
    
    def batch_embed_texts(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        **kwargs
    ) -> np.ndarray:
        """
        Generate embeddings for texts in batches
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            **kwargs: Additional embedding parameters
            
        Returns:
            Array of embeddings with shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embed_texts(batch, **kwargs)
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)