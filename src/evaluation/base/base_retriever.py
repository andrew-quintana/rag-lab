"""Abstract base class for retriever implementations"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from ...core.interfaces import RetrievalResult


class BaseRetriever(ABC):
    """Abstract base class for document retrieval implementations"""
    
    def __init__(self, **config):
        """
        Initialize retriever with configuration
        
        Args:
            **config: Retriever-specific configuration parameters
        """
        self.config = config
    
    @abstractmethod
    def retrieve(
        self, 
        query: str, 
        k: int = 10,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Query text
            k: Number of documents to retrieve
            **kwargs: Additional retrieval parameters
            
        Returns:
            List of retrieval results sorted by relevance score (descending)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name/identifier of this retriever implementation"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get a human-readable description of this retriever"""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration used by this retriever"""
        return self.config.copy()
    
    def batch_retrieve(
        self, 
        queries: List[str], 
        k: int = 10,
        **kwargs
    ) -> List[List[RetrievalResult]]:
        """
        Retrieve documents for multiple queries
        
        Args:
            queries: List of query texts
            k: Number of documents to retrieve per query
            **kwargs: Additional retrieval parameters
            
        Returns:
            List of retrieval result lists, one per query
        """
        results = []
        for query in queries:
            query_results = self.retrieve(query, k=k, **kwargs)
            results.append(query_results)
        return results
    
    def get_retrieval_statistics(
        self, 
        results: List[RetrievalResult]
    ) -> Dict[str, Any]:
        """
        Get statistics about retrieval results
        
        Args:
            results: List of retrieval results
            
        Returns:
            Dictionary with retrieval statistics
        """
        if not results:
            return {
                'total_results': 0,
                'avg_score': 0.0,
                'min_score': 0.0,
                'max_score': 0.0,
                'score_variance': 0.0
            }
        
        scores = [result.score for result in results]
        
        return {
            'total_results': len(results),
            'avg_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'score_variance': sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores),
            'retriever_name': self.get_name(),
            'retriever_config': self.get_config()
        }


class VectorRetriever(BaseRetriever):
    """Base class for vector-based retrieval implementations"""
    
    @abstractmethod
    def add_documents(
        self, 
        documents: List[str], 
        document_ids: List[str],
        **kwargs
    ) -> None:
        """
        Add documents to the retrieval index
        
        Args:
            documents: List of document texts
            document_ids: List of document identifiers
            **kwargs: Additional indexing parameters
        """
        pass
    
    @abstractmethod
    def get_index_size(self) -> int:
        """Get the number of documents in the index"""
        pass
    
    def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the retrieval index
        
        Returns:
            Dictionary with index information
        """
        return {
            'index_size': self.get_index_size(),
            'retriever_name': self.get_name(),
            'config': self.get_config()
        }