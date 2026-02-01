"""Abstract base class for chunking implementations"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ...core.interfaces import Chunk


class BaseChunker(ABC):
    """Abstract base class for text chunking implementations"""
    
    def __init__(self, **config):
        """
        Initialize chunker with configuration
        
        Args:
            **config: Chunker-specific configuration parameters
        """
        self.config = config
    
    @abstractmethod
    def chunk_text(
        self, 
        text: str, 
        document_id: Optional[str] = None,
        **kwargs
    ) -> List[Chunk]:
        """
        Chunk text into smaller segments
        
        Args:
            text: Text to chunk
            document_id: Optional document identifier
            **kwargs: Additional chunking parameters
            
        Returns:
            List of Chunk objects
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name/identifier of this chunker implementation"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get a human-readable description of this chunker"""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration used by this chunker"""
        return self.config.copy()
    
    def chunk_documents(
        self, 
        documents: List[str], 
        document_ids: Optional[List[str]] = None
    ) -> List[Chunk]:
        """
        Chunk multiple documents
        
        Args:
            documents: List of document texts
            document_ids: Optional list of document identifiers
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for i, doc_text in enumerate(documents):
            doc_id = document_ids[i] if document_ids else f"doc_{i}"
            chunks = self.chunk_text(doc_text, document_id=doc_id)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def get_chunk_statistics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Get statistics about the chunking results
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_chunk_length': 0,
                'min_chunk_length': 0,
                'max_chunk_length': 0,
                'total_text_length': 0
            }
        
        chunk_lengths = [len(chunk.text) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_length': sum(chunk_lengths) / len(chunk_lengths),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'total_text_length': sum(chunk_lengths),
            'chunker_name': self.get_name(),
            'chunker_config': self.get_config()
        }