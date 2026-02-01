"""Simple text chunking implementation"""

from typing import List, Dict, Any, Optional
from ..base.base_chunker import BaseChunker
from ...core.interfaces import Chunk
from ...core.registry import register_chunker


@register_chunker(
    name="simple_chunker", 
    description="Simple sentence-based text chunking",
    method="sentence_split"
)
class SimpleChunker(BaseChunker):
    """Simple chunker that splits text by sentences"""
    
    def __init__(self, max_chunk_size: int = 500, overlap_size: int = 50, **config):
        """
        Initialize simple chunker
        
        Args:
            max_chunk_size: Maximum characters per chunk
            overlap_size: Number of characters to overlap between chunks
            **config: Additional configuration
        """
        super().__init__(max_chunk_size=max_chunk_size, overlap_size=overlap_size, **config)
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
    
    def get_name(self) -> str:
        return "simple_chunker"
    
    def get_description(self) -> str:
        return f"Simple sentence-based chunker (max_size={self.max_chunk_size}, overlap={self.overlap_size})"
    
    def chunk_text(
        self, 
        text: str, 
        document_id: Optional[str] = None,
        **kwargs
    ) -> List[Chunk]:
        """
        Chunk text using simple sentence splitting
        
        Args:
            text: Text to chunk
            document_id: Optional document identifier
            **kwargs: Additional parameters
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        sentences = self._split_sentences(text)
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= self.max_chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunk_id = f"{document_id}_chunk_{chunk_index}" if document_id else f"chunk_{chunk_index}"
                    chunks.append(Chunk(
                        chunk_id=chunk_id,
                        text=current_chunk.strip(),
                        document_id=document_id,
                        start_char=None,
                        end_char=None,
                        metadata={"chunk_index": chunk_index, "method": "simple"}
                    ))
                    chunk_index += 1
                
                current_chunk = sentence
        
        if current_chunk:
            chunk_id = f"{document_id}_chunk_{chunk_index}" if document_id else f"chunk_{chunk_index}"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=current_chunk.strip(),
                document_id=document_id,
                start_char=None,
                end_char=None,
                metadata={"chunk_index": chunk_index, "method": "simple"}
            ))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        
        sentence_endings = re.compile(r'[.!?]+')
        sentences = sentence_endings.split(text)
        
        result = []
        for i, sentence in enumerate(sentences[:-1]):
            if sentence.strip():
                result.append(sentence.strip() + '. ')
        
        if sentences[-1].strip():
            result.append(sentences[-1].strip())
            
        return result


@register_chunker(
    name="fixed_chunker",
    description="Fixed-size character chunking with overlap",
    method="fixed_size"
)
class FixedChunker(BaseChunker):
    """Fixed-size chunker with character-based splitting"""
    
    def __init__(self, chunk_size: int = 512, overlap_size: int = 50, **config):
        """
        Initialize fixed chunker
        
        Args:
            chunk_size: Fixed size for each chunk
            overlap_size: Number of characters to overlap
            **config: Additional configuration
        """
        super().__init__(chunk_size=chunk_size, overlap_size=overlap_size, **config)
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
    
    def get_name(self) -> str:
        return "fixed_chunker"
    
    def get_description(self) -> str:
        return f"Fixed-size character chunker (size={self.chunk_size}, overlap={self.overlap_size})"
    
    def chunk_text(
        self, 
        text: str, 
        document_id: Optional[str] = None,
        **kwargs
    ) -> List[Chunk]:
        """
        Chunk text using fixed-size approach
        
        Args:
            text: Text to chunk
            document_id: Optional document identifier
            **kwargs: Additional parameters
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            if chunk_text.strip():
                chunk_id = f"{document_id}_chunk_{chunk_index}" if document_id else f"chunk_{chunk_index}"
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text.strip(),
                    document_id=document_id,
                    start_char=start,
                    end_char=min(end, len(text)),
                    metadata={"chunk_index": chunk_index, "method": "fixed_size"}
                ))
                chunk_index += 1
            
            start = end - self.overlap_size
            
            if start >= len(text):
                break
        
        return chunks