"""Text chunking using fixed-size algorithm (deterministic)"""

import logging
from typing import List, Optional
from .interfaces import Chunk

logger = logging.getLogger(__name__)


def chunk_text_fixed_size(text: str, document_id: Optional[str] = None, chunk_size: int = 1000, overlap: int = 200) -> List[Chunk]:
    """
    Chunk text using a simple fixed-size algorithm (no LLM required).
    
    This function provides **deterministic** chunking: the same input text with the
    same parameters will always produce identical chunks. This is the recommended
    method for reproducible RAG pipeline testing.
    
    The algorithm:
    1. Creates chunks of exactly `chunk_size` characters (except the last chunk)
    2. Overlaps adjacent chunks by `overlap` characters
    3. Continues until the entire text is processed
    
    **Deterministic Behavior**: This function is fully deterministic. Same input text
    and parameters will always produce identical chunks with identical chunk IDs and
    metadata.
    
    Args:
        text: Text to chunk
        document_id: Optional document ID for chunk metadata (stored in each chunk)
        chunk_size: Size of each chunk in characters (default: 1000)
        overlap: Overlap between chunks in characters (default: 200)
        
    Returns:
        List of Chunk objects. Each chunk includes:
        - text: The chunk text content
        - chunk_id: Sequential ID (chunk_0, chunk_1, ...)
        - document_id: The provided document_id (or None)
        - metadata: Contains "start", "end" (character positions), and "chunking_method": "fixed_size"
        
    Note:
        - Empty text produces empty list
        - Text shorter than chunk_size produces a single chunk
        - Overlap must be less than chunk_size (otherwise chunks will overlap incorrectly)
        - Fully deterministic: same input = same output
    """
    chunks = []
    start = 0
    chunk_id = 0
    
    # Validate parameters to prevent infinite loops
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size}) to prevent infinite loops")
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        chunks.append(Chunk(
            text=chunk_text,
            chunk_id=f"chunk_{chunk_id}",
            document_id=document_id,
            metadata={"start": start, "end": end, "chunking_method": "fixed_size"}
        ))
        
        # If we've reached the end of text, we're done
        if end >= len(text):
            break
            
        # Advance start position, ensuring we always move forward
        next_start = end - overlap
        if next_start <= start:
            # Safety check: if overlap >= chunk_size, this would cause infinite loop
            raise ValueError(f"Loop would not advance: start={start}, end={end}, overlap={overlap}. This indicates overlap >= chunk_size.")
        start = next_start
        chunk_id += 1
    return chunks


def chunk_text(
    text: str, 
    document_id: Optional[str] = None, 
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[Chunk]:
    """
    Chunk text using fixed-size algorithm (deterministic).
    
    This is the main entry point for text chunking in the RAG pipeline. It uses
    fixed-size chunking which is deterministic and recommended for reproducible testing.
    
    **Deterministic Behavior**: This function is fully deterministic. Same input text
    and parameters will always produce identical chunks with identical chunk IDs and
    metadata.
    
    Args:
        text: Text to chunk
        document_id: Optional document ID for chunk metadata
        chunk_size: Size of each chunk in characters (default: 1000)
        overlap: Overlap between chunks in characters (default: 200)
        
    Returns:
        List of Chunk objects. Chunk metadata indicates "fixed_size" method.
        
    Note:
        - Fully deterministic: same input = same output
        - Fast, no external dependencies
        - Recommended for reproducible RAG pipeline testing
    """
    return chunk_text_fixed_size(text, document_id, chunk_size, overlap)