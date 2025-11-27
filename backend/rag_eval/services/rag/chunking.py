"""Text chunking logic"""

from typing import List
from rag_eval.core.interfaces import Chunk
from rag_eval.core.logging import get_logger

logger = get_logger("services.rag.chunking")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Chunk]:
    """
    Chunk text using a simple fixed-size algorithm.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of Chunk objects
    """
    logger.info(f"Chunking text of length {len(text)} with chunk_size={chunk_size}, overlap={overlap}")
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        chunks.append(Chunk(
            text=chunk_text,
            chunk_id=f"chunk_{chunk_id}",
            metadata={"start": start, "end": end}
        ))
        start = end - overlap
        chunk_id += 1
    
    logger.info(f"Created {len(chunks)} chunks")
    return chunks

