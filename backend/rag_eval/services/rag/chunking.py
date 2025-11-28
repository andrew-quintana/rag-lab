"""Text chunking - supports both LLM-based and fixed-size chunking"""

from typing import List, Optional
import json
import logging
import requests

# Use absolute imports - pytest config ensures pythonpath is set correctly
from rag_eval.core.interfaces import Chunk
from rag_eval.core.exceptions import AzureServiceError
from rag_eval.core.logging import get_logger

# Initialize logger lazily to prevent hangs in test environments
# Only create logger when actually needed (not at module import time)
def _get_logger():
    """Get logger instance - lazy initialization"""
    return get_logger("services.rag.chunking")

# Initialize logger at module level
logger = get_logger("services.rag.chunking")


def chunk_text_with_llm(text: str, config, document_id: Optional[str] = None) -> List[Chunk]:
    """
    Chunk text intelligently using Azure AI Foundry (free tier OpenAI).
    Uses LLM to identify semantic boundaries for chunking.
    
    This function uses an LLM to identify semantic boundaries in text, creating chunks
    that are semantically coherent. The LLM attempts to:
    - Keep chunks between 500-1500 characters
    - Preserve complete sentences (no mid-sentence breaks)
    - Maintain semantic coherence (related ideas together)
    - Minimize overlap between adjacent chunks
    
    **Fallback Behavior**: If LLM chunking fails (network error, JSON parsing error, etc.),
    this function automatically falls back to fixed-size chunking.
    
    **Determinism Note**: While the LLM uses low temperature (0.1) for reproducibility,
    LLM-based chunking is not fully deterministic. For deterministic chunking, use
    chunk_text_fixed_size() instead.
    
    **Text Length Limitation**: Text longer than 10,000 characters is truncated for
    the LLM request (only first 10,000 characters are sent).
    
    Args:
        text: Text to chunk
        config: Application configuration with Azure AI Foundry credentials
               (azure_ai_foundry_endpoint, azure_ai_foundry_api_key)
        document_id: Optional document ID for chunk metadata
        
    Returns:
        List of Chunk objects with metadata indicating "azure_ai_foundry_llm" method.
        Each chunk includes start_index in metadata.
        
    Raises:
        AzureServiceError: If chunking fails (though fallback to fixed-size chunking
                          is attempted first)
        
    Note:
        - Requires valid Azure AI Foundry credentials in config
        - Automatically falls back to fixed-size chunking on any error
        - Text > 10,000 characters is truncated for LLM processing
        - Chunking is not fully deterministic (LLM may produce different results)
    """
    logger.info(f"Chunking text of length {len(text)} using Azure AI Foundry")
    
    try:
        # Use free tier OpenAI model (gpt-3.5-turbo or similar)
        chunking_model = "gpt-3.5-turbo"  # Free tier model
        
        # Create prompt for intelligent chunking
        # Process text in sections if too long
        text_section = text[:10000] if len(text) > 10000 else text
        
        chunking_prompt = f"""Split the following text into semantic chunks. 
Each chunk should be:
- Between 500-1500 characters
- Complete sentences (don't break mid-sentence)
- Semantically coherent (keep related ideas together)
- Have minimal overlap with adjacent chunks

Return a JSON array where each element is a chunk with "text" and "start_index" fields.

Text to chunk:
{text_section}"""

        # Call Azure AI Foundry using REST API
        # Azure AI Foundry uses OpenAI-compatible API
        endpoint = f"{config.azure_ai_foundry_endpoint}/openai/deployments/{chunking_model}/chat/completions?api-version=2024-02-15-preview"
        
        headers = {
            "Content-Type": "application/json",
            "api-key": config.azure_ai_foundry_api_key
        }
        
        payload = {
            "messages": [
                {"role": "system", "content": "You are a text chunking assistant. Return only valid JSON."},
                {"role": "user", "content": chunking_prompt}
            ],
            "temperature": 0.1,  # Low temperature for deterministic chunking
            "max_tokens": 4000
        }
        
        response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        response_text = result["choices"][0]["message"]["content"]
        
        # Extract JSON from response (handle markdown code blocks if present)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        chunk_data = json.loads(response_text)
        
        # Convert to Chunk objects
        chunks = []
        for idx, chunk_info in enumerate(chunk_data):
            chunks.append(Chunk(
                text=chunk_info.get("text", ""),
                chunk_id=f"chunk_{idx}",
                document_id=document_id,
                metadata={
                    "start_index": chunk_info.get("start_index", 0),
                    "chunking_method": "azure_ai_foundry_llm"
                }
            ))
        
        logger.info(f"Created {len(chunks)} chunks using Azure AI Foundry")
        return chunks
        
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM chunking response as JSON, falling back to fixed-size chunking: {e}")
        # Fallback to fixed-size chunking if LLM response is invalid
        return chunk_text_fixed_size(text, document_id)
    except Exception as e:
        logger.warning(f"LLM chunking failed, falling back to fixed-size chunking: {e}")
        # Fallback to fixed-size chunking
        return chunk_text_fixed_size(text, document_id)


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
    config, 
    document_id: Optional[str] = None, 
    use_llm: bool = False,
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[Chunk]:
    """
    Chunk text using either LLM-based intelligent chunking or fixed-size algorithm.
    
    This is the main entry point for text chunking in the RAG pipeline. It delegates
    to either chunk_text_with_llm() or chunk_text_fixed_size() based on the use_llm
    parameter.
    
    **Default Behavior**: By default (use_llm=False), uses fixed-size chunking which
    is deterministic and recommended for reproducible testing.
    
    **When to Use Each Method**:
    - Fixed-size (use_llm=False): Recommended for most use cases. Deterministic,
      fast, no external dependencies. Good for testing and reproducible pipelines.
    - LLM-based (use_llm=True): Use when semantic coherence is critical and you can
      accept non-deterministic results. Automatically falls back to fixed-size on errors.
    
    Args:
        text: Text to chunk
        config: Application configuration (required for LLM chunking, ignored for fixed-size)
        document_id: Optional document ID for chunk metadata
        use_llm: If True, use Azure AI Foundry for intelligent chunking.
                 If False, use fixed-size chunking (default: False, recommended)
        chunk_size: Size of each chunk in characters (only used if use_llm=False, default: 1000)
        overlap: Overlap between chunks in characters (only used if use_llm=False, default: 200)
        
    Returns:
        List of Chunk objects. Chunk metadata indicates the chunking method used.
        
    Note:
        - Default behavior (use_llm=False) is deterministic
        - LLM chunking (use_llm=True) is not deterministic but has automatic fallback
        - chunk_size and overlap parameters are ignored when use_llm=True
    """
    if use_llm:
        return chunk_text_with_llm(text, config, document_id)
    else:
        return chunk_text_fixed_size(text, document_id, chunk_size, overlap)


if __name__ == "__main__":
    """Test chunking functions directly
    
    Run this as: python -m rag_eval.services.rag.chunking
    (from the backend directory)
    
    Or: python rag_eval/services/rag/chunking.py
    (from the backend directory - requires pythonpath configured)
    """
    import sys
    import logging
    
    # Disable logging for direct testing
    logging.disable(logging.CRITICAL)
    
    print("Testing chunk_text_fixed_size()...")
    
    # Test 1: Basic chunking
    text = "This is a test document. " * 50  # ~1200 characters
    chunks = chunk_text_fixed_size(text, document_id="doc_123", chunk_size=100, overlap=20)
    print(f"✓ Test 1: Created {len(chunks)} chunks")
    assert len(chunks) > 0
    
    # Test 2: Deterministic behavior
    chunks1 = chunk_text_fixed_size(text, document_id="doc_123", chunk_size=100, overlap=20)
    chunks2 = chunk_text_fixed_size(text, document_id="doc_123", chunk_size=100, overlap=20)
    assert len(chunks1) == len(chunks2)
    assert all(c1.text == c2.text for c1, c2 in zip(chunks1, chunks2))
    print("✓ Test 2: Deterministic behavior verified")
    
    # Test 3: Empty document
    chunks = chunk_text_fixed_size("", document_id="doc_123")
    assert len(chunks) == 0
    print("✓ Test 3: Empty document handled")
    
    # Test 4: Small text
    chunks = chunk_text_fixed_size("Short text", chunk_size=1000, overlap=200)
    assert len(chunks) == 1
    print("✓ Test 4: Small text handled")
    
    # Test 5: chunk_text() default behavior
    chunks = chunk_text(text, None, document_id="doc_123", use_llm=False)
    assert len(chunks) > 0
    print("✓ Test 5: chunk_text() default behavior works")
    
    print("\n✅ All direct tests passed!")
    sys.exit(0)

