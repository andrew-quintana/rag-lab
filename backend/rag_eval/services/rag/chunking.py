"""Text chunking - supports both LLM-based and fixed-size chunking"""

from typing import List, Optional
import json
import requests
from rag_eval.core.interfaces import Chunk
from rag_eval.core.exceptions import AzureServiceError
from rag_eval.core.logging import get_logger

logger = get_logger("services.rag.chunking")


def chunk_text_with_llm(text: str, config, document_id: Optional[str] = None) -> List[Chunk]:
    """
    Chunk text intelligently using Azure AI Foundry (free tier OpenAI).
    Uses LLM to identify semantic boundaries for chunking.
    
    Args:
        text: Text to chunk
        config: Application configuration
        document_id: Optional document ID for chunk metadata
        
    Returns:
        List of Chunk objects
        
    Raises:
        AzureServiceError: If chunking fails
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
    
    Args:
        text: Text to chunk
        document_id: Optional document ID for chunk metadata
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of Chunk objects
    """
    logger.info(f"Chunking text of length {len(text)} with fixed-size algorithm (chunk_size={chunk_size}, overlap={overlap})")
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        chunks.append(Chunk(
            text=chunk_text,
            chunk_id=f"chunk_{chunk_id}",
            document_id=document_id,
            metadata={"start": start, "end": end, "chunking_method": "fixed_size"}
        ))
        start = end - overlap
        chunk_id += 1
    
    logger.info(f"Created {len(chunks)} chunks")
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
    
    Args:
        text: Text to chunk
        config: Application configuration
        document_id: Optional document ID for chunk metadata
        use_llm: If True, use Azure AI Foundry for intelligent chunking. 
                 If False, use fixed-size chunking (default: False)
        chunk_size: Size of each chunk in characters (only used if use_llm=False)
        overlap: Overlap between chunks in characters (only used if use_llm=False)
        
    Returns:
        List of Chunk objects
    """
    if use_llm:
        return chunk_text_with_llm(text, config, document_id)
    else:
        return chunk_text_fixed_size(text, document_id, chunk_size, overlap)

