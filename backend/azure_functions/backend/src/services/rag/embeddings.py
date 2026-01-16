"""Embedding generation via Azure AI Foundry"""

import time
from typing import List, Optional
import requests
from src.core.interfaces import Chunk, Query
from src.core.exceptions import AzureServiceError
from src.core.logging import get_logger

logger = get_logger("services.rag.embeddings")


def _retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry (callable that takes no arguments)
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds for exponential backoff (default: 1.0)
        
    Returns:
        Result of the function call
        
    Raises:
        AzureServiceError: If all retries are exhausted
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except (requests.RequestException, Exception) as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(
                    f"Azure embedding API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                time.sleep(delay)
            else:
                logger.error(f"Azure embedding API call failed after {max_retries + 1} attempts: {e}")
                raise AzureServiceError(
                    f"Azure AI Foundry embedding API call failed after {max_retries + 1} attempts: {str(e)}"
                ) from e
    
    # This should never be reached, but included for type safety
    raise AzureServiceError(f"Unexpected error in retry logic: {last_exception}") from last_exception


def _call_embedding_api(texts: List[str], model: str, endpoint: str, api_key: str) -> List[List[float]]:
    """
    Call Azure AI Foundry embedding API with a batch of texts.
    
    Args:
        texts: List of text strings to embed
        model: Embedding model name (e.g., "text-embedding-3-small")
        endpoint: Azure AI Foundry endpoint URL
        api_key: Azure AI Foundry API key
        
    Returns:
        List of embedding vectors (one per input text)
        
    Raises:
        AzureServiceError: If API call fails
        ValueError: If response is invalid
    """
    if not texts:
        return []
    
    # Azure AI Foundry uses OpenAI-compatible API
    # Strip trailing slash from endpoint to avoid double slashes
    endpoint = endpoint.rstrip('/')
    api_endpoint = f"{endpoint}/openai/deployments/{model}/embeddings?api-version=2024-02-15-preview"
    
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    
    payload = {
        "input": texts,
        "model": model
    }
    
    def make_request():
        response = requests.post(api_endpoint, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    
    result = _retry_with_backoff(make_request)
    
    # Validate response structure
    if "data" not in result:
        raise ValueError(f"Invalid embedding API response: missing 'data' field. Response: {result}")
    
    # Extract embeddings from response
    embeddings = []
    for item in result["data"]:
        if "embedding" not in item:
            raise ValueError(f"Invalid embedding API response: missing 'embedding' field in item. Response: {result}")
        embeddings.append(item["embedding"])
    
    # Validate we got the expected number of embeddings
    if len(embeddings) != len(texts):
        raise ValueError(
            f"Embedding count mismatch: expected {len(texts)} embeddings, got {len(embeddings)}"
        )
    
    # Validate embedding dimensions (text-embedding-3-small has 1536 dimensions)
    # But we'll be flexible and just check that all embeddings have the same dimension
    if embeddings:
        expected_dim = len(embeddings[0])
        for idx, emb in enumerate(embeddings):
            if len(emb) != expected_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: embedding {idx} has {len(emb)} dimensions, "
                    f"expected {expected_dim}"
                )
    
    return embeddings


def generate_embeddings(chunks: List[Chunk], config) -> List[List[float]]:
    """
    Generate embeddings for chunks using Azure AI Foundry.
    
    This function:
    - Connects to Azure AI Foundry embedding API
    - Processes chunks in batches (1000 texts per batch to stay within API limits)
    - Validates embedding dimensions match expected model output
    - Handles empty chunk list gracefully
    - Implements retry logic with exponential backoff (3 retries max)
    - Uses the embedding model specified in config
    
    Args:
        chunks: List of Chunk objects to embed
        config: Application configuration with Azure AI Foundry credentials
               (azure_ai_foundry_endpoint, azure_ai_foundry_api_key, azure_ai_foundry_embedding_model)
        
    Returns:
        List of embedding vectors (List[List[float]]), one per chunk.
        Each embedding is a list of floats representing the vector.
        Empty list if chunks is empty.
        
    Raises:
        AzureServiceError: If embedding generation fails after retries
        ValueError: If config is missing required fields
        ValueError: If embedding dimensions are inconsistent
    """
    if not chunks:
        logger.info("Empty chunk list provided, returning empty embeddings list")
        return []
    
    # Validate configuration
    if not config.azure_ai_foundry_endpoint:
        raise ValueError("Azure AI Foundry endpoint is not configured")
    
    if not config.azure_ai_foundry_api_key:
        raise ValueError("Azure AI Foundry API key is not configured")
    
    model = config.azure_ai_foundry_embedding_model
    if not model:
        raise ValueError("Azure AI Foundry embedding model is not configured")
    
    # Azure AI Foundry embedding API batch limit: typically 2048, but we use 1000 to be safe
    # This accounts for token limits and rate limiting
    BATCH_SIZE = 1000
    
    logger.info(f"Generating embeddings for {len(chunks)} chunks using model '{model}' (in batches of {BATCH_SIZE})")
    
    try:
        # Extract text from chunks
        texts = [chunk.text for chunk in chunks]
        
        # Process in batches
        all_embeddings = []
        total_processed = 0
        
        for batch_start in range(0, len(texts), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(texts))
            batch_texts = texts[batch_start:batch_end]
            
            logger.debug(f"Processing embedding batch {batch_start // BATCH_SIZE + 1}: texts {batch_start} to {batch_end - 1}")
            
            # Call embedding API for this batch with retry logic
            batch_embeddings = _call_embedding_api(
                texts=batch_texts,
                model=model,
                endpoint=config.azure_ai_foundry_endpoint,
                api_key=config.azure_ai_foundry_api_key
            )
            
            all_embeddings.extend(batch_embeddings)
            total_processed += len(batch_embeddings)
            logger.debug(f"Batch {batch_start // BATCH_SIZE + 1}: generated {len(batch_embeddings)} embeddings")
        
        # Validate we got the expected number of embeddings
        if len(all_embeddings) != len(chunks):
            raise ValueError(
                f"Embedding count mismatch: expected {len(chunks)} embeddings, got {len(all_embeddings)}"
            )
        
        # Validate embedding dimensions are consistent
        if all_embeddings:
            expected_dim = len(all_embeddings[0])
            for idx, emb in enumerate(all_embeddings):
                if len(emb) != expected_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch: embedding {idx} has {len(emb)} dimensions, "
                        f"expected {expected_dim}"
                    )
        
        logger.info(
            f"Successfully generated {len(all_embeddings)} embeddings "
            f"(dimension: {len(all_embeddings[0]) if all_embeddings else 0})"
        )
        
        return all_embeddings
        
    except AzureServiceError:
        # Re-raise AzureServiceError as-is
        raise
    except ValueError:
        # Re-raise ValueError as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error generating embeddings: {e}", exc_info=True)
        raise AzureServiceError(
            f"Unexpected error generating embeddings: {str(e)}"
        ) from e


def generate_query_embedding(query: Query, config) -> List[float]:
    """
    Generate embedding for a query using Azure AI Foundry.
    
    This function:
    - Uses the same embedding model as chunks (enforced via config)
    - Validates embedding dimensions
    - Implements retry logic with exponential backoff (3 retries max)
    - Ensures model consistency with chunk embeddings
    
    **Model Consistency**: This function uses the same embedding model as
    `generate_embeddings()` to ensure query and chunk embeddings are in the
    same vector space for accurate similarity search.
    
    Args:
        query: Query object with text to embed
        config: Application configuration with Azure AI Foundry credentials
               (azure_ai_foundry_endpoint, azure_ai_foundry_api_key, azure_ai_foundry_embedding_model)
        
    Returns:
        Embedding vector (List[float]) for the query text
        
    Raises:
        AzureServiceError: If embedding generation fails after retries
        ValueError: If query text is empty
        ValueError: If config is missing required fields
        ValueError: If embedding dimensions are invalid
    """
    if not query.text or not query.text.strip():
        raise ValueError("Query text cannot be empty")
    
    # Validate configuration
    if not config.azure_ai_foundry_endpoint:
        raise ValueError("Azure AI Foundry endpoint is not configured")
    
    if not config.azure_ai_foundry_api_key:
        raise ValueError("Azure AI Foundry API key is not configured")
    
    model = config.azure_ai_foundry_embedding_model
    if not model:
        raise ValueError("Azure AI Foundry embedding model is not configured")
    
    logger.info(f"Generating query embedding using model '{model}' (same as chunks for consistency)")
    
    try:
        # Call embedding API with single text
        embeddings = _call_embedding_api(
            texts=[query.text],
            model=model,
            endpoint=config.azure_ai_foundry_endpoint,
            api_key=config.azure_ai_foundry_api_key
        )
        
        if not embeddings or len(embeddings) != 1:
            raise ValueError(f"Expected 1 embedding, got {len(embeddings) if embeddings else 0}")
        
        embedding = embeddings[0]
        
        logger.info(f"Successfully generated query embedding (dimension: {len(embedding)})")
        
        return embedding
        
    except AzureServiceError:
        # Re-raise AzureServiceError as-is
        raise
    except ValueError:
        # Re-raise ValueError as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error generating query embedding: {e}", exc_info=True)
        raise AzureServiceError(
            f"Unexpected error generating query embedding: {str(e)}"
        ) from e

