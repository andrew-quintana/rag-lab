"""Query Generator for In-Corpus Evaluation Dataset Generation

This module generates diverse, effective queries from Azure AI Search indexed documents
using LLM-based query generation from sampled chunks.
"""

import json
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError

from rag_eval.core.config import Config
from rag_eval.core.interfaces import Chunk, Query
from rag_eval.core.exceptions import AzureServiceError
from rag_eval.core.logging import get_logger
from rag_eval.services.shared.llm_providers import get_llm_provider, LLMProvider

logger = get_logger("evaluations._shared.scripts.query_generator")


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
        except ResourceNotFoundError:
            # Don't retry ResourceNotFoundError - it's not a transient error
            raise
        except (HttpResponseError, Exception) as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(
                    f"Operation failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                time.sleep(delay)
            else:
                logger.error(f"Operation failed after {max_retries + 1} attempts: {e}")
                raise AzureServiceError(
                    f"Operation failed after {max_retries + 1} attempts: {str(e)}"
                ) from e
    
    # This should never be reached, but included for type safety
    raise AzureServiceError(f"Unexpected error in retry logic: {last_exception}") from last_exception


def sample_chunks_from_index(
    config: Config,
    num_chunks: int = 50,
    max_attempts: int = 10000
) -> List[Chunk]:
    """
    Sample diverse chunks from Azure AI Search index.
    
    This function samples chunks by:
    1. Searching for all chunks in the index (using search_text="*")
    2. Randomly selecting chunks to ensure diversity
    3. Converting search results to Chunk objects
    
    Args:
        config: Application configuration with Azure AI Search credentials
        num_chunks: Number of chunks to sample (default: 50)
        max_attempts: Maximum number of chunks to retrieve before sampling (default: 10000)
        
    Returns:
        List of Chunk objects sampled from the index
        
    Raises:
        AzureServiceError: If sampling fails
        ValueError: If config is missing required fields
    """
    # Validate configuration
    if not config.azure_search_endpoint:
        raise ValueError("Azure AI Search endpoint is not configured")
    
    if not config.azure_search_api_key:
        raise ValueError("Azure AI Search API key is not configured")
    
    if not config.azure_search_index_name:
        raise ValueError("Azure AI Search index name is not configured")
    
    if num_chunks <= 0:
        raise ValueError(f"num_chunks must be positive, got {num_chunks}")
    
    logger.info(f"Sampling {num_chunks} chunks from Azure AI Search index")
    
    try:
        # Initialize search client
        search_client = SearchClient(
            endpoint=config.azure_search_endpoint,
            index_name=config.azure_search_index_name,
            credential=AzureKeyCredential(config.azure_search_api_key)
        )
        
        # Search for all chunks (or up to max_attempts)
        def search_all_chunks():
            results = search_client.search(
                search_text="*",
                select=["id", "chunk_text", "document_id", "metadata"],
                top=min(max_attempts, 10000)  # Azure AI Search has a limit
            )
            return list(results)
        
        try:
            all_results = _retry_with_backoff(search_all_chunks)
        except ResourceNotFoundError:
            logger.warning(f"Index '{config.azure_search_index_name}' does not exist, returning empty list")
            return []
        
        if not all_results:
            logger.warning("No chunks found in index")
            return []
        
        # Randomly sample chunks for diversity
        num_to_sample = min(num_chunks, len(all_results))
        sampled_results = random.sample(all_results, num_to_sample)
        
        # Convert to Chunk objects
        chunks = []
        for result in sampled_results:
            # Parse metadata from JSON string if present
            metadata = None
            if result.get("metadata"):
                try:
                    metadata = json.loads(result["metadata"])
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
            
            chunk = Chunk(
                text=result.get("chunk_text", ""),
                chunk_id=result["id"],
                document_id=result.get("document_id"),
                metadata=metadata
            )
            chunks.append(chunk)
        
        logger.info(f"Successfully sampled {len(chunks)} chunks from index")
        return chunks
        
    except AzureServiceError:
        raise
    except ValueError:
        raise
    except ResourceNotFoundError:
        logger.warning(f"Index '{config.azure_search_index_name}' does not exist, returning empty list")
        return []
    except Exception as e:
        logger.error(f"Unexpected error sampling chunks: {e}", exc_info=True)
        raise AzureServiceError(
            f"Unexpected error sampling chunks: {str(e)}"
        ) from e


def generate_query_from_chunk(
    chunk: Chunk,
    llm_provider: LLMProvider,
    temperature: float = 0.7
) -> str:
    """
    Generate a query from a chunk using LLM.
    
    This function uses an LLM to generate a diverse, effective query that is
    answerable from the chunk content (In-Corpus).
    
    Args:
        chunk: Chunk object to generate query from
        llm_provider: LLM provider instance for query generation
        temperature: Generation temperature for diversity (default: 0.7)
        
    Returns:
        Generated query string
        
    Raises:
        AzureServiceError: If LLM generation fails
        ValueError: If chunk text is empty
    """
    if not chunk.text or not chunk.text.strip():
        raise ValueError("Chunk text cannot be empty")
    
    # Construct prompt for query generation
    prompt = f"""Generate a diverse, effective question that can be answered using the following document chunk.

The question should be:
1. Answerable from the chunk content (In-Corpus)
2. Specific and clear
3. Natural and conversational
4. Effective for evaluating RAG systems

Document chunk:
{chunk.text[:2000]}  # Limit chunk text to avoid token limits

Generate only the question, without any additional explanation or formatting."""

    try:
        query = llm_provider.call_completion(
            prompt=prompt,
            temperature=temperature,
            max_tokens=100
        )
        
        # Clean up the query (remove quotes, extra whitespace, etc.)
        query = query.strip()
        # Remove leading/trailing quotes if present
        if query.startswith('"') and query.endswith('"'):
            query = query[1:-1]
        elif query.startswith("'") and query.endswith("'"):
            query = query[1:-1]
        
        logger.debug(f"Generated query from chunk {chunk.chunk_id}: {query[:100]}...")
        return query
        
    except Exception as e:
        logger.error(f"Failed to generate query from chunk {chunk.chunk_id}: {e}", exc_info=True)
        raise AzureServiceError(
            f"Failed to generate query from chunk: {str(e)}"
        ) from e


def generate_queries_from_index(
    config: Config,
    num_queries: int = 10,
    sample_chunks: Optional[List[Chunk]] = None,
    chunks_per_query: int = 1
) -> List[Dict[str, Any]]:
    """
    Generate diverse queries from Azure AI Search index.
    
    This function:
    1. Samples chunks from the index (or uses provided chunks)
    2. Generates queries from chunks using LLM
    3. Collects metadata linking queries to source chunks
    4. Returns list of query dictionaries with input and metadata
    
    Args:
        config: Application configuration
        num_queries: Number of queries to generate (default: 10)
        sample_chunks: Optional list of chunks to use as seed (if None, samples from index)
        chunks_per_query: Number of chunks to use per query (default: 1)
        
    Returns:
        List of query dictionaries with input and metadata:
        [
            {
                "input": "string - query for RAG evaluation",
                "metadata": {
                    "source_chunk_ids": ["chunk_1", "chunk_2"],
                    "document_id": "doc_123"
                }
            }
        ]
        
    Raises:
        AzureServiceError: If query generation fails
        ValueError: If config is missing required fields or parameters are invalid
    """
    if num_queries <= 0:
        raise ValueError(f"num_queries must be positive, got {num_queries}")
    
    if chunks_per_query <= 0:
        raise ValueError(f"chunks_per_query must be positive, got {chunks_per_query}")
    
    logger.info(f"Generating {num_queries} queries from Azure AI Search index")
    
    # Step 1: Sample chunks if not provided
    if sample_chunks is None:
        # Sample more chunks than needed to ensure diversity
        chunks_to_sample = max(num_queries * chunks_per_query, 50)
        sample_chunks = sample_chunks_from_index(config, num_chunks=chunks_to_sample)
        
        if not sample_chunks:
            raise AzureServiceError("No chunks found in index. Cannot generate queries.")
    
    if len(sample_chunks) < chunks_per_query:
        raise ValueError(
            f"Not enough chunks provided: need at least {chunks_per_query}, got {len(sample_chunks)}"
        )
    
    # Step 2: Initialize LLM provider
    try:
        llm_provider = get_llm_provider(config)
    except Exception as e:
        logger.error(f"Failed to initialize LLM provider: {e}", exc_info=True)
        raise AzureServiceError(f"Failed to initialize LLM provider: {str(e)}") from e
    
    # Step 3: Generate queries from chunks
    queries = []
    used_chunk_indices = set()
    
    for i in range(num_queries):
        # Select chunks for this query (avoid reusing chunks if possible)
        available_indices = [idx for idx in range(len(sample_chunks)) if idx not in used_chunk_indices]
        
        if not available_indices:
            # If we've used all chunks, reset and allow reuse
            logger.warning("All chunks have been used, reusing chunks for remaining queries")
            available_indices = list(range(len(sample_chunks)))
        
        # Select random chunks for this query
        selected_indices = random.sample(
            available_indices,
            min(chunks_per_query, len(available_indices))
        )
        selected_chunks = [sample_chunks[idx] for idx in selected_indices]
        
        # Mark chunks as used
        used_chunk_indices.update(selected_indices)
        
        # Generate query from primary chunk (first chunk)
        primary_chunk = selected_chunks[0]
        try:
            query_text = generate_query_from_chunk(primary_chunk, llm_provider)
        except Exception as e:
            logger.error(f"Failed to generate query {i+1}/{num_queries}: {e}", exc_info=True)
            # Continue with next query instead of failing completely
            continue
        
        # Collect metadata
        source_chunk_ids = [chunk.chunk_id for chunk in selected_chunks]
        document_ids = [chunk.document_id for chunk in selected_chunks if chunk.document_id]
        # Use first document_id if available, otherwise None
        document_id = document_ids[0] if document_ids else None
        
        query_dict = {
            "input": query_text,
            "metadata": {
                "source_chunk_ids": source_chunk_ids,
                "document_id": document_id
            }
        }
        
        queries.append(query_dict)
        logger.info(f"Generated query {i+1}/{num_queries}: {query_text[:100]}...")
    
    if not queries:
        raise AzureServiceError("Failed to generate any queries")
    
    logger.info(f"Successfully generated {len(queries)} queries")
    return queries


def save_queries_to_json(
    queries: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """
    Save queries to eval_inputs.json file.
    
    Args:
        queries: List of query dictionaries with input and metadata
        output_path: Path to save eval_inputs.json
        
    Raises:
        IOError: If file writing fails
        ValueError: If queries list is empty
    """
    if not queries:
        raise ValueError("Cannot save empty queries list")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write queries to JSON file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(queries, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully saved {len(queries)} queries to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save queries to {output_path}: {e}", exc_info=True)
        raise IOError(f"Failed to save queries to {output_path}: {str(e)}") from e


def main(
    eval_name: str = "in_corpus_eval",
    num_queries: int = 10,
    config: Optional[Config] = None
) -> None:
    """
    Main function to generate queries and save to eval_inputs.json.
    
    Args:
        eval_name: Name of evaluation (default: "in_corpus_eval")
        num_queries: Number of queries to generate (default: 10)
        config: Optional Config instance (if None, loads from environment)
        
    Raises:
        AzureServiceError: If query generation fails
        IOError: If file writing fails
    """
    # Load config if not provided
    if config is None:
        config = Config.from_env()
    
    # Generate queries
    queries = generate_queries_from_index(config, num_queries=num_queries)
    
    # Save to eval_inputs.json
    # Output directory is inputs/ subdirectory of the evaluation directory
    eval_root = Path(__file__).parent.parent.parent
    eval_dir = eval_root / eval_name / "inputs"
    output_path = eval_dir / "eval_inputs.json"
    save_queries_to_json(queries, output_path)
    
    logger.info(f"Query generation complete. Saved {len(queries)} queries to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate queries for in-corpus evaluation")
    parser.add_argument(
        "--eval-name",
        type=str,
        default="in_corpus_eval",
        help="Name of evaluation (default: in_corpus_eval)"
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=10,
        help="Number of queries to generate (default: 10)"
    )
    
    args = parser.parse_args()
    
    main(eval_name=args.eval_name, num_queries=args.num_queries)

