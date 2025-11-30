"""Azure AI Search retrieval"""

import time
import json
from typing import List, Optional
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    SearchField,
)
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError

from rag_eval.core.interfaces import Query, RetrievalResult, Chunk
from rag_eval.core.exceptions import AzureServiceError
from rag_eval.core.logging import get_logger
from rag_eval.services.rag.embeddings import generate_query_embedding

logger = get_logger("services.rag.search")


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
                    f"Azure AI Search operation failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                time.sleep(delay)
            else:
                logger.error(f"Azure AI Search operation failed after {max_retries + 1} attempts: {e}")
                raise AzureServiceError(
                    f"Azure AI Search operation failed after {max_retries + 1} attempts: {str(e)}"
                ) from e
    
    # This should never be reached, but included for type safety
    raise AzureServiceError(f"Unexpected error in retry logic: {last_exception}") from last_exception


def _ensure_index_exists(config) -> None:
    """
    Ensure Azure AI Search index exists, creating it if needed (idempotent).
    
    This function:
    - Checks if index exists before creation
    - Creates index with lightweight schema if not exists
    - Never performs destructive resets (preserves existing data)
    - Handles index creation errors gracefully
    
    Index Schema:
    - `id`: Chunk ID (string, key)
    - `chunk_text`: Chunk content (string, searchable)
    - `embedding`: Vector embedding (Collection(Edm.Single), vectorizable)
    - `document_id`: Source document ID (string)
    - `metadata`: Additional metadata (JSON string)
    
    Args:
        config: Application configuration with Azure AI Search credentials
        
    Raises:
        AzureServiceError: If index creation fails
        ValueError: If config is missing required fields
    """
    # Validate configuration
    if not config.azure_search_endpoint:
        raise ValueError("Azure AI Search endpoint is not configured")
    
    if not config.azure_search_api_key:
        raise ValueError("Azure AI Search API key is not configured")
    
    if not config.azure_search_index_name:
        raise ValueError("Azure AI Search index name is not configured")
    
    index_name = config.azure_search_index_name
    
    try:
        # Initialize index client
        index_client = SearchIndexClient(
            endpoint=config.azure_search_endpoint,
            credential=AzureKeyCredential(config.azure_search_api_key)
        )
        
        # Check if index exists
        try:
            existing_index = index_client.get_index(index_name)
            logger.info(f"Index '{index_name}' already exists, skipping creation")
            return
        except ResourceNotFoundError:
            # Index doesn't exist, create it
            logger.info(f"Index '{index_name}' does not exist, creating...")
        
        # Determine embedding dimension (default to 1536 for text-embedding-3-small)
        # This should match the embedding model used
        embedding_dimension = 1536  # Default for text-embedding-3-small
        
        # Create index with lightweight schema
        index_definition = SearchIndex(
            name=index_name,
            fields=[
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchField(
                    name="chunk_text",
                    type=SearchFieldDataType.String,
                    searchable=True,
                    retrievable=True
                ),
                SearchField(
                    name="embedding",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    vector_search_dimensions=embedding_dimension,
                    vector_search_profile_name="default-vector-profile"
                ),
                SimpleField(name="document_id", type=SearchFieldDataType.String, retrievable=True, filterable=True),
                SimpleField(name="metadata", type=SearchFieldDataType.String, retrievable=True)
            ],
            vector_search=VectorSearch(
                profiles=[
                    VectorSearchProfile(
                        name="default-vector-profile",
                        algorithm_configuration_name="default-algorithm-config"
                    )
                ],
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="default-algorithm-config",
                        kind=VectorSearchAlgorithmKind.HNSW,
                        parameters={
                            "m": 4,
                            "efConstruction": 400,
                            "efSearch": 500,
                            "metric": VectorSearchAlgorithmMetric.COSINE
                        }
                    )
                ]
            )
        )
        
        def create_index():
            index_client.create_index(index_definition)
            return None
        
        _retry_with_backoff(create_index)
        logger.info(f"Successfully created index '{index_name}'")
        
    except AzureServiceError:
        # Re-raise AzureServiceError as-is
        raise
    except ValueError:
        # Re-raise ValueError as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error ensuring index exists: {e}", exc_info=True)
        raise AzureServiceError(
            f"Unexpected error ensuring index exists: {str(e)}"
        ) from e


def index_chunks(chunks: List[Chunk], embeddings: List[List[float]], config) -> None:
    """
    Index chunks and embeddings into Azure AI Search.
    
    This function:
    - Ensures index exists (creates if needed, idempotent)
    - Batch indexes chunks with embeddings and metadata
    - Validates chunks and embeddings match in length
    - Handles empty chunk list gracefully
    - Implements retry logic with exponential backoff (3 retries max)
    - Uses the embedding model specified in config
    
    Args:
        chunks: List of chunks to index
        embeddings: Corresponding embeddings (one per chunk)
        config: Application configuration with Azure AI Search credentials
        
    Raises:
        AzureServiceError: If indexing fails
        ValueError: If chunks and embeddings length mismatch
        ValueError: If config is missing required fields
    """
    if not chunks:
        logger.info("Empty chunk list provided, skipping indexing")
        return
    
    # Validate chunks and embeddings match in length
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunks and embeddings length mismatch: {len(chunks)} chunks, {len(embeddings)} embeddings"
        )
    
    # Validate configuration
    if not config.azure_search_endpoint:
        raise ValueError("Azure AI Search endpoint is not configured")
    
    if not config.azure_search_api_key:
        raise ValueError("Azure AI Search API key is not configured")
    
    if not config.azure_search_index_name:
        raise ValueError("Azure AI Search index name is not configured")
    
    logger.info(f"Indexing {len(chunks)} chunks into Azure AI Search")
    
    try:
        # Ensure index exists (idempotent)
        _ensure_index_exists(config)
        
        # Initialize search client
        search_client = SearchClient(
            endpoint=config.azure_search_endpoint,
            index_name=config.azure_search_index_name,
            credential=AzureKeyCredential(config.azure_search_api_key)
        )
        
        # Prepare documents for indexing
        documents = []
        for chunk, embedding in zip(chunks, embeddings):
            # Serialize metadata to JSON string if present
            metadata_str = None
            if chunk.metadata:
                metadata_str = json.dumps(chunk.metadata)
            
            doc = {
                "id": chunk.chunk_id,
                "chunk_text": chunk.text,
                "embedding": embedding,
                "document_id": chunk.document_id or "",
                "metadata": metadata_str or ""
            }
            documents.append(doc)
        
        # Batch index documents with retry logic
        def upload_documents():
            result = search_client.upload_documents(documents=documents)
            # Check for failures
            failed = [r for r in result if not r.succeeded]
            if failed:
                error_messages = [f"Document {r.key}: {r.error_message}" for r in failed]
                raise AzureServiceError(
                    f"Failed to index {len(failed)} documents: {', '.join(error_messages)}"
                )
            return None
        
        _retry_with_backoff(upload_documents)
        logger.info(f"Successfully indexed {len(chunks)} chunks")
        
    except AzureServiceError:
        # Re-raise AzureServiceError as-is
        raise
    except ValueError:
        # Re-raise ValueError as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error indexing chunks: {e}", exc_info=True)
        raise AzureServiceError(
            f"Unexpected error indexing chunks: {str(e)}"
        ) from e


def retrieve_chunks(query: Query, top_k: int = 5, config=None) -> List[RetrievalResult]:
    """
    Retrieve top-k chunks for a query from Azure AI Search using vector similarity search.
    
    This function:
    - Generates query embedding (uses Phase 3 implementation)
    - Performs vector similarity search (cosine similarity)
    - Retrieves top-k chunks with similarity scores
    - Returns RetrievalResult objects with metadata
    - Handles empty index gracefully (returns empty list)
    - Handles query validation errors
    - Implements retry logic with exponential backoff (3 retries max)
    
    **Reproducibility**: Same query with same index produces identical results.
    Vector similarity search is deterministic given the same embeddings.
    
    Args:
        query: Query object with text to search for
        top_k: Number of chunks to retrieve (default: 5)
        config: Application configuration with Azure AI Search credentials
        
    Returns:
        List of RetrievalResult objects, sorted by similarity score (descending).
        Empty list if index is empty or query is invalid.
        
    Raises:
        AzureServiceError: If retrieval fails
        ValueError: If query text is empty
        ValueError: If config is missing required fields
    """
    if not query.text or not query.text.strip():
        raise ValueError("Query text cannot be empty")
    
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    
    # Validate configuration
    if not config:
        raise ValueError("Config is required")
    
    if not config.azure_search_endpoint:
        raise ValueError("Azure AI Search endpoint is not configured")
    
    if not config.azure_search_api_key:
        raise ValueError("Azure AI Search API key is not configured")
    
    if not config.azure_search_index_name:
        raise ValueError("Azure AI Search index name is not configured")
    
    logger.info(f"Retrieving top-{top_k} chunks for query: {query.text}")
    
    try:
        # Generate query embedding (uses same model as chunks for consistency)
        query_embedding = generate_query_embedding(query, config)
        
        # Initialize search client
        search_client = SearchClient(
            endpoint=config.azure_search_endpoint,
            index_name=config.azure_search_index_name,
            credential=AzureKeyCredential(config.azure_search_api_key)
        )
        
        # Perform vector similarity search
        def search_documents():
            # Use dictionary format for vector query (compatible with SDK 11.4.0)
            vector_query = {
                "kind": "vector",
                "vector": query_embedding,
                "k_nearest_neighbors": top_k,
                "fields": "embedding"
            }
            
            results = search_client.search(
                search_text=None,  # Vector-only search
                vector_queries=[vector_query],
                select=["id", "chunk_text", "document_id", "metadata"],
                top=top_k
            )
            return list(results)
        
        try:
            search_results = _retry_with_backoff(search_documents)
        except ResourceNotFoundError:
            # Index doesn't exist, return empty list (graceful handling)
            logger.warning(f"Index '{config.azure_search_index_name}' does not exist, returning empty results")
            return []
        
        # Convert search results to RetrievalResult objects
        retrieval_results = []
        for result in search_results:
            # Parse metadata from JSON string if present
            metadata = None
            if result.get("metadata"):
                try:
                    metadata = json.loads(result["metadata"])
                except (json.JSONDecodeError, TypeError):
                    # If metadata is not valid JSON, use as-is or empty dict
                    metadata = {}
            
            retrieval_result = RetrievalResult(
                chunk_id=result["id"],
                similarity_score=result.get("@search.score", 0.0),
                chunk_text=result.get("chunk_text", ""),
                metadata=metadata
            )
            retrieval_results.append(retrieval_result)
        
        logger.info(f"Retrieved {len(retrieval_results)} chunks for query")
        return retrieval_results
        
    except AzureServiceError:
        # Re-raise AzureServiceError as-is
        raise
    except ValueError:
        # Re-raise ValueError as-is
        raise
    except ResourceNotFoundError:
        # Index doesn't exist, return empty list (graceful handling)
        logger.warning(f"Index '{config.azure_search_index_name}' does not exist, returning empty results")
        return []
    except Exception as e:
        logger.error(f"Unexpected error retrieving chunks: {e}", exc_info=True)
        raise AzureServiceError(
            f"Unexpected error retrieving chunks: {str(e)}"
        ) from e


def delete_chunks_by_document_id(document_id: str, config) -> int:
    """
    Delete all chunks for a document from Azure AI Search.
    
    This function:
    - Searches for all chunks with the given document_id
    - Deletes all matching chunks from the index
    - Returns the number of chunks deleted
    - Implements retry logic with exponential backoff (3 retries max)
    
    Args:
        document_id: Document identifier to delete chunks for
        config: Application configuration with Azure AI Search credentials
        
    Returns:
        Number of chunks deleted
        
    Raises:
        AzureServiceError: If deletion fails
        ValueError: If document_id is empty or config is missing required fields
    """
    if not document_id or not document_id.strip():
        raise ValueError("document_id cannot be empty")
    
    # Validate configuration
    if not config.azure_search_endpoint:
        raise ValueError("Azure AI Search endpoint is not configured")
    
    if not config.azure_search_api_key:
        raise ValueError("Azure AI Search API key is not configured")
    
    if not config.azure_search_index_name:
        raise ValueError("Azure AI Search index name is not configured")
    
    logger.info(f"Deleting chunks for document '{document_id}' from Azure AI Search")
    
    try:
        # Initialize search client
        search_client = SearchClient(
            endpoint=config.azure_search_endpoint,
            index_name=config.azure_search_index_name,
            credential=AzureKeyCredential(config.azure_search_api_key)
        )
        
        # First, find all chunks for this document
        # Try to use filter if document_id is filterable, otherwise search all and filter
        def find_chunks():
            try:
                # Try filtering by document_id (if field is filterable)
                results = search_client.search(
                    search_text="*",
                    filter=f"document_id eq '{document_id}'",
                    select=["id", "document_id"],
                    top=10000
                )
                return list(results)
            except Exception as filter_error:
                # If filtering fails (field not filterable), fall back to searching all and filtering
                logger.warning(f"Could not filter by document_id, searching all chunks: {filter_error}")
                results = search_client.search(
                    search_text="*",
                    select=["id", "document_id"],
                    top=10000  # Large limit to get all chunks
                )
                chunks = [r for r in results if r.get("document_id") == document_id]
                return chunks
        
        try:
            matching_chunks = _retry_with_backoff(find_chunks)
        except ResourceNotFoundError:
            # Index doesn't exist, return 0 (graceful handling)
            logger.warning(f"Index '{config.azure_search_index_name}' does not exist, no chunks to delete")
            return 0
        
        if not matching_chunks:
            logger.info(f"No chunks found for document '{document_id}'")
            return 0
        
        chunk_ids = [chunk["id"] for chunk in matching_chunks]
        logger.info(f"Found {len(chunk_ids)} chunks to delete for document '{document_id}'")
        
        # Delete chunks by ID
        def delete_chunks():
            # Prepare documents for deletion (only need ID field)
            documents_to_delete = [{"id": chunk_id} for chunk_id in chunk_ids]
            result = search_client.delete_documents(documents=documents_to_delete)
            
            # Check for failures
            failed = [r for r in result if not r.succeeded]
            if failed:
                error_messages = [f"Chunk {r.key}: {r.error_message}" for r in failed]
                raise AzureServiceError(
                    f"Failed to delete {len(failed)} chunks: {', '.join(error_messages)}"
                )
            
            return len(chunk_ids) - len(failed)
        
        deleted_count = _retry_with_backoff(delete_chunks)
        logger.info(f"Successfully deleted {deleted_count} chunks for document '{document_id}'")
        return deleted_count
        
    except AzureServiceError:
        # Re-raise AzureServiceError as-is
        raise
    except ValueError:
        # Re-raise ValueError as-is
        raise
    except ResourceNotFoundError:
        # Index doesn't exist, return 0 (graceful handling)
        logger.warning(f"Index '{config.azure_search_index_name}' does not exist, no chunks to delete")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error deleting chunks: {e}", exc_info=True)
        raise AzureServiceError(
            f"Unexpected error deleting chunks: {str(e)}"
        ) from e

