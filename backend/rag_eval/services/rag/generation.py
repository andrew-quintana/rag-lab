"""Answer generation via Azure AI Foundry"""

from typing import List, Dict, Optional
from rag_eval.core.interfaces import Query, RetrievalResult, ModelAnswer
from rag_eval.core.exceptions import AzureServiceError, DatabaseError, ValidationError
from rag_eval.core.logging import get_logger
from rag_eval.db.queries import QueryExecutor

logger = get_logger("services.rag.generation")

# In-memory cache for prompt templates
# Cache key format: "{prompt_type}:{version_name}"
_prompt_cache: Dict[str, str] = {}


def load_prompt_template(
    version: str, 
    query_executor: QueryExecutor,
    prompt_type: str = "rag"
) -> str:
    """
    Load a prompt template from Supabase by version name and type.
    
    Templates are cached in memory to avoid repeated database queries.
    If a template is already cached, it is returned immediately without
    querying the database.
    
    Args:
        version: Prompt version name (e.g., "v1", "v2")
        query_executor: QueryExecutor instance for database operations
        prompt_type: Type of prompt (e.g., "rag", "evaluation", "summarization").
                    Defaults to "rag" for backward compatibility.
        
    Returns:
        Prompt template text from database
        
    Raises:
        ValidationError: If prompt version is not found in database
        DatabaseError: If database query fails
        
    Example:
        >>> query_executor = QueryExecutor(db_conn)
        >>> template = load_prompt_template("v1", query_executor)
        >>> print(template)
        "You are a helpful assistant... {query} ... {context}"
        
        >>> # Load a different type of prompt
        >>> eval_template = load_prompt_template("v1", query_executor, prompt_type="evaluation")
    """
    # Create cache key from prompt_type and version
    cache_key = f"{prompt_type}:{version}"
    
    # Check cache first
    if cache_key in _prompt_cache:
        logger.debug(f"Retrieved prompt template '{prompt_type}:{version}' from cache")
        return _prompt_cache[cache_key]
    
    # Query database for prompt template
    query = """
        SELECT prompt_text
        FROM prompt_versions
        WHERE prompt_type = %s AND version_name = %s
    """
    
    try:
        results = query_executor.execute_query(query, (prompt_type, version))
        
        if not results:
            error_msg = f"Prompt version '{version}' of type '{prompt_type}' not found in database"
            logger.error(error_msg)
            raise ValidationError(error_msg)
        
        prompt_text = results[0]["prompt_text"]
        
        # Cache the template
        _prompt_cache[cache_key] = prompt_text
        logger.info(f"Loaded and cached prompt template '{prompt_type}:{version}' from database")
        
        return prompt_text
        
    except ValidationError:
        raise
    except Exception as e:
        error_msg = f"Failed to load prompt template '{version}': {e}"
        logger.error(error_msg)
        raise DatabaseError(error_msg) from e


def construct_prompt(
    query: Query,
    retrieved_chunks: List[RetrievalResult],
    prompt_version: str,
    query_executor: QueryExecutor,
    prompt_type: str = "rag"
) -> str:
    """
    Construct a complete prompt by loading a template and replacing placeholders.
    
    The template must contain the following placeholders:
    - {query}: Replaced with the query text
    - {context}: Replaced with concatenated retrieved chunk text
    
    Args:
        query: Query object containing the user's question
        retrieved_chunks: List of retrieved chunks to use as context
        prompt_version: Version name of the prompt template to load
        query_executor: QueryExecutor instance for database operations
        prompt_type: Type of prompt (e.g., "rag", "evaluation", "summarization").
                    Defaults to "rag" for backward compatibility.
        
    Returns:
        Complete prompt string ready for LLM generation
        
    Raises:
        ValidationError: If template is missing required placeholders or prompt version not found
        DatabaseError: If database query fails
        ValueError: If query text is empty or invalid
        
    Example:
        >>> query = Query(text="What is the coverage limit?")
        >>> chunks = [RetrievalResult(chunk_id="1", similarity_score=0.9, chunk_text="Coverage limit is $500k")]
        >>> prompt = construct_prompt(query, chunks, "v1", query_executor)
        >>> print(prompt)
        "You are a helpful assistant... What is the coverage limit? ... Coverage limit is $500k"
        
        >>> # Use a different prompt type
        >>> eval_prompt = construct_prompt(query, chunks, "v1", query_executor, prompt_type="evaluation")
    """
    if not query.text or not query.text.strip():
        raise ValueError("Query text cannot be empty")
    
    # Load prompt template
    template = load_prompt_template(prompt_version, query_executor, prompt_type=prompt_type)
    
    # Validate template has required placeholders
    required_placeholders = ["{query}", "{context}"]
    missing_placeholders = [
        placeholder for placeholder in required_placeholders
        if placeholder not in template
    ]
    
    if missing_placeholders:
        error_msg = (
            f"Prompt template '{prompt_type}:{prompt_version}' is missing required placeholders: "
            f"{', '.join(missing_placeholders)}"
        )
        logger.error(error_msg)
        raise ValidationError(error_msg)
    
    # Construct context from retrieved chunks
    if retrieved_chunks:
        context_parts = [chunk.chunk_text for chunk in retrieved_chunks]
        context = "\n\n".join(context_parts)
    else:
        context = "(No context retrieved)"
        logger.warning("No retrieved chunks provided for prompt construction")
    
    # Replace placeholders
    try:
        prompt = template.replace("{query}", query.text)
        prompt = prompt.replace("{context}", context)
    except Exception as e:
        error_msg = f"Failed to format prompt template: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e
    
    logger.debug(f"Constructed prompt for query '{query.text[:50]}...' using template '{prompt_type}:{prompt_version}'")
    
    return prompt


def generate_answer(
    query: Query,
    retrieved_chunks: List[RetrievalResult],
    prompt_version: str,
    config
) -> ModelAnswer:
    """
    Generate an answer using Azure AI Foundry.
    
    Args:
        query: Query object
        retrieved_chunks: Retrieved context chunks
        prompt_version: Version of the prompt to use
        config: Application configuration
        
    Returns:
        ModelAnswer object
        
    Raises:
        AzureServiceError: If generation fails
    """
    logger.info(f"Generating answer for query: {query.text} using prompt version: {prompt_version}")
    # TODO: Implement Azure AI Foundry answer generation
    raise NotImplementedError("Answer generation not yet implemented")

