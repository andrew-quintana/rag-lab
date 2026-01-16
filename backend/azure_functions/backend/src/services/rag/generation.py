"""Answer generation via Azure AI Foundry"""

import time
from typing import List, Dict, Optional
from datetime import datetime, timezone
import requests
from src.core.interfaces import Query, RetrievalResult, ModelAnswer
from src.core.exceptions import AzureServiceError, DatabaseError, ValidationError
from src.core.logging import get_logger
from src.core.config import Config
from src.db.queries import QueryExecutor
from src.utils.ids import generate_id

logger = get_logger("services.rag.generation")

# In-memory cache for prompt templates
# Cache key format: "{prompt_type}:{evaluator_type}:{version_name}" (when evaluator_type provided)
# or "{prompt_type}:{version_name}" (when evaluator_type is None)
_prompt_cache: Dict[str, str] = {}


def load_prompt_template(
    version: Optional[str] = None,
    query_executor: Optional[QueryExecutor] = None,
    prompt_type: str = "rag",
    name: Optional[str] = None,
    live: bool = True
) -> str:
    """
    Load a prompt template from Supabase by type, name, and optionally version or live flag.
    
    Templates are cached in memory to avoid repeated database queries.
    If a template is already cached, it is returned immediately without
    querying the database.
    
    Args:
        version: Optional prompt version (e.g., "v0.1", "v0.2"). If None and live=True,
                loads the live version for the prompt_type and name combination.
        query_executor: QueryExecutor instance for database operations
        prompt_type: Type of prompt (e.g., "rag", "evaluation", "summarization").
                    Defaults to "rag" for backward compatibility.
        name: Optional name for evaluation prompts (e.g., "correctness_evaluator",
              "hallucination_evaluator", "risk_direction_evaluator"). Only used when
              prompt_type="evaluation". Defaults to None.
        live: If True and version is None, loads the live version. If version is provided,
              live is ignored. Defaults to True.
        
    Returns:
        Prompt template text from database
        
    Raises:
        ValidationError: If prompt is not found in database
        DatabaseError: If database query fails
        ValueError: If both version and live are None, or query_executor is None
        
    Example:
        >>> query_executor = QueryExecutor(db_conn)
        >>> # Load live version
        >>> template = load_prompt_template(query_executor=query_executor, live=True)
        
        >>> # Load specific version
        >>> template = load_prompt_template("0.1", query_executor)
        
        >>> # Load live evaluation prompt
        >>> correctness_template = load_prompt_template(
        ...     query_executor=query_executor, prompt_type="evaluation", 
        ...     name="correctness_evaluator", live=True
        ... )
    """
    if query_executor is None:
        raise ValueError("query_executor is required")
    
    # Create cache key from prompt_type, name (if provided), and version/live
    if name:
        if version:
            cache_key = f"{prompt_type}:{name}:{version}"
        else:
            cache_key = f"{prompt_type}:{name}:live" if live else f"{prompt_type}:{name}"
    else:
        if version:
            cache_key = f"{prompt_type}:{version}"
        else:
            cache_key = f"{prompt_type}:live" if live else f"{prompt_type}"
    
    # Check cache first
    if cache_key in _prompt_cache:
        logger.debug(f"Retrieved prompt template '{cache_key}' from cache")
        return _prompt_cache[cache_key]
    
    # Query database for prompt template
    if name:
        # Evaluation prompt query
        if version:
            # Specific version requested
            query = """
                SELECT prompt_text
                FROM prompts
                WHERE prompt_type = %s AND name = %s AND version = %s
            """
            params = (prompt_type, name, version)
        elif live:
            # Live version requested
            query = """
                SELECT prompt_text
                FROM prompts
                WHERE prompt_type = %s AND name = %s AND live = true
                ORDER BY version DESC
                LIMIT 1
            """
            params = (prompt_type, name)
        else:
            # Latest version (not necessarily live)
            query = """
                SELECT prompt_text
                FROM prompts
                WHERE prompt_type = %s AND name = %s
                ORDER BY version DESC
                LIMIT 1
            """
            params = (prompt_type, name)
    else:
        # Non-evaluation prompt query
        if version:
            # Specific version requested
            query = """
                SELECT prompt_text
                FROM prompts
                WHERE prompt_type = %s AND version = %s AND name IS NULL
            """
            params = (prompt_type, version)
        elif live:
            # Live version requested
            query = """
                SELECT prompt_text
                FROM prompts
                WHERE prompt_type = %s AND name IS NULL AND live = true
                ORDER BY version DESC
                LIMIT 1
            """
            params = (prompt_type,)
        else:
            # Latest version (not necessarily live)
            query = """
                SELECT prompt_text
                FROM prompts
                WHERE prompt_type = %s AND name IS NULL
                ORDER BY version DESC
                LIMIT 1
            """
            params = (prompt_type,)
    
    try:
        results = query_executor.execute_query(query, params)
        
        if not results:
            if name:
                if version:
                    error_msg = f"Prompt version '{version}' of type '{prompt_type}' with name '{name}' not found in database"
                else:
                    error_msg = f"Live prompt of type '{prompt_type}' with name '{name}' not found in database"
            else:
                if version:
                    error_msg = f"Prompt version '{version}' of type '{prompt_type}' not found in database"
                else:
                    error_msg = f"Live prompt of type '{prompt_type}' not found in database"
            logger.error(error_msg)
            raise ValidationError(error_msg)
        
        prompt_text = results[0]["prompt_text"]
        
        # Cache the template
        _prompt_cache[cache_key] = prompt_text
        logger.info(f"Loaded and cached prompt template '{cache_key}' from database")
        
        return prompt_text
        
    except ValidationError:
        raise
    except Exception as e:
        error_msg = f"Failed to load prompt template: {e}"
        logger.error(error_msg)
        raise DatabaseError(error_msg) from e


def construct_prompt(
    query: Query,
    retrieved_chunks: List[RetrievalResult],
    prompt_version: Optional[str] = None,
    query_executor: Optional[QueryExecutor] = None,
    prompt_type: str = "rag",
    live: bool = True
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
    template = load_prompt_template(
        version=prompt_version,
        query_executor=query_executor,
        prompt_type=prompt_type,
        live=live
    )
    
    # Validate template has required placeholders
    required_placeholders = ["{query}", "{context}"]
    missing_placeholders = [
        placeholder for placeholder in required_placeholders
        if placeholder not in template
    ]
    
    if missing_placeholders:
        version_str = prompt_version or ("live" if live else "latest")
        error_msg = (
            f"Prompt template '{prompt_type}:{version_str}' is missing required placeholders: "
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
                    f"Azure generation API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                time.sleep(delay)
            else:
                logger.error(f"Azure generation API call failed after {max_retries + 1} attempts: {e}")
                raise AzureServiceError(
                    f"Azure AI Foundry generation API call failed after {max_retries + 1} attempts: {str(e)}"
                ) from e
    
    # This should never be reached, but included for type safety
    raise AzureServiceError(f"Unexpected error in retry logic: {last_exception}") from last_exception


def _call_generation_api(
    prompt: str,
    model: str,
    endpoint: str,
    api_key: str,
    temperature: float = 0.1,
    max_tokens: int = 1000
) -> str:
    """
    Call Azure AI Foundry chat completions API for answer generation.
    
    Args:
        prompt: Complete prompt string to send to the LLM
        model: Generation model name (e.g., "gpt-4o")
        endpoint: Azure AI Foundry endpoint URL
        api_key: Azure AI Foundry API key
        temperature: Generation temperature (default: 0.1 for reproducibility)
        max_tokens: Maximum tokens to generate (default: 1000)
        
    Returns:
        Generated answer text from the LLM
        
    Raises:
        AzureServiceError: If API call fails
        ValueError: If response is invalid
    """
    # Azure AI Foundry uses OpenAI-compatible API
    # Strip trailing slash from endpoint to avoid double slashes
    endpoint = endpoint.rstrip('/')
    api_endpoint = f"{endpoint}/openai/deployments/{model}/chat/completions?api-version=2024-02-15-preview"
    
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    def make_request():
        response = requests.post(api_endpoint, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    
    result = _retry_with_backoff(make_request)
    
    # Validate response structure
    if "choices" not in result or not result["choices"]:
        raise ValueError(f"Invalid generation API response: missing 'choices' field. Response: {result}")
    
    # Extract answer from response
    choice = result["choices"][0]
    if "message" not in choice or "content" not in choice["message"]:
        raise ValueError(f"Invalid generation API response: missing 'content' field. Response: {result}")
    
    answer_text = choice["message"]["content"]
    
    # Validate answer is not empty
    if not answer_text or not answer_text.strip():
        raise ValueError("Generated answer is empty")
    
    return answer_text.strip()


def generate_answer(
    query: Query,
    retrieved_chunks: List[RetrievalResult],
    prompt_version: Optional[str] = None,
    config: Optional[Config] = None,
    query_executor: Optional[QueryExecutor] = None,
    live: bool = True
) -> ModelAnswer:
    """
    Generate an answer using Azure AI Foundry LLM.
    
    This function:
    - Loads and constructs a prompt template using Phase 5 implementation
    - Calls Azure AI Foundry (OpenAI-compatible API) for generation
    - Configures generation parameters (temperature: 0.1, max_tokens: 1000)
    - Parses and validates LLM response
    - Creates ModelAnswer object with metadata
    - Implements retry logic with exponential backoff (3 retries max)
    
    **Generation Parameters:**
    - Model: From config (default: "gpt-4o")
    - Temperature: 0.1 (for reproducibility)
    - Max tokens: 1000 (configurable)
    
    **Non-Determinism Note**: While temperature is set to 0.1 for reproducibility,
    LLM generation is inherently non-deterministic. The same prompt may produce
    slightly different answers across multiple calls. This is acceptable and expected.
    
    Args:
        query: Query object containing the user's question
        retrieved_chunks: List of retrieved chunks to use as context
        prompt_version: Version name of the prompt template to use (e.g., "v1", "v2")
        config: Application configuration with Azure AI Foundry credentials
               (azure_ai_foundry_endpoint, azure_ai_foundry_api_key, azure_ai_foundry_generation_model)
        query_executor: Optional QueryExecutor for prompt template loading.
                      If None, will attempt to create one from config (requires database connection).
                      It's recommended to pass a QueryExecutor instance for better control.
        
    Returns:
        ModelAnswer object with:
        - text: Generated answer
        - query_id: From query object (generated if missing)
        - prompt_version: Prompt version used
        - retrieved_chunk_ids: List of chunk IDs from retrieval results
        - timestamp: Generation timestamp
        
    Raises:
        AzureServiceError: If generation fails after retries
        ValidationError: If prompt version is not found or query is invalid
        ValueError: If config is missing required fields or response is invalid
        DatabaseError: If database query fails (when loading prompt template)
        
    Example:
        >>> from src.core.config import Config
        >>> from src.db.connection import DatabaseConnection
        >>> from src.db.queries import QueryExecutor
        >>> 
        >>> config = Config.from_env()
        >>> query = Query(text="What is the coverage limit?")
        >>> chunks = [RetrievalResult(chunk_id="1", similarity_score=0.9, chunk_text="Coverage is $500k")]
        >>> 
        >>> db_conn = DatabaseConnection(config)
        >>> query_executor = QueryExecutor(db_conn)
        >>> answer = generate_answer(query, chunks, "v1", config, query_executor)
        >>> print(answer.text)
        "The coverage limit is $500,000."
    """
    if not query.text or not query.text.strip():
        raise ValueError("Query text cannot be empty")
    
    # Validate configuration
    if not config.azure_ai_foundry_endpoint:
        raise ValueError("Azure AI Foundry endpoint is not configured")
    
    if not config.azure_ai_foundry_api_key:
        raise ValueError("Azure AI Foundry API key is not configured")
    
    model = config.azure_ai_foundry_generation_model
    if not model:
        raise ValueError("Azure AI Foundry generation model is not configured")
    
    # Generate query_id if missing
    query_id = query.query_id
    if not query_id:
        query_id = generate_id("query")
        logger.debug(f"Generated query_id: {query_id} for query: {query.text[:50]}...")
    
    # Get QueryExecutor if not provided
    if query_executor is None:
        logger.warning("QueryExecutor not provided, attempting to create from config")
        from src.db.connection import DatabaseConnection
        db_conn = DatabaseConnection(config)
        db_conn.connect()
        query_executor = QueryExecutor(db_conn)
        # Note: We don't close the connection here as it may be managed externally
        # The caller should manage the connection lifecycle
    
    logger.info(
        f"Generating answer for query '{query.text[:50]}...' "
        f"using prompt version '{prompt_version}' and model '{model}'"
    )
    
    try:
        # Step 1: Construct prompt using Phase 5 implementation
        prompt = construct_prompt(query, retrieved_chunks, prompt_version, query_executor, live=live)
        logger.debug(f"Constructed prompt ({len(prompt)} characters)")
        
        # Step 2: Call Azure AI Foundry for generation
        # Generation parameters: temperature=0.1 (for reproducibility), max_tokens=1000
        answer_text = _call_generation_api(
            prompt=prompt,
            model=model,
            endpoint=config.azure_ai_foundry_endpoint,
            api_key=config.azure_ai_foundry_api_key,
            temperature=0.1,
            max_tokens=1000
        )
        
        logger.info(f"Successfully generated answer ({len(answer_text)} characters)")
        
        # Step 3: Extract retrieved chunk IDs
        retrieved_chunk_ids = [chunk.chunk_id for chunk in retrieved_chunks]
        
        # Step 4: Create ModelAnswer object
        answer = ModelAnswer(
            text=answer_text,
            query_id=query_id,
            prompt_version=prompt_version,
            retrieved_chunk_ids=retrieved_chunk_ids,
            timestamp=datetime.now(timezone.utc)
        )
        
        logger.info(
            f"Generated ModelAnswer for query_id '{query_id}' "
            f"with {len(retrieved_chunk_ids)} retrieved chunks"
        )
        
        return answer
        
    except AzureServiceError:
        # Re-raise AzureServiceError as-is
        raise
    except ValidationError:
        # Re-raise ValidationError as-is
        raise
    except DatabaseError:
        # Re-raise DatabaseError as-is
        raise
    except ValueError:
        # Re-raise ValueError as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error generating answer: {e}", exc_info=True)
        raise AzureServiceError(
            f"Unexpected error generating answer: {str(e)}"
        ) from e

