"""Cost extraction LLM node for evaluation"""

from typing import Optional, Dict, Any
from pathlib import Path

from src.core.config import Config
from src.core.exceptions import AzureServiceError
from src.services.evaluator.base_evaluator import BaseEvaluatorNode
from src.services.shared.llm_providers import LLMProvider
from src.db.queries import QueryExecutor


class CostExtractionEvaluator(BaseEvaluatorNode):
    """Cost extraction evaluator using LLM to parse structured cost information from text"""
    
    def __init__(
        self,
        prompt_version: Optional[str] = None,
        live: bool = True,
        query_executor: Optional[QueryExecutor] = None,
        config: Optional[Config] = None,
        llm_provider: Optional[LLMProvider] = None,
        prompt_path: Optional[Path] = None  # Backward compatibility for testing
    ):
        """
        Initialize cost extraction evaluator.
        
        Args:
            prompt_version: Prompt version name (e.g., "v1"). Defaults to None (uses live version)
            query_executor: QueryExecutor instance for database operations (required for database prompt loading)
            config: Application configuration (optional, defaults to Config.from_env())
            llm_provider: LLM provider instance (optional, defaults to Azure from config)
            prompt_path: Path to prompt template file (optional, for testing only).
                        Either query_executor or prompt_path must be provided.
        """
        super().__init__(
            prompt_version=prompt_version,
            prompt_type="evaluation",
            name="cost_extraction_evaluator",
            live=live,
            query_executor=query_executor,
            config=config,
            llm_provider=llm_provider,
            prompt_path=prompt_path
        )
    
    def _construct_prompt(self, text: str) -> str:
        """
        Construct cost extraction prompt by loading template and replacing placeholders.
        
        Args:
            text: Text containing cost information to extract
            
        Returns:
            Complete prompt string ready for LLM
            
        Raises:
            ValueError: If prompt template is missing required placeholders or cannot be loaded
        """
        # Load prompt template
        template = self._load_prompt_template()
        
        # Validate template has required placeholders
        required_placeholders = ["{text}"]
        missing_placeholders = [
            placeholder for placeholder in required_placeholders
            if placeholder not in template
        ]
        
        if missing_placeholders:
            error_msg = (
                f"Prompt template is missing required placeholders: "
                f"{', '.join(missing_placeholders)}"
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Replace placeholders
        try:
            prompt = template.replace("{text}", text)
        except Exception as e:
            error_msg = f"Failed to format prompt template: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg) from e
        
        self.logger.debug(f"Constructed cost extraction prompt for text length: {len(text)}")
        return prompt
    
    def extract_costs(self, text: str) -> Dict[str, Any]:
        """
        Extract structured cost information (time, money, steps) from unstructured text.
        
        Parses natural language text to identify time-based costs, money-based costs,
        and step-based costs. All cost fields are optional - if a cost type is not present
        in the text, that field will be omitted from the result.
        
        **Temperature Setting**: Uses temperature=0.1 for reproducibility, as specified
        in the evaluation system requirements.
        
        Args:
            text: Text containing cost information to extract
            
        Returns:
            Dictionary with optional fields:
            - `time`: Optional string or number representing time cost
            - `money`: Optional string or number representing monetary cost
            - `steps`: Optional integer or string representing step count
            - `reasoning`: Required string explaining what cost information was found
            
        Raises:
            AzureServiceError: If LLM call fails
            ValueError: If input is invalid or empty, or if response cannot be parsed
            
        Example:
            >>> from src.core.config import Config
            >>> 
            >>> config = Config.from_env()
            >>> evaluator = CostExtractionEvaluator(config=config)
            >>> text = "The procedure takes 2 hours and costs $500."
            >>> result = evaluator.extract_costs(text)
            >>> print(result)
            {'time': '2 hours', 'money': '$500', 'reasoning': '...'}
        """
        # Validate input
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Get model name for logging
        model_name = "unknown"
        if hasattr(self.llm_provider, 'model'):
            model_name = self.llm_provider.model
        
        self.logger.info(
            f"Extracting costs from text (length: {len(text)}) "
            f"using model '{model_name}'"
        )
        
        try:
            # Step 1: Construct prompt
            prompt = self._construct_prompt(text)
            self.logger.debug(f"Constructed cost extraction prompt ({len(prompt)} characters)")
            
            # Step 2: Call LLM for extraction
            # Temperature=0.1 for reproducibility (as specified in requirements)
            response_text = self._call_llm(
                prompt=prompt,
                temperature=0.1,
                max_tokens=500
            )
            
            # Step 3: Parse JSON response
            parsed = self._parse_json_response(response_text)
            
            # Step 4: Validate required reasoning field
            if "reasoning" not in parsed:
                raise ValueError(
                    f"Cost extraction response missing 'reasoning' field. "
                    f"Response: {parsed}"
                )
            
            # Step 5: Build result dictionary with only present fields
            result = {
                "reasoning": parsed["reasoning"]
            }
            
            # Add optional fields only if they exist in the parsed response
            if "time" in parsed and parsed["time"] is not None:
                result["time"] = parsed["time"]
            
            if "money" in parsed and parsed["money"] is not None:
                result["money"] = parsed["money"]
            
            if "steps" in parsed and parsed["steps"] is not None:
                result["steps"] = parsed["steps"]
            
            reasoning = parsed.get("reasoning", "No reasoning provided")
            
            self.logger.info(
                f"Cost extraction completed. Found fields: {list(result.keys())} "
                f"(reasoning: {reasoning[:100]}...)"
            )
            
            return result
            
        except AzureServiceError:
            # Re-raise AzureServiceError as-is
            raise
        except ValueError:
            # Re-raise ValueError as-is
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error extracting costs: {e}", exc_info=True)
            raise AzureServiceError(
                f"Unexpected error extracting costs: {str(e)}"
            ) from e


# Backward compatibility: module-level function
def extract_costs(
    text: str,
    config: Optional[Config] = None,
    query_executor: Optional[QueryExecutor] = None,
    prompt_version: Optional[str] = None,
    live: bool = True
) -> Dict[str, Any]:
    """
    Extract structured cost information (time, money, steps) from unstructured text.
    
    This is a convenience function that maintains backward compatibility.
    It creates a CostExtractionEvaluator instance and calls extract_costs().
    
    Parses natural language text to identify time-based costs, money-based costs,
    and step-based costs. All cost fields are optional - if a cost type is not present
    in the text, that field will be omitted from the result.
    
    **Temperature Setting**: Uses temperature=0.1 for reproducibility, as specified
    in the evaluation system requirements.
    
    Args:
        text: Text containing cost information to extract
        config: Application configuration (optional, defaults to Config.from_env())
        query_executor: QueryExecutor instance for database operations (optional, for database prompt loading)
        prompt_version: Prompt version name (e.g., "v1"). Defaults to None (uses live version)
        live: If True and prompt_version is None, loads the live version. Defaults to True
        
    Returns:
        Dictionary with optional fields:
        - `time`: Optional string or number representing time cost
        - `money`: Optional string or number representing monetary cost
        - `steps`: Optional integer or string representing step count
        - `reasoning`: Required string explaining what cost information was found
        
    Raises:
        AzureServiceError: If Azure Foundry LLM call fails
        ValueError: If input is invalid or empty, or if response cannot be parsed
        
    Example:
        >>> from src.core.config import Config
        >>> 
        >>> config = Config.from_env()
        >>> text = "The procedure takes 2 hours and costs $500."
        >>> result = extract_costs(text, config)
        >>> print(result)
        {'time': '2 hours', 'money': '$500', 'reasoning': '...'}
    """
    evaluator = CostExtractionEvaluator(
        config=config,
        query_executor=query_executor,
        prompt_version=prompt_version,
        live=live
    )
    return evaluator.extract_costs(text)

