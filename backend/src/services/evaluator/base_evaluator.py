"""Base class for LLM-based evaluation nodes"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from src.core.config import Config
from src.core.exceptions import AzureServiceError
from src.core.logging import get_logger
from src.services.shared.llm_providers import LLMProvider, get_llm_provider

logger = get_logger("services.evaluator.base_evaluator")


class BaseEvaluatorNode(ABC):
    """Base class for LLM-based evaluation nodes"""
    
    def __init__(
        self,
        prompt_version: Optional[str] = None,
        prompt_type: str = "evaluation",
        name: Optional[str] = None,
        live: bool = True,
        query_executor: Optional['QueryExecutor'] = None,
        config: Optional[Config] = None,
        llm_provider: Optional[LLMProvider] = None,
        prompt_path: Optional[Path] = None  # Backward compatibility for tests
    ):
        """
        Initialize base evaluator node.
        
        Args:
            prompt_version: Optional prompt version (e.g., "v0.1", "v0.2"). If None, uses live version.
            prompt_type: Type of prompt (e.g., "evaluation", "rag"). Defaults to "evaluation"
            name: Optional name for evaluation prompts (e.g., "correctness_evaluator",
                  "hallucination_evaluator", "risk_direction_evaluator"). Only used when
                  prompt_type="evaluation". Defaults to None.
            live: If True and prompt_version is None, loads the live version. Defaults to True.
            query_executor: QueryExecutor instance for database operations (optional, for database prompt loading)
            config: Application configuration (optional, defaults to Config.from_env())
            llm_provider: LLM provider instance (optional, defaults to Azure from config)
            prompt_path: Path to prompt template file (optional, for backward compatibility/testing).
                        Used only if query_executor is None.
        """
        self.prompt_version = prompt_version
        self.prompt_type = prompt_type
        self.name = name
        self.live = live
        self.query_executor = query_executor
        self.prompt_path = prompt_path
        self.config = config or Config.from_env()
        self.llm_provider = llm_provider or get_llm_provider(self.config)
        
        # Get logger name from class name
        class_name = self.__class__.__name__
        self.logger = get_logger(f"services.evaluator.{class_name}")
    
    def _load_prompt_template(self) -> str:
        """
        Load prompt template from database or file.
        
        Loads from Supabase database if query_executor is provided, otherwise
        falls back to file loading if prompt_path is provided (for backward compatibility/testing).
        
        Returns:
            Prompt template text
            
        Raises:
            ValueError: If prompt cannot be loaded (neither database nor file path available)
        """
        if self.query_executor is not None:
            # Load from database
            from src.services.rag.generation import load_prompt_template
            return load_prompt_template(
                version=self.prompt_version,
                query_executor=self.query_executor,
                prompt_type=self.prompt_type,
                name=self.name,
                live=self.live
            )
        elif self.prompt_path is not None:
            # Fallback to file loading (for backward compatibility/testing)
            if not self.prompt_path.exists():
                raise ValueError(f"Prompt template not found at {self.prompt_path}")
            
            try:
                with open(self.prompt_path, 'r', encoding='utf-8') as f:
                    template = f.read()
                self.logger.debug(f"Loaded prompt template from {self.prompt_path}")
                return template
            except Exception as e:
                raise ValueError(f"Failed to load prompt template from {self.prompt_path}: {e}") from e
        else:
            raise ValueError(
                "Either query_executor (for database loading) or prompt_path (for file loading) must be provided"
            )
    
    @abstractmethod
    def _construct_prompt(self, **kwargs) -> str:
        """
        Construct prompt by loading template and replacing placeholders.
        
        This method must be implemented by subclasses to handle
        their specific prompt construction logic.
        
        Args:
            **kwargs: Prompt-specific arguments (e.g., query, model_answer, etc.)
            
        Returns:
            Complete prompt string ready for LLM
        """
        pass
    
    def _call_llm(
        self, 
        prompt: str, 
        temperature: float = 0.1, 
        max_tokens: int = 500
    ) -> str:
        """
        Call LLM using the configured provider.
        
        Args:
            prompt: Complete prompt string to send to the LLM
            temperature: Generation temperature (default: 0.1 for reproducibility)
            max_tokens: Maximum tokens to generate (default: 500)
            
        Returns:
            Raw response text from the LLM
            
        Raises:
            AzureServiceError: If LLM call fails
            ValueError: If response is invalid
        """
        try:
            response = self.llm_provider.call_completion(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            self.logger.debug(f"LLM call successful ({len(response)} characters)")
            return response
        except AzureServiceError:
            # Re-raise AzureServiceError as-is
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error calling LLM: {e}", exc_info=True)
            raise AzureServiceError(
                f"Unexpected error calling LLM: {str(e)}"
            ) from e
    
    def _parse_json_response(self, response_text: str) -> dict:
        """
        Parse JSON from LLM response text.
        
        Handles responses that may be wrapped in markdown code blocks.
        
        Args:
            response_text: Raw response text from LLM
            
        Returns:
            Parsed JSON as dict
            
        Raises:
            ValueError: If response cannot be parsed as JSON or is not a dict
        """
        if not response_text or not response_text.strip():
            raise ValueError("LLM response is empty")
        
        # Try to extract JSON from the response (may be wrapped in markdown code blocks)
        answer_text = response_text.strip()
        if answer_text.startswith("```json"):
            # Remove markdown code block markers
            answer_text = answer_text[7:]  # Remove "```json"
            if answer_text.endswith("```"):
                answer_text = answer_text[:-3]  # Remove trailing ```
            answer_text = answer_text.strip()
        elif answer_text.startswith("```"):
            # Generic code block
            answer_text = answer_text[3:]
            if answer_text.endswith("```"):
                answer_text = answer_text[:-3]
            answer_text = answer_text.strip()
        
        # Parse JSON
        try:
            parsed = json.loads(answer_text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON from LLM response: {e}. "
                f"Response text: {answer_text[:200]}..."
            ) from e
        
        # Validate JSON structure
        if not isinstance(parsed, dict):
            raise ValueError(
                f"LLM response must be a JSON object, got {type(parsed)}"
            )
        
        self.logger.debug(f"Parsed JSON response with {len(parsed)} fields")
        return parsed

