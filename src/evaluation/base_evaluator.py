"""Base class for LLM-based evaluation nodes - Simplified for local use"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Callable, Dict, Any

logger = logging.getLogger(__name__)


class BaseEvaluatorNode(ABC):
    """Base class for LLM-based evaluation nodes - simplified for local use"""
    
    def __init__(
        self,
        llm_call_function: Callable[[str, float, int], str],
        prompt_template: Optional[str] = None,
        prompt_path: Optional[Path] = None,
        name: Optional[str] = None
    ):
        """
        Initialize base evaluator node.
        
        Args:
            llm_call_function: Function to call LLM (prompt: str, temperature: float, max_tokens: int) -> str
            prompt_template: Optional prompt template string
            prompt_path: Optional path to prompt template file
            name: Optional name for this evaluator
        """
        self.llm_call_function = llm_call_function
        self.prompt_template = prompt_template
        self.prompt_path = prompt_path
        self.name = name or self.__class__.__name__
        
        # Set up logger
        self.logger = logging.getLogger(f"evaluator.{self.name}")
    
    def _load_prompt_template(self) -> str:
        """
        Load prompt template from provided template or file.
        
        Returns:
            Prompt template text
            
        Raises:
            ValueError: If prompt cannot be loaded
        """
        if self.prompt_template is not None:
            return self.prompt_template
        elif self.prompt_path is not None:
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
            raise ValueError("Either prompt_template or prompt_path must be provided")
    
    @abstractmethod
    def _construct_prompt(self, **kwargs) -> str:
        """
        Construct prompt by loading template and replacing placeholders.
        
        This method must be implemented by subclasses to handle
        their specific prompt construction logic.
        
        Args:
            **kwargs: Prompt-specific arguments
            
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
        Call LLM using the provided function.
        
        Args:
            prompt: Complete prompt string to send to the LLM
            temperature: Generation temperature (default: 0.1 for reproducibility)
            max_tokens: Maximum tokens to generate (default: 500)
            
        Returns:
            Raw response text from the LLM
            
        Raises:
            Exception: If LLM call fails
        """
        try:
            response = self.llm_call_function(prompt, temperature, max_tokens)
            self.logger.debug(f"LLM call successful ({len(response)} characters)")
            return response
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}", exc_info=True)
            raise Exception(f"LLM call failed: {str(e)}") from e
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
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