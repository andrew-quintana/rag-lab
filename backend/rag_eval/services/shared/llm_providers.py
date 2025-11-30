"""LLM provider abstraction layer for multiple AI service providers"""

import json
import time
from abc import ABC, abstractmethod
from typing import Optional
import requests

from rag_eval.core.config import Config
from rag_eval.core.exceptions import AzureServiceError
from rag_eval.core.logging import get_logger

logger = get_logger("services.shared.llm_providers")


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def call_completion(
        self, 
        prompt: str, 
        temperature: float = 0.1, 
        max_tokens: int = 500
    ) -> str:
        """
        Call LLM and return raw response text.
        
        Args:
            prompt: Complete prompt string to send to the LLM
            temperature: Generation temperature (default: 0.1 for reproducibility)
            max_tokens: Maximum tokens to generate (default: 500)
            
        Returns:
            Raw response text from the LLM
            
        Raises:
            AzureServiceError: If API call fails
            ValueError: If response is invalid
        """
        pass


class AzureFoundryProvider(LLMProvider):
    """Azure AI Foundry LLM provider using OpenAI-compatible API"""
    
    def __init__(
        self, 
        endpoint: str, 
        api_key: str, 
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        base_delay: float = 1.0
    ):
        """
        Initialize Azure Foundry provider.
        
        Args:
            endpoint: Azure AI Foundry endpoint URL
            api_key: Azure AI Foundry API key
            model: Model name (default: "gpt-4o-mini")
            max_retries: Maximum number of retry attempts (default: 3)
            base_delay: Base delay in seconds for exponential backoff (default: 1.0)
        """
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.logger = get_logger("services.shared.llm_providers.AzureFoundryProvider")
        
        if not self.endpoint:
            raise ValueError("Azure AI Foundry endpoint is required")
        if not self.api_key:
            raise ValueError("Azure AI Foundry API key is required")
        if not self.model:
            raise ValueError("Model name is required")
    
    def _retry_with_backoff(self, func):
        """
        Retry a function with exponential backoff.
        
        Args:
            func: Function to retry (callable that takes no arguments)
            
        Returns:
            Result of the function call
            
        Raises:
            AzureServiceError: If all retries are exhausted
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func()
            except (requests.RequestException, Exception) as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(
                        f"Azure LLM API call failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(f"Azure LLM API call failed after {self.max_retries + 1} attempts: {e}")
                    raise AzureServiceError(
                        f"Azure AI Foundry API call failed after {self.max_retries + 1} attempts: {str(e)}"
                    ) from e
        
        # This should never be reached, but included for type safety
        raise AzureServiceError(f"Unexpected error in retry logic: {last_exception}") from last_exception
    
    def call_completion(
        self, 
        prompt: str, 
        temperature: float = 0.1, 
        max_tokens: int = 500
    ) -> str:
        """
        Call Azure AI Foundry chat completions API.
        
        Args:
            prompt: Complete prompt string to send to the LLM
            temperature: Generation temperature (default: 0.1 for reproducibility)
            max_tokens: Maximum tokens to generate (default: 500)
            
        Returns:
            Generated response text from the LLM
            
        Raises:
            AzureServiceError: If API call fails
            ValueError: If response is invalid
        """
        api_endpoint = f"{self.endpoint}/openai/deployments/{self.model}/chat/completions?api-version=2024-02-15-preview"
        
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
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
        
        result = self._retry_with_backoff(make_request)
        
        # Validate response structure
        if "choices" not in result or not result["choices"]:
            raise ValueError(f"Invalid Azure API response: missing 'choices' field. Response: {result}")
        
        # Extract answer from response
        choice = result["choices"][0]
        if "message" not in choice or "content" not in choice["message"]:
            raise ValueError(f"Invalid Azure API response: missing 'content' field. Response: {result}")
        
        answer_text = choice["message"]["content"]
        
        # Validate answer is not empty
        if not answer_text or not answer_text.strip():
            raise ValueError("Generated response is empty")
        
        return answer_text.strip()


class OpenAIProvider(LLMProvider):
    """OpenAI direct API provider"""
    
    def __init__(
        self, 
        api_key: str, 
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        base_delay: float = 1.0
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            model: Model name (default: "gpt-4o-mini")
            max_retries: Maximum number of retry attempts (default: 3)
            base_delay: Base delay in seconds for exponential backoff (default: 1.0)
        """
        try:
            import openai
            self.openai = openai
        except ImportError:
            raise ImportError(
                "OpenAI provider requires the 'openai' package. "
                "Install it with: pip install openai"
            )
        
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.logger = get_logger("services.shared.llm_providers.OpenAIProvider")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        if not self.model:
            raise ValueError("Model name is required")
    
    def call_completion(
        self, 
        prompt: str, 
        temperature: float = 0.1, 
        max_tokens: int = 500
    ) -> str:
        """
        Call OpenAI chat completions API.
        
        Args:
            prompt: Complete prompt string to send to the LLM
            temperature: Generation temperature (default: 0.1 for reproducibility)
            max_tokens: Maximum tokens to generate (default: 500)
            
        Returns:
            Generated response text from the LLM
            
        Raises:
            AzureServiceError: If API call fails (for consistency with other providers)
            ValueError: If response is invalid
        """
        # TODO: Implement OpenAI API call
        # This is a stub implementation
        raise NotImplementedError(
            "OpenAI provider is not yet fully implemented. "
            "Use AzureFoundryProvider for now."
        )


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider"""
    
    def __init__(
        self, 
        api_key: str, 
        model: str = "claude-3-haiku-20240307",
        max_retries: int = 3,
        base_delay: float = 1.0
    ):
        """
        Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key
            model: Model name (default: "claude-3-haiku-20240307")
            max_retries: Maximum number of retry attempts (default: 3)
            base_delay: Base delay in seconds for exponential backoff (default: 1.0)
        """
        try:
            import anthropic
            self.anthropic = anthropic
        except ImportError:
            raise ImportError(
                "Anthropic provider requires the 'anthropic' package. "
                "Install it with: pip install anthropic"
            )
        
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.logger = get_logger("services.shared.llm_providers.AnthropicProvider")
        
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        if not self.model:
            raise ValueError("Model name is required")
    
    def call_completion(
        self, 
        prompt: str, 
        temperature: float = 0.1, 
        max_tokens: int = 500
    ) -> str:
        """
        Call Anthropic Claude API.
        
        Args:
            prompt: Complete prompt string to send to the LLM
            temperature: Generation temperature (default: 0.1 for reproducibility)
            max_tokens: Maximum tokens to generate (default: 500)
            
        Returns:
            Generated response text from the LLM
            
        Raises:
            AzureServiceError: If API call fails (for consistency with other providers)
            ValueError: If response is invalid
        """
        # TODO: Implement Anthropic API call
        # This is a stub implementation
        raise NotImplementedError(
            "Anthropic provider is not yet fully implemented. "
            "Use AzureFoundryProvider for now."
        )


def get_llm_provider(
    config: Config, 
    provider_type: Optional[str] = None
) -> LLMProvider:
    """
    Factory function to create appropriate LLM provider based on config.
    
    Args:
        config: Application configuration
        provider_type: Optional provider type ("azure", "openai", "anthropic").
                      If None, defaults to "azure" or reads from environment variable.
        
    Returns:
        LLMProvider instance
        
    Raises:
        ValueError: If provider type is invalid or required config is missing
        ImportError: If optional dependencies are missing for non-Azure providers
    """
    import os
    
    # Determine provider type
    if provider_type is None:
        provider_type = os.getenv("LLM_PROVIDER", "azure").lower()
    else:
        provider_type = provider_type.lower()
    
    if provider_type == "azure":
        # Validate Azure config
        if not config.azure_ai_foundry_endpoint:
            raise ValueError("Azure AI Foundry endpoint is not configured")
        if not config.azure_ai_foundry_api_key:
            raise ValueError("Azure AI Foundry API key is not configured")
        
        # Get model from config (default to evaluation model, fallback to generation model)
        model = getattr(config, 'azure_ai_foundry_evaluation_model', None) or \
                getattr(config, 'azure_ai_foundry_generation_model', None) or \
                "gpt-4o-mini"
        
        return AzureFoundryProvider(
            endpoint=config.azure_ai_foundry_endpoint,
            api_key=config.azure_ai_foundry_api_key,
            model=model
        )
    
    elif provider_type == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return OpenAIProvider(api_key=api_key, model=model)
    
    elif provider_type == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
        
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        return AnthropicProvider(api_key=api_key, model=model)
    
    else:
        raise ValueError(
            f"Unknown provider type: {provider_type}. "
            f"Supported types: 'azure', 'openai', 'anthropic'"
        )

