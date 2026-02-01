"""Abstract base class for generator implementations"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ...core.interfaces import GenerationResult


class BaseGenerator(ABC):
    """Abstract base class for text generation implementations"""
    
    def __init__(self, **config):
        """
        Initialize generator with configuration
        
        Args:
            **config: Generator-specific configuration parameters
        """
        self.config = config
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        context_chunks: Optional[List[str]] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate text response for a prompt
        
        Args:
            prompt: Input prompt
            context_chunks: Optional context chunks for RAG
            **kwargs: Additional generation parameters
            
        Returns:
            Generation result with response and metadata
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name/identifier of this generator implementation"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get a human-readable description of this generator"""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration used by this generator"""
        return self.config.copy()
    
    def generate_batch(
        self,
        prompts: List[str],
        context_chunks_list: Optional[List[List[str]]] = None,
        **kwargs
    ) -> List[GenerationResult]:
        """
        Generate responses for multiple prompts
        
        Args:
            prompts: List of input prompts
            context_chunks_list: Optional list of context chunks for each prompt
            **kwargs: Additional generation parameters
            
        Returns:
            List of generation results
        """
        results = []
        for i, prompt in enumerate(prompts):
            context = context_chunks_list[i] if context_chunks_list else None
            result = self.generate(prompt, context_chunks=context, **kwargs)
            results.append(result)
        return results
    
    def get_generation_statistics(
        self, 
        results: List[GenerationResult]
    ) -> Dict[str, Any]:
        """
        Get statistics about generation results
        
        Args:
            results: List of generation results
            
        Returns:
            Dictionary with generation statistics
        """
        if not results:
            return {
                'total_generations': 0,
                'avg_response_length': 0,
                'min_response_length': 0,
                'max_response_length': 0
            }
        
        response_lengths = [len(result.response) for result in results]
        
        return {
            'total_generations': len(results),
            'avg_response_length': sum(response_lengths) / len(response_lengths),
            'min_response_length': min(response_lengths),
            'max_response_length': max(response_lengths),
            'generator_name': self.get_name(),
            'generator_config': self.get_config()
        }


class PromptBasedGenerator(BaseGenerator):
    """Base class for prompt-based generation implementations"""
    
    @abstractmethod
    def format_prompt(
        self,
        question: str,
        context_chunks: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Format the prompt for generation
        
        Args:
            question: User question
            context_chunks: Optional context chunks
            **kwargs: Additional formatting parameters
            
        Returns:
            Formatted prompt string
        """
        pass
    
    def generate_from_question(
        self,
        question: str,
        context_chunks: Optional[List[str]] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate response for a question with optional context
        
        Args:
            question: User question
            context_chunks: Optional context chunks for RAG
            **kwargs: Additional generation parameters
            
        Returns:
            Generation result with response and metadata
        """
        formatted_prompt = self.format_prompt(question, context_chunks, **kwargs)
        return self.generate(formatted_prompt, context_chunks=context_chunks, **kwargs)


class ChatBasedGenerator(BaseGenerator):
    """Base class for chat-based generation implementations"""
    
    @abstractmethod
    def format_messages(
        self,
        question: str,
        context_chunks: Optional[List[str]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> List[Dict[str, str]]:
        """
        Format messages for chat-based generation
        
        Args:
            question: User question
            context_chunks: Optional context chunks
            conversation_history: Optional conversation history
            **kwargs: Additional formatting parameters
            
        Returns:
            List of message dictionaries
        """
        pass
    
    def generate_chat_response(
        self,
        question: str,
        context_chunks: Optional[List[str]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate chat response for a question
        
        Args:
            question: User question
            context_chunks: Optional context chunks for RAG
            conversation_history: Optional conversation history
            **kwargs: Additional generation parameters
            
        Returns:
            Generation result with response and metadata
        """
        messages = self.format_messages(question, context_chunks, conversation_history, **kwargs)
        prompt = self._messages_to_prompt(messages)
        return self.generate(prompt, context_chunks=context_chunks, **kwargs)
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to prompt string (default implementation)"""
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            prompt_parts.append(f"{role}: {content}")
        return "\n\n".join(prompt_parts)