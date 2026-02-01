"""OpenAI-based text generation implementation"""

from typing import List, Dict, Any, Optional, Callable
from ..base.base_generator import PromptBasedGenerator, ChatBasedGenerator
from ...core.interfaces import GenerationResult
from ...core.registry import register_generator


@register_generator(
    name="openai_generator",
    description="OpenAI GPT models for text generation",
    provider="OpenAI"
)
class OpenAIGenerator(ChatBasedGenerator):
    """OpenAI generator implementation"""
    
    def __init__(
        self,
        llm_function: Callable[[str], str],
        model_name: str = "gpt-4",
        **config
    ):
        """
        Initialize OpenAI generator
        
        Args:
            llm_function: Function that takes prompt and returns LLM response
            model_name: Name of the OpenAI model
            **config: Additional configuration
        """
        super().__init__(model_name=model_name, **config)
        self.llm_function = llm_function
        self.model_name = model_name
    
    def get_name(self) -> str:
        return "openai_generator"
    
    def get_description(self) -> str:
        return f"OpenAI generator using {self.model_name}"
    
    def generate(
        self,
        prompt: str,
        context_chunks: Optional[List[str]] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate response using OpenAI API"""
        response = self.llm_function(prompt)
        
        return GenerationResult(
            response=response,
            model_name=self.model_name,
            prompt_tokens=len(prompt.split()),  # Rough estimate
            completion_tokens=len(response.split()),  # Rough estimate
            context_chunks=context_chunks or [],
            metadata={
                'generator': 'openai',
                'model': self.model_name,
                **kwargs
            }
        )
    
    def format_messages(
        self,
        question: str,
        context_chunks: Optional[List[str]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> List[Dict[str, str]]:
        """Format messages for OpenAI chat format"""
        messages = []
        
        if conversation_history:
            messages.extend(conversation_history)
        
        if context_chunks:
            context = "\n\n".join(context_chunks)
            system_message = f"Use the following context to answer the user's question:\n\n{context}"
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": question})
        return messages


@register_generator(
    name="simple_rag_generator",
    description="Simple RAG generator with context injection",
    approach="context_injection"
)
class SimpleRAGGenerator(PromptBasedGenerator):
    """Simple RAG generator with context injection"""
    
    def __init__(
        self,
        llm_function: Callable[[str], str],
        **config
    ):
        """
        Initialize simple RAG generator
        
        Args:
            llm_function: Function that takes prompt and returns response
            **config: Additional configuration
        """
        super().__init__(**config)
        self.llm_function = llm_function
    
    def get_name(self) -> str:
        return "simple_rag_generator"
    
    def get_description(self) -> str:
        return "Simple RAG generator with context injection"
    
    def generate(
        self,
        prompt: str,
        context_chunks: Optional[List[str]] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate response with simple context injection"""
        response = self.llm_function(prompt)
        
        return GenerationResult(
            response=response,
            model_name="simple_rag",
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(response.split()),
            context_chunks=context_chunks or [],
            metadata={
                'generator': 'simple_rag',
                **kwargs
            }
        )
    
    def format_prompt(
        self,
        question: str,
        context_chunks: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Format prompt with context injection"""
        if context_chunks:
            context = "\n\n".join(context_chunks)
            prompt = f"""Context information:
{context}

Question: {question}

Answer the question based on the context provided above. If the answer cannot be found in the context, say "I cannot answer this question based on the provided context."

Answer:"""
        else:
            prompt = f"Question: {question}\n\nAnswer:"
        
        return prompt