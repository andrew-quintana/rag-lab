"""Optional agent runner wrapper for RAG systems"""

import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from .core.interfaces import Query, ModelAnswer
from .indexing.index import RAGRetriever

logger = logging.getLogger(__name__)


class SimpleRAGAgent:
    """Simple RAG agent that combines retrieval and generation"""
    
    def __init__(
        self,
        retriever: RAGRetriever,
        generator_function: Callable[[str, List[str]], str],
        system_prompt: Optional[str] = None
    ):
        """
        Initialize RAG agent
        
        Args:
            retriever: RAG retrieval system
            generator_function: Function that takes (query, context_chunks) -> answer
            system_prompt: Optional system prompt to prepend to queries
        """
        self.retriever = retriever
        self.generator_function = generator_function
        self.system_prompt = system_prompt or "You are a helpful AI assistant. Answer the user's question based on the provided context."
        
        logger.info("Initialized SimpleRAGAgent")
    
    def answer_query(
        self,
        query: str,
        retrieval_k: int = 5,
        query_id: Optional[str] = None
    ) -> ModelAnswer:
        """
        Answer a query using RAG pipeline
        
        Args:
            query: User query text
            retrieval_k: Number of chunks to retrieve
            query_id: Optional query identifier
            
        Returns:
            ModelAnswer with generated response
        """
        logger.debug(f"Processing query: {query[:100]}...")
        
        # Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(query, k=retrieval_k)
        context_texts = [chunk.chunk_text for chunk in retrieved_chunks]
        chunk_ids = [chunk.chunk_id for chunk in retrieved_chunks]
        
        logger.debug(f"Retrieved {len(retrieved_chunks)} chunks")
        
        # Generate answer
        answer_text = self.generator_function(query, context_texts)
        
        # Create ModelAnswer
        answer = ModelAnswer(
            text=answer_text,
            query_id=query_id or f"query_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            prompt_version="simple_rag_v1",
            retrieved_chunk_ids=chunk_ids,
            timestamp=datetime.now()
        )
        
        logger.debug(f"Generated answer: {answer_text[:100]}...")
        return answer
    
    def batch_answer_queries(
        self,
        queries: List[str],
        retrieval_k: int = 5
    ) -> List[ModelAnswer]:
        """
        Answer multiple queries in batch
        
        Args:
            queries: List of query strings
            retrieval_k: Number of chunks to retrieve per query
            
        Returns:
            List of ModelAnswer objects
        """
        logger.info(f"Processing batch of {len(queries)} queries")
        
        answers = []
        for i, query in enumerate(queries):
            query_id = f"batch_query_{i}"
            answer = self.answer_query(query, retrieval_k, query_id)
            answers.append(answer)
        
        logger.info(f"Completed batch processing of {len(queries)} queries")
        return answers


class ConversationalRAGAgent:
    """RAG agent with conversation memory"""
    
    def __init__(
        self,
        retriever: RAGRetriever,
        generator_function: Callable[[str, List[str], List[Dict[str, str]]], str],
        max_history: int = 5
    ):
        """
        Initialize conversational RAG agent
        
        Args:
            retriever: RAG retrieval system
            generator_function: Function that takes (query, context_chunks, conversation_history) -> answer
            max_history: Maximum number of conversation turns to keep
        """
        self.retriever = retriever
        self.generator_function = generator_function
        self.max_history = max_history
        self.conversation_history: List[Dict[str, str]] = []
        
        logger.info(f"Initialized ConversationalRAGAgent with max_history={max_history}")
    
    def answer_query(
        self,
        query: str,
        retrieval_k: int = 5,
        query_id: Optional[str] = None
    ) -> ModelAnswer:
        """
        Answer a query with conversation context
        
        Args:
            query: User query text
            retrieval_k: Number of chunks to retrieve
            query_id: Optional query identifier
            
        Returns:
            ModelAnswer with generated response
        """
        logger.debug(f"Processing conversational query: {query[:100]}...")
        
        # Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(query, k=retrieval_k)
        context_texts = [chunk.chunk_text for chunk in retrieved_chunks]
        chunk_ids = [chunk.chunk_id for chunk in retrieved_chunks]
        
        # Generate answer with conversation history
        answer_text = self.generator_function(query, context_texts, self.conversation_history)
        
        # Update conversation history
        self.conversation_history.append({
            "role": "user", 
            "content": query
        })
        self.conversation_history.append({
            "role": "assistant", 
            "content": answer_text
        })
        
        # Trim history if needed
        if len(self.conversation_history) > self.max_history * 2:  # *2 for user+assistant pairs
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
        
        # Create ModelAnswer
        answer = ModelAnswer(
            text=answer_text,
            query_id=query_id or f"conv_query_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            prompt_version="conversational_rag_v1",
            retrieved_chunk_ids=chunk_ids,
            timestamp=datetime.now()
        )
        
        logger.debug(f"Generated conversational answer: {answer_text[:100]}...")
        return answer
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
        logger.info("Reset conversation history")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get current conversation history"""
        return self.conversation_history.copy()


class AgentEvaluator:
    """Evaluator for agent performance"""
    
    def __init__(self, agent, evaluation_dataset: List[Dict[str, Any]]):
        """
        Initialize agent evaluator
        
        Args:
            agent: Agent to evaluate (SimpleRAGAgent or ConversationalRAGAgent)
            evaluation_dataset: List of evaluation examples with 'question' and 'expected_answer' keys
        """
        self.agent = agent
        self.evaluation_dataset = evaluation_dataset
        
        logger.info(f"Initialized AgentEvaluator with {len(evaluation_dataset)} examples")
    
    def evaluate_agent(
        self,
        retrieval_k: int = 5,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate agent on the dataset
        
        Args:
            retrieval_k: Number of chunks to retrieve per query
            save_results: Whether to save detailed results
            
        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Starting agent evaluation on {len(self.evaluation_dataset)} examples")
        
        results = []
        start_time = datetime.now()
        
        for i, example in enumerate(self.evaluation_dataset):
            example_start = datetime.now()
            
            # Get agent answer
            query = example['question']
            answer = self.agent.answer_query(query, retrieval_k)
            
            example_end = datetime.now()
            duration = (example_end - example_start).total_seconds()
            
            # Store result
            result = {
                "example_id": example.get('example_id', f"example_{i}"),
                "question": query,
                "expected_answer": example.get('expected_answer', ''),
                "agent_answer": answer.text,
                "retrieved_chunks": len(answer.retrieved_chunk_ids),
                "query_id": answer.query_id,
                "duration_seconds": duration,
                "timestamp": example_end.isoformat()
            }
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{len(self.evaluation_dataset)} examples")
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Compute summary statistics
        summary = {
            "total_examples": len(self.evaluation_dataset),
            "total_duration_seconds": total_time,
            "average_duration_seconds": total_time / len(self.evaluation_dataset),
            "examples_per_second": len(self.evaluation_dataset) / total_time,
            "timestamp": datetime.now().isoformat()
        }
        
        evaluation_result = {
            "summary": summary,
            "results": results if save_results else []
        }
        
        logger.info(f"Agent evaluation completed in {total_time:.2f} seconds")
        return evaluation_result


# Convenience functions for creating agents
def create_simple_rag_agent(
    retriever: RAGRetriever,
    generator_function: Callable[[str, List[str]], str],
    system_prompt: Optional[str] = None
) -> SimpleRAGAgent:
    """
    Create a simple RAG agent
    
    Args:
        retriever: RAG retrieval system
        generator_function: Function that takes (query, context_chunks) -> answer
        system_prompt: Optional system prompt
        
    Returns:
        SimpleRAGAgent instance
    """
    return SimpleRAGAgent(retriever, generator_function, system_prompt)


def create_conversational_rag_agent(
    retriever: RAGRetriever,
    generator_function: Callable[[str, List[str], List[Dict[str, str]]], str],
    max_history: int = 5
) -> ConversationalRAGAgent:
    """
    Create a conversational RAG agent
    
    Args:
        retriever: RAG retrieval system
        generator_function: Function that takes (query, context_chunks, history) -> answer
        max_history: Maximum conversation turns to remember
        
    Returns:
        ConversationalRAGAgent instance
    """
    return ConversationalRAGAgent(retriever, generator_function, max_history)


def evaluate_agent_performance(
    agent,
    evaluation_dataset: List[Dict[str, Any]],
    retrieval_k: int = 5
) -> Dict[str, Any]:
    """
    Evaluate agent performance on dataset
    
    Args:
        agent: Agent to evaluate
        evaluation_dataset: List of evaluation examples
        retrieval_k: Number of chunks to retrieve
        
    Returns:
        Evaluation results
    """
    evaluator = AgentEvaluator(agent, evaluation_dataset)
    return evaluator.evaluate_agent(retrieval_k)