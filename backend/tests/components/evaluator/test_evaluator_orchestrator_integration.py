"""Integration tests for Evaluation Pipeline Orchestrator with external services

These tests verify the complete evaluation pipeline with real external services:
- Azure AI Foundry (for judge evaluation)
- Azure AI Search (for retrieval)
- Supabase (for prompt loading)

These tests will skip gracefully if credentials are missing.
"""

import pytest
from unittest.mock import Mock
from typing import List

from src.core.config import Config
from src.core.interfaces import (
    EvaluationExample,
    EvaluationResult,
    RetrievalResult,
    ModelAnswer,
    Query
)
from src.services.evaluator.orchestrator import evaluate_rag_system
from src.services.rag.search import retrieve_chunks
from src.services.rag.generation import generate_answer
from src.db.connection import DatabaseConnection
from src.db.queries import QueryExecutor


# Adapter functions to convert between orchestrator interface and RAG system interface
def create_rag_retriever_adapter(config: Config):
    """
    Create a RAG retriever adapter that matches orchestrator interface.
    
    The orchestrator expects: (query: str, k: int) -> List[RetrievalResult]
    The RAG system provides: retrieve_chunks(query: Query, top_k: int, config) -> List[RetrievalResult]
    """
    def rag_retriever(query: str, k: int) -> List[RetrievalResult]:
        """Adapter function for RAG retriever"""
        query_obj = Query(text=query)
        return retrieve_chunks(query_obj, top_k=k, config=config)
    
    return rag_retriever


def create_rag_generator_adapter(config: Config, query_executor: QueryExecutor = None):
    """
    Create a RAG generator adapter that matches orchestrator interface.
    
    The orchestrator expects: (query: str, chunks: List[RetrievalResult]) -> ModelAnswer
    The RAG system provides: generate_answer(query: Query, retrieved_chunks, prompt_version, config, query_executor) -> ModelAnswer
    """
    def rag_generator(query: str, chunks: List[RetrievalResult]) -> ModelAnswer:
        """Adapter function for RAG generator"""
        query_obj = Query(text=query)
        return generate_answer(
            query=query_obj,
            retrieved_chunks=chunks,
            prompt_version="v1",
            config=config,
            query_executor=query_executor
        )
    
    return rag_generator


@pytest.fixture
def sample_evaluation_example():
    """Create a sample evaluation example"""
    return EvaluationExample(
        example_id="val_001",
        question="What is the copay for specialist visits?",
        reference_answer="The copay for specialist visits is $50 per visit, and the deductible is waived.",
        ground_truth_chunk_ids=["chunk_1"],
        beir_failure_scale_factor=0.3
    )


@pytest.fixture
def sample_evaluation_dataset(sample_evaluation_example):
    """Create a minimal evaluation dataset for integration testing"""
    return [sample_evaluation_example]


class TestOrchestratorIntegration:
    """Integration tests for evaluation pipeline with real external services"""
    
    @pytest.mark.skipif(
        not Config.from_env().azure_ai_foundry_endpoint or not Config.from_env().azure_ai_foundry_api_key,
        reason="Azure AI Foundry credentials not configured"
    )
    @pytest.mark.skipif(
        not Config.from_env().azure_search_endpoint or not Config.from_env().azure_search_api_key,
        reason="Azure AI Search credentials not configured"
    )
    def test_full_pipeline_integration_with_real_services(
        self,
        sample_evaluation_dataset
    ):
        """Test complete evaluation pipeline with real external services
        
        This test verifies:
        1. Real Azure AI Search retrieval
        2. Real Azure AI Foundry generation
        3. Real Azure AI Foundry judge evaluation
        4. Meta-evaluation
        5. BEIR metrics computation
        
        Note: This test requires:
        - Azure AI Foundry credentials
        - Azure AI Search credentials
        - Supabase credentials (for prompt loading)
        - Indexed documents in Azure AI Search
        """
        config = Config.from_env()
        
        # Skip if credentials are missing
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure AI Foundry credentials not configured")
        
        if not config.azure_search_endpoint or not config.azure_search_api_key:
            pytest.skip("Azure AI Search credentials not configured")
        
        # Create database connection for prompt loading
        db_conn = None
        query_executor = None
        try:
            if config.database_url:
                db_conn = DatabaseConnection(config)
                db_conn.connect()
                query_executor = QueryExecutor(db_conn)
        except Exception as e:
            print(f"⚠ Database connection failed (prompt loading may use defaults): {e}")
        
        # Create RAG adapters
        rag_retriever = create_rag_retriever_adapter(config)
        rag_generator = create_rag_generator_adapter(config, query_executor)
        
        try:
            # Run evaluation pipeline
            results = evaluate_rag_system(
                evaluation_dataset=sample_evaluation_dataset,
                rag_retriever=rag_retriever,
                rag_generator=rag_generator,
                config=config
            )
            
            # Verify results
            assert len(results) == 1
            result = results[0]
            
            # Verify result structure
            assert isinstance(result, EvaluationResult)
            assert result.example_id == "val_001"
            assert result.judge_output is not None
            assert result.meta_eval_output is not None
            assert result.beir_metrics is not None
            assert result.timestamp is not None
            
            # Verify judge output
            assert isinstance(result.judge_output.correctness_binary, bool)
            assert isinstance(result.judge_output.hallucination_binary, bool)
            assert result.judge_output.reasoning is not None
            
            # Verify meta-eval output
            assert isinstance(result.meta_eval_output.judge_correct, bool)
            
            # Verify BEIR metrics
            assert 0.0 <= result.beir_metrics.recall_at_k <= 1.0
            assert 0.0 <= result.beir_metrics.precision_at_k <= 1.0
            assert 0.0 <= result.beir_metrics.ndcg_at_k <= 1.0
            
            print(f"✓ Integration test passed:")
            print(f"  - Judge correctness: {result.judge_output.correctness_binary}")
            print(f"  - Judge hallucination: {result.judge_output.hallucination_binary}")
            print(f"  - Meta-eval judge correct: {result.meta_eval_output.judge_correct}")
            print(f"  - BEIR recall@5: {result.beir_metrics.recall_at_k:.3f}")
            print(f"  - BEIR precision@5: {result.beir_metrics.precision_at_k:.3f}")
            print(f"  - BEIR nDCG@5: {result.beir_metrics.ndcg_at_k:.3f}")
            
        except Exception as e:
            # Check if error is due to missing prompt or other setup issues
            error_msg = str(e).lower()
            if "prompt" in error_msg and ("not found" in error_msg or "missing" in error_msg):
                pytest.skip(f"Integration test skipped: Prompt template not available ({e})")
            elif "index" in error_msg and ("empty" in error_msg or "not found" in error_msg):
                pytest.skip(f"Integration test skipped: Azure AI Search index not available ({e})")
            else:
                pytest.fail(f"Integration test failed: {e}")
        finally:
            if db_conn is not None:
                try:
                    db_conn.close()
                except Exception:
                    pass
    
    @pytest.mark.skipif(
        not Config.from_env().azure_ai_foundry_endpoint or not Config.from_env().azure_ai_foundry_api_key,
        reason="Azure AI Foundry credentials not configured"
    )
    @pytest.mark.skipif(
        not Config.from_env().azure_search_endpoint or not Config.from_env().azure_search_api_key,
        reason="Azure AI Search credentials not configured"
    )
    def test_pipeline_with_multiple_examples(
        self
    ):
        """Test pipeline with multiple evaluation examples"""
        config = Config.from_env()
        
        # Skip if credentials are missing
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure AI Foundry credentials not configured")
        
        if not config.azure_search_endpoint or not config.azure_search_api_key:
            pytest.skip("Azure AI Search credentials not configured")
        
        # Create minimal evaluation dataset (2 examples)
        evaluation_dataset = [
            EvaluationExample(
                example_id="val_001",
                question="What is the copay for specialist visits?",
                reference_answer="The copay for specialist visits is $50 per visit.",
                ground_truth_chunk_ids=["chunk_1"],
                beir_failure_scale_factor=0.3
            ),
            EvaluationExample(
                example_id="val_002",
                question="What is the deductible?",
                reference_answer="The deductible is $1,500.",
                ground_truth_chunk_ids=["chunk_0"],
                beir_failure_scale_factor=0.2
            )
        ]
        
        # Create database connection for prompt loading
        db_conn = None
        query_executor = None
        try:
            if config.database_url:
                db_conn = DatabaseConnection(config)
                db_conn.connect()
                query_executor = QueryExecutor(db_conn)
        except Exception as e:
            print(f"⚠ Database connection failed (prompt loading may use defaults): {e}")
        
        # Create RAG adapters
        rag_retriever = create_rag_retriever_adapter(config)
        rag_generator = create_rag_generator_adapter(config, query_executor)
        
        try:
            # Run evaluation pipeline
            results = evaluate_rag_system(
                evaluation_dataset=evaluation_dataset,
                rag_retriever=rag_retriever,
                rag_generator=rag_generator,
                config=config
            )
            
            # Verify results
            assert len(results) == 2
            assert all(isinstance(r, EvaluationResult) for r in results)
            assert results[0].example_id == "val_001"
            assert results[1].example_id == "val_002"
            
            # Verify all results have complete data
            for result in results:
                assert result.judge_output is not None
                assert result.meta_eval_output is not None
                assert result.beir_metrics is not None
                assert result.timestamp is not None
            
            print(f"✓ Integration test passed: Processed {len(results)} examples successfully")
            
        except Exception as e:
            # Check if error is due to missing prompt or other setup issues
            error_msg = str(e).lower()
            if "prompt" in error_msg and ("not found" in error_msg or "missing" in error_msg):
                pytest.skip(f"Integration test skipped: Prompt template not available ({e})")
            elif "index" in error_msg and ("empty" in error_msg or "not found" in error_msg):
                pytest.skip(f"Integration test skipped: Azure AI Search index not available ({e})")
            else:
                pytest.fail(f"Integration test failed: {e}")
        finally:
            if db_conn is not None:
                try:
                    db_conn.close()
                except Exception:
                    pass
    
    @pytest.mark.skipif(
        not Config.from_env().azure_ai_foundry_endpoint or not Config.from_env().azure_ai_foundry_api_key,
        reason="Azure AI Foundry credentials not configured"
    )
    @pytest.mark.skipif(
        not Config.from_env().azure_search_endpoint or not Config.from_env().azure_search_api_key,
        reason="Azure AI Search credentials not configured"
    )
    def test_judge_metrics_integration_with_real_services(
        self
    ):
        """Test judge performance metrics calculation with real pipeline results"""
        from src.services.evaluator.meta_eval import calculate_judge_metrics
        
        config = Config.from_env()
        
        # Skip if credentials are missing
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure AI Foundry credentials not configured")
        
        if not config.azure_search_endpoint or not config.azure_search_api_key:
            pytest.skip("Azure AI Search credentials not configured")
        
        # Create evaluation dataset
        evaluation_dataset = [
            EvaluationExample(
                example_id="val_001",
                question="What is the copay for specialist visits?",
                reference_answer="The copay for specialist visits is $50 per visit.",
                ground_truth_chunk_ids=["chunk_1"],
                beir_failure_scale_factor=0.3
            )
        ]
        
        # Create database connection for prompt loading
        db_conn = None
        query_executor = None
        try:
            if config.database_url:
                db_conn = DatabaseConnection(config)
                db_conn.connect()
                query_executor = QueryExecutor(db_conn)
        except Exception as e:
            print(f"⚠ Database connection failed (prompt loading may use defaults): {e}")
        
        # Create RAG adapters
        rag_retriever = create_rag_retriever_adapter(config)
        rag_generator = create_rag_generator_adapter(config, query_executor)
        
        try:
            # Run evaluation pipeline
            results = evaluate_rag_system(
                evaluation_dataset=evaluation_dataset,
                rag_retriever=rag_retriever,
                rag_generator=rag_generator,
                config=config
            )
            
            # Calculate judge performance metrics
            evaluation_pairs = [
                (r.judge_output, r.meta_eval_output) for r in results
            ]
            metrics = calculate_judge_metrics(evaluation_pairs)
            
            # Verify metrics structure
            assert metrics is not None
            assert metrics.correctness is not None
            assert metrics.hallucination is not None
            assert metrics.correctness.total_samples == len(results)
            assert metrics.hallucination.total_samples == len(results)
            
            # Verify metrics values are valid
            assert 0.0 <= metrics.correctness.precision <= 1.0
            assert 0.0 <= metrics.correctness.recall <= 1.0
            assert 0.0 <= metrics.correctness.f1_score <= 1.0
            assert 0.0 <= metrics.hallucination.precision <= 1.0
            assert 0.0 <= metrics.hallucination.recall <= 1.0
            assert 0.0 <= metrics.hallucination.f1_score <= 1.0
            
            print(f"✓ Judge metrics integration test passed:")
            print(f"  - Correctness precision: {metrics.correctness.precision:.3f}")
            print(f"  - Correctness recall: {metrics.correctness.recall:.3f}")
            print(f"  - Correctness F1: {metrics.correctness.f1_score:.3f}")
            print(f"  - Hallucination precision: {metrics.hallucination.precision:.3f}")
            print(f"  - Hallucination recall: {metrics.hallucination.recall:.3f}")
            print(f"  - Hallucination F1: {metrics.hallucination.f1_score:.3f}")
            
        except Exception as e:
            # Check if error is due to missing prompt or other setup issues
            error_msg = str(e).lower()
            if "prompt" in error_msg and ("not found" in error_msg or "missing" in error_msg):
                pytest.skip(f"Integration test skipped: Prompt template not available ({e})")
            elif "index" in error_msg and ("empty" in error_msg or "not found" in error_msg):
                pytest.skip(f"Integration test skipped: Azure AI Search index not available ({e})")
            else:
                pytest.fail(f"Integration test failed: {e}")
        finally:
            if db_conn is not None:
                try:
                    db_conn.close()
                except Exception:
                    pass


class TestOrchestratorConnection:
    """Connection tests for orchestrator external services"""
    
    def test_azure_services_connection_status(self):
        """Test connection status for all required Azure services"""
        config = Config.from_env()
        
        services_status = {
            "azure_ai_foundry": bool(config.azure_ai_foundry_endpoint and config.azure_ai_foundry_api_key),
            "azure_ai_search": bool(config.azure_search_endpoint and config.azure_search_api_key),
            "supabase": bool(config.database_url)
        }
        
        print("\n=== External Services Connection Status ===")
        for service, available in services_status.items():
            status = "✓ Available" if available else "✗ Not configured"
            print(f"  {service}: {status}")
        
        # This test always passes - it's informational only
        assert True
    
    @pytest.mark.skipif(
        not Config.from_env().azure_ai_foundry_endpoint or not Config.from_env().azure_ai_foundry_api_key,
        reason="Azure AI Foundry credentials not configured"
    )
    def test_azure_ai_foundry_judge_connection(self):
        """Test connection to Azure AI Foundry for judge evaluation"""
        from src.services.evaluator.judge import evaluate_answer_with_judge
        from src.core.interfaces import RetrievalResult
        
        config = Config.from_env()
        
        if not config.azure_ai_foundry_endpoint or not config.azure_ai_foundry_api_key:
            pytest.skip("Azure AI Foundry credentials not configured")
        
        # Create database connection for prompt loading
        db_conn = None
        query_executor = None
        try:
            if config.database_url:
                db_conn = DatabaseConnection(config)
                db_conn.connect()
                query_executor = QueryExecutor(db_conn)
        except Exception as e:
            print(f"⚠ Database connection failed (prompt loading may use defaults): {e}")
        
        try:
            # Test judge evaluation with minimal inputs
            retrieved_context = [
                RetrievalResult(
                    chunk_id="test_chunk",
                    similarity_score=0.9,
                    chunk_text="Test context for connection test."
                )
            ]
            
            result = evaluate_answer_with_judge(
                query="Test query",
                retrieved_context=retrieved_context,
                model_answer="Test answer",
                reference_answer="Test reference",
                config=config,
                query_executor=query_executor
            )
            
            # Verify result structure
            assert isinstance(result.correctness_binary, bool)
            assert isinstance(result.hallucination_binary, bool)
            assert result.reasoning is not None
            
            print(f"✓ Azure AI Foundry judge connection test passed")
            print(f"  - Correctness: {result.correctness_binary}")
            print(f"  - Hallucination: {result.hallucination_binary}")
            
        except Exception as e:
            pytest.fail(f"Azure AI Foundry judge connection test failed: {e}")
        finally:
            if db_conn is not None:
                try:
                    db_conn.close()
                except Exception:
                    pass
    
    @pytest.mark.skipif(
        not Config.from_env().azure_search_endpoint or not Config.from_env().azure_search_api_key,
        reason="Azure AI Search credentials not configured"
    )
    def test_azure_ai_search_retrieval_connection(self):
        """Test connection to Azure AI Search for retrieval"""
        config = Config.from_env()
        
        if not config.azure_search_endpoint or not config.azure_search_api_key:
            pytest.skip("Azure AI Search credentials not configured")
        
        try:
            # Test retrieval with a simple query
            query = Query(text="test query for connection test")
            results = retrieve_chunks(query, top_k=5, config=config)
            
            # Verify results (may be empty if index is empty, but connection should work)
            assert isinstance(results, list)
            
            print(f"✓ Azure AI Search retrieval connection test passed")
            print(f"  - Retrieved {len(results)} chunks")
            
        except Exception as e:
            pytest.fail(f"Azure AI Search retrieval connection test failed: {e}")

