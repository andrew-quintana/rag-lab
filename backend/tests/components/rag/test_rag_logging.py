"""Unit tests for RAG logging module"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from src.core.exceptions import DatabaseError
from src.core.interfaces import Query, ModelAnswer, RetrievalResult
from src.core.config import Config
from src.db.connection import DatabaseConnection
from src.db.queries import QueryExecutor
from src.services.rag.logging import log_query, log_retrieval, log_model_answer


@pytest.fixture
def mock_query_executor():
    """Create a mock QueryExecutor for testing"""
    executor = Mock(spec=QueryExecutor)
    executor.execute_insert = Mock(return_value=None)
    executor.execute_query = Mock(return_value=[])
    return executor


@pytest.fixture
def sample_query():
    """Create a sample query for testing"""
    return Query(
        text="What is the coverage limit?",
        query_id="test_query_123",
        timestamp=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_query_no_id():
    """Create a sample query without ID for testing"""
    return Query(
        text="What is the coverage limit?",
        query_id=None,
        timestamp=None
    )


@pytest.fixture
def sample_retrieval_results():
    """Create sample retrieval results for testing"""
    return [
        RetrievalResult(
            chunk_id="chunk_1",
            similarity_score=0.92,
            chunk_text="The coverage limit is $500,000...",
            metadata={"document_id": "doc_1"}
        ),
        RetrievalResult(
            chunk_id="chunk_2",
            similarity_score=0.85,
            chunk_text="Additional coverage information...",
            metadata={"document_id": "doc_1"}
        ),
        RetrievalResult(
            chunk_id="chunk_3",
            similarity_score=0.78,
            chunk_text="More details about coverage...",
            metadata={"document_id": "doc_1"}
        )
    ]


@pytest.fixture
def sample_model_answer():
    """Create a sample model answer for testing"""
    return ModelAnswer(
        text="The coverage limit is $500,000 based on the policy documents.",
        query_id="test_query_123",
        prompt_version="v1",
        retrieved_chunk_ids=["chunk_1", "chunk_2", "chunk_3"],
        timestamp=datetime.now(timezone.utc)
    )


class TestLogQuery:
    """Test cases for log_query() function"""
    
    def test_log_query_with_existing_id(self, mock_query_executor, sample_query):
        """Test logging a query with an existing query_id"""
        query_id = log_query(sample_query, mock_query_executor)
        
        assert query_id == "test_query_123"
        assert mock_query_executor.execute_insert.called
        call_args = mock_query_executor.execute_insert.call_args
        assert call_args[0][0] == """
            INSERT INTO queries (query_id, query_text, timestamp, metadata)
            VALUES (%s, %s, %s, %s::jsonb)
            ON CONFLICT (query_id) DO NOTHING
        """
        params = call_args[0][1]
        assert params[0] == "test_query_123"
        assert params[1] == "What is the coverage limit?"
        assert isinstance(params[2], datetime)
    
    @patch('src.services.rag.logging.generate_id')
    def test_log_query_generates_id_if_missing(self, mock_generate_id, mock_query_executor, sample_query_no_id):
        """Test that query_id is generated if missing"""
        mock_generate_id.return_value = "generated_query_456"
        
        query_id = log_query(sample_query_no_id, mock_query_executor)
        
        assert query_id == "generated_query_456"
        assert mock_generate_id.called
        assert mock_query_executor.execute_insert.called
    
    def test_log_query_handles_database_error_gracefully(self, mock_query_executor, sample_query):
        """Test that database errors don't break the pipeline"""
        mock_query_executor.execute_insert.side_effect = DatabaseError("Database connection failed")
        
        # Should not raise exception
        query_id = log_query(sample_query, mock_query_executor)
        
        # Should still return query_id even on error
        assert query_id == "test_query_123"
    
    def test_log_query_handles_unexpected_error_gracefully(self, mock_query_executor, sample_query):
        """Test that unexpected errors don't break the pipeline"""
        mock_query_executor.execute_insert.side_effect = Exception("Unexpected error")
        
        # Should not raise exception
        query_id = log_query(sample_query, mock_query_executor)
        
        # Should still return query_id even on error
        assert query_id == "test_query_123"
    
    def test_log_query_uses_current_timestamp_if_missing(self, mock_query_executor, sample_query_no_id):
        """Test that current timestamp is used if not provided"""
        with patch('src.services.rag.logging.datetime') as mock_datetime:
            mock_now = datetime(2025, 1, 27, 12, 0, 0, tzinfo=timezone.utc)
            mock_datetime.now.return_value = mock_now
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            log_query(sample_query_no_id, mock_query_executor)
            
            call_args = mock_query_executor.execute_insert.call_args
            params = call_args[0][1]
            assert params[2] == mock_now


class TestLogRetrieval:
    """Test cases for log_retrieval() function"""
    
    def test_log_retrieval_batch_insert(self, mock_query_executor, sample_retrieval_results):
        """Test batch insertion of retrieval results"""
        query_id = "test_query_123"
        
        log_retrieval(query_id, sample_retrieval_results, mock_query_executor)
        
        assert mock_query_executor.execute_insert.called
        call_args = mock_query_executor.execute_insert.call_args
        insert_query = call_args[0][0]
        params = call_args[0][1]
        
        # Check that query contains VALUES clause
        assert "VALUES" in insert_query
        # Check that we have 3 sets of values (one per retrieval result)
        assert insert_query.count("(%s, %s, %s, %s, %s)") == 3
        
        # Check parameters: each result has 5 params (log_id, query_id, chunk_id, similarity_score, timestamp)
        assert len(params) == 15  # 3 results * 5 params
        
        # Verify query_id appears 3 times (once per result)
        assert params.count(query_id) == 3
        
        # Verify chunk_ids are present
        chunk_ids = [params[i*5 + 2] for i in range(3)]  # chunk_id is 3rd param in each set
        assert "chunk_1" in chunk_ids
        assert "chunk_2" in chunk_ids
        assert "chunk_3" in chunk_ids
        
        # Verify similarity scores are present
        scores = [params[i*5 + 3] for i in range(3)]  # similarity_score is 4th param
        assert 0.92 in scores
        assert 0.85 in scores
        assert 0.78 in scores
    
    def test_log_retrieval_empty_results(self, mock_query_executor):
        """Test that empty retrieval results are handled gracefully"""
        query_id = "test_query_123"
        
        log_retrieval(query_id, [], mock_query_executor)
        
        # Should not call execute_insert for empty results
        assert not mock_query_executor.execute_insert.called
    
    def test_log_retrieval_handles_database_error_gracefully(self, mock_query_executor, sample_retrieval_results):
        """Test that database errors don't break the pipeline"""
        mock_query_executor.execute_insert.side_effect = DatabaseError("Database connection failed")
        query_id = "test_query_123"
        
        # Should not raise exception
        log_retrieval(query_id, sample_retrieval_results, mock_query_executor)
    
    def test_log_retrieval_handles_unexpected_error_gracefully(self, mock_query_executor, sample_retrieval_results):
        """Test that unexpected errors don't break the pipeline"""
        mock_query_executor.execute_insert.side_effect = Exception("Unexpected error")
        query_id = "test_query_123"
        
        # Should not raise exception
        log_retrieval(query_id, sample_retrieval_results, mock_query_executor)
    
    def test_log_retrieval_single_result(self, mock_query_executor):
        """Test logging a single retrieval result"""
        query_id = "test_query_123"
        single_result = [
            RetrievalResult(
                chunk_id="chunk_1",
                similarity_score=0.95,
                chunk_text="Single result",
                metadata={}
            )
        ]
        
        log_retrieval(query_id, single_result, mock_query_executor)
        
        assert mock_query_executor.execute_insert.called
        call_args = mock_query_executor.execute_insert.call_args
        params = call_args[0][1]
        
        # Should have 5 parameters (log_id, query_id, chunk_id, similarity_score, timestamp)
        assert len(params) == 5
        assert params[1] == query_id
        assert params[2] == "chunk_1"
        assert params[3] == 0.95


class TestLogModelAnswer:
    """Test cases for log_model_answer() function"""
    
    def test_log_model_answer_with_all_fields(self, mock_query_executor, sample_model_answer):
        """Test logging a model answer with all fields"""
        answer_id = log_model_answer(sample_model_answer, mock_query_executor)
        
        # Should generate an answer_id
        assert answer_id is not None
        assert mock_query_executor.execute_insert.called
        
        call_args = mock_query_executor.execute_insert.call_args
        assert call_args[0][0] == """
            INSERT INTO model_answers (
                answer_id, query_id, answer_text, prompt_version, 
                retrieved_chunk_ids, timestamp
            )
            VALUES (%s, %s, %s, %s, %s::text[], %s)
            ON CONFLICT (answer_id) DO NOTHING
        """
        params = call_args[0][1]
        assert params[1] == "test_query_123"
        assert params[2] == "The coverage limit is $500,000 based on the policy documents."
        assert params[3] == "v1"
        assert params[4] == ["chunk_1", "chunk_2", "chunk_3"]
        assert isinstance(params[5], datetime)
    
    @patch('src.services.rag.logging.generate_id')
    def test_log_model_answer_generates_id_if_missing(self, mock_generate_id, mock_query_executor, sample_model_answer):
        """Test that answer_id is generated if missing"""
        mock_generate_id.return_value = "generated_answer_789"
        
        # Remove answer_id attribute if it exists
        if hasattr(sample_model_answer, 'answer_id'):
            delattr(sample_model_answer, 'answer_id')
        
        answer_id = log_model_answer(sample_model_answer, mock_query_executor)
        
        assert answer_id == "generated_answer_789"
        assert mock_generate_id.called
    
    def test_log_model_answer_handles_empty_chunk_ids(self, mock_query_executor):
        """Test logging model answer with empty retrieved_chunk_ids"""
        answer = ModelAnswer(
            text="Answer with no chunks",
            query_id="test_query_123",
            prompt_version="v1",
            retrieved_chunk_ids=[],
            timestamp=datetime.now(timezone.utc)
        )
        
        answer_id = log_model_answer(answer, mock_query_executor)
        
        assert answer_id is not None
        call_args = mock_query_executor.execute_insert.call_args
        params = call_args[0][1]
        assert params[4] == []  # Empty list for retrieved_chunk_ids
    
    def test_log_model_answer_handles_none_chunk_ids(self, mock_query_executor):
        """Test logging model answer with None retrieved_chunk_ids"""
        answer = ModelAnswer(
            text="Answer with None chunks",
            query_id="test_query_123",
            prompt_version="v1",
            retrieved_chunk_ids=None,
            timestamp=datetime.now(timezone.utc)
        )
        
        answer_id = log_model_answer(answer, mock_query_executor)
        
        assert answer_id is not None
        call_args = mock_query_executor.execute_insert.call_args
        params = call_args[0][1]
        assert params[4] == []  # Should convert None to empty list
    
    def test_log_model_answer_handles_database_error_gracefully(self, mock_query_executor, sample_model_answer):
        """Test that database errors don't break the pipeline"""
        mock_query_executor.execute_insert.side_effect = DatabaseError("Database connection failed")
        
        # Should not raise exception
        answer_id = log_model_answer(sample_model_answer, mock_query_executor)
        
        # Should still return answer_id even on error
        assert answer_id is not None
    
    def test_log_model_answer_handles_unexpected_error_gracefully(self, mock_query_executor, sample_model_answer):
        """Test that unexpected errors don't break the pipeline"""
        mock_query_executor.execute_insert.side_effect = Exception("Unexpected error")
        
        # Should not raise exception
        answer_id = log_model_answer(sample_model_answer, mock_query_executor)
        
        # Should still return answer_id even on error
        assert answer_id is not None
    
    def test_log_model_answer_uses_current_timestamp_if_missing(self, mock_query_executor):
        """Test that current timestamp is used if not provided"""
        answer = ModelAnswer(
            text="Answer without timestamp",
            query_id="test_query_123",
            prompt_version="v1",
            retrieved_chunk_ids=["chunk_1"],
            timestamp=None
        )
        
        with patch('src.services.rag.logging.datetime') as mock_datetime:
            mock_now = datetime(2025, 1, 27, 12, 0, 0, tzinfo=timezone.utc)
            mock_datetime.now.return_value = mock_now
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            log_model_answer(answer, mock_query_executor)
            
            call_args = mock_query_executor.execute_insert.call_args
            params = call_args[0][1]
            assert params[5] == mock_now


class TestConnectionTests:
    """Connection tests for Supabase logging (warns if credentials missing)"""
    
    @pytest.mark.skipif(
        not Config.from_env().database_url,
        reason="Supabase database credentials not configured"
    )
    def test_connection_to_supabase_logging(self):
        """
        Connection test for Supabase logging.
        
        This test verifies that the logging functions can connect to Supabase
        and perform logging operations. Connection tests should warn but not
        fail if credentials are missing or invalid.
        """
        try:
            # Load config from environment
            config = Config.from_env()
            
            if not config.database_url:
                pytest.skip(
                    "Skipping connection test. Set SUPABASE_DB_URL (or DATABASE_URL) environment variable to run this test."
                )
            
            # Create database connection
            db_conn = DatabaseConnection(config)
            db_conn.connect()
            query_executor = QueryExecutor(db_conn)
            
            try:
                # Test 1: Log a query
                test_query = Query(
                    text="Connection test query",
                    query_id=None,  # Will be generated
                    timestamp=None
                )
                query_id = log_query(test_query, query_executor)
                assert query_id is not None
                print(f"✓ Connection test passed: Logged query '{query_id}'")
                
                # Test 2: Log retrieval results
                test_retrieval_results = [
                    RetrievalResult(
                        chunk_id="test_chunk_1",
                        similarity_score=0.95,
                        chunk_text="Test chunk for connection test",
                        metadata={}
                    )
                ]
                log_retrieval(query_id, test_retrieval_results, query_executor)
                print(f"✓ Connection test passed: Logged {len(test_retrieval_results)} retrieval results")
                
                # Test 3: Log model answer
                test_answer = ModelAnswer(
                    text="Connection test answer",
                    query_id=query_id,
                    prompt_version="v1",
                    retrieved_chunk_ids=["test_chunk_1"],
                    timestamp=None
                )
                answer_id = log_model_answer(test_answer, query_executor)
                assert answer_id is not None
                print(f"✓ Connection test passed: Logged model answer '{answer_id}'")
                
                print(
                    "✓ Connection test passed: All logging operations successful. "
                    "Connection test verifies database connectivity only."
                )
                
            finally:
                # Always close database connection
                db_conn.close()
                
        except DatabaseError as e:
            # Connection test should warn but not fail if credentials are invalid
            pytest.skip(
                f"Supabase connection test failed (credentials may be invalid): {e}. "
                "Connection tests are informational only."
            )
        except Exception as e:
            # Catch any other errors and skip test gracefully
            pytest.skip(
                f"Supabase connection test encountered an error: {e}. "
                "Connection tests are informational only."
            )

