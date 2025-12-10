"""End-to-end tests for complete upload and query pipelines"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from fastapi import UploadFile, HTTPException
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path

# Disable logging during tests
logging.disable(logging.CRITICAL)

from src.core.exceptions import AzureServiceError, DatabaseError, ValidationError
from src.core.config import Config
from src.core.interfaces import Query, ModelAnswer, Chunk, RetrievalResult
from src.api.routes.upload import handle_upload, UploadResponse
from src.api.routes.query import handle_query, QueryRequest, QueryResponse
from src.services.rag.pipeline import run_rag


@pytest.fixture
def mock_config():
    """Create a mock config with all required Azure credentials"""
    config = Mock(spec=Config)
    config.azure_document_intelligence_endpoint = "https://test-docint.cognitiveservices.azure.com"
    config.azure_document_intelligence_api_key = "test-docint-key"
    config.azure_ai_foundry_endpoint = "https://test-foundry.openai.azure.com"
    config.azure_ai_foundry_api_key = "test-foundry-key"
    config.azure_ai_foundry_embedding_model = "text-embedding-3-small"
    config.azure_ai_foundry_generation_model = "gpt-4o"
    config.azure_search_endpoint = "https://test-search.search.windows.net"
    config.azure_search_api_key = "test-search-key"
    config.azure_search_index_name = "test-index"
    config.supabase_url = "https://test.supabase.co"
    config.supabase_key = "test-supabase-key"
    config.database_url = "postgresql://test:test@localhost/test"
    config.azure_blob_connection_string = ""
    config.azure_blob_container_name = ""
    return config


@pytest.fixture
def sample_pdf_content():
    """Sample PDF file content for testing"""
    # Return minimal PDF content
    return b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\nxref\n0 0\ntrailer\n<< /Size 0 /Root 1 0 R >>\nstartxref\n0\n%%EOF"


@pytest.fixture
def sample_extracted_text():
    """Sample extracted text from document"""
    return "This is a sample document for testing. " * 50  # ~2000 characters


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing"""
    return [
        Chunk(
            text="This is chunk 1 with relevant information.",
            chunk_id="chunk_0",
            document_id="doc_123",
            metadata={"start": 0, "end": 50, "chunking_method": "fixed_size"}
        ),
        Chunk(
            text="This is chunk 2 with more information.",
            chunk_id="chunk_1",
            document_id="doc_123",
            metadata={"start": 50, "end": 100, "chunking_method": "fixed_size"}
        ),
        Chunk(
            text="This is chunk 3 with additional details.",
            chunk_id="chunk_2",
            document_id="doc_123",
            metadata={"start": 100, "end": 150, "chunking_method": "fixed_size"}
        )
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing"""
    return [
        [0.1] * 1536,  # Mock embedding for chunk 1
        [0.2] * 1536,  # Mock embedding for chunk 2
        [0.3] * 1536   # Mock embedding for chunk 3
    ]


@pytest.fixture
def sample_retrieval_results():
    """Sample retrieval results for testing"""
    return [
        RetrievalResult(
            chunk_id="chunk_0",
            similarity_score=0.92,
            chunk_text="This is chunk 1 with relevant information.",
            metadata={"start": 0, "end": 50}
        ),
        RetrievalResult(
            chunk_id="chunk_1",
            similarity_score=0.85,
            chunk_text="This is chunk 2 with more information.",
            metadata={"start": 50, "end": 100}
        )
    ]


@pytest.fixture
def sample_model_answer():
    """Sample model answer for testing"""
    return ModelAnswer(
        text="Based on the retrieved context, the answer is: This is a comprehensive response.",
        query_id="query_123",
        prompt_version="v1",
        retrieved_chunk_ids=["chunk_0", "chunk_1"],
        timestamp=datetime.now(timezone.utc)
    )


class TestEndToEndUploadPipeline:
    """End-to-end tests for complete upload pipeline"""
    
    @patch('src.api.routes.upload.doc_service')
    @patch('src.api.routes.upload.generate_image_preview')
    @patch('src.api.routes.upload.upload_document_to_storage')
    @patch('src.api.routes.upload.index_chunks')
    @patch('src.api.routes.upload.generate_embeddings')
    @patch('src.api.routes.upload.chunk_text')
    @patch('src.api.routes.upload.ingest_document')
    @patch('src.api.routes.upload.generate_id')
    @patch('src.api.routes.upload.config')
    @pytest.mark.asyncio
    async def test_upload_pipeline_complete_flow(
        self,
        mock_config_obj,
        mock_generate_id,
        mock_ingest_document,
        mock_chunk_text,
        mock_generate_embeddings,
        mock_index_chunks,
        mock_upload_storage,
        mock_generate_preview,
        mock_doc_service,
        mock_config,
        sample_pdf_content,
        sample_extracted_text,
        sample_chunks,
        sample_embeddings
    ):
        """Test complete upload pipeline end-to-end"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_generate_id.return_value = "doc_e2e_123"
        mock_upload_storage.return_value = "doc_e2e_123"
        mock_generate_preview.return_value = None
        mock_doc_service.insert_document.return_value = None
        mock_doc_service.update_document_status.return_value = None
        mock_doc_service.update_document_chunks.return_value = None
        mock_ingest_document.return_value = sample_extracted_text
        mock_chunk_text.return_value = sample_chunks
        mock_generate_embeddings.return_value = sample_embeddings
        mock_index_chunks.return_value = None
        
        # Create mock upload file
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "test_document.pdf"
        mock_file.read = AsyncMock(return_value=sample_pdf_content)
        
        # Execute upload
        result = await handle_upload(mock_file)
        
        # Verify complete pipeline flow
        assert isinstance(result, UploadResponse)
        assert result.document_id == "doc_e2e_123"
        assert result.status == "success"
        assert result.chunks_created == 3
        
        # Verify all pipeline steps were called in order
        mock_ingest_document.assert_called_once()
        assert mock_ingest_document.call_args[0][0] == sample_pdf_content
        
        mock_chunk_text.assert_called_once()
        assert mock_chunk_text.call_args[0][0] == sample_extracted_text
        assert mock_chunk_text.call_args[1]['document_id'] == "doc_e2e_123"
        
        mock_generate_embeddings.assert_called_once()
        assert mock_generate_embeddings.call_args[0][0] == sample_chunks
        
        mock_index_chunks.assert_called_once()
        assert mock_index_chunks.call_args[0][0] == sample_chunks
        assert mock_index_chunks.call_args[0][1] == sample_embeddings
    
    @patch('src.api.routes.upload.generate_image_preview')
    @patch('src.api.routes.upload.upload_document_to_storage')
    @patch('src.api.routes.upload.doc_service')
    @patch('src.api.routes.upload.index_chunks')
    @patch('src.api.routes.upload.generate_embeddings')
    @patch('src.api.routes.upload.chunk_text')
    @patch('src.api.routes.upload.ingest_document')
    @patch('src.api.routes.upload.generate_id')
    @patch('src.api.routes.upload.config')
    @pytest.mark.asyncio
    async def test_upload_pipeline_with_real_sample_document(
        self,
        mock_config_obj,
        mock_generate_id,
        mock_ingest_document,
        mock_chunk_text,
        mock_generate_embeddings,
        mock_index_chunks,
        mock_doc_service,
        mock_upload_storage,
        mock_generate_preview,
        sample_chunks,
        sample_embeddings
    ):
        """Test upload pipeline with actual sample PDF file"""
        # Try to load actual sample document
        sample_doc_path = Path(__file__).parent / "fixtures" / "sample_documents" / "healthguard_select_ppo_plan.pdf"
        
        if not sample_doc_path.exists():
            pytest.skip("Sample document not found")
        
        # Read actual PDF content
        with open(sample_doc_path, "rb") as f:
            pdf_content = f.read()
        
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_generate_id.return_value = "doc_real_123"
        mock_ingest_document.return_value = "Sample extracted text from real PDF document. " * 100
        mock_chunk_text.return_value = sample_chunks
        mock_generate_embeddings.return_value = sample_embeddings
        mock_index_chunks.return_value = None
        mock_upload_storage.return_value = "doc_real_123"
        mock_generate_preview.return_value = None
        mock_doc_service.insert_document.return_value = None
        mock_doc_service.update_document_status.return_value = None
        mock_doc_service.update_document_chunks.return_value = None
        
        # Create mock upload file
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "healthguard_select_ppo_plan.pdf"
        mock_file.read = AsyncMock(return_value=pdf_content)
        
        # Execute upload
        result = await handle_upload(mock_file)
        
        # Verify response
        assert isinstance(result, UploadResponse)
        assert result.status == "success"
        assert result.chunks_created == 3


class TestEndToEndQueryPipeline:
    """End-to-end tests for complete query pipeline"""
    
    @patch('src.api.routes.query.run_rag')
    @patch('src.api.routes.query.config')
    @pytest.mark.asyncio
    async def test_query_pipeline_complete_flow(
        self,
        mock_config_obj,
        mock_run_rag,
        mock_config,
        sample_model_answer
    ):
        """Test complete query pipeline end-to-end"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_run_rag.return_value = sample_model_answer
        
        # Create query request
        request = QueryRequest(
            text="What is the coverage limit?",
            prompt_version="v1"
        )
        
        # Execute query
        result = await handle_query(request)
        
        # Verify complete pipeline flow
        assert isinstance(result, QueryResponse)
        assert result.answer == sample_model_answer.text
        assert result.query_id == sample_model_answer.query_id
        assert result.prompt_version == sample_model_answer.prompt_version
        
        # Verify pipeline was called correctly
        mock_run_rag.assert_called_once()
        call_args = mock_run_rag.call_args
        assert call_args[0][0].text == request.text
        assert call_args[1]['prompt_version'] == request.prompt_version
    
    @patch('src.services.rag.pipeline.generate_query_embedding')
    @patch('src.services.rag.pipeline.retrieve_chunks')
    @patch('src.services.rag.pipeline.generate_answer')
    @patch('src.services.rag.pipeline.log_query')
    @patch('src.services.rag.pipeline.log_retrieval')
    @patch('src.services.rag.pipeline.log_model_answer')
    @patch('src.services.rag.pipeline.DatabaseConnection')
    @patch('src.services.rag.pipeline.QueryExecutor')
    def test_query_pipeline_internal_flow(
        self,
        mock_query_executor_class,
        mock_db_connection_class,
        mock_log_query,
        mock_log_retrieval,
        mock_log_model_answer,
        mock_generate_answer,
        mock_retrieve_chunks,
        mock_generate_query_embedding,
        mock_config,
        sample_retrieval_results,
        sample_model_answer
    ):
        """Test query pipeline internal flow with all components"""
        # Setup mocks
        mock_query_executor = Mock()
        mock_query_executor_class.return_value = mock_query_executor
        mock_db_connection = Mock()
        mock_db_connection.connect = Mock()
        mock_db_connection.close = Mock()
        mock_db_connection_class.return_value = mock_db_connection
        
        query_embedding = [0.5] * 1536
        mock_generate_query_embedding.return_value = query_embedding
        mock_retrieve_chunks.return_value = sample_retrieval_results
        mock_generate_answer.return_value = sample_model_answer
        mock_log_query.return_value = "query_123"
        mock_log_retrieval.return_value = None
        mock_log_model_answer.return_value = "answer_123"
        
        # Create query
        query = Query(text="What is the coverage limit?")
        
        # Execute pipeline
        result = run_rag(query, prompt_version="v1", config=mock_config)
        
        # Verify complete flow
        assert isinstance(result, ModelAnswer)
        assert result.text == sample_model_answer.text
        assert result.query_id == sample_model_answer.query_id
        
        # Verify all steps were called
        mock_generate_query_embedding.assert_called_once()
        mock_retrieve_chunks.assert_called_once()
        mock_generate_answer.assert_called_once()
        mock_log_query.assert_called_once()
        mock_log_retrieval.assert_called_once()
        mock_log_model_answer.assert_called_once()


class TestEndToEndPromptVersions:
    """Test query pipeline with multiple prompt versions"""
    
    @patch('src.api.routes.query.run_rag')
    @patch('src.api.routes.query.config')
    @pytest.mark.asyncio
    async def test_query_with_v1_prompt(
        self,
        mock_config_obj,
        mock_run_rag,
        mock_config
    ):
        """Test query with v1 prompt version"""
        # Setup
        mock_config_obj.return_value = mock_config
        answer_v1 = ModelAnswer(
            text="Answer with v1 prompt",
            query_id="query_v1",
            prompt_version="v1",
            retrieved_chunk_ids=["chunk_1"],
            timestamp=datetime.now(timezone.utc)
        )
        mock_run_rag.return_value = answer_v1
        
        # Execute
        request = QueryRequest(text="Test query", prompt_version="v1")
        result = await handle_query(request)
        
        # Verify
        assert result.prompt_version == "v1"
        assert result.answer == "Answer with v1 prompt"
        mock_run_rag.assert_called_once()
        call_args = mock_run_rag.call_args
        assert call_args[1]['prompt_version'] == "v1"
    
    @patch('src.api.routes.query.run_rag')
    @patch('src.api.routes.query.config')
    @pytest.mark.asyncio
    async def test_query_with_v2_prompt(
        self,
        mock_config_obj,
        mock_run_rag,
        mock_config
    ):
        """Test query with v2 prompt version"""
        # Setup
        mock_config_obj.return_value = mock_config
        answer_v2 = ModelAnswer(
            text="Answer with v2 prompt",
            query_id="query_v2",
            prompt_version="v2",
            retrieved_chunk_ids=["chunk_1"],
            timestamp=datetime.now(timezone.utc)
        )
        mock_run_rag.return_value = answer_v2
        
        # Execute
        request = QueryRequest(text="Test query", prompt_version="v2")
        result = await handle_query(request)
        
        # Verify
        assert result.prompt_version == "v2"
        assert result.answer == "Answer with v2 prompt"
        mock_run_rag.assert_called_once()
        call_args = mock_run_rag.call_args
        assert call_args[1]['prompt_version'] == "v2"


class TestEndToEndErrorScenarios:
    """Test error scenarios in end-to-end pipelines"""
    
    @patch('src.api.routes.upload.generate_image_preview')
    @patch('src.api.routes.upload.upload_document_to_storage')
    @patch('src.api.routes.upload.doc_service')
    @patch('src.api.routes.upload.ingest_document')
    @patch('src.api.routes.upload.generate_id')
    @patch('src.api.routes.upload.config')
    @pytest.mark.asyncio
    async def test_upload_invalid_document(
        self,
        mock_config_obj,
        mock_generate_id,
        mock_ingest_document,
        mock_doc_service,
        mock_upload_storage,
        mock_generate_preview,
        mock_config,
        sample_pdf_content
    ):
        """Test upload with invalid document (no text extracted)"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_generate_id.return_value = "doc_invalid"
        mock_ingest_document.return_value = ""  # Empty text
        mock_upload_storage.return_value = "doc_invalid"
        mock_generate_preview.return_value = None
        mock_doc_service.insert_document.return_value = None
        mock_doc_service.update_document_status.return_value = None
        
        # Create mock upload file
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "invalid.pdf"
        mock_file.read = AsyncMock(return_value=sample_pdf_content)
        
        # Execute and verify exception
        with pytest.raises(HTTPException) as exc_info:
            await handle_upload(mock_file)
        
        assert exc_info.value.status_code == 400
        assert "No text could be extracted" in exc_info.value.detail
    
    @patch('src.api.routes.query.run_rag')
    @patch('src.api.routes.query.config')
    @pytest.mark.asyncio
    async def test_query_invalid_query(
        self,
        mock_config_obj,
        mock_run_rag,
        mock_config
    ):
        """Test query with invalid input"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_run_rag.side_effect = ValueError("Invalid query text")
        
        # Execute
        request = QueryRequest(text="", prompt_version="v1")
        
        # Verify exception
        with pytest.raises(HTTPException) as exc_info:
            await handle_query(request)
        
        assert exc_info.value.status_code == 500
        assert "Invalid query text" in exc_info.value.detail
    
    @patch('src.api.routes.query.run_rag')
    @patch('src.api.routes.query.config')
    @pytest.mark.asyncio
    async def test_query_missing_prompt_version(
        self,
        mock_config_obj,
        mock_run_rag,
        mock_config
    ):
        """Test query with missing prompt version"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_run_rag.side_effect = ValidationError("Prompt version 'v99' not found in database")
        
        # Execute
        request = QueryRequest(text="Test query", prompt_version="v99")
        
        # Verify exception
        with pytest.raises(HTTPException) as exc_info:
            await handle_query(request)
        
        assert exc_info.value.status_code == 500
        assert "not found" in exc_info.value.detail.lower()
    
    @patch('src.api.routes.upload.ingest_document')
    @patch('src.api.routes.upload.generate_id')
    @patch('src.api.routes.upload.config')
    @pytest.mark.asyncio
    async def test_upload_azure_service_failure(
        self,
        mock_config_obj,
        mock_generate_id,
        mock_ingest_document,
        mock_config,
        sample_pdf_content
    ):
        """Test upload with Azure service failure"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_generate_id.return_value = "doc_fail"
        mock_ingest_document.side_effect = AzureServiceError("Azure Document Intelligence service unavailable")
        
        # Create mock upload file
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "test.pdf"
        mock_file.read = AsyncMock(return_value=sample_pdf_content)
        
        # Execute and verify exception
        with pytest.raises(HTTPException) as exc_info:
            await handle_upload(mock_file)
        
        assert exc_info.value.status_code == 500
        assert "Upload processing failed" in exc_info.value.detail
    
    @patch('src.api.routes.query.run_rag')
    @patch('src.api.routes.query.config')
    @pytest.mark.asyncio
    async def test_query_azure_service_failure(
        self,
        mock_config_obj,
        mock_run_rag,
        mock_config
    ):
        """Test query with Azure service failure"""
        # Setup mocks
        mock_config_obj.return_value = mock_config
        mock_run_rag.side_effect = AzureServiceError("Azure AI Foundry service unavailable")
        
        # Execute
        request = QueryRequest(text="Test query", prompt_version="v1")
        
        # Verify exception
        with pytest.raises(HTTPException) as exc_info:
            await handle_query(request)
        
        assert exc_info.value.status_code == 500
        assert "service unavailable" in exc_info.value.detail.lower()


class TestEndToEndIntegration:
    """Integration tests for complete system"""
    
    @patch('src.api.routes.upload.index_chunks')
    @patch('src.api.routes.upload.generate_embeddings')
    @patch('src.api.routes.upload.chunk_text')
    @patch('src.api.routes.upload.ingest_document')
    @patch('src.api.routes.upload.generate_id')
    @patch('src.api.routes.upload.config')
    @patch('src.api.routes.query.run_rag')
    @patch('src.api.routes.query.config')
    @pytest.mark.asyncio
    async def test_upload_then_query_flow(
        self,
        mock_query_config_obj,
        mock_run_rag,
        mock_upload_config_obj,
        mock_generate_id,
        mock_ingest_document,
        mock_chunk_text,
        mock_generate_embeddings,
        mock_index_chunks,
        mock_config,
        sample_pdf_content,
        sample_extracted_text,
        sample_chunks,
        sample_embeddings,
        sample_model_answer
    ):
        """Test complete flow: upload document, then query it"""
        # Setup upload mocks
        mock_upload_config_obj.return_value = mock_config
        mock_generate_id.return_value = "doc_integration_123"
        mock_ingest_document.return_value = sample_extracted_text
        mock_chunk_text.return_value = sample_chunks
        mock_generate_embeddings.return_value = sample_embeddings
        mock_index_chunks.return_value = None
        
        # Setup query mocks
        mock_query_config_obj.return_value = mock_config
        mock_run_rag.return_value = sample_model_answer
        
        # Step 1: Upload document
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "test_document.pdf"
        mock_file.read = AsyncMock(return_value=sample_pdf_content)
        
        upload_result = await handle_upload(mock_file)
        assert upload_result.status == "success"
        assert upload_result.chunks_created == 3
        
        # Step 2: Query the uploaded document
        query_request = QueryRequest(
            text="What information is in the document?",
            prompt_version="v1"
        )
        
        query_result = await handle_query(query_request)
        assert query_result.answer == sample_model_answer.text
        assert query_result.prompt_version == "v1"
        
        # Verify all components were called
        mock_ingest_document.assert_called_once()
        mock_chunk_text.assert_called_once()
        mock_generate_embeddings.assert_called_once()
        mock_index_chunks.assert_called_once()
        mock_run_rag.assert_called_once()

