#!/usr/bin/env python3
"""Generate synthetic test data for Azure Functions testing

This script creates test documents at various pipeline stages:
- uploaded: Ready for ingestion-worker
- parsed: Ready for chunking-worker (with extracted text)
- chunked: Ready for embedding-worker (with chunks)
- embedded: Ready for indexing-worker (with embeddings)

Usage:
    python scripts/generate_test_data.py --stage uploaded
    python scripts/generate_test_data.py --stage parsed
    python scripts/generate_test_data.py --stage chunked
    python scripts/generate_test_data.py --stage embedded
    python scripts/generate_test_data.py --all  # Create one document at each stage
"""

import os
import sys
import uuid
import argparse
import base64
from pathlib import Path
from datetime import datetime
from typing import List

# Add backend to path for imports
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from src.core.config import Config
from src.db.connection import DatabaseConnection
from src.db.documents import DocumentService
from src.db.models import Document
from src.services.workers.persistence import (
    persist_extracted_text,
    persist_chunks,
    persist_embeddings
)
from src.services.rag.chunking import Chunk
from src.services.rag.supabase_storage import _get_supabase_client
from dotenv import load_dotenv

# Load environment
env_file = Path(__file__).parent.parent / ".env.local"
if env_file.exists():
    load_dotenv(env_file)

DOCUMENTS_BUCKET = "documents"


def create_test_document(stage: str, config: Config) -> str:
    """
    Create a test document at a specific pipeline stage.
    
    Args:
        stage: One of 'uploaded', 'parsed', 'chunked', 'embedded'
        config: Application configuration
        
    Returns:
        Document ID (UUID)
    """
    document_id = str(uuid.uuid4())
    
    print(f"\n{'='*70}")
    print(f"Creating test document at stage: {stage}")
    print(f"{'='*70}")
    print(f"Document ID: {document_id}")
    
    # Step 1: Create database record
    db_conn = DatabaseConnection(config)
    db_conn.connect()
    doc_service = DocumentService(db_conn)
    
    try:
        # Create document record
        document = Document(
            id=document_id,
            filename=f"test-{stage}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.txt",
            file_size=100,
            mime_type="text/plain",
            upload_timestamp=datetime.utcnow(),
            status=stage,
            storage_path=document_id,  # Storage path is the document_id itself
            metadata={"test": True, "stage": stage}
        )
        doc_service.insert_document(document)
        print(f"✅ Created database record with status='{stage}'")
        
        # Step 2: Upload file to storage if stage is uploaded
        if stage in ["uploaded", "parsed", "chunked", "embedded"]:
            upload_test_file(document_id, config)
        
        # Step 3: Add stage-specific data
        if stage in ["parsed", "chunked", "embedded"]:
            create_extracted_text(document_id, config)
        
        if stage in ["chunked", "embedded"]:
            create_chunks(document_id, config)
        
        if stage == "embedded":
            create_embeddings(document_id, config)
        
        print(f"\n✅ Test document created successfully!")
        print(f"   Document ID: {document_id}")
        print(f"   Stage: {stage}")
        print(f"   Ready for: {get_next_worker(stage)}")
        
        return document_id
        
    except Exception as e:
        print(f"❌ Error creating test document: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        db_conn.close()


def upload_test_file(document_id: str, config: Config) -> None:
    """Upload a small test file to Supabase storage"""
    try:
        client = _get_supabase_client(config)
        
        # Create dummy file content
        test_content = b"This is a test document for Azure Functions validation."
        
        # Upload to Supabase storage using document_id as the path
        # (ingestion worker downloads using document_id directly)
        storage_path = document_id
        client.storage.from_(DOCUMENTS_BUCKET).upload(
            path=storage_path,
            file=test_content,
            file_options={"content-type": "text/plain", "upsert": "true"}
        )
        
        print(f"✅ Uploaded test file to Supabase storage: {storage_path}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not upload to storage (non-fatal): {e}")


def create_extracted_text(document_id: str, config: Config) -> None:
    """Create sample extracted text for the document"""
    extracted_text = """This is sample extracted text for testing purposes.

It contains multiple paragraphs and sections to simulate real document text.

Section 1: Introduction
This section introduces the test document and explains its purpose.

Section 2: Details
Here we provide more detailed information about the document content.

Section 3: Conclusion
This concludes the test document content."""
    
    try:
        persist_extracted_text(document_id, extracted_text, config)
        print(f"✅ Created extracted text ({len(extracted_text)} chars)")
    except Exception as e:
        print(f"❌ Failed to create extracted text: {e}")
        raise


def create_chunks(document_id: str, config: Config) -> None:
    """Create sample chunks for the document"""
    chunks = [
        Chunk(
            chunk_id=f"{document_id}-chunk-0",
            document_id=document_id,
            text="This is sample extracted text for testing purposes. It contains multiple paragraphs and sections to simulate real document text.",
            metadata={"chunk_index": 0, "page": 1}
        ),
        Chunk(
            chunk_id=f"{document_id}-chunk-1",
            document_id=document_id,
            text="Section 1: Introduction. This section introduces the test document and explains its purpose.",
            metadata={"chunk_index": 1, "page": 1}
        ),
        Chunk(
            chunk_id=f"{document_id}-chunk-2",
            document_id=document_id,
            text="Section 2: Details. Here we provide more detailed information about the document content.",
            metadata={"chunk_index": 2, "page": 1}
        ),
        Chunk(
            chunk_id=f"{document_id}-chunk-3",
            document_id=document_id,
            text="Section 3: Conclusion. This concludes the test document content.",
            metadata={"chunk_index": 3, "page": 1}
        )
    ]
    
    try:
        persist_chunks(document_id, chunks, config)
        print(f"✅ Created {len(chunks)} chunks")
    except Exception as e:
        print(f"❌ Failed to create chunks: {e}")
        raise


def create_embeddings(document_id: str, config: Config) -> None:
    """Create sample embeddings for the document chunks"""
    # Create dummy embeddings (1536 dimensions for text-embedding-3-small)
    # In reality these would come from the embedding model
    embeddings = [
        [0.1] * 1536,  # chunk 0
        [0.2] * 1536,  # chunk 1
        [0.3] * 1536,  # chunk 2
        [0.4] * 1536,  # chunk 3
    ]
    
    try:
        persist_embeddings(document_id, embeddings, config)
        print(f"✅ Created {len(embeddings)} embeddings (1536 dims each)")
    except Exception as e:
        print(f"❌ Failed to create embeddings: {e}")
        raise


def get_next_worker(stage: str) -> str:
    """Get the name of the next worker for this stage"""
    stage_map = {
        "uploaded": "ingestion-worker (extracts text)",
        "parsed": "chunking-worker (creates chunks)",
        "chunked": "embedding-worker (generates embeddings)",
        "embedded": "indexing-worker (indexes to Azure AI Search)"
    }
    return stage_map.get(stage, "unknown")


def send_test_message(document_id: str, stage: str) -> str:
    """
    Generate base64-encoded queue message for testing.
    
    Returns:
        Base64-encoded JSON message ready for Azure Storage Queue
    """
    import json
    
    queue_map = {
        "uploaded": "ingestion-uploads",
        "parsed": "ingestion-chunking",
        "chunked": "ingestion-embeddings",
        "embedded": "ingestion-indexing"
    }
    
    message = {
        "document_id": document_id,
        "source_storage": "supabase",
        "filename": f"test-{stage}.txt",
        "attempt": 1,
        "stage": stage,
        "metadata": {"test": True}
    }
    
    # Encode message as base64 (required by Azure Functions host.json)
    message_json = json.dumps(message)
    message_b64 = base64.b64encode(message_json.encode()).decode()
    
    queue_name = queue_map.get(stage, "unknown")
    
    print(f"\n{'='*70}")
    print(f"Queue Message Ready")
    print(f"{'='*70}")
    print(f"Queue: {queue_name}")
    print(f"Document ID: {document_id}")
    print(f"\nTo send this message to Azure Queue:")
    print(f'\naz storage message put \\')
    print(f'  --queue-name {queue_name} \\')
    print(f'  --connection-string "$AZURE_STORAGE_CONNECTION_STRING" \\')
    print(f"  --content '{message_b64}'")
    
    return message_b64


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic test data for Azure Functions testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_test_data.py --stage uploaded
  python scripts/generate_test_data.py --stage parsed
  python scripts/generate_test_data.py --all
        """
    )
    
    parser.add_argument("--stage", choices=["uploaded", "parsed", "chunked", "embedded"],
                       help="Pipeline stage to create test document for")
    parser.add_argument("--all", action="store_true",
                       help="Create one test document at each stage")
    parser.add_argument("--send-message", action="store_true",
                       help="Generate queue message command after creating document")
    
    args = parser.parse_args()
    
    if not args.stage and not args.all:
        parser.error("Must specify --stage or --all")
    
    # Load configuration
    config = Config.from_env()
    
    if args.all:
        stages = ["uploaded", "parsed", "chunked", "embedded"]
        doc_ids = {}
        for stage in stages:
            doc_id = create_test_document(stage, config)
            doc_ids[stage] = doc_id
            if args.send_message:
                send_test_message(doc_id, stage)
        
        print(f"\n{'='*70}")
        print("All test documents created successfully!")
        print(f"{'='*70}")
        for stage, doc_id in doc_ids.items():
            print(f"  {stage:12} → {doc_id}")
    else:
        doc_id = create_test_document(args.stage, config)
        if args.send_message:
            send_test_message(doc_id, args.stage)


if __name__ == "__main__":
    main()

