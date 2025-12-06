#!/usr/bin/env python3
"""Script to check how many chunks a document has in the database and Azure AI Search"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_eval.core.config import Config
from rag_eval.db.connection import DatabaseConnection
from rag_eval.db.documents import DocumentService
from rag_eval.services.rag.search import delete_chunks_by_document_id
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

def check_document_chunks(document_id: str):
    """Check chunk count for a document"""
    config = Config.from_env()
    
    # Check database
    print(f"\n=== Checking document {document_id} ===\n")
    
    db_conn = DatabaseConnection(config)
    db_conn.connect()
    doc_service = DocumentService(db_conn)
    
    document = doc_service.get_document(document_id)
    if not document:
        print(f"❌ Document not found in database")
        return
    
    print(f"📄 Document Info:")
    print(f"   Filename: {document.filename}")
    print(f"   File size: {document.file_size:,} bytes")
    print(f"   Status: {document.status}")
    print(f"   Chunks created (DB): {document.chunks_created}")
    
    # Check Azure AI Search
    print(f"\n🔍 Checking Azure AI Search for chunks...")
    
    try:
        search_client = SearchClient(
            endpoint=config.azure_search_endpoint,
            index_name=config.azure_search_index_name,
            credential=AzureKeyCredential(config.azure_search_api_key)
        )
        
        # Search for chunks with this document_id
        # Try filtering first, fall back to searching all if filterable field not available
        try:
            results = search_client.search(
                search_text="*",
                filter=f"document_id eq '{document_id}'",
                select=["id", "chunk_text", "document_id", "metadata"],
                top=10000
            )
            chunks = list(results)
        except Exception as filter_error:
            # If filtering fails, search all and filter in memory
            print(f"   ⚠️  Filtering not available, searching all chunks...")
            results = search_client.search(
                search_text="*",
                select=["id", "chunk_text", "document_id", "metadata"],
                top=10000
            )
            all_chunks = list(results)
            chunks = [r for r in all_chunks if r.get("document_id") == document_id]
        print(f"   Chunks found in Azure AI Search: {len(chunks)}")
        
        if chunks:
            print(f"\n📋 Chunk IDs found:")
            for i, chunk in enumerate(chunks[:20], 1):  # Show first 20
                chunk_id = chunk.get("id", "unknown")
                chunk_text_preview = chunk.get("chunk_text", "")[:50]
                print(f"   {i}. {chunk_id}: {chunk_text_preview}...")
            
            if len(chunks) > 20:
                print(f"   ... and {len(chunks) - 20} more chunks")
        
        # Compare
        print(f"\n📊 Comparison:")
        if document.chunks_created is None:
            print(f"   ⚠️  Database chunks_created is NULL")
        elif document.chunks_created != len(chunks):
            print(f"   ⚠️  MISMATCH: DB says {document.chunks_created} chunks, but Azure AI Search has {len(chunks)} chunks")
        else:
            print(f"   ✅ Match: Both DB and Azure AI Search report {len(chunks)} chunks")
            
        if len(chunks) == 1:
            print(f"\n⚠️  WARNING: Only 1 chunk found for a document of {document.file_size:,} bytes")
            print(f"   This is likely incorrect - large documents should be split into multiple chunks")
            print(f"   Default chunk size is 1000 characters with 200 overlap")
            print(f"   Expected chunks: ~{document.file_size // 800} (rough estimate)")
        
    except Exception as e:
        print(f"   ❌ Error querying Azure AI Search: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_document_chunks.py <document_id>")
        sys.exit(1)
    
    document_id = sys.argv[1]
    check_document_chunks(document_id)

