#!/usr/bin/env python3
"""Verify document is indexed in Azure AI Search"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add backend to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from src.core.config import Config

# Load cloud environment
env_file = Path(__file__).parent.parent / ".env.prod"
if env_file.exists():
    load_dotenv(env_file)

def verify_indexing(document_id: str):
    """Verify document chunks are in Azure AI Search"""
    config = Config.from_env()
    
    print(f"\n{'='*70}")
    print(f"🔍 VERIFYING INDEXING IN AZURE AI SEARCH")
    print(f"{'='*70}")
    print(f"Document ID: {document_id}")
    print(f"Index: {config.azure_search_index_name}")
    print(f"{'='*70}\n")
    
    try:
        search_client = SearchClient(
            endpoint=config.azure_search_endpoint,
            index_name=config.azure_search_index_name,
            credential=AzureKeyCredential(config.azure_search_api_key)
        )
        
        # Try filtering by document_id
        print("🔍 Searching for chunks...")
        try:
            results = search_client.search(
                search_text="*",
                filter=f"document_id eq '{document_id}'",
                select=["id", "chunk_text", "document_id", "metadata"],
                top=100
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
        
        print(f"\n✅ Found {len(chunks)} chunk(s) in Azure AI Search")
        
        if chunks:
            print(f"\n📋 Sample chunks:")
            for i, chunk in enumerate(chunks[:5], 1):
                chunk_id = chunk.get("id", "unknown")
                chunk_text = chunk.get("chunk_text", "")
                preview = chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
                print(f"\n   {i}. Chunk ID: {chunk_id}")
                print(f"      Preview: {preview}")
            
            if len(chunks) > 5:
                print(f"\n   ... and {len(chunks) - 5} more chunks")
            
            print(f"\n✅ VERIFICATION PASSED: Document is indexed in Azure AI Search")
            return True
        else:
            print(f"\n❌ VERIFICATION FAILED: No chunks found for document {document_id}")
            print(f"   The document may not have been indexed, or indexing may have failed.")
            return False
            
    except Exception as e:
        print(f"\n❌ Error querying Azure AI Search: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/verify_indexing.py <document_id>")
        sys.exit(1)
    
    document_id = sys.argv[1]
    success = verify_indexing(document_id)
    sys.exit(0 if success else 1)
