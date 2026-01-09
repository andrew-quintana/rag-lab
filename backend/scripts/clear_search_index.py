#!/usr/bin/env python3
"""Script to clear all chunks from Azure AI Search index"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import Config
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.core.exceptions import ResourceNotFoundError

def clear_search_index():
    """Clear all chunks from Azure AI Search index"""
    config = Config.from_env()
    
    print(f"\n=== Clearing Azure AI Search Index ===\n")
    print(f"Index name: {config.azure_search_index_name}")
    
    try:
        search_client = SearchClient(
            endpoint=config.azure_search_endpoint,
            index_name=config.azure_search_index_name,
            credential=AzureKeyCredential(config.azure_search_api_key)
        )
        
        # Get all chunks
        print("🔍 Retrieving all chunks from index...")
        results = search_client.search(
            search_text="*",
            select=["id"],
            top=10000
        )
        
        all_chunks = list(results)
        print(f"   Found {len(all_chunks)} chunks")
        
        if not all_chunks:
            print("✅ Index is already empty")
            return
        
        # Delete all chunks
        print(f"\n🗑️  Deleting {len(all_chunks)} chunks...")
        chunk_ids = [chunk["id"] for chunk in all_chunks]
        
        # Delete in batches (Azure AI Search has limits)
        batch_size = 1000
        deleted_count = 0
        
        for i in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[i:i + batch_size]
            documents_to_delete = [{"id": chunk_id} for chunk_id in batch]
            
            result = search_client.delete_documents(documents=documents_to_delete)
            
            failed = [r for r in result if not r.succeeded]
            if failed:
                print(f"   ⚠️  Failed to delete {len(failed)} chunks in batch {i // batch_size + 1}")
                for f in failed:
                    print(f"      - {f.key}: {f.error_message}")
            else:
                deleted_count += len(batch)
                print(f"   ✅ Deleted batch {i // batch_size + 1} ({len(batch)} chunks)")
        
        print(f"\n✅ Successfully deleted {deleted_count} chunks")
        print(f"   Total chunks processed: {len(chunk_ids)}")
        
        # Verify index is empty
        print(f"\n🔍 Verifying index is empty...")
        verify_results = search_client.search(
            search_text="*",
            select=["id"],
            top=10
        )
        remaining = list(verify_results)
        
        if remaining:
            print(f"   ⚠️  Warning: {len(remaining)} chunks still remain in index")
        else:
            print(f"   ✅ Index is now empty")
        
    except ResourceNotFoundError:
        print(f"   ℹ️  Index '{config.azure_search_index_name}' does not exist (already empty)")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    clear_search_index()




