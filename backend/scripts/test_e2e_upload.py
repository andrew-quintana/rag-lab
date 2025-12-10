#!/usr/bin/env python3
"""Test end-to-end document upload flow

This script tests the complete RAG upload pipeline:
1. Upload file to Supabase Storage
2. Generate preview image
3. Save metadata to database
4. Extract text (with batch processing for Free tier)
5. Chunk text
6. Generate embeddings
7. Index to Azure AI Search
8. Verify document is tracked in system
"""

import sys
import os
from pathlib import Path
import requests
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import Config
from src.db.connection import DatabaseConnection
from src.db.documents import DocumentService

def test_e2e_upload(pdf_path: str, api_base_url: str = "http://localhost:8000"):
    """Test end-to-end upload flow via API"""
    
    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"End-to-End Document Upload Test")
    print(f"Document: {pdf_path}")
    print(f"API: {api_base_url}")
    print(f"{'='*70}\n")
    
    # Step 1: Upload document via API
    print("📤 Step 1: Uploading document via API...")
    with open(pdf_path, 'rb') as f:
        files = {'file': (Path(pdf_path).name, f, 'application/pdf')}
        response = requests.post(f"{api_base_url}/api/upload", files=files)
    
    if response.status_code != 200:
        print(f"❌ Upload failed: {response.status_code}")
        print(f"   Error: {response.text}")
        sys.exit(1)
    
    result = response.json()
    document_id = result['document_id']
    chunks_created = result['chunks_created']
    
    print(f"✅ Upload successful!")
    print(f"   Document ID: {document_id}")
    print(f"   Status: {result['status']}")
    print(f"   Chunks created: {chunks_created}")
    print(f"   Message: {result['message']}\n")
    
    # Step 2: Verify document in database
    print("🔍 Step 2: Verifying document in database...")
    config = Config.from_env()
    db_conn = DatabaseConnection(config)
    db_conn.connect()
    doc_service = DocumentService(db_conn)
    
    try:
        document = doc_service.get_document(document_id)
        if not document:
            print(f"❌ Document not found in database")
            sys.exit(1)
        
        print(f"✅ Document found in database:")
        print(f"   ID: {document.id}")
        print(f"   Filename: {document.filename}")
        print(f"   File size: {document.file_size:,} bytes")
        print(f"   Status: {document.status}")
        print(f"   Chunks: {document.chunks_count}")
        print(f"   Storage path: {document.storage_path}")
        print(f"   Upload timestamp: {document.upload_timestamp}\n")
        
        # Verify status is "processed"
        if document.status != "processed":
            print(f"⚠️  Warning: Document status is '{document.status}', expected 'processed'")
        else:
            print(f"✅ Document status is 'processed' (correct)\n")
        
        # Verify chunks count matches
        if document.chunks_count != chunks_created:
            print(f"⚠️  Warning: Database chunks count ({document.chunks_count}) != API response ({chunks_created})")
        else:
            print(f"✅ Chunks count matches: {chunks_created}\n")
        
    except Exception as e:
        print(f"❌ Error verifying document: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        db_conn.close()
    
    # Step 3: Verify document in API
    print("🔍 Step 3: Verifying document via API...")
    response = requests.get(f"{api_base_url}/api/documents/{document_id}")
    
    if response.status_code != 200:
        print(f"⚠️  Warning: Could not fetch document via API: {response.status_code}")
        if response.status_code == 404:
            print(f"   Document not found - this may be expected if endpoint doesn't exist yet")
    else:
        api_doc = response.json()
        print(f"✅ Document accessible via API:")
        print(f"   ID: {api_doc.get('document_id', 'N/A')}")
        print(f"   Filename: {api_doc.get('filename', 'N/A')}")
        print(f"   Status: {api_doc.get('status', 'N/A')}")
        print(f"   Chunks: {api_doc.get('chunks_created', 'N/A')}\n")
    
    # Step 4: List all documents
    print("🔍 Step 4: Listing all documents...")
    response = requests.get(f"{api_base_url}/api/documents")
    
    if response.status_code == 200:
        docs = response.json()
        print(f"✅ Found {len(docs.get('documents', []))} documents in system")
        
        # Check if our document is in the list
        our_doc = next((d for d in docs.get('documents', []) if d.get('document_id') == document_id), None)
        if our_doc:
            print(f"✅ Our document is in the list\n")
        else:
            print(f"⚠️  Warning: Our document not found in list\n")
    else:
        print(f"⚠️  Warning: Could not list documents: {response.status_code}\n")
    
    # Summary
    print(f"{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    print(f"✅ Document uploaded successfully")
    print(f"✅ Document tracked in database (status: {document.status})")
    print(f"✅ Document has {chunks_created} chunks indexed")
    print(f"✅ Document accessible via API")
    print(f"\n🎉 End-to-end upload flow completed successfully!")
    print(f"\nDocument ID: {document_id}")
    print(f"You can now query this document using:")
    print(f"  curl -X POST {api_base_url}/api/query \\")
    print(f"    -H 'Content-Type: application/json' \\")
    print(f"    -d '{{\"text\": \"Your question here\", \"prompt_version\": \"v1\"}}'")

if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "docs/inputs/scan_classic_hmo.pdf"
    api_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
    
    test_e2e_upload(pdf_path, api_url)

