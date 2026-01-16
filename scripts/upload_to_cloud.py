#!/usr/bin/env python3
"""Upload document to cloud RAG system

Uploads a document via the API endpoint, which will:
1. Upload file to Supabase Storage (cloud)
2. Save metadata to database (cloud Supabase)
3. Enqueue message to ingestion-uploads queue
4. Azure Functions will process it via queue trigger
"""

import sys
import os
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load cloud environment
env_file = Path(__file__).parent.parent / ".env.prod"
if env_file.exists():
    load_dotenv(env_file)

# Default API URL - can be overridden
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

def upload_document(pdf_path: str, api_url: str = None):
    """Upload document to cloud RAG system"""
    
    if api_url is None:
        api_url = API_BASE_URL
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        return None
    
    print("\n" + "="*70)
    print("📤 UPLOADING DOCUMENT TO CLOUD RAG SYSTEM")
    print("="*70)
    print(f"File: {pdf_path}")
    print(f"API: {api_url}")
    print("="*70 + "\n")
    
    # Upload via API
    print("📤 Uploading document via API...")
    try:
        with open(pdf_path, 'rb') as f:
            files = {'file': (pdf_path.name, f, 'application/pdf')}
            response = requests.post(
                f"{api_url}/api/upload",
                files=files,
                timeout=60
            )
        
        if response.status_code != 200:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
        
        result = response.json()
        document_id = result['document_id']
        
        print(f"✅ Upload successful!")
        print(f"   Document ID: {document_id}")
        print(f"   Status: {result['status']}")
        print(f"   Message: {result.get('message', 'N/A')}")
        print(f"\n📋 Next Steps:")
        print(f"   1. Check queue status: python scripts/monitor_queue_health.py")
        print(f"   2. Monitor Azure Functions logs in Azure Portal")
        print(f"   3. Check document status in database")
        print(f"\n💡 The document will be processed by Azure Functions queue workers")
        print(f"   Processing happens asynchronously - check logs for progress")
        
        return document_id
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to API at {api_url}")
        print(f"   Make sure the API server is running:")
        print(f"   cd backend && uvicorn src.api.main:app --reload")
        return None
    except Exception as e:
        print(f"❌ Upload error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default test file
        test_file = Path(__file__).parent.parent / "backend" / "tests" / "fixtures" / "sample_documents" / "healthguard_select_ppo_plan.pdf"
        if test_file.exists():
            pdf_path = str(test_file)
            print(f"Using default test file: {pdf_path}")
        else:
            print("Usage: python scripts/upload_to_cloud.py <pdf_path> [api_url]")
            sys.exit(1)
    else:
        pdf_path = sys.argv[1]
    
    api_url = sys.argv[2] if len(sys.argv) > 2 else None
    
    document_id = upload_document(pdf_path, api_url)
    
    if document_id:
        print(f"\n✅ Upload completed! Document ID: {document_id}")
        sys.exit(0)
    else:
        print("\n❌ Upload failed")
        sys.exit(1)
