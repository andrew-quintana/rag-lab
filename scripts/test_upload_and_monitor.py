#!/usr/bin/env python3
"""Test document upload and monitor queue processing

This script:
1. Uploads a test document
2. Monitors queue status
3. Checks if Azure Functions are processing messages
"""

import sys
import time
import requests
from pathlib import Path
from datetime import datetime

# Add backend to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from azure.storage.queue import QueueServiceClient
from dotenv import load_dotenv
import os

# Load environment
env_file = Path(__file__).parent.parent / ".env.prod"
if env_file.exists():
    load_dotenv(env_file)

API_URL = "http://localhost:8000"

def upload_document(pdf_path: str):
    """Upload document via API"""
    print("\n" + "="*70)
    print("📤 STEP 1: Uploading Document")
    print("="*70)
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        return None
    
    try:
        with open(pdf_path, 'rb') as f:
            files = {'file': (pdf_path.name, f, 'application/pdf')}
            response = requests.post(
                f"{API_URL}/api/upload",
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
        return document_id
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to API at {API_URL}")
        print(f"   Make sure API is running: make backend-cloud")
        return None
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return None

def check_queue_status():
    """Check queue status"""
    print("\n" + "="*70)
    print("📊 STEP 2: Checking Queue Status")
    print("="*70)
    
    conn_str = os.getenv("AZURE_STORAGE_QUEUES_CONNECTION_STRING")
    if not conn_str:
        print("   ⚠️  AZURE_STORAGE_QUEUES_CONNECTION_STRING not set")
        return
    
    try:
        queue_service = QueueServiceClient.from_connection_string(conn_str)
        queue_client = queue_service.get_queue_client("ingestion-uploads")
        props = queue_client.get_queue_properties()
        count = props.approximate_message_count or 0
        
        print(f"   Queue: ingestion-uploads")
        print(f"   Messages: {count}")
        
        if count > 0:
            print(f"   ✅ Messages in queue - Azure Functions should process them")
        else:
            print(f"   ✅ Queue is empty - messages may have been processed")
        
        return count
    except Exception as e:
        print(f"   ⚠️  Error checking queue: {e}")
        return None

def monitor_queue_changes(initial_count, duration=60):
    """Monitor queue for changes"""
    print("\n" + "="*70)
    print(f"👀 STEP 3: Monitoring Queue (for {duration} seconds)")
    print("="*70)
    
    conn_str = os.getenv("AZURE_STORAGE_QUEUES_CONNECTION_STRING")
    if not conn_str:
        print("   ⚠️  Cannot monitor - connection string not set")
        return
    
    queue_service = QueueServiceClient.from_connection_string(conn_str)
    queue_client = queue_service.get_queue_client("ingestion-uploads")
    
    start_time = time.time()
    last_count = initial_count
    
    print(f"   Starting count: {initial_count}")
    print(f"   Monitoring for {duration} seconds...")
    print("")
    
    while time.time() - start_time < duration:
        try:
            props = queue_client.get_queue_properties()
            current_count = props.approximate_message_count or 0
            
            if current_count != last_count:
                elapsed = int(time.time() - start_time)
                print(f"   [{elapsed}s] Queue count changed: {last_count} → {current_count}")
                if current_count < last_count:
                    print(f"      ✅ Messages being processed!")
                last_count = current_count
            
            time.sleep(5)
        except Exception as e:
            print(f"   ⚠️  Error monitoring: {e}")
            break
    
    final_count = queue_client.get_queue_properties().approximate_message_count or 0
    print(f"\n   Final count: {final_count}")
    
    if final_count < initial_count:
        print(f"   ✅ SUCCESS: Messages were processed!")
    elif final_count == initial_count:
        print(f"   ⚠️  No change: Messages may still be processing or function not triggered")
    else:
        print(f"   ⚠️  Count increased: New messages added")

def main():
    """Run full test"""
    print("\n" + "="*70)
    print("🧪 DOCUMENT UPLOAD & QUEUE MONITORING TEST")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Upload
    test_file = Path(__file__).parent.parent / "backend" / "tests" / "fixtures" / "sample_documents" / "healthguard_select_ppo_plan.pdf"
    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1])
    
    document_id = upload_document(test_file)
    if not document_id:
        print("\n❌ Upload failed - cannot continue")
        sys.exit(1)
    
    # Wait a moment for message to be enqueued
    time.sleep(2)
    
    # Step 2: Check initial queue status
    initial_count = check_queue_status()
    
    # Step 3: Monitor for changes
    if initial_count is not None:
        monitor_queue_changes(initial_count, duration=60)
    
    print("\n" + "="*70)
    print("📋 NEXT STEPS")
    print("="*70)
    print("1. Check Azure Portal Log Stream for function execution")
    print("2. Monitor queue: python scripts/monitor_queue_health.py")
    print("3. Check database status for document processing")
    print(f"4. Document ID: {document_id}")
    print("="*70)

if __name__ == "__main__":
    main()
