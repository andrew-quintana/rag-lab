#!/usr/bin/env python3
"""Monitor batch processing progress for a document"""

import sys
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load cloud environment
env_file = Path(__file__).parent.parent / ".env.prod"
if env_file.exists():
    load_dotenv(env_file)

API_URL = "http://localhost:8000"

def monitor_document(document_id: str, watch: bool = False):
    """Monitor batch processing progress"""
    
    print(f"\n{'='*70}")
    print(f"📊 MONITORING BATCH PROCESSING")
    print(f"{'='*70}")
    print(f"Document ID: {document_id}")
    print(f"API: {API_URL}")
    print(f"{'='*70}\n")
    
    while True:
        try:
            response = requests.get(f"{API_URL}/api/documents/{document_id}")
            if response.status_code != 200:
                print(f"❌ Error: {response.status_code} - {response.text}")
                break
            
            doc = response.json()
            status = doc.get("status", "unknown")
            meta = doc.get("metadata", {}).get("ingestion", {})
            
            num_pages = meta.get("num_pages", 0)
            num_batches_total = meta.get("num_batches_total", 0)
            batches_completed = meta.get("batches_completed", {})
            last_page = meta.get("last_successful_page", 0)
            next_batch = meta.get("next_unparsed_batch_index", 0)
            parsing_status = meta.get("parsing_status", "unknown")
            
            # Calculate progress
            progress_pct = (last_page / num_pages * 100) if num_pages > 0 else 0
            batches_done = len(batches_completed) if batches_completed else 0
            
            print(f"📄 Status: {status}")
            print(f"🔄 Parsing Status: {parsing_status}")
            print(f"📊 Progress: {last_page}/{num_pages} pages ({progress_pct:.1f}%)")
            print(f"📦 Batches: {batches_done}/{num_batches_total} completed")
            print(f"📍 Last Page: {last_page}")
            print(f"⏭️  Next Batch: {next_batch}")
            
            if status == "parsed":
                print(f"\n✅ Ingestion complete!")
                break
            elif status.startswith("failed"):
                print(f"\n❌ Processing failed!")
                break
            elif parsing_status == "completed":
                print(f"\n✅ All batches processed!")
                break
            
            if not watch:
                break
            
            print(f"\n{'─'*70}")
            time.sleep(10)
            
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            break

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/monitor_batch_processing.py <document_id> [--watch]")
        sys.exit(1)
    
    document_id = sys.argv[1]
    watch = "--watch" in sys.argv or "-w" in sys.argv
    
    monitor_document(document_id, watch=watch)
