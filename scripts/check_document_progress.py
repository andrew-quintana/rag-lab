#!/usr/bin/env python3
"""
Comprehensive document progress checker for Azure Functions RAG pipeline.

Provides detailed status of a document through all pipeline stages with
multiple independent verification methods.

Usage:
    python scripts/check_document_progress.py <document_id>
    python scripts/check_document_progress.py --latest  # Check most recent document
    python scripts/check_document_progress.py --watch <document_id>  # Auto-refresh
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Add backend to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from azure.storage.queue import QueueServiceClient
from dotenv import load_dotenv
from src.core.config import Config
from src.db.connection import DatabaseConnection
from src.db.documents import DocumentService

# Load environment
env_file = Path(__file__).parent.parent / ".env.local"
if env_file.exists():
    load_dotenv(env_file)

QUEUE_NAMES = [
    "ingestion-uploads",
    "ingestion-chunking",
    "ingestion-embeddings",
    "ingestion-indexing"
]

POISON_QUEUES = [f"{q}-poison" for q in QUEUE_NAMES]

STAGE_MAP = {
    "uploaded": {"current": "ingestion-uploads", "next": "ingestion-chunking"},
    "parsed": {"current": "ingestion-chunking", "next": "ingestion-embeddings"},
    "chunked": {"current": "ingestion-embeddings", "next": "ingestion-indexing"},
    "embedded": {"current": "ingestion-indexing", "next": None},
    "indexed": {"current": None, "next": None}
}


def get_connection_string():
    """Get Azure Storage connection string"""
    conn_str = (
        os.getenv("AZURE_STORAGE_QUEUES_CONNECTION_STRING") or
        os.getenv("AZURE_BLOB_CONNECTION_STRING") or
        os.getenv("AzureWebJobsStorage")
    )
    if not conn_str:
        print("❌ Error: No Azure Storage connection string found")
        sys.exit(1)
    return conn_str


def check_database_status(doc_id: str, config: Config) -> Dict:
    """Check document status in database"""
    db_conn = DatabaseConnection(config)
    db_conn.connect()
    doc_service = DocumentService(db_conn)
    
    try:
        doc = doc_service.get_document(doc_id)
        if not doc:
            return {"found": False}
        
        # Calculate time in current status
        time_in_status = None
        if doc.upload_timestamp:
            time_in_status = datetime.utcnow() - doc.upload_timestamp
        
        return {
            "found": True,
            "status": doc.status,
            "filename": doc.filename,
            "file_size": doc.file_size,
            "chunks_created": doc.chunks_created or 0,
            "upload_timestamp": doc.upload_timestamp,
            "time_in_status": time_in_status,
            "storage_path": doc.storage_path
        }
    finally:
        db_conn.close()


def check_queue_position(doc_id: str, queue_name: str, conn_str: str) -> Dict:
    """Check if document is in a specific queue"""
    client = QueueServiceClient.from_connection_string(conn_str)
    queue = client.get_queue_client(queue_name)
    
    try:
        messages = list(queue.peek_messages(max_messages=32))
        for idx, msg in enumerate(messages):
            try:
                import json
                content = json.loads(msg.content)
                if content.get("document_id") == doc_id:
                    return {
                        "found": True,
                        "position": idx + 1,
                        "total": len(messages),
                        "dequeue_count": msg.dequeue_count,
                        "message_id": msg.id
                    }
            except:
                pass
        
        return {"found": False}
    except:
        return {"error": True}


def check_poison_queues(doc_id: str, conn_str: str) -> List[str]:
    """Check if document is in any poison queue"""
    found_in = []
    client = QueueServiceClient.from_connection_string(conn_str)
    
    for queue_name in POISON_QUEUES:
        try:
            queue = client.get_queue_client(queue_name)
            messages = list(queue.peek_messages(max_messages=32))
            
            for msg in messages:
                try:
                    import json
                    content = json.loads(msg.content)
                    if content.get("document_id") == doc_id:
                        found_in.append({
                            "queue": queue_name,
                            "dequeue_count": msg.dequeue_count,
                            "message_id": msg.id
                        })
                except:
                    pass
        except:
            pass
    
    return found_in


def get_application_insights_data(doc_id: str) -> Dict:
    """Get Application Insights data for document"""
    # This would call Azure CLI to get logs
    # Simplified version for now
    import subprocess
    
    try:
        # Check for recent errors
        result = subprocess.run([
            "az", "monitor", "app-insights", "query",
            "--app", "ai-rag-evaluator",
            "--resource-group", "rag-lab",
            "--analytics-query", 
            f"union traces, exceptions | where timestamp > ago(30m) | where message contains '{doc_id}' | where message contains 'Error' or message contains 'Failed' | take 5 | project timestamp, message",
            "--output", "json"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            rows = data.get("tables", [{}])[0].get("rows", [])
            return {
                "recent_errors": len(rows),
                "errors": [{"time": r[0], "message": r[1][:200]} for r in rows[:3]]
            }
    except:
        pass
    
    return {"recent_errors": 0, "errors": []}


def print_status_report(doc_id: str, watch_mode: bool = False):
    """Print comprehensive status report"""
    config = Config.from_env()
    conn_str = get_connection_string()
    
    if watch_mode:
        os.system('clear' if os.name != 'nt' else 'cls')
    
    print("=" * 70)
    print(f"📊 Document Progress Report")
    print(f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Document ID: {doc_id}")
    print("=" * 70)
    
    # 1. Database Status
    print("\n📁 Database Status")
    print("-" * 70)
    db_status = check_database_status(doc_id, config)
    
    if not db_status.get("found"):
        print("❌ Document not found in database")
        return
    
    print(f"✅ Document Found")
    print(f"   Filename:        {db_status['filename']}")
    print(f"   Current Status:  {db_status['status']}")
    print(f"   File Size:       {db_status['file_size']:,} bytes")
    print(f"   Chunks Created:  {db_status['chunks_created']}")
    print(f"   Upload Time:     {db_status['upload_timestamp']}")
    
    if db_status['time_in_status']:
        minutes = db_status['time_in_status'].total_seconds() / 60
        print(f"   Time in Status:  {int(minutes)} minutes")
        
        # Warning for stuck documents
        if minutes > 30 and db_status['status'] != 'indexed':
            print(f"   ⚠️  WARNING: Document has been in '{db_status['status']}' status for {int(minutes)} minutes")
    
    # 2. Current Stage & Expected Queue
    print("\n🔄 Pipeline Stage")
    print("-" * 70)
    current_status = db_status['status']
    stage_info = STAGE_MAP.get(current_status, {})
    
    if current_status == "indexed":
        print("✅ Document is FULLY PROCESSED and indexed")
    else:
        expected_queue = stage_info.get("current")
        next_queue = stage_info.get("next")
        
        print(f"Current Stage:    {current_status}")
        print(f"Should be in:     {expected_queue}")
        print(f"Next stage:       {next_queue}")
    
    # 3. Queue Position
    print("\n📥 Queue Status")
    print("-" * 70)
    
    found_in_queue = False
    for queue_name in QUEUE_NAMES:
        queue_status = check_queue_position(doc_id, queue_name, conn_str)
        if queue_status.get("found"):
            found_in_queue = True
            print(f"✅ Found in: {queue_name}")
            print(f"   Position:      {queue_status['position']} of {queue_status['total']}")
            print(f"   Dequeue Count: {queue_status['dequeue_count']}")
            if queue_status['dequeue_count'] > 2:
                print(f"   ⚠️  WARNING: High dequeue count indicates repeated failures")
    
    if not found_in_queue:
        print("ℹ️  Not currently in any processing queue")
        if current_status != "indexed":
            print("   This could mean:")
            print("   - Document is being processed right now")
            print("   - Document failed and is in poison queue")
            print("   - Document is stuck")
    
    # 4. Poison Queue Check
    print("\n☠️  Poison Queue Check")
    print("-" * 70)
    
    poison_findings = check_poison_queues(doc_id, conn_str)
    if poison_findings:
        print("🚨 ALERT: Document found in poison queue(s)!")
        for finding in poison_findings:
            print(f"   Queue:         {finding['queue']}")
            print(f"   Dequeue Count: {finding['dequeue_count']}")
            print(f"   Message ID:    {finding['message_id']}")
        print("\n   This indicates the worker failed to process this document.")
        print("   Check Application Insights for error details.")
    else:
        print("✅ Not in any poison queue")
    
    # 5. Application Insights
    print("\n📊 Recent Activity (Application Insights)")
    print("-" * 70)
    
    ai_data = get_application_insights_data(doc_id)
    if ai_data['recent_errors'] > 0:
        print(f"⚠️  Found {ai_data['recent_errors']} recent errors:")
        for error in ai_data['errors']:
            print(f"   {error['time']}: {error['message']}")
    else:
        print("✅ No recent errors in logs (last 30 minutes)")
    
    # 6. Overall Assessment
    print("\n🎯 Overall Assessment")
    print("-" * 70)
    
    if poison_findings:
        print("❌ STATUS: FAILED")
        print("   Action: Check Application Insights for error details")
        print("   Command: az monitor app-insights query --app ai-rag-evaluator --resource-group rag-lab \\")
        print(f"            --analytics-query \"exceptions | where message contains '{doc_id}' | take 5\"")
    elif current_status == "indexed":
        print("✅ STATUS: COMPLETE")
        print("   Document has been fully processed and indexed")
    elif db_status.get('time_in_status') and db_status['time_in_status'].total_seconds() > 1800:
        print("⚠️  STATUS: LIKELY STUCK")
        print(f"   Document has been in '{current_status}' for {int(db_status['time_in_status'].total_seconds() / 60)} minutes")
        print("   Expected processing time: < 30 minutes per stage")
        print("   Recommended: Check poison queues and Application Insights")
    elif found_in_queue:
        print("🔄 STATUS: QUEUED FOR PROCESSING")
        print("   Document is waiting in queue")
    else:
        print("🔄 STATUS: PROCESSING")
        print("   Document may be actively being processed by a worker")
        print("   Check again in 2-3 minutes for progress")
    
    print("=" * 70)


def watch_mode(doc_id: str, interval: int = 30):
    """Watch mode - continuously monitor document"""
    print(f"🔄 Watch Mode - Refreshing every {interval} seconds")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            print_status_report(doc_id, watch_mode=True)
            print(f"\nRefreshing in {interval} seconds... (Ctrl+C to stop)")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n👋 Watch mode stopped")


def get_latest_document(config: Config) -> Optional[str]:
    """Get the most recently uploaded document ID"""
    db_conn = DatabaseConnection(config)
    db_conn.connect()
    
    try:
        # Query for most recent document
        from src.db.queries import QueryExecutor
        executor = QueryExecutor(db_conn)
        
        query = "SELECT id FROM documents ORDER BY upload_timestamp DESC LIMIT 1"
        results = executor.execute_query(query)
        
        if results:
            return results[0]['id']
        return None
    finally:
        db_conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Check document progress through RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/check_document_progress.py abc-123-def
  python scripts/check_document_progress.py --latest
  python scripts/check_document_progress.py --watch abc-123-def
        """
    )
    
    parser.add_argument("document_id", nargs="?", help="Document ID to check")
    parser.add_argument("--latest", action="store_true", help="Check most recent document")
    parser.add_argument("--watch", action="store_true", help="Watch mode (auto-refresh)")
    parser.add_argument("--interval", type=int, default=30, help="Watch mode refresh interval (seconds)")
    
    args = parser.parse_args()
    
    if args.latest:
        config = Config.from_env()
        doc_id = get_latest_document(config)
        if not doc_id:
            print("❌ No documents found in database")
            sys.exit(1)
        print(f"ℹ️  Using most recent document: {doc_id}\n")
    elif args.document_id:
        doc_id = args.document_id
    else:
        parser.error("Must provide document_id or use --latest")
    
    if args.watch:
        watch_mode(doc_id, args.interval)
    else:
        print_status_report(doc_id)


if __name__ == "__main__":
    main()

