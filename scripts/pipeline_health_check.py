#!/usr/bin/env python3
"""
Overall pipeline health check for Azure Functions RAG system.

Checks all critical components and provides a comprehensive health report.

Usage:
    python scripts/pipeline_health_check.py
    python scripts/pipeline_health_check.py --verbose
    python scripts/pipeline_health_check.py --watch
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List

# Add backend to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from azure.storage.queue import QueueServiceClient
from dotenv import load_dotenv
from src.core.config import Config
from src.db.connection import DatabaseConnection
from src.db.queries import QueryExecutor

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


def check_queues() -> Dict:
    """Check all queue health"""
    conn_str = (
        os.getenv("AZURE_STORAGE_QUEUES_CONNECTION_STRING") or
        os.getenv("AZURE_BLOB_CONNECTION_STRING") or
        os.getenv("AzureWebJobsStorage")
    )
    
    if not conn_str:
        return {"status": "error", "message": "No connection string"}
    
    client = QueueServiceClient.from_connection_string(conn_str)
    queue_stats = {}
    total_messages = 0
    poison_messages = 0
    
    # Check main queues
    for queue_name in QUEUE_NAMES:
        try:
            queue = client.get_queue_client(queue_name)
            props = queue.get_queue_properties()
            count = props.approximate_message_count
            queue_stats[queue_name] = count
            total_messages += count
        except:
            queue_stats[queue_name] = -1
    
    # Check poison queues
    for queue_name in QUEUE_NAMES:
        poison_name = f"{queue_name}-poison"
        try:
            queue = client.get_queue_client(poison_name)
            props = queue.get_queue_properties()
            count = props.approximate_message_count
            queue_stats[poison_name] = count
            poison_messages += count
        except:
            queue_stats[poison_name] = 0
    
    status = "healthy"
    if poison_messages > 0:
        status = "degraded"
    if poison_messages > 5:
        status = "critical"
    
    return {
        "status": status,
        "total_queued": total_messages,
        "poison_count": poison_messages,
        "queues": queue_stats
    }


def check_stuck_documents() -> Dict:
    """Check for documents stuck in processing"""
    config = Config.from_env()
    db_conn = DatabaseConnection(config)
    db_conn.connect()
    executor = QueryExecutor(db_conn)
    
    try:
        # Find documents older than 30 minutes not in final state
        query = """
            SELECT id, filename, status, upload_timestamp, 
                   EXTRACT(EPOCH FROM (NOW() - upload_timestamp))/60 as minutes_old
            FROM documents 
            WHERE status != 'indexed' 
              AND upload_timestamp < NOW() - INTERVAL '30 minutes'
            ORDER BY upload_timestamp DESC
            LIMIT 10
        """
        
        results = executor.execute_query(query)
        
        stuck_docs = []
        for row in results:
            stuck_docs.append({
                "id": row['id'],
                "filename": row['filename'],
                "status": row['status'],
                "minutes_old": int(row['minutes_old'])
            })
        
        status = "healthy"
        if len(stuck_docs) > 0:
            status = "warning"
        if len(stuck_docs) > 5:
            status = "critical"
        
        return {
            "status": status,
            "count": len(stuck_docs),
            "documents": stuck_docs
        }
    finally:
        db_conn.close()


def check_recent_failures() -> Dict:
    """Check Application Insights for recent failures"""
    import subprocess
    
    try:
        result = subprocess.run([
            "az", "monitor", "app-insights", "query",
            "--app", "ai-rag-evaluator",
            "--resource-group", "rag-lab",
            "--analytics-query",
            "requests | where timestamp > ago(1h) | summarize total=count(), failures=countif(success==false) by name",
            "--output", "json"
        ], capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            rows = data.get("tables", [{}])[0].get("rows", [])
            
            worker_stats = {}
            total_failures = 0
            
            for row in rows:
                worker_name = row[0]
                total = row[1]
                failures = row[2]
                success_rate = ((total - failures) / total * 100) if total > 0 else 0
                
                worker_stats[worker_name] = {
                    "total": total,
                    "failures": failures,
                    "success_rate": round(success_rate, 1)
                }
                total_failures += failures
            
            status = "healthy"
            for worker, stats in worker_stats.items():
                if stats['success_rate'] < 50:
                    status = "critical"
                elif stats['success_rate'] < 90:
                    status = "warning"
            
            return {
                "status": status,
                "total_failures": total_failures,
                "workers": worker_stats
            }
    except:
        pass
    
    return {"status": "unknown", "message": "Could not query Application Insights"}


def check_document_counts() -> Dict:
    """Check document counts by status"""
    config = Config.from_env()
    db_conn = DatabaseConnection(config)
    db_conn.connect()
    executor = QueryExecutor(db_conn)
    
    try:
        query = """
            SELECT status, COUNT(*) as count
            FROM documents
            GROUP BY status
            ORDER BY 
                CASE status
                    WHEN 'uploaded' THEN 1
                    WHEN 'parsed' THEN 2
                    WHEN 'chunked' THEN 3
                    WHEN 'embedded' THEN 4
                    WHEN 'indexed' THEN 5
                    ELSE 6
                END
        """
        
        results = executor.execute_query(query)
        
        status_counts = {}
        for row in results:
            status_counts[row['status']] = row['count']
        
        return {
            "status": "healthy",
            "counts": status_counts,
            "total": sum(status_counts.values())
        }
    finally:
        db_conn.close()


def print_health_report(verbose: bool = False):
    """Print comprehensive health report"""
    print("=" * 70)
    print(f"🏥 RAG Pipeline Health Check")
    print(f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 70)
    
    overall_status = "healthy"
    
    # 1. Queue Health
    print("\n📥 Queue Health")
    print("-" * 70)
    queue_health = check_queues()
    
    if queue_health['status'] == "error":
        print(f"❌ ERROR: {queue_health['message']}")
        overall_status = "critical"
    else:
        status_icon = {"healthy": "✅", "degraded": "⚠️", "critical": "🚨"}[queue_health['status']]
        print(f"{status_icon} Status: {queue_health['status'].upper()}")
        print(f"   Queued Messages:  {queue_health['total_queued']}")
        print(f"   Poison Messages:  {queue_health['poison_count']}")
        
        if verbose or queue_health['poison_count'] > 0:
            print("\n   Queue Details:")
            for queue_name, count in queue_health['queues'].items():
                if "-poison" in queue_name and count > 0:
                    print(f"     {queue_name:35} 🚨 {count} message(s)")
                elif "-poison" not in queue_name and count > 0:
                    print(f"     {queue_name:35} 📝 {count} message(s)")
                elif verbose:
                    print(f"     {queue_name:35} ✅ Empty")
        
        if queue_health['status'] in ["degraded", "critical"]:
            overall_status = queue_health['status']
    
    # 2. Stuck Documents
    print("\n⏱️  Stuck Documents Check")
    print("-" * 70)
    stuck_check = check_stuck_documents()
    
    status_icon = {"healthy": "✅", "warning": "⚠️", "critical": "🚨"}[stuck_check['status']]
    print(f"{status_icon} Status: {stuck_check['status'].upper()}")
    print(f"   Documents stuck (>30min): {stuck_check['count']}")
    
    if stuck_check['count'] > 0:
        print("\n   Stuck Documents:")
        for doc in stuck_check['documents'][:5]:
            print(f"     {doc['id']}")
            print(f"       Status: {doc['status']}, Age: {doc['minutes_old']} minutes")
            print(f"       File: {doc['filename']}")
        
        if stuck_check['count'] > 5:
            print(f"     ... and {stuck_check['count'] - 5} more")
        
        print("\n   💡 Check these documents with:")
        print(f"      python scripts/check_document_progress.py <document_id>")
        
        if stuck_check['status'] == "critical" and overall_status == "healthy":
            overall_status = "warning"
    
    # 3. Worker Success Rates
    print("\n⚙️  Worker Success Rates (Last Hour)")
    print("-" * 70)
    failures = check_recent_failures()
    
    if failures['status'] == "unknown":
        print(f"⚠️  Status: UNKNOWN")
        print(f"   {failures.get('message', 'Could not check')}")
    else:
        status_icon = {"healthy": "✅", "warning": "⚠️", "critical": "🚨"}[failures['status']]
        print(f"{status_icon} Status: {failures['status'].upper()}")
        print(f"   Total Failures: {failures['total_failures']}")
        
        if failures.get('workers'):
            print("\n   Worker Details:")
            for worker_name, stats in failures['workers'].items():
                rate = stats['success_rate']
                if rate >= 90:
                    icon = "✅"
                elif rate >= 50:
                    icon = "⚠️"
                else:
                    icon = "🚨"
                
                print(f"     {worker_name:30} {icon} {rate}% ({stats['total'] - stats['failures']}/{stats['total']})")
        
        if failures['status'] in ["warning", "critical"]:
            if overall_status == "healthy":
                overall_status = failures['status']
            elif overall_status == "warning" and failures['status'] == "critical":
                overall_status = "critical"
    
    # 4. Document Status Distribution
    print("\n📊 Document Status Distribution")
    print("-" * 70)
    doc_counts = check_document_counts()
    
    print(f"✅ Total Documents: {doc_counts['total']}")
    print("\n   By Status:")
    for status, count in doc_counts['counts'].items():
        print(f"     {status:15} {count:5} documents")
    
    # 5. Overall Assessment
    print("\n🎯 Overall Assessment")
    print("-" * 70)
    
    status_icons = {
        "healthy": "✅",
        "warning": "⚠️",
        "degraded": "⚠️",
        "critical": "🚨"
    }
    
    print(f"{status_icons.get(overall_status, '❓')} SYSTEM STATUS: {overall_status.upper()}")
    
    if overall_status == "healthy":
        print("   All systems operational")
    elif overall_status == "warning":
        print("   System functional but requires attention")
        print("   Review warnings above")
    else:
        print("   System experiencing issues")
        print("   Immediate action required - review errors above")
    
    print("\n💡 Monitoring Commands:")
    print("   python scripts/pipeline_health_check.py --verbose")
    print("   python scripts/monitor_queue_health.py --watch")
    print("   python scripts/check_document_progress.py --latest")
    
    print("=" * 70)


def watch_mode(interval: int = 60):
    """Watch mode - continuously monitor health"""
    print(f"🔄 Watch Mode - Refreshing every {interval} seconds")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            os.system('clear' if os.name != 'nt' else 'cls')
            print_health_report(verbose=False)
            print(f"\nRefreshing in {interval} seconds... (Ctrl+C to stop)")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n👋 Watch mode stopped")


def main():
    parser = argparse.ArgumentParser(
        description="Check overall RAG pipeline health"
    )
    
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed information")
    parser.add_argument("--watch", "-w", action="store_true",
                       help="Watch mode (auto-refresh)")
    parser.add_argument("--interval", type=int, default=60,
                       help="Watch mode refresh interval (seconds)")
    
    args = parser.parse_args()
    
    if args.watch:
        watch_mode(args.interval)
    else:
        print_health_report(args.verbose)


if __name__ == "__main__":
    main()

