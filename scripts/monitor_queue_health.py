#!/usr/bin/env python3
"""Queue Health Monitoring Script

Standalone script to monitor Azure Storage Queue health.
Can be run anytime to check queue status.

Usage:
    python scripts/monitor_queue_health.py              # One-time check
    python scripts/monitor_queue_health.py --watch      # Watch mode (refresh every 10s)
    python scripts/monitor_queue_health.py --clear-poison  # Clear poison queues
    python scripts/monitor_queue_health.py --peek QUEUE_NAME  # Peek at messages
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add backend to path for imports
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from azure.storage.queue import QueueServiceClient
from dotenv import load_dotenv


# Load environment (prefer .env.prod for cloud, fallback to .env.local)
env_file = Path(__file__).parent.parent / ".env.prod"
if not env_file.exists():
    env_file = Path(__file__).parent.parent / ".env.local"
if env_file.exists():
    load_dotenv(env_file)


QUEUE_NAMES = [
    "ingestion-uploads",
    "ingestion-chunking",
    "ingestion-embeddings",
    "ingestion-indexing"
]


def get_connection_string():
    """Get Azure Storage connection string from environment"""
    conn_str = (
        os.getenv("AZURE_STORAGE_QUEUES_CONNECTION_STRING") or
        os.getenv("AZURE_BLOB_CONNECTION_STRING") or
        os.getenv("AzureWebJobsStorage")
    )
    
    if not conn_str:
        print("❌ Error: No Azure Storage connection string found")
        print("   Set one of: AZURE_STORAGE_QUEUES_CONNECTION_STRING, AZURE_BLOB_CONNECTION_STRING, AzureWebJobsStorage")
        sys.exit(1)
    
    return conn_str


def format_timestamp():
    """Format current timestamp"""
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def check_queue_health(show_timestamp=True):
    """Check and display queue health status"""
    if show_timestamp:
        print(f"\n⏰ {format_timestamp()}")
    
    print("=" * 70)
    print("📊 Queue Health Status")
    print("=" * 70)
    
    conn_str = get_connection_string()
    client = QueueServiceClient.from_connection_string(conn_str)
    
    queue_data = []
    total_messages = 0
    poison_messages = 0
    
    # Check all queues (main + poison)
    all_queues = QUEUE_NAMES + [f"{q}-poison" for q in QUEUE_NAMES]
    
    for queue_name in all_queues:
        try:
            queue = client.get_queue_client(queue_name)
            props = queue.get_queue_properties()
            count = props.approximate_message_count
            
            is_poison = "-poison" in queue_name
            if is_poison:
                poison_messages += count
            else:
                total_messages += count
            
            queue_data.append({
                'name': queue_name,
                'count': count,
                'is_poison': is_poison,
                'status': 'ok' if count == 0 else 'warn' if is_poison else 'active'
            })
        except Exception as e:
            queue_data.append({
                'name': queue_name,
                'count': -1,
                'is_poison': "-poison" in queue_name,
                'status': 'error',
                'error': str(e)
            })
    
    # Display main queues
    print("\n🔵 Main Queues:")
    for q in queue_data:
        if not q['is_poison']:
            if q['status'] == 'error':
                print(f"  {q['name']:<35} ❌ Error: {q.get('error', 'Unknown')}")
            elif q['count'] == 0:
                print(f"  {q['name']:<35} ✅ Empty")
            else:
                print(f"  {q['name']:<35} 📝 {q['count']} message(s)")
    
    # Display poison queues
    print("\n☠️  Poison Queues:")
    poison_with_messages = []
    for q in queue_data:
        if q['is_poison']:
            if q['status'] == 'error':
                print(f"  {q['name']:<35} ⚠️  Queue may not exist")
            elif q['count'] == 0:
                print(f"  {q['name']:<35} ✅ Empty")
            else:
                print(f"  {q['name']:<35} ⚠️  {q['count']} message(s)")
                poison_with_messages.append(q['name'])
    
    # Summary
    print("\n" + "=" * 70)
    if poison_messages > 0:
        print(f"⚠️  WARNING: {poison_messages} message(s) in poison queues")
        print(f"💡 To inspect: python scripts/monitor_queue_health.py --peek {poison_with_messages[0]}")
        print(f"💡 To clear: python scripts/monitor_queue_health.py --clear-poison")
    elif total_messages > 0:
        print(f"📝 {total_messages} message(s) queued for processing")
    else:
        print("✅ All queues are empty and healthy")
    
    print("=" * 70)
    
    return queue_data, total_messages, poison_messages


def peek_queue(queue_name, max_messages=5):
    """Peek at messages in a queue"""
    print(f"\n🔍 Peeking at queue: {queue_name}")
    print("=" * 70)
    
    conn_str = get_connection_string()
    client = QueueServiceClient.from_connection_string(conn_str)
    
    try:
        queue = client.get_queue_client(queue_name)
        messages = list(queue.peek_messages(max_messages=max_messages))
        
        if not messages:
            print("✅ Queue is empty")
            return
        
        print(f"Found {len(messages)} message(s):\n")
        
        for i, msg in enumerate(messages, 1):
            print(f"Message {i}:")
            print(f"  ID: {msg.id}")
            print(f"  Insertion Time: {msg.insertion_time}")
            print(f"  Expiration Time: {msg.expiration_time}")
            print(f"  Dequeue Count: {msg.dequeue_count}")
            
            # Try to parse content as JSON
            try:
                content = json.loads(msg.content)
                print(f"  Content (parsed):")
                for key, value in content.items():
                    if isinstance(value, dict):
                        print(f"    {key}: {json.dumps(value, indent=6)}")
                    else:
                        print(f"    {key}: {value}")
            except:
                print(f"  Content (raw): {msg.content[:200]}")
                if len(msg.content) > 200:
                    print(f"    ... ({len(msg.content)} total characters)")
            
            print()
    
    except Exception as e:
        print(f"❌ Error peeking queue: {e}")


def clear_poison_queues(confirm=True):
    """Clear all poison queues"""
    print("\n☠️  Clear Poison Queues")
    print("=" * 70)
    
    conn_str = get_connection_string()
    client = QueueServiceClient.from_connection_string(conn_str)
    
    poison_queues = [f"{q}-poison" for q in QUEUE_NAMES]
    
    # Check what's there
    to_clear = []
    total_to_delete = 0
    for queue_name in poison_queues:
        try:
            queue = client.get_queue_client(queue_name)
            props = queue.get_queue_properties()
            count = props.approximate_message_count
            if count > 0:
                to_clear.append((queue_name, count))
                total_to_delete += count
        except:
            pass
    
    if not to_clear:
        print("✅ No messages in poison queues")
        return
    
    print(f"\nFound {total_to_delete} message(s) in poison queues:")
    for queue_name, count in to_clear:
        print(f"  {queue_name}: {count} message(s)")
    
    if confirm:
        response = input("\n⚠️  Are you sure you want to clear these queues? (yes/no): ")
        if response.lower() != "yes":
            print("❌ Cancelled")
            return
    
    # Clear queues
    print("\n🗑️  Clearing poison queues...")
    for queue_name, count in to_clear:
        try:
            queue = client.get_queue_client(queue_name)
            queue.clear_messages()
            print(f"  ✅ Cleared {queue_name}")
        except Exception as e:
            print(f"  ❌ Failed to clear {queue_name}: {e}")
    
    print("\n✅ Poison queue cleanup complete")


def watch_mode(interval=10):
    """Watch mode - continuously monitor queues"""
    print("🔄 Watch Mode (Ctrl+C to stop)")
    print(f"   Refreshing every {interval} seconds")
    
    try:
        iteration = 0
        while True:
            if iteration > 0:
                # Clear screen (works on most terminals)
                os.system('clear' if os.name != 'nt' else 'cls')
            
            check_queue_health(show_timestamp=True)
            
            iteration += 1
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n\n👋 Watch mode stopped")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor Azure Storage Queue health",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/monitor_queue_health.py                    # One-time check
  python scripts/monitor_queue_health.py --watch            # Watch mode
  python scripts/monitor_queue_health.py --peek ingestion-uploads-poison
  python scripts/monitor_queue_health.py --clear-poison     # Clear poison queues
        """
    )
    
    parser.add_argument("--watch", action="store_true", 
                       help="Watch mode (refresh every 10 seconds)")
    parser.add_argument("--interval", type=int, default=10,
                       help="Watch mode refresh interval (default: 10s)")
    parser.add_argument("--peek", type=str, metavar="QUEUE_NAME",
                       help="Peek at messages in specified queue")
    parser.add_argument("--clear-poison", action="store_true",
                       help="Clear all poison queues")
    parser.add_argument("--yes", action="store_true",
                       help="Skip confirmation prompts")
    
    args = parser.parse_args()
    
    if args.peek:
        peek_queue(args.peek)
    elif args.clear_poison:
        clear_poison_queues(confirm=not args.yes)
    elif args.watch:
        watch_mode(interval=args.interval)
    else:
        check_queue_health(show_timestamp=False)


if __name__ == "__main__":
    main()

