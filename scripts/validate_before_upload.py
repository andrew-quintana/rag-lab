#!/usr/bin/env python3
"""Pre-Upload Validation Script

Validates Azure Functions setup before attempting document upload.
Checks:
1. Function app status
2. Queue status
3. Function registration
4. Environment configuration
"""

import os
import sys
import subprocess
import json
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from azure.storage.queue import QueueServiceClient
from dotenv import load_dotenv

# Load environment
env_file = Path(__file__).parent.parent / ".env.prod"
if env_file.exists():
    load_dotenv(env_file)

FUNCTION_APP = "func-raglab-uploadworkers"
RESOURCE_GROUP = "rag-lab"

def run_az_command(cmd):
    """Run Azure CLI command and return result"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return None

def check_function_app_status():
    """Check if function app is running"""
    print("\n" + "="*70)
    print("1️⃣  Checking Function App Status")
    print("="*70)
    
    status = run_az_command(
        f"az functionapp show --name {FUNCTION_APP} --resource-group {RESOURCE_GROUP} "
        f"--query '{{state:state, runtime:siteConfig.linuxFxVersion, enabled:enabled}}' -o json"
    )
    
    if not status:
        print("❌ Failed to get function app status")
        return False
    
    data = json.loads(status)
    print(f"   State: {data.get('state', 'Unknown')}")
    print(f"   Runtime: {data.get('runtime', 'Unknown')}")
    print(f"   Enabled: {data.get('enabled', 'Unknown')}")
    
    if data.get('state') == 'Running' and data.get('enabled'):
        print("   ✅ Function app is running and enabled")
        return True
    else:
        print("   ⚠️  Function app may not be ready")
        return False

def check_function_registration():
    """Check if all functions are registered"""
    print("\n" + "="*70)
    print("2️⃣  Checking Function Registration")
    print("="*70)
    
    functions = run_az_command(
        f"az functionapp function list --name {FUNCTION_APP} "
        f"--resource-group {RESOURCE_GROUP} -o json"
    )
    
    if not functions:
        print("❌ Failed to list functions")
        return False
    
    func_list = json.loads(functions)
    expected = ["ingestion-worker", "chunking-worker", "embedding-worker", "indexing-worker"]
    found = [f.get('name', '').replace(f"{FUNCTION_APP}/", "") for f in func_list]
    
    print(f"   Found {len(found)} functions:")
    for func in found:
        print(f"      - {func}")
    
    missing = set(expected) - set(found)
    if missing:
        print(f"   ⚠️  Missing functions: {', '.join(missing)}")
        return False
    
    print("   ✅ All expected functions are registered")
    return True

def check_queue_status():
    """Check queue status"""
    print("\n" + "="*70)
    print("3️⃣  Checking Queue Status")
    print("="*70)
    
    conn_str = os.getenv("AZURE_STORAGE_QUEUES_CONNECTION_STRING")
    if not conn_str:
        print("   ⚠️  AZURE_STORAGE_QUEUES_CONNECTION_STRING not set")
        return False
    
    try:
        queue_service = QueueServiceClient.from_connection_string(conn_str)
        queues = ["ingestion-uploads", "ingestion-chunking", "ingestion-embeddings", "ingestion-indexing"]
        
        all_ok = True
        for queue_name in queues:
            try:
                queue_client = queue_service.get_queue_client(queue_name)
                props = queue_client.get_queue_properties()
                count = props.approximate_message_count or 0
                
                if count == 0:
                    print(f"   ✅ {queue_name}: Empty")
                else:
                    print(f"   📝 {queue_name}: {count} message(s)")
                    if queue_name == "ingestion-uploads" and count > 5:
                        print(f"      ⚠️  High message count - may indicate processing issues")
                        all_ok = False
            except Exception as e:
                print(f"   ⚠️  {queue_name}: Error checking ({str(e)[:50]})")
                all_ok = False
        
        return all_ok
    except Exception as e:
        print(f"   ❌ Failed to connect to queues: {e}")
        return False

def check_environment_variables():
    """Check critical environment variables"""
    print("\n" + "="*70)
    print("4️⃣  Checking Environment Variables")
    print("="*70)
    
    required_vars = [
        "AzureWebJobsStorage",
        "DATABASE_URL",
        "SUPABASE_URL",
        "SUPABASE_KEY"
    ]
    
    settings = run_az_command(
        f"az functionapp config appsettings list --name {FUNCTION_APP} "
        f"--resource-group {RESOURCE_GROUP} -o json"
    )
    
    if not settings:
        print("   ⚠️  Could not retrieve app settings")
        return False
    
    app_settings = {s['name']: s.get('value', '') for s in json.loads(settings)}
    
    all_present = True
    for var in required_vars:
        if var in app_settings and app_settings[var]:
            print(f"   ✅ {var}: Set")
        else:
            print(f"   ❌ {var}: Missing or empty")
            all_present = False
    
    return all_present

def main():
    """Run all validation checks"""
    print("\n" + "="*70)
    print("🔍 PRE-UPLOAD VALIDATION")
    print("="*70)
    print(f"Function App: {FUNCTION_APP}")
    print(f"Resource Group: {RESOURCE_GROUP}")
    
    results = {
        "function_app": check_function_app_status(),
        "function_registration": check_function_registration(),
        "queue_status": check_queue_status(),
        "environment": check_environment_variables()
    }
    
    print("\n" + "="*70)
    print("📊 VALIDATION SUMMARY")
    print("="*70)
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {check.replace('_', ' ').title()}: {status}")
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Ready for upload!")
        print("="*70)
        return 0
    else:
        print("⚠️  SOME CHECKS FAILED - Review issues above before uploading")
        print("="*70)
        return 1

if __name__ == "__main__":
    sys.exit(main())
