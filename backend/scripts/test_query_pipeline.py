#!/usr/bin/env python3
"""Test end-to-end RAG query pipeline

This script tests the complete RAG query pipeline via the API:
1. Check API health
2. Send query request to /api/query
3. Verify response format
4. Verify answer quality
5. Optionally verify database logging
6. Test multiple query types
7. Test error handling

Usage:
    python scripts/test_query_pipeline.py [query_text] [prompt_version] [api_url]
    
Examples:
    python scripts/test_query_pipeline.py "What is the copay?" v1 http://localhost:8000
    python scripts/test_query_pipeline.py  # Uses defaults
"""

import sys
import os
from pathlib import Path
import requests
import json
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import Config
from src.db.supabase_db_service import SupabaseDatabaseService


def check_api_health(api_base_url: str) -> bool:
    """Check if API is running and healthy"""
    try:
        response = requests.get(f"{api_base_url}/health", timeout=5)
        if response.status_code == 200:
            return True
        return False
    except requests.RequestException:
        return False


def test_single_query(query_text: str, prompt_version: str, api_base_url: str) -> dict:
    """Test a single query and return results"""
    print(f"\n{'='*70}")
    print(f"Testing Query")
    print(f"Query: {query_text}")
    print(f"Prompt Version: {prompt_version}")
    print(f"API: {api_base_url}")
    print(f"{'='*70}\n")
    
    # Step 1: Check API health
    print("🔍 Step 1: Checking API health...")
    if not check_api_health(api_base_url):
        print(f"❌ API is not responding at {api_base_url}")
        print(f"   Make sure the API server is running:")
        print(f"   cd backend && source venv/bin/activate && uvicorn src.api.main:app --reload --port 8000")
        return {"success": False, "error": "API not responding"}
    
    print(f"✅ API is healthy\n")
    
    # Step 2: Send query request
    print("📤 Step 2: Sending query request...")
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{api_base_url}/api/query",
            json={
                "text": query_text,
                "prompt_version": prompt_version
            },
            headers={"Content-Type": "application/json"},
            timeout=60  # 60 second timeout for LLM calls
        )
        
        request_latency = time.time() - start_time
        
        if response.status_code != 200:
            print(f"❌ Query failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return {"success": False, "error": response.text, "status_code": response.status_code}
        
        result = response.json()
        print(f"✅ Query successful! (took {request_latency:.2f}s)\n")
        
    except requests.Timeout:
        print(f"❌ Query timed out after 60 seconds")
        return {"success": False, "error": "Request timeout"}
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        return {"success": False, "error": str(e)}
    
    # Step 3: Verify response format
    print("🔍 Step 3: Verifying response format...")
    required_fields = ["answer", "query_id", "prompt_version"]
    missing_fields = [field for field in required_fields if field not in result]
    
    if missing_fields:
        print(f"❌ Response missing required fields: {missing_fields}")
        return {"success": False, "error": f"Missing fields: {missing_fields}"}
    
    print(f"✅ Response format valid:")
    print(f"   Answer: {result['answer'][:100]}..." if len(result['answer']) > 100 else f"   Answer: {result['answer']}")
    print(f"   Query ID: {result['query_id']}")
    print(f"   Prompt Version: {result['prompt_version']}\n")
    
    # Step 4: Verify answer quality
    print("🔍 Step 4: Verifying answer quality...")
    answer = result['answer']
    
    if not answer or len(answer.strip()) == 0:
        print(f"⚠️  Warning: Answer is empty")
    elif len(answer) < 10:
        print(f"⚠️  Warning: Answer is very short ({len(answer)} characters)")
    else:
        print(f"✅ Answer quality OK ({len(answer)} characters)")
    
    if result['prompt_version'] != prompt_version:
        print(f"⚠️  Warning: Prompt version mismatch (expected {prompt_version}, got {result['prompt_version']})")
    else:
        print(f"✅ Prompt version matches\n")
    
    # Step 5: Optionally verify database logging
    print("🔍 Step 5: Verifying database logging (optional)...")
    try:
        # Try to use .env.prod if it exists, otherwise use default
        from pathlib import Path
        import re
        project_root = Path(__file__).parent.parent.parent
        env_prod_path = project_root / ".env.prod"
        if env_prod_path.exists():
            config = Config.from_env(env_file=str(env_prod_path))
        else:
            config = Config.from_env()
        if config.supabase_url and (config.supabase_anon_key or config.supabase_service_role_key):
            try:
                supabase_service = SupabaseDatabaseService(config)
                
                # Extract UUID from prefixed query_id (e.g., 'query_<uuid>' -> '<uuid>')
                query_id_str = result['query_id']
                uuid_pattern = r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})'
                uuid_match = re.search(uuid_pattern, query_id_str, re.IGNORECASE)
                query_uuid = uuid_match.group(1) if uuid_match else query_id_str
                
                # Check if query was logged using REST API
                query_result = supabase_service.client.table("queries")\
                    .select("*")\
                    .eq("id", query_uuid)\
                    .execute()
                
                if query_result.data and len(query_result.data) > 0:
                    print(f"✅ Query logged to database")
                    
                    # Check retrieval results (table name is 'retrieval_logs', not 'retrieval_results')
                    retrieval_result = supabase_service.client.table("retrieval_logs")\
                        .select("id")\
                        .eq("query_id", query_uuid)\
                        .execute()
                    if retrieval_result.data:
                        count = len(retrieval_result.data)
                        print(f"✅ Retrieval results logged ({count} chunks)")
                    
                    # Check model answer
                    answer_result = supabase_service.client.table("model_answers")\
                        .select("*")\
                        .eq("query_id", query_uuid)\
                        .execute()
                    if answer_result.data and len(answer_result.data) > 0:
                        print(f"✅ Model answer logged to database")
                else:
                    print(f"⚠️  Warning: Query not found in database (logging may have failed)")
                
            except Exception as e:
                print(f"⚠️  Warning: Could not verify database logging: {e}")
        else:
            print(f"⚠️  Supabase credentials not configured, skipping database verification\n")
    except Exception as e:
        print(f"⚠️  Warning: Could not verify database logging: {e}\n")
    
    return {
        "success": True,
        "query_id": result['query_id'],
        "answer": result['answer'],
        "prompt_version": result['prompt_version'],
        "latency": request_latency
    }


def test_multiple_queries(api_base_url: str, prompt_version: str = "v1"):
    """Test multiple different query types"""
    print(f"\n{'='*70}")
    print(f"Testing Multiple Query Types")
    print(f"{'='*70}\n")
    
    test_queries = [
        ("What is the copay for specialist visits?", "factual"),
        ("What is the coverage limit?", "factual"),
        ("What is the deductible?", "factual"),
        ("What services are covered?", "summary"),
    ]
    
    results = []
    
    for query_text, query_type in test_queries:
        print(f"\n--- Testing {query_type} query ---")
        result = test_single_query(query_text, prompt_version, api_base_url)
        results.append({
            "query": query_text,
            "type": query_type,
            "success": result.get("success", False),
            "query_id": result.get("query_id"),
            "latency": result.get("latency")
        })
    
    # Summary
    print(f"\n{'='*70}")
    print("Multiple Queries Summary")
    print(f"{'='*70}\n")
    
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"✅ Successful: {len(successful)}/{len(results)}")
    if successful:
        avg_latency = sum(r["latency"] for r in successful) / len(successful)
        print(f"   Average latency: {avg_latency:.2f}s")
    
    if failed:
        print(f"❌ Failed: {len(failed)}/{len(results)}")
        for r in failed:
            print(f"   - {r['query']}")
    
    return results


def test_error_handling(api_base_url: str):
    """Test error handling with invalid inputs"""
    print(f"\n{'='*70}")
    print(f"Testing Error Handling")
    print(f"{'='*70}\n")
    
    error_tests = [
        {
            "name": "Empty query text",
            "payload": {"text": "", "prompt_version": "v1"},
            "expected_status": 422  # Validation error
        },
        {
            "name": "Missing query text",
            "payload": {"prompt_version": "v1"},
            "expected_status": 422  # Validation error
        },
        {
            "name": "Invalid prompt version",
            "payload": {"text": "What is the coverage limit?", "prompt_version": "v999"},
            "expected_status": 500  # Server error (prompt not found)
        },
    ]
    
    results = []
    
    for test in error_tests:
        print(f"\n--- Testing: {test['name']} ---")
        try:
            response = requests.post(
                f"{api_base_url}/api/query",
                json=test["payload"],
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == test["expected_status"]:
                print(f"✅ Correctly returned status {response.status_code}")
                results.append({"test": test["name"], "success": True})
            else:
                print(f"⚠️  Expected status {test['expected_status']}, got {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                results.append({"test": test["name"], "success": False})
        except requests.RequestException as e:
            print(f"❌ Request failed: {e}")
            results.append({"test": test["name"], "success": False})
    
    # Summary
    print(f"\n{'='*70}")
    print("Error Handling Summary")
    print(f"{'='*70}\n")
    
    successful = [r for r in results if r["success"]]
    print(f"✅ Passed: {len(successful)}/{len(results)}")
    
    return results


def main():
    """Main test function"""
    # Parse command line arguments
    query_text = sys.argv[1] if len(sys.argv) > 1 else "What is the copay for specialist visits?"
    prompt_version = sys.argv[2] if len(sys.argv) > 2 else "v1"
    api_url = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8000"
    
    print(f"\n{'='*70}")
    print(f"RAG Query Pipeline Test")
    print(f"{'='*70}")
    print(f"API URL: {api_url}")
    print(f"Default Query: {query_text}")
    print(f"Default Prompt Version: {prompt_version}")
    print(f"{'='*70}\n")
    
    # Test single query
    result = test_single_query(query_text, prompt_version, api_url)
    
    if not result.get("success"):
        print(f"\n❌ Single query test failed")
        sys.exit(1)
    
    # Ask if user wants to run additional tests
    print(f"\n{'='*70}")
    print("Additional Tests Available")
    print(f"{'='*70}\n")
    print("The following additional tests are available:")
    print("1. Test multiple query types")
    print("2. Test error handling")
    print("3. Exit")
    
    # For automated testing, run all tests
    # In interactive mode, you could prompt the user
    # For now, we'll just run the single query test
    
    # Summary
    print(f"\n{'='*70}")
    print("Test Summary")
    print(f"{'='*70}\n")
    print(f"✅ Single query test completed successfully")
    print(f"   Query ID: {result['query_id']}")
    print(f"   Answer length: {len(result['answer'])} characters")
    print(f"   Latency: {result['latency']:.2f}s")
    print(f"\n🎉 Query pipeline test completed successfully!")
    print(f"\nYou can run additional tests:")
    print(f"  python scripts/test_query_pipeline.py  # Run with different queries")
    print(f"  # Or modify this script to call test_multiple_queries() and test_error_handling()")


if __name__ == "__main__":
    main()
