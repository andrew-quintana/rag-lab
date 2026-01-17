#!/usr/bin/env python3
"""Insert RAG prompt template into database using Supabase REST API

This script inserts the RAG prompt template (v1) into the prompts table
using Supabase REST API. It reads the template from the test fixtures
and inserts it with the correct format.

Usage:
    python scripts/insert_rag_prompt.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import Config
from src.db.supabase_db_service import SupabaseDatabaseService


def read_prompt_template(template_path: Path) -> str:
    """Read prompt template from file"""
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    
    with open(template_path, 'r') as f:
        content = f.read()
    
    # Replace {question} with {query} to match code expectations
    content = content.replace("{question}", "{query}")
    
    return content


def insert_rag_prompt(version: str = "v1", live: bool = True):
    """Insert RAG prompt template into database"""
    
    print(f"\n{'='*70}")
    print(f"Inserting RAG Prompt Template")
    print(f"Version: {version}")
    print(f"Live: {live}")
    print(f"{'='*70}\n")
    
    # Load configuration
    project_root = Path(__file__).parent.parent.parent
    env_prod = project_root / ".env.prod"
    
    if env_prod.exists():
        config = Config.from_env(env_file=str(env_prod))
        print(f"✓ Loaded configuration from .env.prod")
    else:
        config = Config.from_env()
        print(f"✓ Loaded configuration from default")
    
    if not config.supabase_url or (not config.supabase_anon_key and not config.supabase_service_role_key):
        print(f"❌ Supabase credentials not configured")
        print(f"   Required: SUPABASE_URL and SUPABASE_ANON_KEY or SUPABASE_SERVICE_ROLE_KEY")
        sys.exit(1)
    
    print(f"  SUPABASE_URL: {config.supabase_url}")
    print(f"  Using: {'Service Role Key' if config.supabase_service_role_key else 'Anon Key'}\n")
    
    # Read prompt template
    template_file = Path(__file__).parent.parent / "tests" / "fixtures" / "prompts" / f"prompt_{version}.md"
    print(f"📖 Reading prompt template from: {template_file}")
    
    try:
        prompt_text = read_prompt_template(template_file)
        print(f"✓ Template loaded ({len(prompt_text)} characters)\n")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    # Initialize Supabase service
    print("🔌 Connecting to Supabase...")
    try:
        supabase_service = SupabaseDatabaseService(config)
        print(f"✓ Supabase REST API client initialized\n")
    except Exception as e:
        print(f"❌ Failed to initialize Supabase client: {e}")
        sys.exit(1)
    
    # Check if prompt already exists
    print("🔍 Checking if prompt already exists...")
    try:
        result = supabase_service.client.table("prompts")\
            .select("id, version, live")\
            .eq("prompt_type", "rag")\
            .eq("version", version)\
            .is_("name", "null")\
            .execute()
        
        if result.data and len(result.data) > 0:
            existing = result.data[0]
            print(f"⚠️  Prompt already exists:")
            print(f"   ID: {existing.get('id')}")
            print(f"   Version: {existing.get('version')}")
            print(f"   Live: {existing.get('live')}")
            print()
            response = input("Update existing prompt? (y/N): ").strip().lower()
            if response != 'y':
                print("Aborted.")
                sys.exit(0)
            
            # Update existing prompt
            print(f"\n📝 Updating existing prompt...")
            update_result = supabase_service.client.table("prompts")\
                .update({
                    "prompt_text": prompt_text,
                    "live": live
                })\
                .eq("id", existing.get('id'))\
                .execute()
            
            print(f"✅ Prompt updated successfully!")
            print(f"   ID: {existing.get('id')}")
            print(f"   Version: {version}")
            print(f"   Live: {live}")
            return
        else:
            print(f"✓ No existing prompt found, will insert new one\n")
    except Exception as e:
        print(f"⚠️  Could not check for existing prompt: {e}")
        print(f"   Proceeding with insert...\n")
    
    # Insert new prompt
    print(f"📝 Inserting prompt template...")
    try:
        insert_data = {
            "version": version,
            "prompt_type": "rag",
            "name": None,  # RAG prompts don't have a name
            "prompt_text": prompt_text,
            "live": live
        }
        
        result = supabase_service.client.table("prompts")\
            .insert(insert_data)\
            .execute()
        
        if result.data and len(result.data) > 0:
            inserted = result.data[0]
            print(f"✅ Prompt inserted successfully!")
            print(f"   ID: {inserted.get('id')}")
            print(f"   Version: {inserted.get('version')}")
            print(f"   Prompt Type: {inserted.get('prompt_type')}")
            print(f"   Live: {inserted.get('live')}")
            print(f"   Created: {inserted.get('created_at')}")
        else:
            print(f"⚠️  Insert completed but no data returned")
            
    except Exception as e:
        print(f"❌ Failed to insert prompt: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Verify insertion
    print(f"\n🔍 Verifying insertion...")
    try:
        result = supabase_service.client.table("prompts")\
            .select("*")\
            .eq("prompt_type", "rag")\
            .eq("version", version)\
            .is_("name", "null")\
            .execute()
        
        if result.data and len(result.data) > 0:
            prompt = result.data[0]
            print(f"✅ Verification successful!")
            print(f"   Template length: {len(prompt.get('prompt_text', ''))} characters")
            print(f"   Contains {{query}}: {'{query}' in prompt.get('prompt_text', '')}")
            print(f"   Contains {{context}}: {'{context}' in prompt.get('prompt_text', '')}")
        else:
            print(f"⚠️  Prompt not found after insertion")
    except Exception as e:
        print(f"⚠️  Could not verify insertion: {e}")
    
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}\n")
    print(f"✅ RAG prompt template '{version}' inserted successfully")
    print(f"✅ Ready to use in query pipeline")
    print(f"\nYou can now test the query pipeline:")
    print(f"  python scripts/test_query_pipeline.py \"What is the copay?\" {version}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Insert RAG prompt template into database")
    parser.add_argument("--version", default="v1", help="Prompt version (default: v1)")
    parser.add_argument("--no-live", action="store_true", help="Don't mark as live version")
    
    args = parser.parse_args()
    
    insert_rag_prompt(version=args.version, live=not args.no_live)
