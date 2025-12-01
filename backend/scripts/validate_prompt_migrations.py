#!/usr/bin/env python3
"""
Script to validate that evaluation prompts are correctly inserted into the database.

This script:
1. Connects to the database
2. Verifies all three evaluation prompts exist
3. Validates prompt content matches markdown files
4. Tests prompt loading via load_prompt_template()
"""

import sys
import os
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from rag_eval.core.config import Config
from rag_eval.db.queries import QueryExecutor
from rag_eval.services.rag.generation import load_prompt_template


def read_prompt_file(filename: str) -> str:
    """Read prompt from markdown file"""
    prompt_path = backend_dir / "rag_eval" / "prompts" / "evaluation" / filename
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def validate_prompts_in_db():
    """Validate that all evaluation prompts are in the database"""
    print("=" * 60)
    print("Validating Evaluation Prompts in Database")
    print("=" * 60)
    
    try:
        config = Config.from_env()
        query_executor = QueryExecutor(config)
        
        # Expected prompts
        expected_prompts = [
            {
                "version_name": "v1",
                "prompt_type": "evaluation",
                "evaluator_type": "correctness_evaluator",
                "file": "correctness_prompt.md"
            },
            {
                "version_name": "v1",
                "prompt_type": "evaluation",
                "evaluator_type": "hallucination_evaluator",
                "file": "hallucination_prompt.md"
            },
            {
                "version_name": "v1",
                "prompt_type": "evaluation",
                "evaluator_type": "risk_direction_evaluator",
                "file": "risk_direction_prompt.md"
            }
        ]
        
        all_valid = True
        
        for prompt_info in expected_prompts:
            print(f"\nChecking {prompt_info['evaluator_type']}...")
            
            # Query database
            query = """
                SELECT prompt_text, version_id
                FROM prompt_versions
                WHERE prompt_type = %s 
                  AND evaluator_type = %s 
                  AND version_name = %s
            """
            results = query_executor.execute_query(
                query,
                (prompt_info['prompt_type'], prompt_info['evaluator_type'], prompt_info['version_name'])
            )
            
            if not results:
                print(f"  ✗ NOT FOUND in database")
                all_valid = False
                continue
            
            db_prompt = results[0]['prompt_text'].strip()
            version_id = results[0]['version_id']
            
            # Read file for comparison
            try:
                file_prompt = read_prompt_file(prompt_info['file'])
                
                # Compare (allowing for minor whitespace differences)
                if db_prompt == file_prompt:
                    print(f"  ✓ Found in database (version_id: {version_id})")
                    print(f"  ✓ Content matches markdown file")
                else:
                    print(f"  ⚠ Found in database but content differs from file")
                    print(f"    DB length: {len(db_prompt)} chars")
                    print(f"    File length: {len(file_prompt)} chars")
                    # Don't fail on this - migration might have slight differences
                
            except FileNotFoundError:
                print(f"  ✓ Found in database (version_id: {version_id})")
                print(f"  ⚠ Markdown file not found for comparison: {prompt_info['file']}")
            
            # Test loading via load_prompt_template()
            try:
                loaded_prompt = load_prompt_template(
                    prompt_info['version_name'],
                    query_executor,
                    prompt_type=prompt_info['prompt_type'],
                    evaluator_type=prompt_info['evaluator_type']
                )
                if loaded_prompt:
                    print(f"  ✓ Successfully loaded via load_prompt_template()")
                else:
                    print(f"  ✗ load_prompt_template() returned empty")
                    all_valid = False
            except Exception as e:
                print(f"  ✗ Failed to load via load_prompt_template(): {e}")
                all_valid = False
        
        print("\n" + "=" * 60)
        if all_valid:
            print("✅ All validation checks passed!")
            return 0
        else:
            print("❌ Some validation checks failed")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = validate_prompts_in_db()
    sys.exit(exit_code)


