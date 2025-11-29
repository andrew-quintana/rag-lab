#!/usr/bin/env python3
"""Comprehensive validation script for package structure standards"""

import sys
from pathlib import Path

def test_imports():
    """Test all package imports"""
    print("=" * 60)
    print("Testing Package Imports")
    print("=" * 60)
    
    tests = [
        ("Core", "from rag_eval.core import Chunk, Query, Config, get_logger, RAGEvalException"),
        ("RAG Services", "from rag_eval.services.rag import chunk_text, ingest_document, extract_text_from_document, upload_document_to_blob"),
        ("DB", "from rag_eval.db import QueryExecutor, DatabaseConnection, QueryRecord"),
        ("Utils", "from rag_eval.utils import generate_id, timer, read_prompt_file"),
        ("Evaluator", "from rag_eval.services.evaluator import evaluate_answer, normalize_score, aggregate_scores"),
        ("Meta-eval", "from rag_eval.services.meta_eval import generate_summary, compute_judge_consistency, detect_drift, compare_versions"),
        ("Main Package", "from rag_eval import Chunk, Query, Config, get_logger, __version__"),
    ]
    
    results = []
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"✓ {name:20} - Imports work")
            results.append((name, True, None))
        except Exception as e:
            print(f"✗ {name:20} - FAILED: {e}")
            results.append((name, False, str(e)))
    
    return results

def test_all_lists():
    """Test that __all__ lists are properly defined"""
    print("\n" + "=" * 60)
    print("Testing __all__ Lists")
    print("=" * 60)
    
    tests = [
        ("rag_eval", "rag_eval"),
        ("rag_eval.core", "rag_eval.core"),
        ("rag_eval.services.rag", "rag_eval.services.rag"),
        ("rag_eval.db", "rag_eval.db"),
        ("rag_eval.utils", "rag_eval.utils"),
        ("rag_eval.services.evaluator", "rag_eval.services.evaluator"),
        ("rag_eval.services.meta_eval", "rag_eval.services.meta_eval"),
    ]
    
    results = []
    for name, module_name in tests:
        try:
            module = __import__(module_name, fromlist=['__all__'])
            if hasattr(module, '__all__'):
                exports = module.__all__
                print(f"✓ {name:30} - {len(exports)} exports defined")
                results.append((name, True, len(exports)))
            else:
                print(f"✗ {name:30} - Missing __all__ list")
                results.append((name, False, "Missing __all__"))
        except Exception as e:
            print(f"✗ {name:30} - FAILED: {e}")
            results.append((name, False, str(e)))
    
    return results

def test_exports_validity():
    """Test that all items in __all__ are actually exported"""
    print("\n" + "=" * 60)
    print("Testing Export Validity")
    print("=" * 60)
    
    try:
        import rag_eval
        main_all = rag_eval.__all__
        
        missing = []
        for item in main_all:
            if not item.startswith('_'):
                if not hasattr(rag_eval, item):
                    missing.append(item)
        
        if missing:
            print(f"✗ Main package - Missing exports: {missing}")
            return False
        else:
            print(f"✓ Main package - All {len(main_all)} exports are valid")
            return True
    except Exception as e:
        print(f"✗ Export validation - FAILED: {e}")
        return False

def test_module_execution():
    """Test that modules can be executed as Python modules"""
    print("\n" + "=" * 60)
    print("Testing Module Execution")
    print("=" * 60)
    
    import subprocess
    import os
    
    backend_dir = Path(__file__).parent
    os.chdir(backend_dir)
    
    modules = [
        "rag_eval.core.config",
        "rag_eval.core.logging",
        "rag_eval.utils.ids",
    ]
    
    results = []
    for module in modules:
        try:
            # Try to import it first
            __import__(module)
            print(f"✓ {module:30} - Can be imported")
            results.append((module, True, None))
        except Exception as e:
            print(f"✗ {module:30} - FAILED: {e}")
            results.append((module, False, str(e)))
    
    return results

def test_pytest_config():
    """Test that pytest configuration is correct"""
    print("\n" + "=" * 60)
    print("Testing pytest Configuration")
    print("=" * 60)
    
    try:
        import tomli
        with open("pyproject.toml", "rb") as f:
            config = tomli.load(f)
        
        pytest_config = config.get("tool", {}).get("pytest", {}).get("ini_options", {})
        
        if "pythonpath" in pytest_config:
            pythonpath = pytest_config["pythonpath"]
            if "." in pythonpath:
                print(f"✓ pytest pythonpath configured: {pythonpath}")
                return True
            else:
                print(f"✗ pytest pythonpath missing '.'")
                return False
        else:
            print(f"✗ pytest pythonpath not configured")
            return False
    except ImportError:
        # Try with tomllib (Python 3.11+)
        try:
            import tomllib
            with open("pyproject.toml", "rb") as f:
                config = tomllib.load(f)
            
            pytest_config = config.get("tool", {}).get("pytest", {}).get("ini_options", {})
            
            if "pythonpath" in pytest_config:
                pythonpath = pytest_config["pythonpath"]
                if "." in pythonpath:
                    print(f"✓ pytest pythonpath configured: {pythonpath}")
                    return True
                else:
                    print(f"✗ pytest pythonpath missing '.'")
                    return False
            else:
                print(f"✗ pytest pythonpath not configured")
                return False
        except Exception as e:
            print(f"✗ Could not validate pytest config: {e}")
            return False
    except Exception as e:
        print(f"✗ Could not validate pytest config: {e}")
        return False

def main():
    """Run all validation tests"""
    print("\n" + "=" * 60)
    print("Package Structure Validation")
    print("=" * 60)
    print()
    
    # Run all tests
    import_results = test_imports()
    all_results = test_all_lists()
    export_valid = test_exports_validity()
    module_results = test_module_execution()
    pytest_valid = test_pytest_config()
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    import_passed = sum(1 for _, passed, _ in import_results if passed)
    import_total = len(import_results)
    
    all_passed = sum(1 for _, passed, _ in all_results if passed)
    all_total = len(all_results)
    
    module_passed = sum(1 for _, passed, _ in module_results if passed)
    module_total = len(module_results)
    
    print(f"Imports:        {import_passed}/{import_total} passed")
    print(f"__all__ lists:  {all_passed}/{all_total} passed")
    print(f"Export validity: {'✓' if export_valid else '✗'}")
    print(f"Module execution: {module_passed}/{module_total} passed")
    print(f"pytest config:   {'✓' if pytest_valid else '✗'}")
    
    total_passed = (
        import_passed + 
        all_passed + 
        (1 if export_valid else 0) + 
        module_passed + 
        (1 if pytest_valid else 0)
    )
    total_tests = import_total + all_total + 1 + module_total + 1
    
    print(f"\nOverall: {total_passed}/{total_tests} checks passed")
    
    if total_passed == total_tests:
        print("\n✅ All package structure validations passed!")
        return 0
    else:
        print("\n❌ Some validations failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())



