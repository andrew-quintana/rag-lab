#!/usr/bin/env python3
"""Configuration validation script

Validates configuration files and environment variables for local and cloud environments.

Usage:
    python scripts/validate_config.py [--local] [--cloud] [--all]
    
Options:
    --local    Validate local configuration (.env.local, local.settings.json)
    --cloud    Validate Azure Function App settings (requires Azure CLI)
    --all      Validate both local and cloud (default)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
FUNCTIONS_DIR = BACKEND_DIR / "azure_functions"
LOCAL_SETTINGS_PATH = FUNCTIONS_DIR / "local.settings.json"
ENV_LOCAL_PATH = PROJECT_ROOT / ".env.local"

# Required environment variables (from Config class)
REQUIRED_VARS = [
    "SUPABASE_URL",
    "SUPABASE_KEY",
    "DATABASE_URL",
    "AZURE_AI_FOUNDRY_ENDPOINT",
    "AZURE_AI_FOUNDRY_API_KEY",
    "AZURE_SEARCH_ENDPOINT",
    "AZURE_SEARCH_API_KEY",
    "AZURE_SEARCH_INDEX_NAME",
    "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
    "AZURE_DOCUMENT_INTELLIGENCE_API_KEY",
]

# Optional environment variables (with defaults)
OPTIONAL_VARS = [
    "AZURE_AI_FOUNDRY_EMBEDDING_MODEL",  # default: "text-embedding-3-small"
    "AZURE_AI_FOUNDRY_GENERATION_MODEL",  # default: "gpt-4o"
    "AZURE_BLOB_CONNECTION_STRING",
    "AZURE_BLOB_CONTAINER_NAME",
]

# Allowed variables in local.settings.json
ALLOWED_LOCAL_SETTINGS = {
    "AzureWebJobsStorage",
    "AZURE_STORAGE_QUEUES_CONNECTION_STRING",
    "FUNCTIONS_WORKER_RUNTIME",
    "IsEncrypted",  # metadata, not a value
}


def validate_local_settings() -> Tuple[bool, List[str]]:
    """Validate local.settings.json has only allowed variables"""
    errors = []
    
    if not LOCAL_SETTINGS_PATH.exists():
        errors.append(f"local.settings.json not found at {LOCAL_SETTINGS_PATH}")
        return False, errors
    
    try:
        with open(LOCAL_SETTINGS_PATH, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON in local.settings.json: {e}")
        return False, errors
    except Exception as e:
        errors.append(f"Error reading local.settings.json: {e}")
        return False, errors
    
    # Check structure
    if "Values" not in data:
        errors.append("local.settings.json missing 'Values' key")
        return False, errors
    
    values = data["Values"]
    
    # Check for disallowed variables
    for key in values.keys():
        if key not in ALLOWED_LOCAL_SETTINGS:
            errors.append(
                f"Disallowed variable in local.settings.json: {key}. "
                f"Only {', '.join(sorted(ALLOWED_LOCAL_SETTINGS - {'IsEncrypted'}))} are allowed."
            )
    
    # Check required variables are present
    required_in_local = {"AzureWebJobsStorage", "AZURE_STORAGE_QUEUES_CONNECTION_STRING", "FUNCTIONS_WORKER_RUNTIME"}
    missing = required_in_local - set(values.keys())
    if missing:
        errors.append(f"Missing required variables in local.settings.json: {', '.join(missing)}")
    
    # Check values are not empty
    for key in required_in_local:
        if key in values and not values[key]:
            errors.append(f"Empty value for required variable in local.settings.json: {key}")
    
    return len(errors) == 0, errors


def validate_env_local() -> Tuple[bool, List[str], List[str]]:
    """Validate .env.local has required variables"""
    errors = []
    warnings = []
    
    if not ENV_LOCAL_PATH.exists():
        warnings.append(f".env.local not found at {ENV_LOCAL_PATH} (optional for local development)")
        return True, errors, warnings
    
    # Load .env.local
    from dotenv import load_dotenv
    load_dotenv(ENV_LOCAL_PATH, override=False)
    
    # Check required variables
    missing = []
    for var in REQUIRED_VARS:
        value = os.getenv(var)
        if not value:
            missing.append(var)
    
    if missing:
        errors.append(f"Missing required variables in .env.local: {', '.join(missing)}")
    
    # Check for empty values
    for var in REQUIRED_VARS:
        value = os.getenv(var)
        if value and value.strip() == "":
            errors.append(f"Empty value for required variable in .env.local: {var}")
    
    return len(errors) == 0, errors, warnings


def validate_azure_settings() -> Tuple[bool, List[str]]:
    """Validate Azure Function App settings (requires Azure CLI)"""
    errors = []
    
    # Check Azure CLI is available
    import subprocess
    try:
        result = subprocess.run(
            ["az", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            errors.append("Azure CLI not available or not working")
            return False, errors
    except FileNotFoundError:
        errors.append("Azure CLI not found. Install Azure CLI to validate cloud settings.")
        return False, errors
    except Exception as e:
        errors.append(f"Error checking Azure CLI: {e}")
        return False, errors
    
    # Get Function App name and resource group from environment or config
    function_app_name = os.getenv("AZURE_FUNCTION_APP_NAME", "func-raglab-uploadworkers")
    resource_group = os.getenv("AZURE_RESOURCE_GROUP", "rag-lab")
    
    # Get Function App settings
    try:
        result = subprocess.run(
            [
                "az", "functionapp", "config", "appsettings", "list",
                "--name", function_app_name,
                "--resource-group", resource_group,
                "--output", "json"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            errors.append(
                f"Failed to get Azure Function App settings: {result.stderr}. "
                f"Check that Function App '{function_app_name}' exists in resource group '{resource_group}'."
            )
            return False, errors
        
        settings = json.loads(result.stdout)
        settings_dict = {s["name"]: s.get("value", "") for s in settings}
        
        # Check required variables
        missing = []
        for var in REQUIRED_VARS:
            if var not in settings_dict or not settings_dict[var]:
                missing.append(var)
        
        if missing:
            errors.append(f"Missing required variables in Azure Function App settings: {', '.join(missing)}")
        
        # Check Azure Functions runtime variables
        if "AzureWebJobsStorage" not in settings_dict or not settings_dict["AzureWebJobsStorage"]:
            errors.append("Missing AzureWebJobsStorage in Azure Function App settings")
        
        if "AZURE_STORAGE_QUEUES_CONNECTION_STRING" not in settings_dict or not settings_dict["AZURE_STORAGE_QUEUES_CONNECTION_STRING"]:
            errors.append("Missing AZURE_STORAGE_QUEUES_CONNECTION_STRING in Azure Function App settings")
        
    except json.JSONDecodeError:
        errors.append("Failed to parse Azure Function App settings JSON")
    except subprocess.TimeoutExpired:
        errors.append("Timeout getting Azure Function App settings")
    except Exception as e:
        errors.append(f"Error validating Azure Function App settings: {e}")
    
    return len(errors) == 0, errors


def print_results(valid: bool, errors: List[str], warnings: List[str] = None, context: str = ""):
    """Print validation results"""
    if warnings is None:
        warnings = []
    
    print(f"\n{'='*60}")
    if context:
        print(f"Validation: {context}")
    else:
        print("Validation Results")
    print(f"{'='*60}")
    
    if valid:
        print("✅ Validation PASSED")
    else:
        print("❌ Validation FAILED")
    
    if warnings:
        print(f"\n⚠️  Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"   - {warning}")
    
    if errors:
        print(f"\n❌ Errors ({len(errors)}):")
        for error in errors:
            print(f"   - {error}")
    
    print()


def main():
    parser = argparse.ArgumentParser(description="Validate configuration files and environment variables")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Validate local configuration only"
    )
    parser.add_argument(
        "--cloud",
        action="store_true",
        help="Validate Azure Function App settings only"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=True,
        help="Validate both local and cloud (default)"
    )
    
    args = parser.parse_args()
    
    # Determine what to validate
    validate_local = args.local or (args.all and not args.cloud)
    validate_cloud = args.cloud or (args.all and not args.local)
    
    all_valid = True
    
    if validate_local:
        print("Validating local configuration...")
        
        # Validate local.settings.json
        valid, errors = validate_local_settings()
        print_results(valid, errors, context="local.settings.json")
        if not valid:
            all_valid = False
        
        # Validate .env.local
        valid, errors, warnings = validate_env_local()
        print_results(valid, errors, warnings, context=".env.local")
        if not valid:
            all_valid = False
    
    if validate_cloud:
        print("Validating Azure Function App settings...")
        valid, errors = validate_azure_settings()
        print_results(valid, errors, context="Azure Function App Settings")
        if not valid:
            all_valid = False
    
    # Final summary
    print(f"\n{'='*60}")
    if all_valid:
        print("✅ All validations PASSED")
        sys.exit(0)
    else:
        print("❌ Some validations FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()

