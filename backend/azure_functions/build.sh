#!/bin/bash
# Build script for Azure Functions Git deployment
# This script runs automatically when Azure Functions pulls from Git
# It validates prerequisites and prepares the deployment package
# Note: Functions now import directly from project root, so no code copying is needed

set -e

echo "=========================================="
echo "Azure Functions Build Script"
echo "=========================================="
echo "Validating prerequisites and preparing deployment package..."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Calculate project root
# Script is in: backend/azure_functions/build.sh
# Project root is 2 levels up: backend/azure_functions -> backend -> project_root
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKEND_SOURCE="$PROJECT_ROOT/backend/src"

echo "Project root: $PROJECT_ROOT"
echo "Backend source: $BACKEND_SOURCE"
echo ""

# Validate prerequisites
echo "Validating prerequisites..."

# Check if backend source exists
if [ ! -d "$BACKEND_SOURCE" ]; then
  echo "Error: Backend source directory not found: $BACKEND_SOURCE"
  exit 1
fi
echo "✓ Backend source directory found"

# Check if requirements.txt exists
if [ ! -f "$SCRIPT_DIR/requirements.txt" ]; then
  echo "Error: requirements.txt not found in $SCRIPT_DIR"
  exit 1
fi
echo "✓ requirements.txt found"

# Validate function entry points exist
FUNCTIONS=("ingestion-worker" "chunking-worker" "embedding-worker" "indexing-worker")
for func_name in "${FUNCTIONS[@]}"; do
  func_dir="$SCRIPT_DIR/$func_name"
  if [ ! -d "$func_dir" ]; then
    echo "Error: Function directory not found: $func_dir"
    exit 1
  fi
  if [ ! -f "$func_dir/__init__.py" ]; then
    echo "Error: Function entry point not found: $func_dir/__init__.py"
    exit 1
  fi
  if [ ! -f "$func_dir/function.json" ]; then
    echo "Error: Function binding not found: $func_dir/function.json"
    exit 1
  fi
  echo "✓ $func_name entry point validated"
done

echo ""
echo "=========================================="
echo "Build validation complete!"
echo "=========================================="
echo ""
echo "Deployment package structure:"
echo "  - Function entry points: *-worker/__init__.py (import from project root)"
echo "  - Function bindings: *-worker/function.json"
echo "  - Dependencies: requirements.txt"
echo "  - Configuration: host.json"
echo ""
echo "Note: Functions import directly from project root backend/src/"
echo "      No code copying is required."
echo ""

