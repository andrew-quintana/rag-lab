#!/bin/bash
# Build script for Azure Functions Git deployment
# This script runs automatically when Azure Functions pulls from Git
# It prepares the deployment package by copying backend code and updating paths

set -e

echo "=========================================="
echo "Azure Functions Build Script"
echo "=========================================="
echo "Preparing deployment package..."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Calculate paths relative to script location
# Script is in: infra/azure/azure_functions/build.sh
# Project root is 3 levels up: infra/azure/azure_functions -> infra/azure -> infra -> project_root
# Backend is in: backend/rag_eval (relative to project root)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BACKEND_SOURCE="$PROJECT_ROOT/backend/rag_eval"
BACKEND_TARGET="$SCRIPT_DIR/backend"

echo "Project root: $PROJECT_ROOT"
echo "Backend source: $BACKEND_SOURCE"
echo "Backend target: $BACKEND_TARGET"
echo ""

# Copy backend code (rag_eval package)
if [ ! -d "$BACKEND_SOURCE" ]; then
  echo "Error: Backend source directory not found: $BACKEND_SOURCE"
  exit 1
fi

echo "Copying backend code..."
mkdir -p "$BACKEND_TARGET"
cp -r "$BACKEND_SOURCE" "$BACKEND_TARGET/"
echo "✓ Backend code copied"
echo ""

# Update function __init__.py files to use local backend
echo "Updating function entry points..."
for func_dir in "$SCRIPT_DIR"/*-worker; do
  if [ -d "$func_dir" ] && [ -f "$func_dir/__init__.py" ]; then
    func_name=$(basename "$func_dir")
    echo "  Updating $func_name..."
    
    # Update the path in __init__.py to use local backend
    # Original: backend_dir = Path(__file__).parent.parent.parent.parent.parent / "backend"
    # Updated: backend_dir = Path(__file__).parent.parent / "backend"
    sed -i.bak 's|backend_dir = Path(__file__).parent.parent.parent.parent.parent / "backend"|backend_dir = Path(__file__).parent.parent / "backend"|g' "$func_dir/__init__.py"
    rm -f "$func_dir/__init__.py.bak"
    echo "    ✓ Updated $func_name/__init__.py"
  fi
done

echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="
echo ""
echo "Deployment package structure:"
echo "  - Function entry points: *-worker/__init__.py"
echo "  - Function bindings: *-worker/function.json"
echo "  - Backend code: backend/rag_eval/"
echo "  - Dependencies: requirements.txt"
echo "  - Configuration: host.json"
echo ""

