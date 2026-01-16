#!/bin/bash
# Build script for Azure Functions deployment
# This script prepares the deployment package by copying backend code

set -e

echo "=========================================="
echo "Azure Functions Build Script"
echo "=========================================="
echo "Preparing deployment package..."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Calculate project root
# Script is in: backend/azure_functions/build.sh
# Project root is 2 levels up
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKEND_SOURCE="$PROJECT_ROOT/backend/src"

echo "Project root: $PROJECT_ROOT"
echo "Backend source: $BACKEND_SOURCE"
echo ""

# Validate prerequisites
echo "Validating prerequisites..."

# Check if backend source exists
if [ ! -d "$BACKEND_SOURCE" ]; then
  echo "❌ Error: Backend source directory not found: $BACKEND_SOURCE"
  exit 1
fi
echo "✓ Backend source directory found"

# Check if requirements.txt exists
if [ ! -f "$SCRIPT_DIR/requirements.txt" ]; then
  echo "❌ Error: requirements.txt not found in $SCRIPT_DIR"
  exit 1
fi
echo "✓ requirements.txt found"

# Validate function entry points exist
FUNCTIONS=("ingestion-worker" "chunking-worker" "embedding-worker" "indexing-worker")
for func_name in "${FUNCTIONS[@]}"; do
  func_dir="$SCRIPT_DIR/$func_name"
  if [ ! -d "$func_dir" ]; then
    echo "❌ Error: Function directory not found: $func_dir"
    exit 1
  fi
  if [ ! -f "$func_dir/__init__.py" ]; then
    echo "❌ Error: Function entry point not found: $func_dir/__init__.py"
    exit 1
  fi
  if [ ! -f "$func_dir/function.json" ]; then
    echo "❌ Error: Function binding not found: $func_dir/function.json"
    exit 1
  fi
  echo "✓ $func_name entry point validated"
done

echo ""
echo "=========================================="
echo "Copying Backend Code to Deployment Package"
echo "=========================================="
echo ""

# Remove existing backend directory if it exists
if [ -d "$SCRIPT_DIR/backend" ]; then
  echo "Removing old backend directory..."
  rm -rf "$SCRIPT_DIR/backend"
fi

# Create backend directory structure
echo "Creating backend directory..."
mkdir -p "$SCRIPT_DIR/backend"

# Copy source code
echo "Copying backend/src to deployment package..."
cp -r "$BACKEND_SOURCE" "$SCRIPT_DIR/backend/"

echo "✓ Backend code copied successfully"
echo ""

# Verify the copy
if [ -d "$SCRIPT_DIR/backend/src" ]; then
  FILE_COUNT=$(find "$SCRIPT_DIR/backend/src" -type f -name "*.py" | wc -l | tr -d ' ')
  echo "Verification:"
  echo "  - backend/src directory: ✓ Present"
  echo "  - Python files copied: $FILE_COUNT"
else
  echo "❌ Error: backend/src directory not found after copy"
  exit 1
fi

echo ""
echo "=========================================="
echo "Merging Requirements"
echo "=========================================="
echo ""

# Backup original requirements
if [ ! -f "$SCRIPT_DIR/requirements.txt.original" ]; then
  cp "$SCRIPT_DIR/requirements.txt" "$SCRIPT_DIR/requirements.txt.original"
  echo "✓ Backed up original requirements.txt"
fi

# Merge backend requirements if they exist
if [ -f "$PROJECT_ROOT/backend/requirements.txt" ]; then
  echo "Merging backend requirements..."
  
  # Restore original and merge
  cp "$SCRIPT_DIR/requirements.txt.original" "$SCRIPT_DIR/requirements.txt"
  cat "$PROJECT_ROOT/backend/requirements.txt" >> "$SCRIPT_DIR/requirements.txt"
  
  # Remove duplicates by package name (keep last occurrence, which is usually the more restrictive version)
  # This handles cases where same package has different version requirements
  awk -F'[=<>]' '{
    pkg = $1
    if (pkg in seen) {
      # Package seen before, replace with current line (usually more restrictive)
      seen[pkg] = $0
    } else {
      seen[pkg] = $0
    }
  } END {
    for (pkg in seen) {
      print seen[pkg]
    }
  }' "$SCRIPT_DIR/requirements.txt" | sort > "$SCRIPT_DIR/requirements.txt.tmp"
  mv "$SCRIPT_DIR/requirements.txt.tmp" "$SCRIPT_DIR/requirements.txt"
  
  REQUIREMENT_COUNT=$(wc -l < "$SCRIPT_DIR/requirements.txt" | tr -d ' ')
  echo "✓ Requirements merged ($REQUIREMENT_COUNT total packages)"
else
  echo "⚠ Backend requirements.txt not found, using azure_functions requirements only"
fi

echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo ""
echo "Deployment package structure:"
echo "  ✓ Function entry points: *-worker/__init__.py"
echo "  ✓ Function bindings: *-worker/function.json"
echo "  ✓ Backend code: backend/src/"
echo "  ✓ Dependencies: requirements.txt"
echo "  ✓ Configuration: host.json"
echo ""
echo "Package is ready for deployment!"
echo ""
echo "To deploy, run:"
echo "  func azure functionapp publish func-raglab-uploadworkers --python"
echo ""
