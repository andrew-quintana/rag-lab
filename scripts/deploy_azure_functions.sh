#!/bin/bash
# Script to deploy Azure Functions with backend code included

set -e

RESOURCE_GROUP="rag-lab"
FUNCTION_APP_NAME="func-raglab-uploadworkers"
FUNCTIONS_DIR="backend/azure_functions"
BACKEND_DIR="backend"

echo "=========================================="
echo "Deploying Azure Functions"
echo "=========================================="
echo "Function App: $FUNCTION_APP_NAME"
echo "Resource Group: $RESOURCE_GROUP"
echo ""

# Check if we're in the project root
if [ ! -d "$FUNCTIONS_DIR" ]; then
  echo "Error: $FUNCTIONS_DIR not found. Please run from project root."
  exit 1
fi

# Create a temporary deployment directory
TEMP_DIR=$(mktemp -d)
echo "Creating deployment package in: $TEMP_DIR"

# Copy Azure Functions files
echo "Copying Azure Functions files..."
cp -r "$FUNCTIONS_DIR"/* "$TEMP_DIR/"

# Copy backend code (src package)
echo "Copying backend code..."
mkdir -p "$TEMP_DIR/backend"
cp -r "$BACKEND_DIR/src" "$TEMP_DIR/backend/"

# Update function __init__.py files to use local backend
echo "Updating function entry points..."
for func_dir in "$TEMP_DIR"/*-worker; do
  if [ -d "$func_dir" ]; then
    func_name=$(basename "$func_dir")
    echo "  Updating $func_name..."
    
    # Update the path in __init__.py to use local backend
    sed -i.bak 's|backend_dir = Path(__file__).parent.parent.parent.parent.parent / "backend"|backend_dir = Path(__file__).parent.parent / "backend"|g' "$func_dir/__init__.py"
    rm -f "$func_dir/__init__.py.bak"
  fi
done

# Deploy using func command
echo ""
echo "Deploying to Azure..."
cd "$TEMP_DIR"
func azure functionapp publish "$FUNCTION_APP_NAME" --python

# Cleanup
echo ""
echo "Cleaning up temporary files..."
cd - > /dev/null
rm -rf "$TEMP_DIR"

echo ""
echo "=========================================="
echo "Deployment complete!"
echo "=========================================="
echo ""
echo "Verifying deployment..."
az functionapp function list \
  --name "$FUNCTION_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query "[].{Name:name, Status:status}" -o table

