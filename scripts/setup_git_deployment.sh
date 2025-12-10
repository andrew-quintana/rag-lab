#!/bin/bash
# Script to configure Azure Functions Git-based deployment
# This sets up automatic deployment from your Git repository

set -e

RESOURCE_GROUP="rag-lab"
FUNCTION_APP_NAME="func-raglab-uploadworkers"
REPO_URL="${1:-}"  # Optional: pass repo URL as first argument
BRANCH="${2:-main}"  # Optional: pass branch as second argument (default: main)

echo "=========================================="
echo "Setting up Azure Functions Git Deployment"
echo "=========================================="
echo "Function App: $FUNCTION_APP_NAME"
echo "Resource Group: $RESOURCE_GROUP"
echo ""

# Check if Azure CLI is authenticated
if ! az account show &>/dev/null; then
  echo "Error: Not logged in to Azure. Please run 'az login' first."
  exit 1
fi

# Get repository URL if not provided
if [ -z "$REPO_URL" ]; then
  # Try to get from git remote
  if git remote get-url origin &>/dev/null; then
    REPO_URL=$(git remote get-url origin)
    echo "Detected repository URL: $REPO_URL"
  else
    echo "Error: Repository URL not provided and could not detect from git remote."
    echo "Usage: $0 <repo-url> [branch]"
    echo "Example: $0 https://github.com/your-org/srcuator.git main"
    exit 1
  fi
fi

echo "Repository: $REPO_URL"
echo "Branch: $BRANCH"
echo ""

# Check if Function App exists
if ! az functionapp show --name "$FUNCTION_APP_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
  echo "Error: Function App '$FUNCTION_APP_NAME' not found in resource group '$RESOURCE_GROUP'"
  exit 1
fi

# Configure deployment source
echo "Configuring Git deployment source..."
echo "This will connect your Function App to the Git repository."
echo ""

# For GitHub, we need to use manual integration (user will authorize in Azure Portal)
if [[ "$REPO_URL" == *"github.com"* ]] || [[ "$REPO_URL" == *"github"* ]]; then
  echo "GitHub repository detected. Using manual integration."
  echo ""
  echo "Step 1: Configuring deployment source..."
  echo "Note: For GitHub, you'll need to complete authorization in Azure Portal."
  echo ""
  echo "The build script is configured via .deployment file in the repository."
  echo "Azure will automatically use backend/azure_functions/build.sh"
  echo ""
  
  # Try to configure, but GitHub usually requires Portal authorization
  az functionapp deployment source config \
    --name "$FUNCTION_APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --repo-url "$REPO_URL" \
    --branch "$BRANCH" \
    --manual-integration 2>&1 || {
    echo ""
    echo "⚠️  CLI configuration may require Portal authorization for GitHub."
    echo ""
    echo "Please complete setup in Azure Portal:"
    echo "  1. Go to Azure Portal → Function App → Deployment Center"
    echo "  2. Select 'GitHub' as source"
    echo "  3. Authorize and select repository: $REPO_URL"
    echo "  4. Set branch to: $BRANCH"
    echo "  5. The build script (backend/azure_functions/build.sh) is configured via .deployment file"
    echo ""
    echo "Alternatively, you can use the Portal UI to complete the setup."
    exit 0  # Don't fail - Portal setup is expected for GitHub
  }
  
  echo ""
  echo "=========================================="
  echo "Git deployment configured!"
  echo "=========================================="
  echo ""
  echo "⚠️  IMPORTANT: For GitHub, complete authorization in Azure Portal:"
  echo ""
  echo "1. Go to Azure Portal → Function App → Deployment Center"
  echo "2. Click 'Authorize' to connect GitHub"
  echo "3. Select repository: $REPO_URL"
  echo "4. Select branch: $BRANCH"
  echo "5. The build script is automatically configured via .deployment file"
  echo ""
  echo "After authorization, deployments will trigger automatically on push to $BRANCH"
  echo ""
  
elif [[ "$REPO_URL" == *"dev.azure.com"* ]] || [[ "$REPO_URL" == *"azure.com"* ]]; then
  echo "Azure DevOps repository detected."
  echo "Note: Build script is configured via .deployment file in repository."
  az functionapp deployment source config \
    --name "$FUNCTION_APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --repo-url "$REPO_URL" \
    --branch "$BRANCH"
  
  echo ""
  echo "=========================================="
  echo "Git deployment configured!"
  echo "=========================================="
  echo ""
  echo "Deployments will trigger automatically on push to $BRANCH branch"
  echo ""
  
else
  echo "Other Git repository detected. Attempting to configure..."
  echo "Note: Build script is configured via .deployment file in repository."
  az functionapp deployment source config \
    --name "$FUNCTION_APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --repo-url "$REPO_URL" \
    --branch "$BRANCH" \
    --manual-integration || {
    echo ""
    echo "Note: Manual integration may be required. Check Azure Portal → Deployment Center"
    exit 1
  }
  
  echo ""
  echo "=========================================="
  echo "Git deployment configured!"
  echo "=========================================="
  echo ""
  echo "Deployments will trigger automatically on push to $BRANCH branch"
  echo ""
fi

# Verify configuration
echo "Verifying deployment configuration..."
DEPLOYMENT_INFO=$(az functionapp deployment source show \
  --name "$FUNCTION_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" 2>/dev/null || echo "")

if [ -n "$DEPLOYMENT_INFO" ]; then
  echo ""
  echo "Current deployment source:"
  echo "$DEPLOYMENT_INFO" | jq -r '{repo: .repoUrl, branch: .branch}' 2>/dev/null || echo "$DEPLOYMENT_INFO"
  echo ""
  echo "Note: Build script is configured via .deployment file:"
  echo "  - File: backend/azure_functions/.deployment"
  echo "  - Build script: backend/azure_functions/build.sh"
else
  echo "Could not retrieve deployment info. Check Azure Portal → Deployment Center"
fi

echo ""
echo "To trigger a deployment, push changes to the $BRANCH branch:"
echo "  git push origin $BRANCH"
echo ""

