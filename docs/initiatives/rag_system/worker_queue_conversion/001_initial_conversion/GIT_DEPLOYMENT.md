# Azure Functions Git-Based Deployment Guide

This guide explains how to use Git-based deployment for Azure Functions, which provides automatic deployments synced with your Git repository.

## Overview

Git-based deployment is the **recommended approach** for Azure Functions because it:
- ✅ Automatically deploys on git push
- ✅ Provides native Azure integration
- ✅ Shows deployment history in Azure Portal
- ✅ Simplifies deployment management
- ✅ No CI/CD pipeline maintenance needed

## Architecture

```
Git Repository (GitHub/Azure DevOps/etc.)
    ↓ (push to main branch)
Azure Functions Deployment Service
    ↓ (runs build.sh)
Deployment Package
    ├── Function entry points (*-worker/__init__.py)
    ├── Function bindings (*-worker/function.json)
    ├── Backend code (backend/rag_eval/)
    ├── Dependencies (requirements.txt)
    └── Configuration (host.json)
    ↓
Azure Function App
```

## Setup

### Step 1: Configure Git Deployment

Run the setup script:

```bash
./scripts/setup_git_deployment.sh
```

The script will:
1. Auto-detect your git remote URL
2. Configure Azure Functions to deploy from your repository
3. Set up the build script path
4. Configure automatic deployments

**Manual setup** (if script doesn't work):

```bash
# Configure Git deployment (build script is configured via .deployment file)
az functionapp deployment source config \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --repo-url https://github.com/your-org/rag_evaluator.git \
  --branch main \
  --manual-integration
```

**Note**: The build script path is configured via the `.deployment` file in your repository, not via CLI parameters. The `.deployment` file tells Azure to run `build.sh` during deployment.

### Step 2: Authorize Git Access (GitHub)

If using GitHub, you'll need to authorize access:

1. Go to Azure Portal → Function App → Deployment Center
2. Select "GitHub" as source
3. Authorize Azure to access your repository
4. Select repository and branch
5. Verify build script path: `infra/azure/azure_functions/build.sh`
6. Save configuration

### Step 3: Verify Configuration

Check deployment source:

```bash
az functionapp deployment source show \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab
```

## How It Works

### Build Script

The build script (`infra/azure/azure_functions/build.sh`) runs automatically during deployment:

1. **Copies backend code**: `backend/rag_eval/` → `backend/rag_eval/` in deployment package
2. **Updates import paths**: Modifies `__init__.py` files to use local backend
3. **Prepares package**: Creates complete deployment package with all dependencies

### Deployment Process

1. **Push to Git**: `git push origin main`
2. **Azure detects change**: Deployment service triggers
3. **Build script runs**: Prepares deployment package
4. **Dependencies installed**: `pip install -r requirements.txt`
5. **Functions deployed**: All 4 workers deployed
6. **Status updated**: Deployment history in Azure Portal

### Deployment Configuration

The `.deployment` file configures the build:

```ini
[config]
SCM_DO_BUILD_DURING_DEPLOYMENT=true
BUILD_COMMAND=./build.sh
POST_BUILD_COMMAND=
```

## Deployment Workflow

### Normal Deployment

```bash
# Make changes to code
git add .
git commit -m "Update worker logic"
git push origin main

# Azure automatically deploys!
```

### View Deployment Status

**Azure Portal**:
- Navigate to Function App → Deployment Center
- View deployment history
- See build logs
- Check deployment status

**Azure CLI**:
```bash
# List recent deployments
az functionapp deployment list \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab

# View deployment logs
az functionapp log deployment list \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab
```

## Troubleshooting

### Build Script Not Running

1. **Check build script path**:
   ```bash
   az functionapp deployment source show \
     --name func-raglab-uploadworkers \
     --resource-group rag-lab
   ```
   Verify `buildScriptPath` is set to `infra/azure/azure_functions/build.sh`

2. **Check build script permissions**:
   ```bash
   ls -la infra/azure/azure_functions/build.sh
   ```
   Should be executable (`-rwxr-xr-x`)

3. **Check deployment logs**:
   - Azure Portal → Function App → Deployment Center → Logs
   - Look for build script errors

### Backend Code Not Included

If functions fail with import errors:

1. **Check build script output** in deployment logs
2. **Verify backend path** in build script
3. **Check `.funcignore`** - ensure `backend/` is not ignored

### Deployment Not Triggering

1. **Check branch configuration**:
   ```bash
   az functionapp deployment source show \
     --name func-raglab-uploadworkers \
     --resource-group rag-lab
   ```
   Verify branch matches your push

2. **Check Git integration**:
   - Azure Portal → Deployment Center
   - Verify repository connection
   - Re-authorize if needed

3. **Manual trigger**:
   ```bash
   az functionapp deployment source sync \
     --name func-raglab-uploadworkers \
     --resource-group rag-lab
   ```

## Best Practices

### 1. Use Branch Protection

- Protect `main` branch
- Require pull requests
- Run tests before merge

### 2. Monitor Deployments

- Check deployment logs after each push
- Set up alerts for failed deployments
- Review Application Insights for errors

### 3. Environment-Specific Deployments

For multiple environments:

```bash
# Production: main branch
az functionapp deployment source config \
  --name func-raglab-uploadworkers-prod \
  --repo-url https://github.com/your-org/rag_evaluator.git \
  --branch main

# Staging: staging branch
az functionapp deployment source config \
  --name func-raglab-uploadworkers-staging \
  --repo-url https://github.com/your-org/rag_evaluator.git \
  --branch staging
```

### 4. Deployment Slots

Use deployment slots for zero-downtime deployments:

```bash
# Create staging slot
az functionapp deployment slot create \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --slot staging

# Deploy to staging slot
az functionapp deployment source config \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --slot staging \
  --repo-url https://github.com/your-org/rag_evaluator.git \
  --branch staging

# Swap slots after validation
az functionapp deployment slot swap \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --slot staging \
  --target-slot production
```

## Comparison: Git vs Manual Deployment

| Feature | Git-Based Deployment | Manual Deployment |
|---------|---------------------|-------------------|
| **Setup** | One-time configuration | Run script each time |
| **Automation** | Automatic on git push | Manual execution |
| **Integration** | Native Azure integration | External script |
| **History** | Azure Portal deployment history | Manual tracking |
| **Rollback** | Easy via Azure Portal | Manual redeployment |
| **Flexibility** | Limited to git workflow | Full control |

## Migration from Manual to Git Deployment

If you're currently using manual deployment:

1. **Set up Git deployment**:
   ```bash
   ./scripts/setup_git_deployment.sh
   ```

2. **Verify first deployment**:
   - Push a test commit
   - Verify deployment succeeds
   - Check functions are deployed

3. **Remove manual deployment script** (optional):
   - Keep `scripts/deploy_azure_functions.sh` as backup
   - Or remove if not needed

## Related Documentation

- `DEPLOYMENT_GUIDE.md` - Complete deployment guide
- `phase_5_azure_functions_deployment.md` - Detailed deployment steps
- `../../setup/environment_variables.md` - Environment variable reference

