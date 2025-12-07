# Azure Functions Deployment Refactor Summary

## Overview

Refactored Azure Functions deployment from manual script-based approach to **Git-based deployment** (Option 2), which provides native Azure integration and automatic deployments synced with Git.

## Changes Made

### 1. Created Build Script

**File**: `infra/azure/azure_functions/build.sh`

- Automatically runs during Git deployment
- Copies `backend/rag_eval/` to deployment package
- Updates function import paths to use local backend
- Prepares complete deployment package

**Key Features**:
- Calculates paths relative to script location
- Works in Azure deployment environment
- Updates all worker `__init__.py` files automatically

### 2. Created Deployment Configuration

**File**: `infra/azure/azure_functions/.deployment`

```ini
[config]
SCM_DO_BUILD_DURING_DEPLOYMENT=true
BUILD_COMMAND=./build.sh
POST_BUILD_COMMAND=
```

Tells Azure Functions to:
- Run build during deployment
- Execute `build.sh` script
- Use the prepared deployment package

### 3. Created Setup Script

**File**: `scripts/setup_git_deployment.sh`

- Configures Git-based deployment
- Auto-detects git remote URL
- Sets up build script path
- Handles GitHub/Azure DevOps/other Git providers

**Usage**:
```bash
# Auto-detect repository
./scripts/setup_git_deployment.sh

# Manual specification
./scripts/setup_git_deployment.sh https://github.com/your-org/rag_evaluator.git main
```

### 4. Updated Documentation

**New Files**:
- `GIT_DEPLOYMENT.md` - Complete Git-based deployment guide
- `DEPLOYMENT_REFACTOR_SUMMARY.md` - This file

**Updated Files**:
- `DEPLOYMENT_GUIDE.md` - Now prioritizes Git-based deployment
- `phase_5_azure_functions_deployment.md` - Added Git deployment section
- `.funcignore` - Added comment about backend/ directory

## Migration Path

### For New Setups

1. **Set up Git deployment**:
   ```bash
   ./scripts/setup_git_deployment.sh
   ```

2. **Authorize Git access** (if GitHub):
   - Azure Portal → Function App → Deployment Center
   - Authorize and select repository

3. **Push to deploy**:
   ```bash
   git push origin main
   ```

### For Existing Manual Deployments

1. **Set up Git deployment** (same as above)
2. **Verify first deployment** works
3. **Keep manual script as backup** (optional)

## Benefits

### Before (Manual Deployment)
- ❌ Manual script execution required
- ❌ Not synced with Git
- ❌ No deployment history in Azure
- ❌ Risk of deploying uncommitted code

### After (Git-Based Deployment)
- ✅ Automatic deployment on git push
- ✅ Synced with Git repository
- ✅ Deployment history in Azure Portal
- ✅ Native Azure integration
- ✅ Easy rollback via Azure Portal
- ✅ Build script handles code preparation

## How It Works

### Deployment Flow

```
1. Developer pushes to Git
   ↓
2. Azure detects change
   ↓
3. Azure pulls code from Git
   ↓
4. Build script runs (build.sh)
   - Copies backend/rag_eval/
   - Updates import paths
   ↓
5. Dependencies installed (requirements.txt)
   ↓
6. Functions deployed
   ↓
7. Deployment complete
```

### Build Script Details

The build script:
1. **Locates backend code**: Finds `backend/rag_eval/` relative to project root
2. **Copies to deployment package**: `backend/rag_eval/` → `backend/rag_eval/`
3. **Updates import paths**: Changes `__init__.py` files from:
   ```python
   # Original (assumes backend 4 levels up)
   backend_dir = Path(__file__).parent.parent.parent.parent.parent / "backend"
   ```
   To:
   ```python
   # Updated (backend is 2 levels up in deployment)
   backend_dir = Path(__file__).parent.parent / "backend"
   ```

## File Structure

### New Files
```
infra/azure/azure_functions/
├── .deployment              # Deployment configuration
├── build.sh                 # Build script (runs during deployment)
└── ...

scripts/
└── setup_git_deployment.sh  # Git deployment setup script

docs/initiatives/rag_system/worker_queue_conversion/
├── GIT_DEPLOYMENT.md        # Git deployment guide
└── DEPLOYMENT_REFACTOR_SUMMARY.md  # This file
```

### Modified Files
```
infra/azure/azure_functions/
└── .funcignore              # Updated with backend/ comment

docs/initiatives/rag_system/worker_queue_conversion/
├── DEPLOYMENT_GUIDE.md      # Updated to prioritize Git deployment
└── phase_5_azure_functions_deployment.md  # Added Git deployment section
```

## Next Steps

1. **Set up Git deployment**:
   ```bash
   ./scripts/setup_git_deployment.sh
   ```

2. **Test deployment**:
   - Make a small change
   - Push to main branch
   - Verify deployment in Azure Portal

3. **Monitor first deployment**:
   - Check Azure Portal → Deployment Center
   - Review build logs
   - Verify functions are deployed

4. **Update team documentation**:
   - Share `GIT_DEPLOYMENT.md` with team
   - Update onboarding docs
   - Document deployment workflow

## Troubleshooting

See `GIT_DEPLOYMENT.md` for detailed troubleshooting guide.

Common issues:
- Build script not running → Check `.deployment` file
- Backend code missing → Check build script logs
- Deployment not triggering → Verify Git integration

## Rollback

If Git deployment doesn't work, you can still use manual deployment:

```bash
./scripts/deploy_azure_functions.sh
```

The manual deployment script remains available as a fallback.

