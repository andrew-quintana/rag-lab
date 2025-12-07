# Quick Start: Git-Based Deployment

## One-Time Setup

### Step 1: Configure Git Deployment

```bash
./scripts/setup_git_deployment.sh
```

This will:
- Auto-detect your git repository
- Configure Azure Functions for Git-based deployment
- Set up automatic deployments

### Step 2: Authorize Git Access (GitHub only)

If using GitHub, complete authorization in Azure Portal:

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to: Function App → `func-raglab-uploadworkers` → Deployment Center
3. Click "Authorize" if prompted
4. Select your repository and branch (default: `main`)
5. Verify build script path: `infra/azure/azure_functions/build.sh`
6. Click "Save"

### Step 3: Verify Configuration

```bash
az functionapp deployment source show \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab
```

## Daily Workflow

### Deploy Changes

Simply push to your main branch:

```bash
git add .
git commit -m "Update worker logic"
git push origin main
```

Azure Functions will automatically:
1. Detect the push
2. Pull the code
3. Run the build script
4. Deploy all functions

### Check Deployment Status

**Azure Portal**:
- Function App → Deployment Center → Logs
- View deployment history and build logs

**Azure CLI**:
```bash
az functionapp deployment list \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --query "[0].{Status:status, Message:message, Time:endTime}" -o json
```

## What Gets Deployed

When you push to Git, the build script automatically:

1. ✅ Copies `backend/rag_eval/` to deployment package
2. ✅ Updates function import paths
3. ✅ Installs dependencies from `requirements.txt`
4. ✅ Deploys all 4 worker functions

## Troubleshooting

### Deployment Not Triggering

```bash
# Manually trigger sync
az functionapp deployment source sync \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab
```

### Build Script Errors

Check deployment logs in Azure Portal:
- Function App → Deployment Center → Logs
- Look for build script output
- Verify backend code was copied

### Functions Not Deployed

```bash
# Check if functions exist
az functionapp function list \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab
```

## Next Steps

After Git deployment is set up:

1. **Configure environment variables** (if not done):
   ```bash
   ./scripts/configure_function_app_env.sh
   ```

2. **Test deployment**:
   - Make a small change
   - Push to main
   - Verify deployment succeeds

3. **Monitor deployments**:
   - Set up alerts for failed deployments
   - Review Application Insights logs

## Related Documentation

- `GIT_DEPLOYMENT.md` - Complete Git deployment guide
- `DEPLOYMENT_GUIDE.md` - Full deployment documentation
- `DEPLOYMENT_REFACTOR_SUMMARY.md` - What changed and why

