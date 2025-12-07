# Azure Functions Deployment

This directory contains Azure Functions for the RAG ingestion pipeline workers.

## Structure

```
azure_functions/
├── host.json                 # Function App configuration
├── requirements.txt          # Python dependencies
├── .deployment              # Deployment configuration (Git deployment)
├── build.sh                 # Build script (runs during Git deployment)
├── .funcignore             # Files to exclude from deployment
├── ingestion-worker/        # Ingestion worker function
│   ├── __init__.py         # Function entry point
│   └── function.json       # Queue trigger binding
├── chunking-worker/         # Chunking worker function
│   ├── __init__.py
│   └── function.json
├── embedding-worker/        # Embedding worker function
│   ├── __init__.py
│   └── function.json
└── indexing-worker/         # Indexing worker function
    ├── __init__.py
    └── function.json
```

## Deployment

### Git-Based Deployment (Recommended)

Deployments happen automatically when you push to Git:

1. **Set up Git deployment** (one-time):
   ```bash
   ./scripts/setup_git_deployment.sh
   ```

2. **Push to deploy**:
   ```bash
   git push origin main
   ```

The build script (`build.sh`) runs automatically during deployment and:
- Copies `backend/rag_eval/` to the deployment package
- Updates function import paths
- Prepares the complete deployment package

### Manual Deployment (Alternative)

If you need to deploy manually:

```bash
./scripts/deploy_azure_functions.sh
```

## Build Script

The `build.sh` script is executed automatically by Azure Functions during Git deployment. It:

1. **Locates backend code**: Finds `backend/rag_eval/` relative to project root
2. **Copies to deployment package**: `backend/rag_eval/` → `backend/rag_eval/`
3. **Updates import paths**: Modifies `__init__.py` files to use local backend

**Note**: The build script modifies `__init__.py` files during deployment. The original paths in git are preserved - the build script updates them in the deployment environment.

## Testing Build Script Locally

To test the build script locally:

```bash
cd infra/azure/azure_functions
./build.sh
```

This will:
- Copy backend code to `backend/` directory
- Update `__init__.py` files (restore them from git after testing)

**Warning**: Running the build script locally modifies the `__init__.py` files. Restore them from git after testing.

## Function Entry Points

Each worker function:
- Has a `function.json` that defines the queue trigger
- Has an `__init__.py` with a `main()` function that:
  - Parses the queue message
  - Adds backend to Python path
  - Calls the worker function from `rag_eval.services.workers`

## Queue Triggers

| Function | Queue Name |
|----------|------------|
| `ingestion-worker` | `ingestion-uploads` |
| `chunking-worker` | `ingestion-chunking` |
| `embedding-worker` | `ingestion-embeddings` |
| `indexing-worker` | `ingestion-indexing` |

## Related Documentation

- `../../docs/initiatives/rag_system/worker_queue_conversion/GIT_DEPLOYMENT.md` - Git deployment guide
- `../../docs/initiatives/rag_system/worker_queue_conversion/DEPLOYMENT_GUIDE.md` - Complete deployment guide
- `../../docs/initiatives/rag_system/worker_queue_conversion/phase_5_azure_functions_deployment.md` - Detailed deployment steps

