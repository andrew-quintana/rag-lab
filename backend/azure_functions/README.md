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
- Validates prerequisites (backend source, function entry points)
- Verifies function structure
- Prepares the deployment package (no code copying needed - functions import directly from `src`)

### Manual Deployment (Alternative)

If you need to deploy manually:

```bash
./scripts/deploy_azure_functions.sh
```

## Build Script

The `build.sh` script is executed automatically by Azure Functions during Git deployment. It:

1. **Validates prerequisites**: Checks that backend source and function entry points exist
2. **Verifies structure**: Ensures all required files are present
3. **Prepares deployment**: No code copying needed - functions import directly from `src`

**Note**: Functions are now in `backend/azure_functions/` alongside `backend/src/`, so direct imports work naturally without path manipulation.

## Testing Build Script Locally

To test the build script locally:

```bash
cd backend/azure_functions
./build.sh
```

This will:
- Validate prerequisites
- Verify function structure
- Confirm deployment package is ready

## Function Entry Points

Each worker function:
- Has a `function.json` that defines the queue trigger
- Has an `__init__.py` with a `main()` function that:
  - Parses the queue message
  - Calls the worker function from `src.services.workers` (direct import)

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

