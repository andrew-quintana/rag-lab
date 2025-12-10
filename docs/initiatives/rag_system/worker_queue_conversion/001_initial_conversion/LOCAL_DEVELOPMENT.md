# Local Azure Functions Development Guide

This guide explains how to set up and run Azure Functions locally using Azurite (local Azure Storage emulator) and local Supabase for Phase 5 integration testing.

## Overview

Local development uses:
- **Azurite**: Local Azure Storage emulator for queue operations
- **Local Supabase**: Local PostgreSQL database and storage
- **Azure Functions Core Tools**: Run functions locally with `func start`

## Prerequisites

1. **Node.js and npm** (for Azurite)
   ```bash
   npm install -g azurite
   ```

2. **Azure Functions Core Tools**
   ```bash
   brew tap azure/functions
   brew install azure-functions-core-tools@4
   ```

3. **Supabase CLI**
   ```bash
   brew install supabase/tap/supabase
   ```

4. **Python 3.9+** with virtual environment

5. **Python dependencies** (in backend/venv):
   ```bash
   pip install python-dotenv azure-storage-queue
   ```

## Initial Setup

### Step 1: Setup Environment Variables

Run the setup script to create/update `.env.local` with local Supabase credentials:

```bash
./scripts/setup_local_env.sh
```

This script will:
- Extract local Supabase credentials from `supabase status`
- Prompt for Azure AI service credentials (Foundry, Search, Document Intelligence)
- Create/update `.env.local` in the project root
- Preserve existing values if variables are already set

**Important**: The script requires Supabase to be running. Start it first:

```bash
cd infra/supabase
supabase start
```

### Step 2: Verify local.settings.json

The `local.settings.json` file should already exist in `backend/azure_functions/` with:

```json
{
  "IsEncrypted": false,
  "Values": {
    "AzureWebJobsStorage": "UseDevelopmentStorage=true",
    "AZURE_STORAGE_QUEUES_CONNECTION_STRING": "UseDevelopmentStorage=true",
    "FUNCTIONS_WORKER_RUNTIME": "python"
  }
}
```

**Critical**: This file contains ONLY Azurite connection strings and runtime settings. All other environment variables are loaded from `.env.local` (if it exists) via `Config.from_env()` in the backend code.

### Step 3: Start Local Services

Use the convenience script to start everything:

```bash
./scripts/dev_functions_local.sh
```

This script will:
1. Start Azurite (if not running)
2. Create required queues in Azurite
3. Start Azure Functions locally

Or start services manually:

```bash
# Terminal 1: Start Supabase
cd infra/supabase
supabase start

# Terminal 2: Start Azurite
./scripts/start_azurite.sh

# Terminal 3: Start Functions
cd backend/azure_functions
func start
```

## Environment Variable Loading

### How It Works

1. **Azure Functions Runtime**:
   - Reads `local.settings.json` for Azurite connection strings
   - Makes variables available via `os.environ`

2. **Backend Code**:
   - Uses `Config.from_env()` to load configuration
   - `Config.from_env()` checks for `.env.local` in project root (optional)
   - If `.env.local` exists, loads it (using `override=True` to respect precedence)
   - If `.env.local` doesn't exist, uses system environment variables
   - Variables are available via `os.getenv()` after loading

3. **Configuration Precedence**:
   - Azure Function App settings (cloud) > `.env.local` (local) > system environment > test fixtures
   - `local.settings.json` contains ONLY Azurite/runtime settings
   - Application configuration goes in `.env.local` or Azure Function App settings

**Note**: Function entry points use simple direct imports. Configuration loading is handled by `Config.from_env()` when workers need it.

### Required Variables in `.env.local`

```bash
# Local Supabase (auto-filled by setup script)
SUPABASE_URL=http://localhost:54321
SUPABASE_KEY=<from supabase status>
DATABASE_URL=postgresql://postgres:postgres@localhost:54322/postgres

# Azure AI Services (prompted by setup script)
AZURE_AI_FOUNDRY_ENDPOINT=<your-endpoint>
AZURE_AI_FOUNDRY_API_KEY=<your-key>
AZURE_AI_FOUNDRY_EMBEDDING_MODEL=text-embedding-3-small
AZURE_AI_FOUNDRY_GENERATION_MODEL=gpt-4o

AZURE_SEARCH_ENDPOINT=<your-endpoint>
AZURE_SEARCH_API_KEY=<your-key>
AZURE_SEARCH_INDEX_NAME=<your-index>

AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=<your-endpoint>
AZURE_DOCUMENT_INTELLIGENCE_API_KEY=<your-key>

# Optional
AZURE_BLOB_CONNECTION_STRING=
AZURE_BLOB_CONTAINER_NAME=
```

## Running Functions Locally

### Using the Dev Script (Recommended)

```bash
./scripts/dev_functions_local.sh
```

### Manual Start

1. **Start Azurite**:
   ```bash
   ./scripts/start_azurite.sh
   ```

2. **Create Queues** (if not auto-created):
   Queues are created automatically on first use, or you can create them manually using the Python script in `dev_functions_local.sh`.

3. **Start Functions**:
   ```bash
   cd backend/azure_functions
   func start
   ```

### Verify Functions Are Running

You should see output like:

```
Functions:
        ingestion-worker: queueTrigger
        chunking-worker: queueTrigger
        embedding-worker: queueTrigger
        indexing-worker: queueTrigger
```

## Running Tests Locally

### Phase 5 Integration Tests

Run the test script:

```bash
./scripts/test_functions_local.sh
```

Or run tests manually:

```bash
cd backend
source venv/bin/activate
pytest tests/integration/test_phase5_e2e_pipeline.py -v -m local
```

### Test Markers

- `@pytest.mark.local`: Tests that run in local development
- `@pytest.mark.cloud`: Tests that require cloud Azure Functions
- `@pytest.mark.integration`: All integration tests

## Debugging

### VS Code Debug Configuration

A debug configuration is provided in `infra/azure/azure_functions/.vscode/launch.json`:

1. Set breakpoints in function code
2. Start functions with `func start`
3. Attach debugger using "Attach to Python Functions" configuration

### Function Logs

View function logs in the terminal where `func start` is running, or check Application Insights (if configured for local development).

## Queue Operations

### Azurite Connection

Functions use `UseDevelopmentStorage=true` which connects to Azurite on:
- Blob service: `http://127.0.0.1:10000`
- Queue service: `http://127.0.0.1:10001`
- Table service: `http://127.0.0.1:10002`

### Required Queues

The following queues are created automatically:
- `ingestion-uploads`
- `ingestion-chunking`
- `ingestion-embeddings`
- `ingestion-indexing`
- `ingestion-dead-letter`

### Testing Queue Operations

You can test queue operations using the Python SDK:

```python
from azure.storage.queue import QueueServiceClient

client = QueueServiceClient.from_connection_string("UseDevelopmentStorage=true")
queue = client.get_queue_client("ingestion-uploads")
queue.send_message('{"document_id": "test-123", ...}')
```

## Troubleshooting

### Functions Not Starting

1. **Check local.settings.json exists**:
   ```bash
   ls backend/azure_functions/local.settings.json
   ```

2. **Check .env.local exists**:
   ```bash
   ls .env.local
   ```

3. **Check Azurite is running**:
   ```bash
   lsof -i :10000
   ```

4. **Check function logs** for import errors

### Import Errors

If you see import errors:
1. Verify backend code is accessible (functions are in `backend/azure_functions/` and import from `src/`)
2. Check Python path includes backend directory
3. Verify all dependencies are installed in function environment
4. Functions should use simple direct imports: `from src.services.workers...`

### Queue Connection Errors

1. **Verify Azurite is running**:
   ```bash
   ./scripts/stop_azurite.sh
   ./scripts/start_azurite.sh
   ```

2. **Check connection string**:
   - Should be `UseDevelopmentStorage=true` in `local.settings.json`
   - Should NOT be in `.env.local` (functions use local.settings.json for queues)

3. **Verify queues exist**:
   Check Azurite logs or create queues manually

### Database Connection Errors

1. **Verify Supabase is running**:
   ```bash
   cd infra/supabase
   supabase status
   ```

2. **Check DATABASE_URL in .env.local**:
   Should be: `postgresql://postgres:postgres@localhost:54322/postgres`

3. **Verify .env.local is loaded**:
   Check function logs for dotenv loading messages

### Environment Variables Not Loading

1. **Check .env.local path**:
   `.env.local` should be in project root (same directory as `backend/`)

2. **Verify dotenv is installed**:
   ```bash
   pip list | grep python-dotenv
   ```

3. **Check configuration loading**:
   - `Config.from_env()` loads `.env.local` if it exists
   - Workers call `Config.from_env()` when they need configuration
   - Run validation script: `python scripts/validate_config.py --local`

4. **Verify configuration precedence**:
   - Azure Function App settings (cloud) > `.env.local` (local) > system environment
   - `local.settings.json` contains only Azurite/runtime settings

## Differences from Cloud

### Connection Strings

- **Local**: `UseDevelopmentStorage=true` (Azurite)
- **Cloud**: Full Azure Storage connection string

### Queue Creation

- **Local**: Queues created automatically or manually
- **Cloud**: Queues must exist or be created via Azure CLI/Portal

### Environment Variables

- **Local**: Loaded from `.env.local` via dotenv
- **Cloud**: Set in Azure Function App configuration

### Database

- **Local**: Local Supabase on `localhost:54322`
- **Cloud**: Remote Supabase instance

## Next Steps

After local testing passes:

1. Document test results in `phase_5_testing.md`
2. Update `fracas.md` with any issues found
3. Deploy to cloud and run cloud tests
4. Compare local vs cloud test results

## Related Documentation

- [Configuration Guide](../../002_codebase_consolidation/notes/deployment/configuration_guide.md) - Complete configuration strategy and precedence
- [Azure Function App Settings](../../002_codebase_consolidation/notes/deployment/azure_function_app_settings.md) - Required Azure settings
- [Environment Variables Reference](../../../../setup/environment_variables.md) - Complete variable reference
- [Phase 5 Testing Guide](phase_5_testing.md)
- [Azure Functions Deployment](phase_5_azure_functions_deployment.md)

