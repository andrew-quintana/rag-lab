# Local Azure Functions Development - Quick Start

Quick reference for running Azure Functions locally with Azurite and local Supabase.

## Prerequisites

- Azurite: `npm install -g azurite`
- Azure Functions Core Tools: `brew install azure-functions-core-tools@4`
- Supabase CLI: `brew install supabase/tap/supabase`

## Quick Start

1. **Setup environment**:
   ```bash
   ./scripts/setup_local_env.sh
   ```

2. **Start services**:
   ```bash
   ./scripts/dev_functions_local.sh
   ```

3. **Run tests**:
   ```bash
   ./scripts/test_functions_local.sh
   ```

## Essential Commands

### Start/Stop Azurite

```bash
./scripts/start_azurite.sh   # Start Azurite
./scripts/stop_azurite.sh    # Stop Azurite
```

### Start Functions Manually

```bash
cd infra/azure/azure_functions
func start
```

### Run Tests

```bash
cd backend
source venv/bin/activate
pytest tests/integration/test_phase5_e2e_pipeline.py -v -m local
```

## File Locations

- **Environment variables**: `.env.local` (project root)
- **Function settings**: `infra/azure/azure_functions/local.settings.json`
- **Function code**: `infra/azure/azure_functions/*-worker/__init__.py`

## Common Issues

### Functions won't start
- Check `local.settings.json` exists
- Check `.env.local` exists
- Check Azurite is running: `lsof -i :10000`

### Import errors
- Verify `backend/` directory exists in `infra/azure/azure_functions/`
- Check Python path includes backend directory

### Queue errors
- Restart Azurite: `./scripts/stop_azurite.sh && ./scripts/start_azurite.sh`
- Verify connection string is `UseDevelopmentStorage=true`

## Full Documentation

See [LOCAL_DEVELOPMENT.md](../../../../docs/initiatives/rag_system/worker_queue_conversion/LOCAL_DEVELOPMENT.md) for complete guide.

