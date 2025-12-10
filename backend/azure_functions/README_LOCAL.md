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
cd backend/azure_functions
func start
```

### Run Tests

```bash
cd backend
source venv/bin/activate
pytest tests/integration/test_phase5_e2e_pipeline.py -v -m local
```

## File Locations

- **Environment variables**: `.env.local` (project root, optional)
- **Function settings**: `backend/azure_functions/local.settings.json` (Azurite/runtime only)
- **Function code**: `backend/azure_functions/*-worker/__init__.py`
- **Configuration guide**: `docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/notes/deployment/configuration_guide.md`

## Common Issues

### Functions won't start
- Check `local.settings.json` exists (contains only Azurite/runtime settings)
- Check `.env.local` exists (optional - functions can use system environment variables)
- Check Azurite is running: `lsof -i :10000`
- Validate configuration: `python scripts/validate_config.py --local`

### Import errors
- Verify functions are in `backend/azure_functions/` (alongside `backend/src/`)
- Functions should import directly from `src.*` (no path manipulation needed)

### Queue errors
- Restart Azurite: `./scripts/stop_azurite.sh && ./scripts/start_azurite.sh`
- Verify connection string is `UseDevelopmentStorage=true`

## Configuration

Configuration loading follows a clear precedence order:
1. Azure Function App settings (cloud) - highest precedence
2. `.env.local` (local) - optional, loaded if exists
3. System environment variables - fallback

See [Configuration Guide](../../../docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/notes/deployment/configuration_guide.md) for complete details.

## Full Documentation

See [LOCAL_DEVELOPMENT.md](../../../../docs/initiatives/rag_system/worker_queue_conversion/001_initial_conversion/LOCAL_DEVELOPMENT.md) for complete guide.

