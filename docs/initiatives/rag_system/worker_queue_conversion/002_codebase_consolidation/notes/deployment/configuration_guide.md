# Configuration Guide

This guide explains how configuration and environment variables are loaded and managed across local and cloud environments for Azure Functions.

## Overview

The configuration system uses a flexible, precedence-based approach that supports multiple configuration sources:

1. **Azure Function App Settings** (Cloud) - Highest precedence
2. **`.env.local`** (Local Development) - Optional, loaded if exists
3. **System Environment Variables** - Fallback
4. **Test Fixtures** - For unit tests

## Configuration Precedence

The configuration loading follows this precedence order (highest to lowest):

1. **Azure Function App Settings** (Cloud only)
   - Automatically available via `os.environ` in Azure Functions
   - Set via Azure Portal or Azure CLI
   - Highest precedence - always used in cloud

2. **`.env.local`** (Local Development)
   - Optional file in project root
   - Loaded by `Config.from_env()` if file exists
   - Used for local development with Azurite and local Supabase
   - Not required - functions can use system environment variables

3. **System Environment Variables**
   - Variables set in the shell environment
   - Used when `.env.local` doesn't exist or variable not in file

4. **Test Fixtures**
   - Environment variables set via pytest fixtures or `os.environ` in test setup
   - Used for unit tests (no file dependency)

## Configuration Files

### `local.settings.json`

**Location**: `backend/azure_functions/local.settings.json`

**Purpose**: Contains only Azurite connection strings and Azure Functions runtime settings for local development.

**Required Variables**:
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

**Important**: This file contains **ONLY** Azurite/runtime settings. All other application configuration (database, Azure AI services, etc.) should be in `.env.local` or system environment variables.

**When Used**: Only for local Azure Functions development with Azurite.

### `.env.local`

**Location**: Project root (same directory as `backend/`)

**Purpose**: Application configuration for local development (database, Azure AI services, etc.).

**Status**: Optional - functions can use system environment variables instead.

**Required Variables** (for local development):
```bash
# Database (Supabase Postgres)
SUPABASE_URL=http://localhost:54321
SUPABASE_KEY=<your-supabase-anon-key>
DATABASE_URL=postgresql://postgres:postgres@localhost:54322/postgres

# Azure AI Foundry
AZURE_AI_FOUNDRY_ENDPOINT=<your-endpoint>
AZURE_AI_FOUNDRY_API_KEY=<your-key>
AZURE_AI_FOUNDRY_EMBEDDING_MODEL=text-embedding-3-small
AZURE_AI_FOUNDRY_GENERATION_MODEL=gpt-4o

# Azure AI Search
AZURE_SEARCH_ENDPOINT=<your-endpoint>
AZURE_SEARCH_API_KEY=<your-key>
AZURE_SEARCH_INDEX_NAME=<your-index-name>

# Azure Document Intelligence
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=<your-endpoint>
AZURE_DOCUMENT_INTELLIGENCE_API_KEY=<your-key>

# Azure Blob Storage (optional)
AZURE_BLOB_CONNECTION_STRING=
AZURE_BLOB_CONTAINER_NAME=
```

**When Used**: Local development with Azurite and local Supabase. Not required - functions can use system environment variables.

**Loading**: Loaded by `Config.from_env()` if file exists. Not loaded if file doesn't exist (no error).

### Azure Function App Settings

**Location**: Azure Portal → Function App → Configuration → Application settings

**Purpose**: Production configuration (secrets, endpoints, etc.) for deployed Azure Functions.

**Status**: Automatically available via `os.environ` in Azure Functions runtime.

**Required Variables**: See `azure_function_app_settings.md` for complete list.

**When Used**: Always in cloud Azure Functions (highest precedence).

## How Configuration Loading Works

### Local Development

1. **Azure Functions Runtime**:
   - Reads `local.settings.json` for Azurite connection strings
   - Makes variables available via `os.environ`

2. **Function Entry Points**:
   - Functions import backend code directly
   - Backend code uses `Config.from_env()` to load configuration
   - `Config.from_env()` checks for `.env.local` in project root
   - If `.env.local` exists, loads it (using `override=True` to respect precedence)
   - If `.env.local` doesn't exist, uses system environment variables

3. **Workers**:
   - Workers call `Config.from_env()` to get configuration
   - `Config.from_env()` handles loading `.env.local` if it exists
   - Workers can also access environment variables directly via `os.getenv()`

### Cloud Deployment

1. **Azure Functions Runtime**:
   - Azure Function App settings are automatically available via `os.environ`
   - No file loading needed

2. **Function Entry Points**:
   - Functions import backend code directly
   - Backend code uses `Config.from_env()` to load configuration
   - `Config.from_env()` checks for `.env.local` (won't exist in cloud)
   - Falls back to `os.environ` (Azure Function App settings)

3. **Workers**:
   - Workers call `Config.from_env()` to get configuration
   - `Config.from_env()` reads from `os.environ` (Azure Function App settings)

### Unit Tests

1. **Test Setup**:
   - Tests set environment variables via pytest fixtures or `os.environ`
   - No file dependency

2. **Test Execution**:
   - `Config.from_env()` reads from `os.environ` (test fixtures)
   - No `.env.local` loading in tests

## Configuration Loading Implementation

### `Config.from_env()`

The `Config.from_env()` method in `backend/src/core/config.py` handles configuration loading:

```python
@classmethod
def from_env(cls, env_file: Optional[str] = None) -> "Config":
    """Load configuration from environment variables
    
    Args:
        env_file: Optional path to .env file. If None, looks for .env.local
                 in the project root (backend/.env.local or ../.env.local).
                 If specified, loads from that path.
    
    Returns:
        Config instance with loaded values
    """
    # Determine env file path
    if env_file:
        env_path = Path(env_file)
    else:
        # Default: look for .env.local in project root
        backend_dir = Path(__file__).parent.parent.parent
        project_root = backend_dir.parent
        env_path = backend_dir / ".env.local"
        if not env_path.exists():
            env_path = project_root / ".env.local"
    
    # Load environment variables from file if it exists
    if env_path.exists():
        load_dotenv(env_path, override=True)
    elif env_file:
        # If user specified a file that doesn't exist, warn but continue
        import warnings
        warnings.warn(f"Specified env file not found: {env_file}. Using system environment variables.")
    
    return cls(
        supabase_url=os.getenv("SUPABASE_URL", ""),
        # ... other variables
    )
```

**Key Points**:
- Checks for `.env.local` in project root (optional)
- Uses `override=True` to respect precedence (Azure settings > `.env.local`)
- Falls back to `os.environ` if file doesn't exist
- No error if file doesn't exist (flexible approach)

### Worker Configuration Access

Workers access configuration via `Config.from_env()`:

```python
# Get config from context
if context and hasattr(context, 'config'):
    config = context.config
elif context and isinstance(context, dict) and 'config' in context:
    config = context['config']
else:
    from src.core.config import Config
    try:
        config = Config.from_env()
    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        raise ValidationError("Config not available in context and failed to load from environment")
```

**Key Points**:
- Workers try to get config from context first (for testing)
- Fall back to `Config.from_env()` if context doesn't provide config
- `Config.from_env()` handles all environment variable loading

## Local vs Cloud Configuration

### Local Development

**Configuration Sources**:
1. `local.settings.json` - Azurite connection strings (required)
2. `.env.local` - Application configuration (optional)
3. System environment variables (fallback)

**Setup**:
1. Create `local.settings.json` with Azurite connection strings
2. Optionally create `.env.local` with application configuration
3. Start Azurite and local Supabase
4. Run functions locally with `func start`

**Example**:
```bash
# local.settings.json
{
  "Values": {
    "AzureWebJobsStorage": "UseDevelopmentStorage=true",
    "AZURE_STORAGE_QUEUES_CONNECTION_STRING": "UseDevelopmentStorage=true",
    "FUNCTIONS_WORKER_RUNTIME": "python"
  }
}

# .env.local (optional)
SUPABASE_URL=http://localhost:54321
DATABASE_URL=postgresql://postgres:postgres@localhost:54322/postgres
AZURE_AI_FOUNDRY_ENDPOINT=https://your-endpoint.openai.azure.com/
# ... other variables
```

### Cloud Deployment

**Configuration Sources**:
1. Azure Function App settings (required, highest precedence)
2. System environment variables (fallback)

**Setup**:
1. Set all required variables in Azure Function App settings
2. Deploy functions to Azure
3. Functions automatically use Azure Function App settings

**Example**:
```bash
# Set via Azure CLI
az functionapp config appsettings set \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --settings \
    DATABASE_URL="postgresql://..." \
    SUPABASE_URL="https://..." \
    AZURE_AI_FOUNDRY_ENDPOINT="https://..." \
    # ... other variables
```

## Troubleshooting

### Variables Not Loading

**Problem**: Environment variables not available in functions.

**Solutions**:
1. **Local**: Check `.env.local` exists in project root
2. **Local**: Verify `Config.from_env()` is being called
3. **Cloud**: Check Azure Function App settings are set
4. **Cloud**: Verify variable names match exactly (case-sensitive)
5. **Both**: Check variable names match `Config` class fields

### Precedence Issues

**Problem**: Wrong configuration value being used.

**Solutions**:
1. **Cloud**: Azure Function App settings always take precedence
2. **Local**: Check `.env.local` has correct values
3. **Local**: Verify `override=True` in `load_dotenv()` call
4. **Both**: Check system environment variables aren't overriding

### Missing Variables

**Problem**: Required variables not set.

**Solutions**:
1. Run validation script: `scripts/validate_config.py`
2. Check required variables list in `azure_function_app_settings.md`
3. Verify all variables are set in appropriate location (local vs cloud)

## Related Documentation

- [Azure Function App Settings](./azure_function_app_settings.md) - Complete list of required Azure settings
- [Local Development Guide](../../001_initial_conversion/LOCAL_DEVELOPMENT.md) - Local setup instructions
- [Environment Variables Reference](../../../../setup/environment_variables.md) - Complete variable reference

---

**Last Updated**: 2025-12-09  
**Related Initiative**: 002 (Codebase Consolidation)  
**Status**: Current

