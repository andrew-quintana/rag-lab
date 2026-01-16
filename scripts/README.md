# Project Scripts Documentation

This directory contains operational scripts for managing, testing, and monitoring the RAG Evaluator system.

## Table of Contents
- [Monitoring & Diagnostics](#monitoring--diagnostics)
- [Testing & Validation](#testing--validation)
- [Azure Functions](#azure-functions)
- [Infrastructure Setup](#infrastructure-setup)
- [Database & Storage](#database--storage)

---

## Monitoring & Diagnostics

### `monitor_queue_health.py`
**Purpose**: Monitor Azure Storage Queue health and manage poison queues  
**Usage**:
```bash
python scripts/monitor_queue_health.py        # Check queue status
python scripts/monitor_queue_health.py --clear-poison  # Clear poison queues
```
**Key Features**:
- View message counts in all main queues
- Peek at messages in poison queues
- Clear poison queues for retry
- Monitor queue backlog

### `pipeline_health_check.py` ⭐ NEW
**Purpose**: Overall RAG pipeline health check  
**Usage**:
```bash
python scripts/pipeline_health_check.py
```
**Key Features**:
- Queue status across all stages
- Poison queue monitoring
- Stuck document detection (> 30 min in 'uploaded')
- Recent Application Insights errors
- Consolidated health dashboard

### `check_document_progress.py` ⭐ NEW
**Purpose**: Track a specific document through the pipeline  
**Usage**:
```bash
python scripts/check_document_progress.py <document_id>
```
**Key Features**:
- Database status check
- Queue presence detection
- Recent Application Insights logs
- Single-pane-of-glass for document debugging

### `diagnose_functions.py`
**Purpose**: Quick diagnostic testing of Azure Functions  
**Usage**:
```bash
python scripts/diagnose_functions.py
```
**Key Features**:
- Creates test document with valid UUID
- Tests ingestion worker
- Monitors queue and App Insights
- Quick smoke test

**Note**: Requires real PDF for ingestion worker (not dummy text file)

---

## Testing & Validation

### `generate_test_data.py` ⭐ NEW
**Purpose**: Generate synthetic test data for worker testing  
**Usage**:
```bash
# Generate data at specific stage
python scripts/generate_test_data.py uploaded    # For ingestion-worker
python scripts/generate_test_data.py parsed      # For chunking-worker
python scripts/generate_test_data.py chunked     # For embedding-worker
python scripts/generate_test_data.py embedded    # For indexing-worker

# With custom document ID
python scripts/generate_test_data.py chunked --document-id <uuid>
```
**Key Features**:
- Creates valid database records
- Uploads dummy files to Supabase storage
- Progresses documents through stages
- Generates queue-ready messages
- Enables component-level testing

### `validate_config.py`
**Purpose**: Validate environment configuration  
**Usage**:
```bash
python scripts/validate_config.py
```
**Key Features**:
- Check required environment variables
- Validate connection strings
- Verify Azure resource connectivity

---

## Azure Functions

### Deployment

#### `deploy_azure_functions.sh`
**Purpose**: Deploy Azure Functions to cloud  
**Usage**:
```bash
./scripts/deploy_azure_functions.sh
```

#### `configure_function_app_env.sh`
**Purpose**: Configure Function App environment variables  
**Usage**:
```bash
./scripts/configure_function_app_env.sh
```

### Local Development

#### `dev_functions_local.sh`
**Purpose**: Run Azure Functions locally  
**Usage**:
```bash
./scripts/dev_functions_local.sh
```

### Testing

#### `test_functions_local.sh`
**Purpose**: Test Functions locally with Azurite  
**Usage**:
```bash
./scripts/test_functions_local.sh
```

#### `test_functions_cloud.sh`
**Purpose**: Test deployed Functions in cloud  
**Usage**:
```bash
./scripts/test_functions_cloud.sh
```

#### `test_functions_all.sh`
**Purpose**: Run all Function tests  
**Usage**:
```bash
./scripts/test_functions_all.sh
```

### Monitoring

#### `check_function_status.sh`
**Purpose**: Check Azure Function App status  
**Usage**:
```bash
./scripts/check_function_status.sh
```

---

## Infrastructure Setup

### Ngrok Tunnels (for local Supabase)

#### `start_ngrok_tunnel.sh`
**Purpose**: Start ngrok HTTP tunnel for API  
**Usage**:
```bash
./scripts/start_ngrok_tunnel.sh
```

#### `start_ngrok_postgres_tunnel.sh`
**Purpose**: Start ngrok TCP tunnel for Postgres  
**Usage**:
```bash
./scripts/start_ngrok_postgres_tunnel.sh
```

#### `setup_ngrok_tcp_tunnel.sh`
**Purpose**: Setup ngrok TCP tunnel configuration  
**Usage**:
```bash
./scripts/setup_ngrok_tcp_tunnel.sh
```

#### `stop_ngrok_tunnel.sh`
**Purpose**: Stop all ngrok tunnels  
**Usage**:
```bash
./scripts/stop_ngrok_tunnel.sh
```

### Azurite (Local Azure Storage Emulator)

#### `start_azurite.sh`
**Purpose**: Start Azurite for local development  
**Usage**:
```bash
./scripts/start_azurite.sh
```

#### `stop_azurite.sh`
**Purpose**: Stop Azurite  
**Usage**:
```bash
./scripts/stop_azurite.sh
```

### Environment Setup

#### `setup_local_env.sh`
**Purpose**: Setup local development environment  
**Usage**:
```bash
./scripts/setup_local_env.sh
```

#### `setup_env_local.sh`
**Purpose**: Configure .env.local file  
**Usage**:
```bash
./scripts/setup_env_local.sh
```

#### `get_azure_env_values.sh`
**Purpose**: Retrieve Azure environment values  
**Usage**:
```bash
./scripts/get_azure_env_values.sh
```

---

## Database & Storage

### `update_azure_db_url.sh`
**Purpose**: Update Azure Function App with Supabase DB URL  
**Usage**:
```bash
./scripts/update_azure_db_url.sh
```

### `update_azure_db_url_from_ngrok.sh`
**Purpose**: Update Azure Function App with ngrok-tunneled DB URL  
**Usage**:
```bash
./scripts/update_azure_db_url_from_ngrok.sh
```

---

## Script Categories

### 🆕 New Scripts (Added 2026-01-13)

These scripts were created during the Azure Functions cloud testing session:

- `pipeline_health_check.py` - Overall pipeline monitoring
- `check_document_progress.py` - Document-specific debugging
- `generate_test_data.py` - Synthetic test data generation

See `docs/testing/azure_functions_cloud/README.md` for details.

### 🔍 Monitoring Scripts

Essential for production debugging and health checks:

- `monitor_queue_health.py`
- `pipeline_health_check.py` ⭐
- `check_document_progress.py` ⭐
- `check_function_status.sh`

### 🧪 Testing Scripts

For validation and testing:

- `generate_test_data.py` ⭐
- `diagnose_functions.py`
- `test_functions_*.sh`
- `validate_config.py`

### 🚀 Deployment Scripts

For deploying and configuring:

- `deploy_azure_functions.sh`
- `configure_function_app_env.sh`
- `setup_git_deployment.sh`

### 🔧 Infrastructure Scripts

For local development setup:

- `start_ngrok_*.sh`
- `start_azurite.sh`
- `setup_*.sh`
- `update_azure_db_url*.sh`

---

## Common Workflows

### Debugging a Failed Document

```bash
# 1. Check overall pipeline health
python scripts/pipeline_health_check.py

# 2. Track specific document
python scripts/check_document_progress.py <document_id>

# 3. Check queue status
python scripts/monitor_queue_health.py

# 4. Review Application Insights
# Use queries from docs/testing/azure_functions_cloud/AZURE_FUNCTIONS_TEST_RESULTS.md
```

### Testing a Specific Worker

```bash
# 1. Generate test data at the appropriate stage
python scripts/generate_test_data.py <stage>

# 2. Send message to queue (output from step 1)
az storage message put --queue-name <queue> --content '<message>'

# 3. Monitor progress
python scripts/check_document_progress.py <document_id>
```

### Deploying Functions

```bash
# 1. Configure environment
./scripts/configure_function_app_env.sh

# 2. Deploy functions
./scripts/deploy_azure_functions.sh

# 3. Verify deployment
./scripts/check_function_status.sh

# 4. Test deployed functions
./scripts/test_functions_cloud.sh
```

---

## Environment Requirements

Most scripts require:
- Python 3.11+
- Azure CLI (`az`) installed and authenticated
- Environment variables loaded (`.env.local`)
- Virtual environment activated (`source backend/venv/bin/activate`)

## Related Documentation

- **Testing Documentation**: `docs/testing/azure_functions_cloud/`
- **Backend Scripts**: `backend/scripts/` (separate from these operational scripts)
- **Setup Documentation**: `docs/setup/`
- **Monitoring Documentation**: `docs/monitoring/`

---

**Last Updated**: 2026-01-13  
**Maintained By**: RAG Evaluator Team

