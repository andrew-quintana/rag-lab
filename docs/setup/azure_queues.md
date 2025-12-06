# Azure Queue Setup Guide

## Prerequisites

1. Azure Storage Account with Queue Storage enabled
2. Connection string for the Storage Account

## Step 1: Get Azure Storage Connection String

### Option A: Using Azure Portal
1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to your Storage Account
3. Go to **Security + networking** → **Access keys**
4. Click **Show** next to one of the connection strings
5. Copy the connection string

### Option B: Using Azure CLI
```bash
# Login to Azure (if not already logged in)
az login

# List your storage accounts
az storage account list --output table

# Get connection string (replace <STORAGE_ACCOUNT_NAME> and <RESOURCE_GROUP>)
az storage account show-connection-string \
  --name <STORAGE_ACCOUNT_NAME> \
  --resource-group <RESOURCE_GROUP> \
  --output tsv
```

## Step 2: Add Connection String to .env.local

Add the connection string to your `.env.local` file in the project root:

```bash
AZURE_STORAGE_QUEUES_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net
```

**Note**: You can also use `AZURE_BLOB_CONNECTION_STRING` if you prefer (the queue client supports both).

**Important**: 
- No quotes needed around the connection string
- The connection string works for both Blob Storage and Queue Storage

## Step 3: Create Queues

Run the setup script to create all required queues:

```bash
cd backend
source venv/bin/activate
python scripts/setup_azure_queues.py
```

This will create:
- `ingestion-uploads`
- `ingestion-chunking`
- `ingestion-embeddings`
- `ingestion-indexing`
- `ingestion-dead-letter`

**Note**: The script is idempotent - safe to run multiple times. If queues already exist, it will report them as already existing.

## Step 4: Test Queues

Run the integration test script:

```bash
cd backend
source venv/bin/activate
python scripts/test_azure_queues.py
```

This will test:
- Queue creation
- Message enqueue
- Message peek
- Message dequeue
- Message delete
- Queue length queries

## Troubleshooting

### Connection String Not Found
- Make sure `.env.local` is in the project root (not in `backend/`)
- Verify the connection string is on a single line with no quotes
- Check for typos in the variable name: `AZURE_STORAGE_QUEUES_CONNECTION_STRING` or `AZURE_BLOB_CONNECTION_STRING`
- The queue client checks both variable names, so either will work

### Queue Creation Fails
- Verify your Storage Account has Queue Storage enabled
- Check that your connection string has proper permissions
- Ensure network connectivity to Azure
- If queues already exist, that's fine - the script handles this gracefully

### Authentication Errors
- Verify the connection string is valid and not expired
- Check that the Storage Account key hasn't been rotated

## Manual Queue Creation (Alternative)

If the script fails, you can create queues manually in Azure Portal:

1. Go to Azure Portal → Your Storage Account
2. Navigate to **Data storage** → **Queues**
3. Click **+ Queue** and create each queue:
   - `ingestion-uploads`
   - `ingestion-chunking`
   - `ingestion-embeddings`
   - `ingestion-indexing`
   - `ingestion-dead-letter`

## Related Documentation

- Queue Client Implementation: `backend/rag_eval/services/workers/queue_client.py`
- Phase 2 Integration Testing: `docs/initiatives/rag_system/worker_queue_conversion/phase_2_azure_integration.md`

