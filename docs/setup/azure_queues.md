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

## Step 3: Queues Auto-Create

Queues are automatically created when you first enqueue a message. The queue client includes auto-creation logic that will create the following queues on first use:

- `ingestion-uploads`
- `ingestion-chunking`
- `ingestion-embeddings`
- `ingestion-indexing`
- `ingestion-dead-letter`

**Note**: Queues are created idempotently - if they already exist, creation is skipped gracefully.

### Manual Queue Creation (Optional)

If you prefer to create queues manually before use, you can do so in Azure Portal:

1. Go to Azure Portal → Your Storage Account
2. Navigate to **Data storage** → **Queues**
3. Click **+ Queue** and create each queue:
   - `ingestion-uploads`
   - `ingestion-chunking`
   - `ingestion-embeddings`
   - `ingestion-indexing`
   - `ingestion-dead-letter`

## Step 4: Test Queues

Run the integration tests using pytest:

```bash
cd backend
source venv/bin/activate
pytest tests/components/workers/test_queue_client_integration.py -v -m integration
```

This will test:
- Queue length queries
- Message enqueue
- Message peek (non-destructive)
- Message dequeue with receipt
- Message delete
- Full queue workflow

**Note**: These are integration tests that require real Azure Storage. They are marked with `@pytest.mark.integration` and can be skipped if Azure is not available.

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
- Queues auto-create on first use, but if auto-creation fails, create them manually in Azure Portal

### Authentication Errors
- Verify the connection string is valid and not expired
- Check that the Storage Account key hasn't been rotated

## Related Documentation

- Queue Client Implementation: `backend/rag_eval/services/workers/queue_client.py`
- Phase 2 Integration Testing: `docs/initiatives/rag_system/worker_queue_conversion/phase_2_azure_integration.md`

