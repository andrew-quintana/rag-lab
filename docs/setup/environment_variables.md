# Environment Variables Documentation

**Last Updated**: 2026-01-14  
**Purpose**: Document environment variable structure for local development and cloud deployment

---

## 📋 Environment Files Structure

```
/Users/aq_home/1Projects/rag_evaluator/
├── .env.local          # Local development (local Supabase)
├── .env.prod           # Production/Cloud (cloud Supabase)
└── .env.example        # Template (no secrets)
```

---

## 🔑 Supabase Keys Documentation

### Local Development (.env.local)

```bash
# Local Supabase Instance (via supabase start)
SUPABASE_URL=http://127.0.0.1:54321
SUPABASE_ANON_KEY=<local-anon-key>              # For client-side operations (respects RLS)
SUPABASE_SERVICE_ROLE_KEY=<local-service-role-key>  # For server-side operations (bypasses RLS)
DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:54322/postgres
```

**Local Keys** (from `supabase start` output):
- **ANON_KEY** (anon): For client-side operations, respects Row Level Security policies
- **SERVICE_ROLE_KEY**: For server-side operations, bypasses RLS - use in backend/Azure Functions

### Cloud/Production (.env.prod)

```bash
# Cloud Supabase Instance (oeyivkusvlgyuorcjime.supabase.co)
SUPABASE_URL=https://oeyivkusvlgyuorcjime.supabase.co
SUPABASE_ANON_KEY=<cloud-anon-key>              # For client-side operations (respects RLS)
SUPABASE_SERVICE_ROLE_KEY=<cloud-service-role-key>  # For server-side operations (bypasses RLS)
DATABASE_URL=postgresql://postgres.[ref]:[password]@aws-0-us-east-1.pooler.supabase.com:6543/postgres
```

**Cloud Keys** (from Supabase Dashboard → Settings → API):
- **Project URL**: `https://oeyivkusvlgyuorcjime.supabase.co`
- **anon public key** (`SUPABASE_ANON_KEY`): For client-side operations - respects RLS policies
- **service_role key** (`SUPABASE_SERVICE_ROLE_KEY`): For backend/Azure Functions - bypasses RLS, has full admin access

---

## 🏗️ Key Naming Convention

| Variable Name | Purpose | RLS Behavior | Usage |
|---------------|---------|--------------|-------|
| `SUPABASE_URL` | Base URL for Supabase project | N/A | Both |
| `SUPABASE_ANON_KEY` | Anonymous/public key | Respects RLS | Client-side only |
| `SUPABASE_SERVICE_ROLE_KEY` | Service role key (admin) | Bypasses RLS | Backend/Azure Functions |
| `DATABASE_URL` | Direct PostgreSQL connection string | N/A | Both |

**Important**: 
- **Backend/Azure Functions**: Always use `SUPABASE_SERVICE_ROLE_KEY` - it bypasses Row Level Security and has full database access
- **Client-side**: Use `SUPABASE_ANON_KEY` - it respects RLS policies for security
- **Never use anon key in backend** - it will fail for operations that require admin access

---

## 🔐 Security Best Practices

### ✅ DO:
- Use `SUPABASE_SERVICE_ROLE_KEY` in backend/Azure Functions (bypasses RLS)
- Use `SUPABASE_ANON_KEY` in client-side code (respects RLS)
- Store keys in `.env.local` and `.env.prod` (gitignored)
- Use Azure Key Vault or App Settings for cloud secrets
- Rotate keys periodically

### ❌ DON'T:
- Commit keys to git
- Use `SUPABASE_ANON_KEY` in backend (limited permissions, respects RLS)
- Share `SUPABASE_SERVICE_ROLE_KEY` publicly (full admin access)
- Hardcode keys in source code

---

## 🚀 Azure Functions Configuration

### Required Environment Variables for Azure Functions

```bash
# Supabase
SUPABASE_URL=https://oeyivkusvlgyuorcjime.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<cloud-service-role-key>  # Required for backend operations
DATABASE_URL=postgresql://postgres.[ref]:[password]@aws-0-us-east-1.pooler.supabase.com:6543/postgres

# Azure Storage
AZURE_STORAGE_ENDPOINT=https://raglabqueues.queue.core.windows.net/
AZURE_STORAGE_API_KEY=<storage-key>
AZURE_STORAGE_CONNECTION_STRING=<full-connection-string>
AZURE_STORAGE_QUEUES_CONNECTION_STRING=<queue-connection-string>
AZURE_BLOB_CONNECTION_STRING=<blob-connection-string>
AZURE_BLOB_CONTAINER_NAME=documents
AZURE_QUEUE_STORAGE_NAME=ingestion-uploads
AZURE_DOCUMENT_STORAGE_NAME=documents

# Azure AI Services
AZURE_AI_FOUNDRY_ENDPOINT=<endpoint>
AZURE_AI_FOUNDRY_API_KEY=<key>
AZURE_AI_FOUNDRY_EMBEDDING_MODEL=text-embedding-3-small
AZURE_AI_FOUNDRY_GENERATION_MODEL=gpt-4o-mini
AZURE_SEARCH_ENDPOINT=<endpoint>
AZURE_SEARCH_API_KEY=<key>
AZURE_SEARCH_INDEX_NAME=rag-lab-search
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=<endpoint>
AZURE_DOCUMENT_INTELLIGENCE_API_KEY=<key>

# Functions Runtime
AzureWebJobsStorage=<storage-connection-string>
APPLICATIONINSIGHTS_CONNECTION_STRING=<app-insights-connection>
FUNCTIONS_WORKER_RUNTIME=python
FUNCTIONS_EXTENSION_VERSION=~4
```

---

## 📝 Setting Up Cloud Supabase

### 1. Create Supabase Cloud Project
- Go to https://supabase.com/dashboard
- Create new project: `rag-evaluator-prod`
- Project ref: `oeyivkusvlgyuorcjime`
- Region: US East (closest to Azure East US)

### 2. Get Connection Details
**From Dashboard → Settings → API**:
- Project URL: `https://oeyivkusvlgyuorcjime.supabase.co`
- anon public key (`SUPABASE_ANON_KEY`): For client-side operations (respects RLS)
- service_role key (`SUPABASE_SERVICE_ROLE_KEY`): For backend/Azure Functions (bypasses RLS, full admin access)

**From Dashboard → Settings → Database**:
- Connection string (URI): `postgresql://postgres.[ref]:[password]@aws-0-us-east-1.pooler.supabase.com:6543/postgres`

### 3. Run Migrations
```bash
cd /Users/aq_home/1Projects/rag_evaluator

# Run migrations to cloud instance
supabase db push --db-url "postgresql://postgres.[ref]:[password]@aws-0-us-east-1.pooler.supabase.com:6543/postgres"

# Or apply migrations from infra/supabase/migrations/ manually
```

### 4. Update Azure Functions
```bash
# Set cloud Supabase credentials
az functionapp config appsettings set \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --settings \
    SUPABASE_URL="https://oeyivkusvlgyuorcjime.supabase.co" \
    SUPABASE_SERVICE_ROLE_KEY="<service-role-key>" \
    DATABASE_URL="postgresql://postgres.[ref]:[password]@aws-0-us-east-1.pooler.supabase.com:6543/postgres"
```

---

## 🧪 Testing Configuration

### Local Testing
```bash
# Use .env.local
export $(cat .env.local | xargs)
python -m pytest
```

### Cloud Testing
```bash
# Use .env.prod
export $(cat .env.prod | xargs)
# Test Azure Functions deployment
```

---

## 🔄 Migration Between Environments

### Local → Cloud
```bash
# Export local database
pg_dump postgresql://postgres:postgres@127.0.0.1:54322/postgres > local_backup.sql

# Import to cloud
psql "postgresql://postgres.[ref]:[password]@aws-0-us-east-1.pooler.supabase.com:6543/postgres" < local_backup.sql
```

### Cloud → Local
```bash
# Pull from cloud
supabase db pull --db-url "postgresql://postgres.[ref]:[password]@aws-0-us-east-1.pooler.supabase.com:6543/postgres"
```

---

## 📊 Environment Variable Checklist

### For Local Development
- [ ] .env.local exists with local Supabase keys
- [ ] Local Supabase running (`supabase start`)
- [ ] DATABASE_URL points to localhost:54322
- [ ] Migrations applied to local DB

### For Cloud Deployment
- [ ] .env.prod exists with cloud Supabase keys
- [ ] Cloud Supabase project created
- [ ] DATABASE_URL points to cloud instance
- [ ] Migrations applied to cloud DB
- [ ] Azure Functions configured with cloud keys

### For Both
- [ ] .env files in .gitignore
- [ ] Azure Storage keys configured
- [ ] Azure AI service keys configured
- [ ] Backend uses `SUPABASE_SERVICE_ROLE_KEY` (not anon key)
- [ ] Client-side uses `SUPABASE_ANON_KEY` (respects RLS)

---

## 🆘 Troubleshooting

### "No module named 'src'" Error
- **Problem**: Azure Functions deployment missing backend code
- **Solution**: See `E2E_TESTING_GAPS_CHECKLIST.md` GAP-001

### Database Connection Refused
- **Problem**: Wrong DATABASE_URL or firewall blocking
- **Solution**: Check Supabase Dashboard → Database → Connection pooler enabled

### "Invalid JWT" or "Invalid API Key"
- **Problem**: Using wrong key type or expired key
- **Solution**: Use service_role key, regenerate if needed

### Keys Not Loading
- **Problem**: Environment variables not set in Azure Functions
- **Solution**: Run `az functionapp config appsettings list` to verify

---

**Last Updated**: 2026-01-14  
**Maintained By**: Engineering Team  
**Related Docs**: 
- `E2E_TESTING_GAPS_CHECKLIST.md`
- `docs/setup/environment_variables.md`
