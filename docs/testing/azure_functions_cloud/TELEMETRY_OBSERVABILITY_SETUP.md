# Telemetry Observability Setup - Azure Functions

**Date**: 2026-01-15  
**Status**: ✅ **COMPLETE** - Application Insights KQL queries operational

---

## ✅ What Was Implemented

### 1. KQL Query Infrastructure

**Main Script**: `scripts/query_app_insights.sh`
- Automatically discovers Application Insights App ID
- Supports multiple query types
- Handles JSON parsing and table formatting
- Uses resource name (`ai-rag-evaluator`) for reliable queries

**Quick Access Scripts**:
- `scripts/logs_requests.sh` - HTTP requests
- `scripts/logs_errors.sh` - Exceptions and errors
- `scripts/logs_functions.sh` - Function execution logs
- `scripts/logs_queue.sh` - Queue-related logs

### 2. Makefile Integration

Added observability commands:
```bash
make logs-requests    # View recent HTTP requests
make logs-errors      # View recent errors
make logs-functions   # View function execution logs
make logs-queue       # View queue-related logs
make logs-ingestion   # View ingestion worker logs
```

### 3. Documentation

**Created**:
- `docs/monitoring/TELEMETRY_GUIDE.md` - Complete telemetry guide
- `docs/monitoring/CREATING_KQL_SCRIPTS.md` - Step-by-step guide for creating new scripts

**Updated**:
- `docs/monitoring/QUICKSTART.md` - Added telemetry commands
- `docs/monitoring/TROUBLESHOOTING.md` - Added Application Insights diagnosis section
- `docs/monitoring/IMPLEMENTATION_SUMMARY.md` - Added observability section
- `.cursorrules-azure-functions` - Added observability rules

---

## 🔍 Application Insights Configuration

### Resource Details

- **Resource Name**: `ai-rag-evaluator`
- **Resource Group**: `rag-lab`
- **App ID**: `***` (redacted)
- **Instrumentation Key**: `***` (redacted)

### Connection String

Configured in Azure Functions:
```
APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=***;IngestionEndpoint=https://westus2-2.in.applicationinsights.azure.com/;LiveEndpoint=https://westus2.livediagnostics.monitor.azure.com/;ApplicationId=***
```

### Authentication

- **Method**: Azure AD (via `az login`)
- **No API Key Required**: Azure CLI uses your authenticated identity
- **Permissions**: Requires Reader access to Application Insights resource

---

## 📊 Validated Queries

### Queue Trigger Verification ✅

**Query**: Operation ID `3576e5b499f5b0b67bddfc5c6a8d951b`

**Results**: 14 rows showing:
- ✅ Queue trigger fired: "New queue message detected on 'ingestion-uploads'"
- ✅ Function executed: "Executing 'Functions.ingestion-worker'"
- ✅ Message processed: "Processing ingestion message for document: 73075619-73f7-4bbb-9bdc-8fa4add80ffb"
- ❌ Database connection failed (separate issue, now fixed)

**Query Format Used**:
```kql
union traces
| union exceptions
| where timestamp > ago(30d)
| where operation_Id == '3576e5b499f5b0b67bddfc5c6a8d951b'
| order by timestamp asc
| project
    timestamp,
    message = iff(message != '', message, iff(innermostMessage != '', innermostMessage, customDimensions.['prop__{OriginalFormat}'])),
    logLevel = customDimensions.['LogLevel'],
    severityLevel
```

---

## 🛠️ Query Types Available

### 1. Requests
View HTTP requests to Azure Functions (if HTTP triggers exist)

### 2. Errors
View all exceptions and errors with severity level >= 3

### 3. Traces
View all trace logs (information, warnings, errors)

### 4. Dependencies
View external dependencies (queue operations, storage, database)

### 5. Functions
View function execution logs (filtered for function-related messages)

### 6. Queue
View queue-related logs (trigger activity, message processing)

### 7. Ingestion
View ingestion worker specific logs

### 8. Operation
Query by specific operation_Id (for detailed trace analysis)

---

## 📝 Creating New KQL Scripts

See [CREATING_KQL_SCRIPTS.md](../../monitoring/CREATING_KQL_SCRIPTS.md) for complete step-by-step guide.

**Quick Summary**:
1. Design your KQL query
2. Test it with `az monitor app-insights query`
3. Create quick access script
4. Add query type to `query_app_insights.sh`
5. Add Makefile command (optional)
6. Test and document

---

## 🎯 Observability Rules (Cursor Rules)

**Rule: Runtime behavior must be observable.**

1. **Telemetry is required** - All meaningful workflows must emit telemetry
2. **KQL is the validation layer** - Use KQL queries to inspect behavior
3. **Query before debugging** - Run relevant log queries first
4. **Use bounded time windows** - Always use `ago(...)` time windows
5. **Missing telemetry is a signal** - Note explicitly if telemetry is missing
6. **No claims without evidence** - Reasoning without telemetry is only allowed if telemetry is explicitly missing

**Before making changes or debugging:**
1. Run `make logs-functions` to see current state
2. Run `make logs-errors` to check for failures
3. Use telemetry to validate behavior, not assumptions

---

## 📊 Current Telemetry Status

### Data Available ✅
- Function execution traces
- Exception logs
- Queue trigger activity
- Database connection attempts
- Worker processing logs

### Query Performance ✅
- Queries execute successfully
- Results formatted as readable tables
- Supports time windows from 1h to 30d
- Can query by operation_Id for detailed traces

### Integration ✅
- Makefile commands working
- Scripts executable and tested
- Documentation complete
- Cursor rules updated

---

## 🔗 Related Documentation

- **Telemetry Guide**: [docs/monitoring/TELEMETRY_GUIDE.md](../../monitoring/TELEMETRY_GUIDE.md)
- **Creating Scripts**: [docs/monitoring/CREATING_KQL_SCRIPTS.md](../../monitoring/CREATING_KQL_SCRIPTS.md)
- **Quick Start**: [docs/monitoring/QUICKSTART.md](../../monitoring/QUICKSTART.md)
- **Troubleshooting**: [docs/monitoring/TROUBLESHOOTING.md](../../monitoring/TROUBLESHOOTING.md)

---

**Last Updated**: 2026-01-15  
**Validated**: ✅ Queries tested and working with real Application Insights data
