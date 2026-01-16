# Azure Functions Telemetry Guide

**Complete guide to querying and analyzing Application Insights telemetry for Azure Functions**

**Last Updated**: 2026-01-15  
**Application Insights**: `ai-rag-evaluator` (App ID: `***` - redacted)

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Available Scripts](#available-scripts)
3. [KQL Query Format](#kql-query-format)
4. [Common Queries](#common-queries)
5. [Creating New KQL Scripts](#creating-new-kql-scripts)
6. [Telemetry Data Structure](#telemetry-data-structure)
7. [Troubleshooting Queries](#troubleshooting-queries)

---

## 🚀 Quick Start

### Prerequisites

- Azure CLI installed and authenticated (`az login`)
- Access to Application Insights resource `ai-rag-evaluator`
- Resource group: `rag-lab`

### Basic Usage

```bash
# View recent function execution logs
make logs-functions

# View recent errors
make logs-errors

# View queue-related logs
make logs-queue

# View ingestion worker logs
make logs-ingestion

# Custom time window (last 30 minutes)
TIME_WINDOW=30m make logs-functions
```

### Direct Script Usage

```bash
# View ingestion logs from last 2 days
./scripts/query_app_insights.sh ingestion 2d

# Query specific operation
./scripts/query_app_insights.sh operation 30d 3576e5b499f5b0b67bddfc5c6a8d951b

# View errors from last hour
./scripts/logs_errors.sh 1h
```

---

## 📜 Available Scripts

### Main Query Script

**`scripts/query_app_insights.sh`** - Main KQL query script

**Usage:**
```bash
./scripts/query_app_insights.sh <query_type> [time_window] [operation_id]
```

**Query Types:**
- `requests` - HTTP requests
- `errors` - Exceptions and errors
- `traces` - All trace logs
- `dependencies` - External dependencies (queues, storage, etc.)
- `functions` - Function execution logs
- `queue` - Queue-related logs
- `ingestion` - Ingestion worker logs
- `operation` - Query by operation_Id (requires operation_id as 3rd arg)

**Examples:**
```bash
# View function logs from last hour
./scripts/query_app_insights.sh functions 1h

# View errors from last 2 days
./scripts/query_app_insights.sh errors 2d

# Query specific operation
./scripts/query_app_insights.sh operation 30d 3576e5b499f5b0b67bddfc5c6a8d951b
```

### Quick Access Scripts

**`scripts/logs_requests.sh`** - Quick access to HTTP requests
```bash
./scripts/logs_requests.sh [time_window]
```

**`scripts/logs_errors.sh`** - Quick access to errors
```bash
./scripts/logs_errors.sh [time_window]
```

**`scripts/logs_functions.sh`** - Quick access to function logs
```bash
./scripts/logs_functions.sh [time_window]
```

**`scripts/logs_queue.sh`** - Quick access to queue logs
```bash
./scripts/logs_queue.sh [time_window]
```

### Makefile Commands

```bash
# View recent requests (default: 1h)
make logs-requests

# View recent errors (default: 1h)
make logs-errors

# View function execution logs (default: 1h)
make logs-functions

# View queue-related logs (default: 1h)
make logs-queue

# View ingestion worker logs (default: 1h)
make logs-ingestion

# Custom time window
TIME_WINDOW=30m make logs-functions
TIME_WINDOW=7d make logs-errors
```

---

## 🔍 KQL Query Format

### Standard Query Template

All queries use this format for consistent message extraction:

```kql
union traces
| union exceptions
| where timestamp > ago(<time_window>)
| where <filter_conditions>
| order by timestamp <asc|desc>
| project
    timestamp,
    message = iff(message != '', message, iff(innermostMessage != '', innermostMessage, customDimensions.['prop__{OriginalFormat}'])),
    logLevel = customDimensions.['LogLevel'],
    severityLevel
```

### Key Components

1. **Union traces and exceptions**: Combines both trace logs and exceptions
2. **Time filtering**: `where timestamp > ago(1h)` - filters by time window
3. **Message extraction**: Uses `iff()` to extract message from multiple possible fields
4. **Custom dimensions**: Extracts `LogLevel` from custom dimensions
5. **Projection**: Selects only relevant fields for readability

### Time Window Formats

- `1h` - Last 1 hour
- `30m` - Last 30 minutes
- `2d` - Last 2 days
- `7d` - Last 7 days
- `30d` - Last 30 days

---

## 📊 Common Queries

### 1. Function Execution Summary

```kql
union traces
| union exceptions
| where timestamp > ago(24h)
| where message contains 'Function' or message contains 'ingestion-worker' or message contains 'chunking-worker' or message contains 'embedding-worker' or message contains 'indexing-worker'
| summarize 
    total = count(),
    errors = countif(severityLevel >= 3),
    by bin(timestamp, 1h)
| order by timestamp desc
```

**Usage:**
```bash
./scripts/query_app_insights.sh functions 24h
```

### 2. Error Analysis

```kql
union traces
| union exceptions
| where timestamp > ago(24h)
| where severityLevel >= 3
| project timestamp, message, logLevel, severityLevel, operation_Id
| order by timestamp desc
```

**Usage:**
```bash
make logs-errors
```

### 3. Queue Processing Activity

```kql
union traces
| union exceptions
| where timestamp > ago(24h)
| where message contains 'queue' or message contains 'Queue' or message contains 'ingestion-uploads'
| project timestamp, message, logLevel, severityLevel
| order by timestamp desc
```

**Usage:**
```bash
make logs-queue
```

### 4. Specific Operation Trace

```kql
union traces
| union exceptions
| where timestamp > ago(30d)
| where operation_Id == '<operation_id>'
| order by timestamp asc
| project
    timestamp,
    message = iff(message != '', message, iff(innermostMessage != '', innermostMessage, customDimensions.['prop__{OriginalFormat}'])),
    logLevel = customDimensions.['LogLevel'],
    severityLevel
```

**Usage:**
```bash
./scripts/query_app_insights.sh operation 30d <operation_id>
```

### 5. Database Connection Errors

```kql
union traces
| union exceptions
| where timestamp > ago(24h)
| where message contains 'database' or message contains 'Database' or message contains 'psycopg2' or message contains 'connection'
| where severityLevel >= 3
| project timestamp, message, logLevel, severityLevel
| order by timestamp desc
```

### 6. Function Execution Duration

```kql
traces
| where timestamp > ago(24h)
| where message contains "Executed 'Functions"
| extend duration = extract(@"Duration=(\d+)ms", 1, message, typeof(long))
| summarize 
    avg_duration = avg(duration),
    max_duration = max(duration),
    p95_duration = percentile(duration, 95),
    by bin(timestamp, 1h)
| order by timestamp desc
```

### 7. Queue Trigger Activity

```kql
union traces
| union exceptions
| where timestamp > ago(24h)
| where message contains "New queue message detected" or message contains "Trigger Details"
| project timestamp, message, logLevel, severityLevel
| order by timestamp desc
```

---

## 🛠️ Creating New KQL Scripts

### Step 1: Create the Script File

Create a new script in `scripts/` directory:

```bash
touch scripts/logs_<your_query_name>.sh
chmod +x scripts/logs_<your_query_name>.sh
```

### Step 2: Write the Script

Use this template:

```bash
#!/bin/bash
# Quick script to view <description>
exec "$(dirname "$0")/query_app_insights.sh" <query_type> "${1:-1h}"
```

**Example:**
```bash
#!/bin/bash
# Quick script to view database connection logs
exec "$(dirname "$0")/query_app_insights.sh" database "${1:-1h}"
```

### Step 3: Add Query Type to query_app_insights.sh

Edit `scripts/query_app_insights.sh` and add your query type in the `case` statement:

```bash
  database)
    QUERY="union traces | union exceptions | where timestamp > ago($TIME_WINDOW) | where message contains 'database' or message contains 'Database' or message contains 'psycopg2' | order by timestamp desc | project timestamp, message = iff(message != '', message, iff(innermostMessage != '', innermostMessage, customDimensions.['prop__{OriginalFormat}'])), logLevel = customDimensions.['LogLevel'], severityLevel"
    ;;
```

### Step 4: Add Makefile Command (Optional)

Add to `Makefile`:

```makefile
logs-database:
	@./scripts/query_app_insights.sh database $(TIME_WINDOW)
```

And update the help text:

```makefile
	@echo "  make logs-database - View database connection logs (default: 1h)"
```

### Step 5: Test Your Script

```bash
# Test the script
./scripts/logs_<your_query_name>.sh 1h

# Test via Makefile
make logs-<your_query_name>
```

---

## 📐 KQL Query Best Practices

### 1. Always Use Union for Traces and Exceptions

```kql
union traces
| union exceptions
```

This ensures you capture both log traces and exceptions.

### 2. Use Consistent Message Extraction

```kql
message = iff(message != '', message, iff(innermostMessage != '', innermostMessage, customDimensions.['prop__{OriginalFormat}']))
```

This handles different message formats across trace types.

### 3. Filter by Time Window

Always use `ago()` for time filtering:
```kql
where timestamp > ago(1h)
```

### 4. Order Results

```kql
order by timestamp desc  # Most recent first
order by timestamp asc   # Chronological order
```

### 5. Limit Results for Readability

```kql
| take 50  # Limit to 50 rows
```

### 6. Use Projections

Only select fields you need:
```kql
project timestamp, message, logLevel, severityLevel
```

---

## 📊 Telemetry Data Structure

### Traces Table

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | datetime | When the event occurred |
| `message` | string | Log message content |
| `severityLevel` | int | 1=Information, 2=Warning, 3=Error |
| `operation_Id` | string | Unique operation identifier |
| `operation_Name` | string | Function name (e.g., "ingestion-worker") |
| `customDimensions` | dynamic | Additional metadata |

### Exceptions Table

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | datetime | When the exception occurred |
| `type` | string | Exception type |
| `message` | string | Exception message |
| `innermostMessage` | string | Inner exception message |
| `outerMessage` | string | Outer exception message |
| `operation_Id` | string | Unique operation identifier |

### Common Custom Dimensions

- `LogLevel` - Log level (Information, Warning, Error)
- `prop__{OriginalFormat}` - Original log format
- `FunctionName` - Azure Function name
- `InvocationId` - Function invocation ID

---

## 🔧 Troubleshooting Queries

### Issue: Query Returns No Results

**Possible Causes:**
1. Time window too short - try `7d` or `30d`
2. Filter too restrictive - remove some `where` conditions
3. No telemetry in that time period

**Solution:**
```bash
# Try a longer time window
./scripts/query_app_insights.sh functions 7d

# Check if any telemetry exists
az monitor app-insights query \
  --app ai-rag-evaluator \
  --resource-group rag-lab \
  --analytics-query "traces | where timestamp > ago(7d) | take 1" \
  -o json
```

### Issue: "Could not find Application Insights App ID"

**Solution:**
1. Verify App Insights resource exists:
   ```bash
   az monitor app-insights component list --resource-group rag-lab
   ```

2. Set APP_INSIGHTS_ID environment variable:
   ```bash
   export APP_INSIGHTS_ID=***  # Replace with your App Insights ID
   ```

3. Or update script to use resource name instead of App ID

### Issue: Query Times Out

**Solution:**
- Reduce time window (e.g., `1h` instead of `30d`)
- Add more specific filters to reduce data volume
- Use `take` to limit results:
  ```kql
  | take 100
  ```

### Issue: Missing Fields in Results

**Solution:**
- Check if field exists in the table:
  ```kql
  traces
  | where timestamp > ago(1h)
  | take 1
  | project *
  ```
- Use `extend` to create computed fields if needed

---

## 📝 Example: Creating a Custom Query Script

### Scenario: Monitor Database Connection Failures

**Step 1:** Create script:
```bash
cat > scripts/logs_database.sh << 'EOF'
#!/bin/bash
# Quick script to view database connection logs
exec "$(dirname "$0")/query_app_insights.sh" database "${1:-1h}"
EOF
chmod +x scripts/logs_database.sh
```

**Step 2:** Add query type to `query_app_insights.sh`:
```bash
  database)
    QUERY="union traces | union exceptions | where timestamp > ago($TIME_WINDOW) | where message contains 'database' or message contains 'Database' or message contains 'psycopg2' or message contains 'connection' or message contains 'Connection' | where severityLevel >= 3 | order by timestamp desc | project timestamp, message = iff(message != '', message, iff(innermostMessage != '', innermostMessage, customDimensions.['prop__{OriginalFormat}'])), logLevel = customDimensions.['LogLevel'], severityLevel"
    ;;
```

**Step 3:** Add to Makefile:
```makefile
logs-database:
	@./scripts/query_app_insights.sh database $(TIME_WINDOW)
```

**Step 4:** Test:
```bash
make logs-database
# or
./scripts/logs_database.sh 2d
```

---

## 🎯 Query Patterns Reference

### Pattern 1: Filter by Function Name

```kql
where operation_Name contains 'ingestion-worker'
```

### Pattern 2: Filter by Message Content

```kql
where message contains 'queue' or message contains 'Queue'
```

### Pattern 3: Filter by Severity

```kql
where severityLevel >= 3  # Errors only
where severityLevel == 1  # Information only
```

### Pattern 4: Filter by Operation ID

```kql
where operation_Id == '3576e5b499f5b0b67bddfc5c6a8d951b'
```

### Pattern 5: Time-Based Aggregation

```kql
| summarize count() by bin(timestamp, 1h)
| order by timestamp desc
```

### Pattern 6: Extract Values from Messages

```kql
| extend duration = extract(@"Duration=(\d+)ms", 1, message, typeof(long))
| extend document_id = extract(@"document: ([a-f0-9-]+)", 1, message)
```

---

## 🔗 Related Documentation

- **Creating New Scripts**: [CREATING_KQL_SCRIPTS.md](CREATING_KQL_SCRIPTS.md) - Step-by-step guide
- **Quick Start**: [QUICKSTART.md](QUICKSTART.md)
- **Troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Cursor Rules**: [.cursorrules-azure-functions](../../.cursorrules-azure-functions)
- **Environment Variables**: [ENVIRONMENT_VARIABLES.md](../setup/ENVIRONMENT_VARIABLES.md)

---

## 📚 Additional Resources

- **KQL Documentation**: https://learn.microsoft.com/en-us/azure/data-explorer/kusto/query/
- **Application Insights Query API**: https://learn.microsoft.com/en-us/rest/api/application-insights/query
- **Azure CLI App Insights**: `az monitor app-insights query --help`

---

**Last Updated**: 2026-01-15  
**Maintained By**: Engineering Team
