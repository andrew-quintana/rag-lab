# Creating New KQL Scripts - Step-by-Step Guide

**Complete guide for creating custom KQL query scripts for Application Insights telemetry**

---

## 📋 Overview

This guide walks you through creating a new KQL query script from scratch. You'll learn:
- How to structure a KQL query
- How to create the script file
- How to integrate it with the existing system
- How to test and validate it

---

## 🎯 Prerequisites

- Azure CLI installed and authenticated
- Access to Application Insights resource `ai-rag-evaluator`
- Basic understanding of KQL syntax
- Familiarity with bash scripting

---

## 📝 Step-by-Step Process

### Step 1: Design Your Query

First, design your KQL query. Consider:
- **What data do you want?** (traces, exceptions, dependencies, etc.)
- **What time window?** (1h, 24h, 7d, 30d)
- **What filters?** (function name, message content, severity level, etc.)
- **What fields?** (timestamp, message, logLevel, severityLevel, etc.)

**Example:** Query for database connection errors

```kql
union traces
| union exceptions
| where timestamp > ago(24h)
| where message contains 'database' or message contains 'Database' or message contains 'psycopg2' or message contains 'connection'
| where severityLevel >= 3
| order by timestamp desc
| project
    timestamp,
    message = iff(message != '', message, iff(innermostMessage != '', innermostMessage, customDimensions.['prop__{OriginalFormat}'])),
    logLevel = customDimensions.['LogLevel'],
    severityLevel
```

### Step 2: Test Your Query

Test the query directly using Azure CLI:

```bash
az monitor app-insights query \
  --app ai-rag-evaluator \
  --resource-group rag-lab \
  --analytics-query "union traces | union exceptions | where timestamp > ago(24h) | where message contains 'database' | take 5" \
  -o json
```

Verify you get results before creating the script.

### Step 3: Create the Quick Access Script

Create a simple wrapper script:

```bash
# Create the script file
cat > scripts/logs_database.sh << 'EOF'
#!/bin/bash
# Quick script to view database connection logs
exec "$(dirname "$0")/query_app_insights.sh" database "${1:-1h}"
EOF

# Make it executable
chmod +x scripts/logs_database.sh
```

**Template:**
```bash
#!/bin/bash
# Quick script to view <description>
exec "$(dirname "$0")/query_app_insights.sh" <query_type> "${1:-1h}"
```

### Step 4: Add Query Type to query_app_insights.sh

Edit `scripts/query_app_insights.sh` and add your query type:

**Find the `case` statement** (around line 50-100) and add:

```bash
  database)
    QUERY="union traces | union exceptions | where timestamp > ago($TIME_WINDOW) | where message contains 'database' or message contains 'Database' or message contains 'psycopg2' or message contains 'connection' or message contains 'Connection' | where severityLevel >= 3 | order by timestamp desc | project timestamp, message = iff(message != '', message, iff(innermostMessage != '', innermostMessage, customDimensions.['prop__{OriginalFormat}'])), logLevel = customDimensions.['LogLevel'], severityLevel"
    ;;
```

**Important:**
- Use `$TIME_WINDOW` variable for time filtering
- Use the standard message extraction pattern
- Escape special characters properly in bash

### Step 5: Add Makefile Command (Optional but Recommended)

Edit `Makefile` and add:

**In the `.PHONY` line:**
```makefile
.PHONY: ... logs-database
```

**In the help section:**
```makefile
	@echo "  make logs-database - View database connection logs (default: 1h)"
```

**Add the command:**
```makefile
logs-database:
	@./scripts/query_app_insights.sh database $(TIME_WINDOW)
```

### Step 6: Test Your Script

```bash
# Test the quick access script
./scripts/logs_database.sh 1h

# Test with custom time window
./scripts/logs_database.sh 2d

# Test via Makefile
make logs-database

# Test with custom time window via Makefile
TIME_WINDOW=30m make logs-database
```

### Step 7: Document Your Script

Add documentation to `docs/monitoring/TELEMETRY_GUIDE.md`:

1. Add to "Available Scripts" section
2. Add example query to "Common Queries" section
3. Update "Query Patterns Reference" if it introduces a new pattern

---

## 📐 Query Template Reference

### Standard Trace/Exception Query

```kql
union traces
| union exceptions
| where timestamp > ago($TIME_WINDOW)
| where <your_filters>
| order by timestamp desc
| project
    timestamp,
    message = iff(message != '', message, iff(innermostMessage != '', innermostMessage, customDimensions.['prop__{OriginalFormat}'])),
    logLevel = customDimensions.['LogLevel'],
    severityLevel
```

### Filter Patterns

**By Function Name:**
```kql
where operation_Name contains 'ingestion-worker'
```

**By Message Content:**
```kql
where message contains 'queue' or message contains 'Queue'
```

**By Severity:**
```kql
where severityLevel >= 3  # Errors only
where severityLevel == 1  # Information only
```

**By Operation ID:**
```kql
where operation_Id == '<operation_id>'
```

**Combined Filters:**
```kql
where (message contains 'error' or message contains 'Error')
  and severityLevel >= 3
  and operation_Name contains 'ingestion-worker'
```

### Aggregation Queries

**Count by Time:**
```kql
| summarize count() by bin(timestamp, 1h)
| order by timestamp desc
```

**Count by Function:**
```kql
| summarize count() by operation_Name
| order by count_ desc
```

**Error Rate:**
```kql
| summarize 
    total = count(),
    errors = countif(severityLevel >= 3),
    error_rate = 100.0 * countif(severityLevel >= 3) / count()
    by bin(timestamp, 1h)
| order by timestamp desc
```

---

## 🎨 Example: Complete Script Creation

### Example: Create "logs_chunking.sh"

**Step 1:** Test query
```bash
az monitor app-insights query \
  --app ai-rag-evaluator \
  --resource-group rag-lab \
  --analytics-query "union traces | union exceptions | where timestamp > ago(24h) | where message contains 'chunking' or operation_Name contains 'chunking-worker' | take 5" \
  -o json
```

**Step 2:** Create script
```bash
cat > scripts/logs_chunking.sh << 'EOF'
#!/bin/bash
# Quick script to view chunking worker logs
exec "$(dirname "$0")/query_app_insights.sh" chunking "${1:-1h}"
EOF
chmod +x scripts/logs_chunking.sh
```

**Step 3:** Add to query_app_insights.sh
```bash
  chunking)
    QUERY="union traces | union exceptions | where timestamp > ago($TIME_WINDOW) | where message contains 'chunking' or operation_Name contains 'chunking-worker' | order by timestamp desc | project timestamp, message = iff(message != '', message, iff(innermostMessage != '', innermostMessage, customDimensions.['prop__{OriginalFormat}'])), logLevel = customDimensions.['LogLevel'], severityLevel"
    ;;
```

**Step 4:** Add to Makefile
```makefile
logs-chunking:
	@./scripts/query_app_insights.sh chunking $(TIME_WINDOW)
```

**Step 5:** Test
```bash
make logs-chunking
./scripts/logs_chunking.sh 2d
```

---

## ✅ Checklist

When creating a new KQL script, ensure:

- [ ] Query tested directly with `az monitor app-insights query`
- [ ] Script file created and made executable
- [ ] Query type added to `query_app_insights.sh`
- [ ] Makefile command added (optional but recommended)
- [ ] Script tested with different time windows
- [ ] Documentation updated in `TELEMETRY_GUIDE.md`
- [ ] Help text updated in Makefile

---

## 🔍 Debugging Your Query

### Issue: Query Returns No Results

1. **Check time window** - Try `7d` or `30d`
2. **Remove filters** - Test with minimal query first
3. **Check if data exists:**
   ```bash
   az monitor app-insights query \
     --app ai-rag-evaluator \
     --resource-group rag-lab \
     --analytics-query "traces | where timestamp > ago(7d) | take 1" \
     -o json
   ```

### Issue: Query Syntax Error

1. **Test in Azure Portal** - Use Application Insights Logs in portal to validate syntax
2. **Check escaping** - Ensure special characters are properly escaped in bash
3. **Validate KQL syntax** - Use KQL documentation: https://learn.microsoft.com/en-us/azure/data-explorer/kusto/query/

### Issue: Script Not Found

1. **Check file exists:** `ls -la scripts/logs_*.sh`
2. **Check executable:** `chmod +x scripts/logs_<name>.sh`
3. **Check path:** Use `$(dirname "$0")` for relative paths

---

## 📚 Additional Resources

- **KQL Documentation**: https://learn.microsoft.com/en-us/azure/data-explorer/kusto/query/
- **Application Insights Query API**: https://learn.microsoft.com/en-us/rest/api/application-insights/query
- **Telemetry Guide**: [TELEMETRY_GUIDE.md](TELEMETRY_GUIDE.md)
- **Existing Scripts**: See `scripts/logs_*.sh` for examples

---

## 🎯 Best Practices

1. **Use Standard Message Extraction**: Always use the standard pattern for message extraction
2. **Include Time Window**: Always use `ago($TIME_WINDOW)` for time filtering
3. **Order Results**: Use `order by timestamp desc` for most recent first
4. **Limit Results**: Consider adding `| take 100` for very large result sets
5. **Document Filters**: Add comments explaining what the query filters for
6. **Test Thoroughly**: Test with different time windows and edge cases

---

**Last Updated**: 2026-01-15  
**Maintained By**: Engineering Team
