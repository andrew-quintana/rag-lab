#!/bin/bash
# Query Azure Application Insights logs using KQL
# Usage: ./scripts/query_app_insights.sh <query_type> [time_window]
# Query types: requests, errors, traces, dependencies, functions

set -e

RESOURCE_GROUP="rag-lab"
FUNCTION_APP="func-raglab-uploadworkers"
APP_INSIGHTS_NAME="ai-rag-evaluator"  # Application Insights resource name
TIME_WINDOW="${2:-1h}"  # Default: 1 hour ago

# Get App Insights App ID
echo "🔍 Getting Application Insights App ID..."

# Try to get from connection string first
CONN_STRING=$(az functionapp config appsettings list \
  --name "$FUNCTION_APP" \
  --resource-group "$RESOURCE_GROUP" \
  --query "[?name=='APPLICATIONINSIGHTS_CONNECTION_STRING'].value" -o tsv 2>/dev/null || echo "")

if [ -n "$CONN_STRING" ]; then
  # Extract ApplicationId from connection string (format: InstrumentationKey=...;IngestionEndpoint=...;ApplicationId=...)
  APP_ID=$(echo "$CONN_STRING" | sed -n 's/.*ApplicationId=\([^;]*\).*/\1/p')
  # Also try AppId (alternative format)
  if [ -z "$APP_ID" ]; then
    APP_ID=$(echo "$CONN_STRING" | sed -n 's/.*AppId=\([^;]*\).*/\1/p')
  fi
fi

# If not found, try to get from App Insights resource directly
if [ -z "$APP_ID" ]; then
  APP_ID=$(az monitor app-insights component show \
    --app "$FUNCTION_APP" \
    --resource-group "$RESOURCE_GROUP" \
    --query "appId" -o tsv 2>/dev/null || echo "")
fi

# If still not found, try getting from resource name
if [ -z "$APP_ID" ]; then
  APP_ID=$(az monitor app-insights component show \
    --app "$APP_INSIGHTS_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query "appId" -o tsv 2>/dev/null || echo "")
fi

# If still not found, try listing all App Insights in resource group
if [ -z "$APP_ID" ]; then
  APP_ID=$(az monitor app-insights component list \
    --resource-group "$RESOURCE_GROUP" \
    --query "[0].appId" -o tsv 2>/dev/null || echo "")
fi

if [ -z "$APP_ID" ]; then
  # Try to find by resource group
  APP_ID=$(az monitor app-insights component list \
    --resource-group "$RESOURCE_GROUP" \
    --query "[0].appId" -o tsv 2>/dev/null || echo "")
fi

# Allow override via environment variable
if [ -z "$APP_ID" ] && [ -n "$APP_INSIGHTS_ID" ]; then
  APP_ID="$APP_INSIGHTS_ID"
fi

if [ -z "$APP_ID" ]; then
  echo "❌ Error: Could not find Application Insights App ID"
  echo ""
  echo "Options:"
  echo "  1. Set APP_INSIGHTS_ID environment variable:"
  echo "     export APP_INSIGHTS_ID=<your-app-id>"
  echo ""
  echo "  2. Ensure Application Insights is linked to Function App"
  echo "     Check: az functionapp show --name $FUNCTION_APP --resource-group $RESOURCE_GROUP"
  echo ""
  echo "  3. Find App ID manually:"
  echo "     az monitor app-insights component list --resource-group $RESOURCE_GROUP"
  exit 1
fi

echo "✅ App ID: $APP_ID"
echo ""

# KQL queries based on type
case "$1" in
  requests)
    QUERY="requests | where timestamp > ago($TIME_WINDOW) | order by timestamp desc | project timestamp, name, url, resultCode, duration, operation_Id"
    ;;
  errors)
    QUERY="exceptions | where timestamp > ago($TIME_WINDOW) | order by timestamp desc | project timestamp, type, message, outerMessage, operation_Id"
    ;;
  traces)
    QUERY="union traces | union exceptions | where timestamp > ago($TIME_WINDOW) | order by timestamp desc | project timestamp, message = iff(message != '', message, iff(innermostMessage != '', innermostMessage, customDimensions.['prop__{OriginalFormat}'])), logLevel = customDimensions.['LogLevel'], severityLevel"
    ;;
  dependencies)
    QUERY="dependencies | where timestamp > ago($TIME_WINDOW) | order by timestamp desc | project timestamp, name, type, target, resultCode, duration, operation_Id"
    ;;
  functions)
    QUERY="union traces | union exceptions | where timestamp > ago($TIME_WINDOW) | where message contains 'Function' or message contains 'ingestion-worker' or message contains 'chunking-worker' or message contains 'embedding-worker' or message contains 'indexing-worker' | order by timestamp desc | project timestamp, message = iff(message != '', message, iff(innermostMessage != '', innermostMessage, customDimensions.['prop__{OriginalFormat}'])), logLevel = customDimensions.['LogLevel'], severityLevel"
    ;;
  queue)
    QUERY="union traces | union exceptions | where timestamp > ago($TIME_WINDOW) | where message contains 'queue' or message contains 'Queue' or message contains 'ingestion-uploads' | order by timestamp desc | project timestamp, message = iff(message != '', message, iff(innermostMessage != '', innermostMessage, customDimensions.['prop__{OriginalFormat}'])), logLevel = customDimensions.['LogLevel'], severityLevel"
    ;;
  ingestion)
    QUERY="union traces | union exceptions | where timestamp > ago($TIME_WINDOW) | where message contains 'ingestion' or operation_Name contains 'ingestion-worker' | order by timestamp desc | project timestamp, message = iff(message != '', message, iff(innermostMessage != '', innermostMessage, customDimensions.['prop__{OriginalFormat}'])), logLevel = customDimensions.['LogLevel'], severityLevel"
    ;;
  operation)
    # Query by operation_Id (requires operation_Id as third argument)
    if [ -z "$3" ]; then
      echo "❌ Error: operation_Id required for 'operation' query type"
      echo "Usage: $0 operation <time_window> <operation_Id>"
      echo "Example: $0 operation 30d 3576e5b499f5b0b67bddfc5c6a8d951b"
      exit 1
    fi
    QUERY="union traces | union exceptions | where timestamp > ago($TIME_WINDOW) | where operation_Id == '$3' | order by timestamp asc | project timestamp, message = iff(message != '', message, iff(innermostMessage != '', innermostMessage, customDimensions.['prop__{OriginalFormat}'])), logLevel = customDimensions.['LogLevel'], severityLevel"
    ;;
  *)
    echo "Usage: $0 <query_type> [time_window]"
    echo ""
    echo "Query types:"
    echo "  requests    - HTTP requests"
    echo "  errors      - Exceptions and errors"
    echo "  traces      - All trace logs"
    echo "  dependencies - External dependencies (queues, storage, etc.)"
    echo "  functions   - Function execution logs"
    echo "  queue       - Queue-related logs"
    echo "  ingestion   - Ingestion worker logs"
    echo ""
    echo "Time window examples: 1h, 30m, 1d"
    echo ""
    echo "Example: $0 functions 1h"
    exit 1
    ;;
esac

echo "📊 Query: $1"
echo "⏰ Time window: $TIME_WINDOW"
echo ""

# Execute query - try with resource name first (more reliable)
echo "Executing query..."

# Use temporary file to avoid shell quoting issues
TEMP_FILE=$(mktemp)
az monitor app-insights query \
  --app "$APP_INSIGHTS_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --analytics-query "$QUERY" \
  -o json > "$TEMP_FILE" 2>&1

if [ $? -eq 0 ] && [ -s "$TEMP_FILE" ]; then
  # Parse JSON and display as table
  python3 << PYTHON_SCRIPT
import sys
import json

try:
    with open("$TEMP_FILE", 'r') as f:
        data = json.load(f)
    
    if 'tables' in data and len(data['tables']) > 0:
        table = data['tables'][0]
        if 'rows' in table and len(table['rows']) > 0:
            # Print header
            if 'columns' in table:
                headers = [col.get('name', '') for col in table['columns']]
                # Format header nicely
                header_line = " | ".join(f"{h:30}" for h in headers)
                print(header_line)
                print("-" * min(120, len(header_line)))
            
            # Print rows (limit to first 50 for readability)
            for row in table['rows'][:50]:
                # Format row values - truncate long messages
                formatted_row = []
                for i, val in enumerate(row):
                    val_str = str(val) if val is not None else ''
                    # Truncate message column (usually index 1) more aggressively
                    if i == 1 and len(val_str) > 80:
                        formatted_row.append(val_str[:77] + "...")
                    elif len(val_str) > 30:
                        formatted_row.append(val_str[:27] + "...")
                    else:
                        formatted_row.append(val_str)
                
                # Pad columns for alignment
                formatted_line = " | ".join(f"{v:30}" for v in formatted_row)
                print(formatted_line)
            
            if len(table['rows']) > 50:
                print(f"\n... ({len(table['rows']) - 50} more rows)")
            print(f"\n✅ Found {len(table['rows'])} result(s)")
        else:
            print("⚠️  Query returned no results")
    else:
        print("⚠️  Unexpected response format")
        with open("$TEMP_FILE", 'r') as f:
            content = f.read()
            print(content[:500])
except json.JSONDecodeError as e:
    print(f"❌ Error parsing JSON: {e}")
    with open("$TEMP_FILE", 'r') as f:
        content = f.read()
        print("First 500 chars of response:")
        print(content[:500])
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
PYTHON_SCRIPT
  rm -f "$TEMP_FILE"
else
  # Fallback to App ID or show error
  echo "⚠️  Query failed or returned empty. Trying with App ID..."
  az monitor app-insights query \
    --app "$APP_ID" \
    --analytics-query "$QUERY" \
    -o json 2>&1 | python3 -c "import sys, json; data = json.load(sys.stdin); print('Results:', len(data.get('tables', [{}])[0].get('rows', [])) if data.get('tables') else 0, 'rows')" 2>/dev/null || cat "$TEMP_FILE"
  rm -f "$TEMP_FILE"
fi
