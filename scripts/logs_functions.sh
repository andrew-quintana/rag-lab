#!/bin/bash
# Quick script to view function execution logs
exec "$(dirname "$0")/query_app_insights.sh" functions "${1:-1h}"
