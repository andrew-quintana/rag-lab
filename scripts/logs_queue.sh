#!/bin/bash
# Quick script to view queue-related logs
exec "$(dirname "$0")/query_app_insights.sh" queue "${1:-1h}"
