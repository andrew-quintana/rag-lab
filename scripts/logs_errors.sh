#!/bin/bash
# Quick script to view recent errors
exec "$(dirname "$0")/query_app_insights.sh" errors "${1:-1h}"
