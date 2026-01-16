#!/bin/bash
# Quick script to view recent requests
exec "$(dirname "$0")/query_app_insights.sh" requests "${1:-1h}"
