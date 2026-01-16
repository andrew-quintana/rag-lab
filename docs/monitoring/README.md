# Azure Functions Monitoring Documentation

**Complete monitoring and observability documentation for Azure Functions**

---

## 📚 Documentation Index

### Quick Start Guides
- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute guide to monitoring Azure Functions
  - Quick health checks
  - Common scenarios
  - Command reference

### Telemetry & Observability
- **[TELEMETRY_GUIDE.md](TELEMETRY_GUIDE.md)** - Complete Application Insights telemetry guide
  - KQL query reference
  - Available scripts and commands
  - Common query patterns
  - Telemetry data structure

- **[CREATING_KQL_SCRIPTS.md](CREATING_KQL_SCRIPTS.md)** - Step-by-step guide for creating new KQL scripts
  - Query design
  - Script creation
  - Integration steps
  - Testing and validation

### Troubleshooting
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Troubleshooting playbook
  - 11 common issues with fixes
  - Step-by-step diagnosis
  - Application Insights integration

### Implementation
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Implementation status and history

---

## 🚀 Quick Commands

### Queue Monitoring
```bash
python scripts/monitor_queue_health.py
python scripts/diagnose_functions.py
```

### Application Insights Telemetry
```bash
make logs-functions    # Function execution logs
make logs-errors       # Recent errors
make logs-queue       # Queue-related logs
make logs-ingestion   # Ingestion worker logs
```

### Azure CLI
```bash
az functionapp log tail --name func-raglab-uploadworkers --resource-group rag-lab
```

---

## 📖 Getting Started

1. **New to monitoring?** Start with [QUICKSTART.md](QUICKSTART.md)
2. **Need to query logs?** See [TELEMETRY_GUIDE.md](TELEMETRY_GUIDE.md)
3. **Creating custom queries?** See [CREATING_KQL_SCRIPTS.md](CREATING_KQL_SCRIPTS.md)
4. **Troubleshooting issues?** See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

**Last Updated**: 2026-01-15
