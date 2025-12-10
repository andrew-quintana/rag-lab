# Phase 1 Decisions - Code Duplication Elimination & Structure Reorganization

**Date**: 2025-12-09  
**Phase**: 1 - Code Duplication Elimination & Structure Reorganization  
**Status**: ✅ Complete

## Summary

Phase 1 successfully moved Azure Functions from `infra/azure/azure_functions/` to `backend/azure_functions/` and simplified all function entry points to use direct imports without path manipulation. This establishes a single source of truth for all backend code in one location.

---

## Decision 1: Move Azure Functions to Backend Directory

### Issue
Azure Functions were located in `infra/azure/azure_functions/`, requiring complex path manipulation to import from `backend/src/`. This created maintenance overhead and risk of code drift.

### Decision
Move the entire `infra/azure/azure_functions/` directory to `backend/azure_functions/` so functions are alongside `backend/src/` in the same directory.

### Implementation
```bash
mv infra/azure/azure_functions backend/azure_functions
```

### Rationale
- Functions are now in `backend/` alongside `src/`, enabling direct imports
- No path manipulation needed - Python naturally finds `src` when running from `backend/`
- Single source of truth: all backend code (workers + functions) in one location
- Simpler structure: `backend/src/` (source) and `backend/azure_functions/` (functions)

### Files Moved
- `infra/azure/azure_functions/ingestion-worker/` → `backend/azure_functions/ingestion-worker/`
- `infra/azure/azure_functions/chunking-worker/` → `backend/azure_functions/chunking-worker/`
- `infra/azure/azure_functions/embedding-worker/` → `backend/azure_functions/embedding-worker/`
- `infra/azure/azure_functions/indexing-worker/` → `backend/azure_functions/indexing-worker/`
- `infra/azure/azure_functions/build.sh` → `backend/azure_functions/build.sh`
- `infra/azure/azure_functions/host.json` → `backend/azure_functions/host.json`
- `infra/azure/azure_functions/requirements.txt` → `backend/azure_functions/requirements.txt`
- `infra/azure/azure_functions/.deployment` → `backend/azure_functions/.deployment`
- `infra/azure/azure_functions/.funcignore` → `backend/azure_functions/.funcignore`
- `infra/azure/azure_functions/local.settings.json` → `backend/azure_functions/local.settings.json`
- `infra/azure/azure_functions/README.md` → `backend/azure_functions/README.md`
- `infra/azure/azure_functions/README_LOCAL.md` → `backend/azure_functions/README_LOCAL.md`

---

## Decision 2: Simplify Function Entry Points

### Issue
Function entry points used complex path manipulation:
- `Path(__file__).parent.parent.parent.parent.parent` to find project root
- Dotenv loading from project root
- Adding backend directory to `sys.path`

### Decision
Remove all path manipulation and use simple direct imports. Functions are now in `backend/azure_functions/` alongside `backend/src/`, so imports work naturally.

### Implementation Pattern
```python
"""Azure Function for ingestion worker

This function is triggered by messages in the ingestion-uploads queue.
It processes documents through the ingestion stage (text extraction).
"""
import json
from src.services.workers.ingestion_worker import ingestion_worker
from src.core.logging import get_logger

logger = get_logger("azure_functions.ingestion_worker")

def main(queueMessage: str) -> None:
    """
    Azure Function entry point for ingestion worker.
    
    Args:
        queueMessage: JSON string from queue trigger
    """
    try:
        # Parse queue message
        message_dict = json.loads(queueMessage)
        
        logger.info(f"Processing ingestion message for document: {message_dict.get('document_id')}")
        
        # Call worker function
        ingestion_worker(message_dict, context=None)
        
        logger.info(f"Successfully processed ingestion for document: {message_dict.get('document_id')}")
    
    except Exception as e:
        logger.error(f"Error processing ingestion message: {e}", exc_info=True)
        raise
```

### Rationale
- **No path manipulation**: Functions are in `backend/` alongside `src/`, so Python finds `src` naturally
- **No dotenv loading**: Azure Functions handle environment variables automatically via `os.environ`
- **Simple imports**: Direct `from src.services.workers...` imports work without path manipulation
- **Maintainable**: Clean, simple code that's easy to understand and maintain

### Files Updated
- `backend/azure_functions/ingestion-worker/__init__.py`
- `backend/azure_functions/chunking-worker/__init__.py`
- `backend/azure_functions/embedding-worker/__init__.py`
- `backend/azure_functions/indexing-worker/__init__.py`

### Removed Code
- All `Path(__file__).parent.parent...` path manipulation
- All dotenv loading code (`load_dotenv`)
- All `sys.path` manipulation
- All project root calculation logic

---

## Decision 3: Update Build Script for New Location

### Issue
Build script calculated project root as 3 levels up from `infra/azure/azure_functions/build.sh`, but now it's in `backend/azure_functions/build.sh` (2 levels up).

### Decision
Update build script path calculation to reflect new location.

### Implementation
```bash
# Calculate project root
# Script is in: backend/azure_functions/build.sh
# Project root is 2 levels up: backend/azure_functions -> backend -> project_root
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKEND_SOURCE="$PROJECT_ROOT/backend/src"
```

### Rationale
- Build script still validates prerequisites
- Path calculation updated for new location
- No code copying needed (functions import directly from `src`)

### Files Updated
- `backend/azure_functions/build.sh`

---

## Decision 4: Update Deployment Scripts

### Issue
Multiple deployment and development scripts referenced the old location `infra/azure/azure_functions/`.

### Decision
Update all scripts to reference the new location `backend/azure_functions/`.

### Files Updated
- `scripts/deploy_azure_functions.sh` - Updated `FUNCTIONS_DIR`
- `scripts/setup_git_deployment.sh` - Updated build script path references
- `scripts/test_functions_local.sh` - Updated `local.settings.json` path
- `scripts/dev_functions_local.sh` - Updated `FUNCTIONS_DIR`

### Rationale
- All scripts must reference the correct location
- Consistent path references across all tooling
- Prevents confusion and errors

---

## Decision 5: Update Documentation

### Issue
README files in `azure_functions/` directory referenced old paths and outdated build process.

### Decision
Update README files to reflect new location and simplified build process.

### Files Updated
- `backend/azure_functions/README.md` - Updated paths and build process description
- `backend/azure_functions/README_LOCAL.md` - Updated paths and troubleshooting

### Rationale
- Documentation must match actual implementation
- Clear instructions for developers
- Accurate troubleshooting guides

---

## Decision 6: Update .funcignore

### Issue
`.funcignore` had comments about `backend/` being needed for deployment (from old approach).

### Decision
Update comments to reflect that no duplicate code directory is needed.

### Implementation
```gitignore
# Note: Functions now import directly from backend/src/
# No duplicate code directory is needed
```

### Rationale
- Clear documentation of new approach
- Prevents confusion about deployment structure

---

## Verification

### No Duplicate Code
- ✅ Verified no `backend/azure_functions/backend/src/` directory exists
- ✅ No duplicate code tracked in Git
- ✅ Single source of truth: `backend/src/` only

### Build Script
- ✅ Build script validates prerequisites correctly
- ✅ Path calculation works for new location
- ✅ All function entry points validated

### Function Entry Points
- ✅ All 4 functions use simple direct imports
- ✅ No path manipulation code remains
- ✅ No dotenv loading code remains
- ✅ All functions compile successfully

---

## Summary of Changes

### Structure Changes
- ✅ Azure Functions moved: `infra/azure/azure_functions/` → `backend/azure_functions/`
- ✅ Empty `infra/azure/` directory remains (can be removed if no longer needed)

### Code Changes
- ✅ 4 function entry points simplified (removed path manipulation)
- ✅ Build script updated for new location
- ✅ 4 deployment scripts updated
- ✅ 2 README files updated
- ✅ `.funcignore` comments updated

### No Changes Needed
- ✅ No duplicate code directory existed (already removed in previous work)
- ✅ `.gitignore` already configured correctly

---

**Last Updated**: 2025-12-09  
**Status**: ✅ Phase 1 Complete
