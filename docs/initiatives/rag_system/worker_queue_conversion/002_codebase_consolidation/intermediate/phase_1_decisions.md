# Phase 1 Decisions - Code Duplication Elimination

**Date**: 2025-12-07  
**Phase**: 1 - Code Duplication Elimination  
**Status**: ✅ Complete

## Summary

Phase 1 successfully eliminated code duplication by updating Azure Functions to import directly from project root, removing duplicate backend code, and simplifying the build script.

---

## Decision 1: Path Resolution Approach

### Issue
The RFC Section 1.2 example showed adding project root to `sys.path`, but the actual backend code structure is `backend/rag_eval/`, which requires adding the `backend` directory to the path instead.

### Decision
Add `backend` directory to `sys.path` (not project root) to allow direct imports of `rag_eval.*`.

### Implementation
```python
# Add backend directory to Python path for backend imports
# This allows importing directly from rag_eval.* (backend/rag_eval/)
backend_dir = project_root / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
```

### Rationale
- The backend code is located at `backend/rag_eval/`
- Python imports `rag_eval` directly when `backend/` is in `sys.path`
- This matches the existing import pattern: `from rag_eval.services.workers.ingestion_worker import ingestion_worker`

### Files Updated
- `infra/azure/azure_functions/ingestion-worker/__init__.py`
- `infra/azure/azure_functions/chunking-worker/__init__.py`
- `infra/azure/azure_functions/embedding-worker/__init__.py`
- `infra/azure/azure_functions/indexing-worker/__init__.py`

---

## Decision 2: Dotenv Loading Order

### Issue
Environment variables must be loaded before importing backend code to ensure configuration is available.

### Decision
Load `.env.local` from project root BEFORE path manipulation and imports.

### Implementation
```python
# Load .env.local from project root BEFORE importing backend code
try:
    from dotenv import load_dotenv
    project_root = Path(__file__).parent.parent.parent.parent.parent
    env_file = project_root / ".env.local"
    if env_file.exists():
        load_dotenv(env_file, override=True)
except ImportError:
    # dotenv not available, continue without loading
    pass
```

### Rationale
- Environment variables are needed by backend code during import
- Flexible approach: works with or without `.env.local` file
- Supports Azure Function App settings (which take precedence via `override=True`)

---

## Decision 3: Build Script Simplification

### Issue
The build script was copying backend code and modifying function entry points during deployment, which is no longer needed.

### Decision
Simplify build script to only validate prerequisites and document the new deployment structure.

### Implementation
- Removed all code copying logic
- Removed path modification logic (sed commands)
- Added prerequisite validation:
  - Backend source directory exists
  - `requirements.txt` exists
  - All 4 function entry points exist
  - All function bindings exist

### Rationale
- Functions now import directly from project root
- No code copying needed during deployment
- Build script should validate, not modify code
- Simpler build process reduces deployment time and complexity

### Files Updated
- `infra/azure/azure_functions/build.sh`

---

## Decision 4: Duplicate Code Removal

### Issue
Duplicate backend code existed in `infra/azure/azure_functions/backend/rag_eval/` (1.3MB).

### Decision
Remove duplicate code directory and update `.gitignore` to prevent accidental commits.

### Implementation
1. Removed from Git tracking: `git rm -r infra/azure/azure_functions/backend/rag_eval/`
2. Deleted directory from filesystem
3. Updated `.gitignore` to ignore `infra/azure/azure_functions/backend/`

### Rationale
- Single source of truth: all backend code in `backend/rag_eval/`
- Eliminates code drift risk
- Reduces repository size
- Simplifies maintenance

### Files Updated
- `.gitignore` (added `infra/azure/azure_functions/backend/`)
- Removed: `infra/azure/azure_functions/backend/rag_eval/` (entire directory, 60+ files)

---

## Deviations from RFC

### RFC Section 1.2 Pattern
The RFC example showed:
```python
# Add project root to path for backend imports
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
```

**Actual Implementation**:
```python
# Add backend directory to Python path for backend imports
backend_dir = project_root / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
```

**Rationale**: The backend code structure is `backend/rag_eval/`, so we need to add `backend/` to the path, not the project root. This allows direct imports of `rag_eval.*` as intended.

---

## Path Resolution Details

### Function Entry Point Location
- `infra/azure/azure_functions/ingestion-worker/__init__.py`

### Path Calculation
- `__file__` = `infra/azure/azure_functions/ingestion-worker/__init__.py`
- `Path(__file__).parent` = `infra/azure/azure_functions/ingestion-worker/`
- `Path(__file__).parent.parent` = `infra/azure/azure_functions/`
- `Path(__file__).parent.parent.parent` = `infra/azure/`
- `Path(__file__).parent.parent.parent.parent` = `infra/`
- `Path(__file__).parent.parent.parent.parent.parent` = project root

### Backend Path
- `project_root / "backend"` = `backend/` (relative to project root)
- Adding `backend/` to `sys.path` allows `from rag_eval.*` imports

---

## Git Changes

### Files Removed
- `infra/azure/azure_functions/backend/rag_eval/` (entire directory, 60+ files, 1.3MB)

### Files Modified
- `infra/azure/azure_functions/ingestion-worker/__init__.py`
- `infra/azure/azure_functions/chunking-worker/__init__.py`
- `infra/azure/azure_functions/embedding-worker/__init__.py`
- `infra/azure/azure_functions/indexing-worker/__init__.py`
- `infra/azure/azure_functions/build.sh`
- `.gitignore`

---

## Next Steps

Phase 1 is complete. Ready to proceed to Phase 2: Configuration Consolidation.

---

**Last Updated**: 2025-12-07  
**Status**: ✅ Complete

