# Codebase Management Rules

## Overview

This document defines codebase management rules and standards for the RAG Evaluation Platform. These rules ensure consistency, maintainability, and proper package structure across all modules.

**Last Updated**: 2025-01-27  
**Status**: Active

---

## Package Structure Standards

### 1. `__init__.py` Files

**Requirement**: All packages must have `__init__.py` files with proper exports.

#### Standard Pattern

Every `__init__.py` file must follow this pattern:

```python
"""[Package description]"""

from rag_eval.[module].[submodule] import Class1, Class2, function1

__all__ = [
    "Class1",
    "Class2", 
    "function1",
]
```

#### Requirements

- **Docstring**: Every `__init__.py` must have a docstring describing the package/module purpose
- **Exports**: Export commonly used classes/functions using `from ... import ...` statements
- **`__all__` List**: Define an explicit `__all__` list for all exports
- **No Empty Files**: Even if a package doesn't export anything yet, include a docstring and empty `__all__` list

#### Examples

**Core Package** (`rag_eval/core/__init__.py`):
```python
"""Core primitives shared across modules"""

from rag_eval.core.interfaces import (
    Chunk,
    Query,
    ModelAnswer,
    RetrievalResult,
    EvaluationScore,
)
from rag_eval.core.exceptions import (
    RAGEvalException,
    ConfigurationError,
    DatabaseError,
    AzureServiceError,
    ValidationError,
)
from rag_eval.core.config import Config
from rag_eval.core.logging import get_logger, setup_logging

__all__ = [
    "Chunk", "Query", "ModelAnswer", "RetrievalResult", "EvaluationScore",
    "RAGEvalException", "ConfigurationError", "DatabaseError",
    "AzureServiceError", "ValidationError",
    "Config", "get_logger", "setup_logging",
]
```

**Service Package** (`rag_eval/services/rag/__init__.py`):
```python
"""RAG pipeline engine"""

from rag_eval.services.rag.ingestion import (
    ingest_document,
    extract_text_from_document,
)
from rag_eval.services.rag.chunking import (
    chunk_text,
    chunk_text_fixed_size,
    chunk_text_with_llm,
)

__all__ = [
    "ingest_document", "extract_text_from_document",
    "chunk_text", "chunk_text_fixed_size", "chunk_text_with_llm",
]
```

### 2. Import Conventions

#### Absolute Imports Preferred

Always use absolute imports from the package root:

```python
# ✅ Correct
from rag_eval.core.interfaces import Chunk
from rag_eval.services.rag.chunking import chunk_text

# ❌ Incorrect (relative imports)
from ..core.interfaces import Chunk
from .chunking import chunk_text
```

#### No `sys.path` Manipulation

**Production Code**: Must not manipulate `sys.path`. This is strictly prohibited.

**Test Utilities**: Only acceptable in test utilities if absolutely necessary, and must be clearly documented with rationale.

**Exception Example** (only in test utilities):
```python
# Only acceptable in test utilities with clear documentation
import sys
from pathlib import Path
backend_dir = Path(__file__).parent.parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
```

### 3. Module Execution Pattern

All modules must be runnable as Python modules:

```bash
# From backend/ directory
python -m rag_eval.services.rag.chunking
python -m rag_eval.core.config
```

This ensures:
- Proper import resolution
- Package structure validation
- Testability

### 4. Package Organization

The codebase follows this structure:

```
rag_eval/
├── __init__.py              # Main package exports
├── core/                    # Core primitives
│   ├── __init__.py
│   ├── interfaces.py        # Data structures (Chunk, Query, etc.)
│   ├── exceptions.py        # Custom exceptions
│   ├── config.py            # Configuration management
│   └── logging.py            # Logging utilities
├── services/                # Business logic
│   ├── __init__.py
│   ├── rag/                 # RAG pipeline services
│   ├── evaluator/           # Evaluation services
│   └── meta_eval/           # Meta-evaluation services
├── db/                      # Database layer
│   ├── __init__.py
│   ├── connection.py        # Database connection pool
│   ├── queries.py           # Query executor
│   └── models.py            # Database models
├── api/                     # FastAPI application
│   ├── __init__.py
│   ├── main.py              # FastAPI app
│   └── routes/              # API routes
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── ids.py               # ID generation
│   ├── timing.py            # Timing utilities
│   └── file.py              # File utilities
└── prompts/                 # Prompt templates (non-Python)
```

---

## Testing Standards

### 1. pytest Configuration

The `pyproject.toml` must include:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
pythonpath = ["."]  # REQUIRED: Ensures pytest can find rag_eval package
```

The `pythonpath = ["."]` setting is **required** to allow pytest to resolve `rag_eval.*` imports when run from the `backend/` directory.

### 2. Import Path Handling

Tests should use the same absolute import pattern as production code:

```python
# ✅ Correct
from rag_eval.core.interfaces import Chunk
from rag_eval.services.rag.chunking import chunk_text

# ❌ Incorrect
import sys
sys.path.insert(0, "..")
from core.interfaces import Chunk
```

### 3. Module Execution in Tests

Tests should verify that modules can be executed:

```python
def test_module_execution():
    """Verify module can be run as Python module"""
    import subprocess
    result = subprocess.run(
        ["python", "-m", "rag_eval.services.rag.chunking"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
```

---

## Code Organization

### 1. No `sys.path` Manipulation in Production Code

**Strict Rule**: Production code must never manipulate `sys.path`.

**Rationale**:
- Breaks package structure assumptions
- Makes imports non-portable
- Causes issues in testing and deployment
- Violates Python packaging best practices

**Enforcement**: Code reviews must reject any PR that adds `sys.path` manipulation to production code.

### 2. Absolute Imports Preferred

Always use absolute imports from the package root:

```python
# ✅ Correct
from rag_eval.core.interfaces import Chunk
from rag_eval.services.rag.chunking import chunk_text

# ❌ Avoid (relative imports)
from ..core.interfaces import Chunk
from .chunking import chunk_text
```

**Exception**: Relative imports are acceptable within the same package for internal organization, but absolute imports are preferred for clarity.

### 3. Module Execution via `python -m` Pattern

All modules should be executable as Python modules:

```bash
python -m rag_eval.services.rag.chunking
```

This ensures:
- Proper package structure
- Correct import resolution
- Testability
- Consistency with Python packaging standards

---

## Enforcement

### 1. Checklist for New Implementations

When creating new packages or modules, verify:

- [ ] Package has proper `__init__.py` file with exports
- [ ] `__init__.py` includes docstring describing package purpose
- [ ] `__init__.py` defines `__all__` list for explicit exports
- [ ] Imports work correctly (`from rag_eval.* import ...`)
- [ ] Module can be run as `python -m rag_eval.*.*`
- [ ] No `sys.path` manipulation in production code
- [ ] pytest can discover and run tests
- [ ] All exports are listed in `__all__`

### 2. Code Review Requirements

Code reviews must check:

1. **Package Structure**:
   - Does the package have a proper `__init__.py`?
   - Are exports properly defined?
   - Is `__all__` list present and complete?

2. **Import Patterns**:
   - Are imports absolute from package root?
   - Is there any `sys.path` manipulation?
   - Do imports work when module is run as `python -m`?

3. **Testing**:
   - Can pytest discover and run tests?
   - Do imports work in test files?
   - Can the module be executed directly?

### 3. Linting Rules

Recommended linting configuration (if using tools like `ruff` or `pylint`):

- Enforce absolute imports
- Detect `sys.path` manipulation
- Verify `__all__` lists match exports
- Check for missing `__init__.py` files

---

## Common Issues and Solutions

### Issue 1: Import Errors in Tests

**Symptom**: `ModuleNotFoundError: No module named 'rag_eval'`

**Solution**: 
1. Verify `pyproject.toml` includes `pythonpath = ["."]`
2. Run tests from `backend/` directory: `pytest tests/`
3. Ensure package is properly structured with `__init__.py` files

### Issue 2: Module Can't Be Run Directly

**Symptom**: `ModuleNotFoundError` when running `python rag_eval/services/rag/chunking.py`

**Solution**:
1. Run as module: `python -m rag_eval.services.rag.chunking`
2. Ensure absolute imports are used (not relative)
3. Verify `__init__.py` files exist in all parent packages

### Issue 3: Circular Import Errors

**Symptom**: `ImportError: cannot import name 'X' from partially initialized module`

**Solution**:
1. Review import structure and dependencies
2. Move shared code to a common module
3. Use lazy imports if necessary
4. Restructure to avoid circular dependencies

### Issue 4: Missing Exports

**Symptom**: `from rag_eval.core import X` fails but `from rag_eval.core.interfaces import X` works

**Solution**:
1. Add `X` to `rag_eval/core/__init__.py` exports
2. Add `X` to `__all__` list
3. Verify import statement in `__init__.py`

---

## Migration Guide

### Migrating Existing Code

If you have existing code that doesn't follow these standards:

1. **Add `__init__.py` files**: Create proper `__init__.py` files with exports
2. **Update imports**: Convert relative imports to absolute imports
3. **Remove `sys.path` manipulation**: Refactor to use proper package structure
4. **Add `__all__` lists**: Explicitly define exports in each `__init__.py`
5. **Update pytest config**: Add `pythonpath = ["."]` to `pyproject.toml`
6. **Test module execution**: Verify modules can be run as `python -m`

### Example Migration

**Before**:
```python
# rag_eval/services/rag/chunking.py
import sys
from pathlib import Path
backend_dir = Path(__file__).parent.parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
from core.interfaces import Chunk
```

**After**:
```python
# rag_eval/services/rag/chunking.py
from rag_eval.core.interfaces import Chunk
```

---

## References

- [RFC001.md](../initiatives/rag_system/RFC001.md) - Package Structure Standards section
- [TODO001.md](../initiatives/rag_system/TODO001.md) - Package Structure Requirements for all phases
- [Python Packaging User Guide](https://packaging.python.org/en/latest/)
- [PEP 8 - Style Guide for Python Code](https://peps.python.org/pep-0008/)

---

**Document Status**: Active  
**Last Updated**: 2025-01-27  
**Maintainer**: Development Team




