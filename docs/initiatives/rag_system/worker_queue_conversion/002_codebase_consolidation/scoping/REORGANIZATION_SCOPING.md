# Backend Package Reorganization Scoping

**Date**: 2025-12-07  
**Status**: Scoping  
**Related**: Initiative 002 - Codebase Consolidation

## Current Structure

```
backend/
├── platform/                    # Main package (to be renamed/reorganized)
│   ├── api/                     # API layer (FastAPI)
│   ├── core/                    # Core primitives, config, logging, exceptions
│   ├── db/                      # Database layer (Supabase)
│   ├── services/                 # Business logic
│   │   ├── rag/                 # RAG pipeline services
│   │   ├── evaluator/           # Evaluation services
│   │   ├── workers/             # Worker implementations
│   │   └── shared/              # Shared services (LLM providers)
│   ├── utils/                   # Utilities
│   └── prompts/                 # Prompt templates
└── azure_functions/             # Azure Functions (to be moved here in Phase 1)
```

## Problem Statement

The package name `platform` is redundant - the project goal (RAG evaluation) is already implied. The current structure mixes concerns:
- API, core, db, services all under one package
- No clear separation between platform components
- Hard to understand component boundaries at a glance

## Goals

1. **Better Component Separation**: Clear boundaries between API, services, workers, etc.
2. **More Descriptive Organization**: Package name reflects structure, not project goal
3. **Maintainability**: Easier to understand what each component does
4. **Scalability**: Structure supports future growth

---

## Option 1: Rename to `platform/` (Recommended)

**Structure**:
```
backend/
├── platform/                    # Platform code (renamed from platform)
│   ├── api/                     # API layer
│   ├── core/                    # Core primitives
│   ├── db/                      # Database layer
│   ├── services/                # Business logic
│   │   ├── rag/
│   │   ├── evaluator/
│   │   ├── workers/
│   │   └── shared/
│   ├── utils/
│   └── prompts/
└── azure_functions/             # Azure Functions
```

**Pros**:
- Simple rename - minimal structural changes
- "Platform" clearly indicates it's the platform code
- All imports change from `platform.*` to `platform.*`
- Maintains current organization

**Cons**:
- Still one large package
- Doesn't improve component separation

**Impact**:
- **Files to change**: ~687 import statements across 147 files
- **Migration complexity**: Medium (find/replace imports)
- **Risk**: Medium (need to update all imports)

---

## Option 2: Split into Multiple Top-Level Packages

**Structure**:
```
backend/
├── api/                         # API layer (separate package)
│   └── routes/
├── core/                        # Core primitives (separate package)
│   ├── config.py
│   ├── logging.py
│   ├── exceptions.py
│   └── interfaces.py
├── db/                          # Database layer (separate package)
│   ├── connection.py
│   ├── queries.py
│   └── models.py
├── services/                     # Services (separate package)
│   ├── rag/
│   ├── evaluator/
│   ├── workers/
│   └── shared/
├── utils/                       # Utilities (separate package)
├── prompts/                      # Prompts (separate package)
└── azure_functions/             # Azure Functions
```

**Pros**:
- Clear component separation
- Each component is a distinct package
- Easy to understand boundaries
- Better for large codebases

**Cons**:
- Major structural change
- More complex imports (need to handle cross-package dependencies)
- Higher migration risk
- May need to adjust Python path handling

**Impact**:
- **Files to change**: All files, plus package structure
- **Migration complexity**: High (structural changes + imports)
- **Risk**: High (breaking changes, need careful dependency management)

---

## Option 3: Hybrid - `platform/` with Clear Subpackages

**Structure**:
```
backend/
├── platform/                     # Platform package
│   ├── api/                     # API subpackage
│   ├── core/                    # Core subpackage
│   ├── db/                      # Database subpackage
│   ├── services/                # Services subpackage
│   │   ├── rag/
│   │   ├── evaluator/
│   │   ├── workers/
│   │   └── shared/
│   ├── utils/                   # Utils subpackage
│   └── prompts/                 # Prompts subpackage
└── azure_functions/             # Azure Functions
```

**Pros**:
- Clear that everything under `platform/` is platform code
- Maintains current structure
- Simple rename operation
- Clear namespace: `platform.api`, `platform.services.rag`, etc.

**Cons**:
- Still one main package (just renamed)
- Doesn't improve separation beyond naming

**Impact**:
- **Files to change**: ~687 import statements
- **Migration complexity**: Medium (find/replace)
- **Risk**: Medium

---

## Option 4: Domain-Based Organization

**Structure**:
```
backend/
├── api/                         # API layer
├── core/                        # Core primitives
├── db/                          # Database layer
├── rag/                         # RAG domain
│   ├── services/               # RAG services
│   └── workers/                # RAG workers
├── evaluator/                   # Evaluator domain
│   └── services/               # Evaluator services
├── shared/                      # Shared components
│   ├── utils/
│   ├── prompts/
│   └── services/               # Shared services
└── azure_functions/             # Azure Functions
```

**Pros**:
- Domain-driven organization
- Clear separation by business domain
- Workers grouped with their domain

**Cons**:
- Workers currently span domains (they use both RAG and evaluator services)
- Major restructuring
- High migration complexity

**Impact**:
- **Files to change**: All files, major restructuring
- **Migration complexity**: Very High
- **Risk**: Very High

---

## Recommendation: Option 1 (Rename to `platform/`)

**Rationale**:
1. **Minimal Risk**: Simple rename, no structural changes
2. **Clear Intent**: "platform" clearly indicates it's the platform code
3. **Maintains Organization**: Current structure is actually well-organized
4. **Manageable Migration**: Find/replace imports is straightforward
5. **Future-Proof**: Can always split further later if needed

**Alternative Consideration**: If you want better separation NOW, Option 2 (split packages) is viable but requires more careful planning.

---

## Migration Plan (Option 1: Rename to `platform/`)

### Phase 1: Preparation
1. Create comprehensive list of all files with `platform` imports
2. Document all external references (tests, scripts, docs)
3. Create migration script to update imports

### Phase 2: Rename Package
1. Rename `backend/platform/` → `backend/platform/`
2. Update all `__init__.py` files
3. Update package metadata

### Phase 3: Update Imports
1. Update all Python imports: `from platform.*` → `from platform.*`
2. Update test files
3. Update scripts
4. Update documentation

### Phase 4: Update Configuration
1. Update `pyproject.toml` if present
2. Update any build scripts
3. Update Azure Functions imports (after Phase 1 move)
4. Update deployment scripts

### Phase 5: Validation
1. Run all tests
2. Verify imports work
3. Check documentation
4. Validate Azure Functions can import

---

## Impact Analysis

### Files Affected
- **Python files**: ~147 files with `platform` imports
- **Test files**: All test files
- **Scripts**: All scripts in `backend/scripts/`
- **Documentation**: All docs referencing `platform`
- **Azure Functions**: All 4 function entry points (after Phase 1 move)
- **Configuration**: `pyproject.toml`, build scripts

### Dependencies
- **External**: None (internal package only)
- **Database**: No changes needed
- **API**: No changes needed (just imports)
- **Workers**: No changes needed (just imports)

### Risks
- **Import Errors**: If any imports missed
- **Test Failures**: If test imports not updated
- **Documentation Drift**: If docs not updated
- **Azure Functions**: Need to update after Phase 1 move

### Mitigation
- Comprehensive find/replace with validation
- Run full test suite after migration
- Update documentation in same commit
- Coordinate with Phase 1 (move Azure Functions first, then rename)

---

## Questions to Answer

1. **Timing**: Should this happen before or after Phase 1 (moving Azure Functions)?
   - **Recommendation**: After Phase 1, so we only update Azure Functions imports once

2. **Scope**: Just rename, or also reorganize structure?
   - **Recommendation**: Start with rename, reorganize later if needed

3. **Package Name**: `platform/`, `app/`, `lib/`, or something else?
   - **Recommendation**: `platform/` (clear, descriptive, not too generic)

4. **Gradual Migration**: Can we do this incrementally?
   - **Recommendation**: No - do it all at once to avoid confusion

---

## Next Steps

1. **Decision**: Choose option (recommend Option 1: `platform/`)
2. **Timing**: Decide when (recommend after Phase 1)
3. **Create Migration Plan**: Detailed step-by-step plan
4. **Create Migration Script**: Automated import updates
5. **Execute**: Run migration with validation

---

**Last Updated**: 2025-12-07  
**Status**: Scoping Complete - Prompt Created

## Execution Prompt

A comprehensive execution prompt has been created:
- `prompts/prompt_package_rename_002.md` - Complete execution guide for Option 1 (rename to `platform/`)

The prompt includes:
- Step-by-step tasks for renaming the package
- Updating all imports (~687 imports across 147 files)
- Updating all scoping documents
- Updating all phase prompts
- Validation and testing steps
- Documentation requirements

**Ready for execution when Phase 1 is complete (or can be coordinated with Phase 1).**

