# Phase 0 Decisions - Scoping & Documentation Updates

**Date**: 2025-12-07  
**Phase**: 0 - Scoping & Documentation Updates  
**Status**: ✅ Complete

## Overview

This document records decisions made during Phase 0 of Initiative 002: Codebase Consolidation & Production Readiness. Phase 0 focused on updating foundational cursor rules files to reflect the worker-queue architecture transformation implemented in Initiative 001.

---

## Decisions Made

### 1. Architecture Rules Updates

**Decision**: Update `.cursor/rules/architecture_rules.md` to include worker-queue architecture layer.

**Rationale**: The architecture rules file previously described the system as synchronous and did not include worker infrastructure. Since Initiative 001 successfully implemented the worker-queue architecture, the rules must accurately reflect the current system.

**Changes Made**:
- Added worker infrastructure section (`rag_eval/services/workers/`)
- Added worker execution layer (Azure Functions)
- Documented queue-based communication patterns
- Updated development workflow to include local Azure Functions development with Azurite
- Added Azure Storage Queues to infrastructure components

**Files Modified**: `.cursor/rules/architecture_rules.md`

---

### 2. Scoping Document Updates

**Decision**: Update `.cursor/rules/scoping_document.md` to include worker-queue architecture in system overview and codebase structure.

**Rationale**: The scoping document serves as foundational documentation for the codebase structure. It must accurately reflect the worker-queue architecture and Azure Functions deployment structure.

**Changes Made**:
- Updated "High-Level System Overview" to include worker-queue architecture
- Added worker infrastructure to codebase structure diagram (`rag_eval/services/workers/`)
- Documented Azure Functions deployment structure (`infra/azure/azure_functions/`)
- Updated "Out of Scope" section (removed workers from out-of-scope list)
- Added Azure Storage Queues to infrastructure architecture section
- Updated database scope to include worker status tracking (migrations 0019, 0020)
- Updated "Required Scaffolding" section to include Azure Functions setup
- Updated Meta-Rule to include worker-queue architecture

**Files Modified**: `.cursor/rules/scoping_document.md`

---

### 3. State of Development Updates

**Decision**: Update `.cursor/rules/state_of_development.md` to reflect worker-queue architecture is implemented.

**Rationale**: The state of development file must accurately describe the current phase and what is in scope. Workers are no longer out of scope and must be reflected in the development workflow.

**Changes Made**:
- Updated "CURRENT PHASE" to reflect worker-queue architecture is implemented
- Removed workers from "OUT OF SCOPE FOR THIS PHASE" section
- Added worker-queue architecture to system description
- Updated "PRIMARY OBJECTIVES IN THIS PHASE" to include worker-queue architecture
- Updated "LOCAL DEVELOPMENT WORKFLOW" to include Azure Functions local development
- Added validation gates for worker-queue functionality
- Updated meta-rule to reflect workers are now part of the system
- Documented local development with Azurite and local Supabase

**Files Modified**: `.cursor/rules/state_of_development.md`

---

### 4. Fracas Document Creation

**Decision**: Create `fracas.md` in the root initiative directory for failure tracking.

**Rationale**: Initiative 002 needs its own failure tracking document to track issues discovered during consolidation work. Known issues from Initiative 001 (FM-001, FM-005) are documented for reference.

**Changes Made**:
- Created `fracas.md` with proper structure
- Documented known issues from Initiative 001 (FM-001, FM-005)
- Set up tracking format for new issues discovered during consolidation
- Included severity levels and status tracking

**Files Created**: `docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/fracas.md`

---

## Clarifications Needed

### Environment Variable Loading Strategy Update

**Decision**: Updated environment variable loading strategy to be more flexible and context-aware.

**Rationale**: The original approach hardcoded `.env.local` which doesn't work well for:
- Azure Functions in cloud (use Azure Function App settings via `os.environ`)
- Unit tests (should set environment variables directly, not depend on files)

**Changes Made**:
- Updated architecture rules to describe flexible environment variable loading
- Updated scoping document to include environment variable loading strategy section
- Updated state of development to reflect flexible approach
- Updated RFC002 to reflect flexible precedence order
- Updated context.md to reflect flexible approach

**Implementation Pattern**:
- Function entry points check for `.env.local` and load it if it exists (for local development)
- But don't require it - Azure settings and test fixtures take precedence
- Precedence: Azure Function App settings > `.env.local` (when loaded) > system environment > test fixtures

**Files Modified**:
- `.cursor/rules/architecture_rules.md`
- `.cursor/rules/scoping_document.md`
- `.cursor/rules/state_of_development.md`
- `scoping/RFC002.md`
- `scoping/context.md`

---

## Structural Changes

### Cursor Rules Files

**No structural changes** were made to the cursor rules files. Updates were additive and focused on:
- Adding worker-queue architecture descriptions
- Updating out-of-scope sections
- Adding new sections for worker infrastructure
- Updating existing sections to reflect current state

**Format Consistency**: All updates maintain consistency with existing document structure and formatting.

---

## File Organization Clarifications

### Directory Structure

The file organization requirements in TODO002.md Section 0.2 are clear and complete:
- Directory purposes are clearly defined
- Naming conventions are specified
- Examples are provided for file placement

**No clarifications needed** - the file organization requirements are sufficient for Phase 1 and beyond.

---

## Next Steps

1. **Phase 1**: Code Duplication Elimination
   - Update function entry points to import from project root
   - Remove duplicate backend code
   - Simplify build script

2. **Phase 2**: Configuration Consolidation
   - Unify environment variable loading strategy
   - Standardize configuration files

3. **Phase 3**: Test Infrastructure Consolidation
   - Register pytest markers
   - Consolidate test fixtures

4. **Phase 4**: Production Readiness Validation
   - Complete Phase 5 testing
   - Resolve identified issues

---

**Last Updated**: 2025-12-07  
**Status**: ✅ Phase 0 Complete

