# Summary — In-Corpus Evaluation Dataset Generation System

## Overview

The In-Corpus Evaluation Dataset Generation System provides an end-to-end pipeline for generating evaluation datasets from Azure AI Search indexed documents. The system focuses on "In-Corpus" questions that can be answered from the indexed documents, generating complete evaluation datasets with ground truth retrieval context, system prompts, structured prompts, and RAG outputs.

## System Components

### 1. Query Generator (Phase 1)
- **Location**: `evaluations/{eval_name}/query_generator.py`
- **Purpose**: Generate diverse, effective queries from Azure AI Search indexed documents using LLM
- **Output**: `eval_inputs.json` with queries and metadata linking to source chunks

### 2. Dataset Generator (Phase 2)
- **Location**: `evaluations/{eval_name}/generate_eval_dataset.py`
- **Purpose**: Generate complete evaluation datasets from eval_inputs.json
- **Output**: `eval_dataset.json` with all required fields including:
  - Input queries
  - Retrieval queries
  - Ground truth retrieved context
  - System prompts
  - Structured prompts
  - RAG outputs
  - BEIR metrics
  - Placeholders for LLM and human evaluation fields

### 3. Output Generator (Phase 3)
- **Location**: `evaluations/{eval_name}/generate_eval_outputs.py`
- **Purpose**: Generate RAG outputs for evaluation dataset entries
- **Output**: Updated `eval_dataset.json` with generated outputs

### 4. System Prompt Management (Phase 4)
- **Location**: `infra/supabase/migrations/0016_add_system_prompts.sql`
- **Purpose**: Create and manage system prompts in Supabase with naming convention `{type}_system`
- **Output**: System prompts available in Supabase for use in dataset generation

### 5. Terminology Update (Phase 5)
- **Scope**: Codebase-wide
- **Purpose**: Update all `risk_impact` references to `risk_magnitude` for consistent terminology
- **Output**: Consistent terminology throughout codebase

## Directory Structure

```
evaluations/
  {eval_name}/
    query_generator.py
    generate_eval_dataset.py
    generate_eval_outputs.py
    create_system_prompts.py (optional)
    in_corpus/
      eval_inputs.json
      eval_dataset.json
```

## Documentation Organization

The documentation for this initiative follows a structured organization:

- **scoping/**: Initial scoping documents (PRD, RFC, TODO, context)
- **prompts/**: Phase execution prompts (prompt_phase_X_001.md)
- **intermediate/**: Intermediate documentation created during implementation:
  - phase_X_decisions.md - Decisions made during each phase
  - phase_X_testing.md - Testing documentation for each phase
  - phase_X_handoff.md - Handoff documentation between phases
  - fracas.md - Failure tracking document
- **summary.md**: This document - final summary of the initiative

## Key Features

1. **Automated Query Generation**: Uses LLM to generate diverse, effective queries from indexed documents
2. **Ground Truth Context**: Includes ground truth retrieved context for accurate BEIR metrics
3. **System Prompt Integration**: Queries Supabase for system prompts with naming convention
4. **Complete Dataset Generation**: Generates all required fields for comprehensive evaluation
5. **RAG Pipeline Integration**: Seamlessly integrates with existing RAG pipeline components
6. **BEIR Metrics**: Computes BEIR metrics using ground truth chunk IDs

## Integration Points

- **Azure AI Search**: For sampling chunks and retrieving context
- **Supabase**: For storing and querying system and structured prompts
- **RAG Pipeline**: For generating outputs using existing components
- **Evaluation System**: Generated datasets compatible with existing evaluation system

## Success Criteria

- ✅ Query generator produces diverse, effective queries from indexed documents
- ✅ Dataset generator creates complete evaluation datasets with all required fields
- ✅ Output generator successfully generates RAG outputs for dataset entries
- ✅ System prompts properly managed in Supabase with naming convention
- ✅ All `risk_impact` references updated to `risk_magnitude`
- ✅ All tests passing with >= 80% coverage
- ✅ End-to-end pipeline works with sample data

## Related Documents

- **Scoping Documents**: @docs/initiatives/in_corpus_eval_setup/scoping/
  - PRD001.md - Product requirements
  - RFC001.md - Technical architecture
  - TODO001.md - Implementation breakdown
  - context.md - Project context
- **Phase Prompts**: @docs/initiatives/in_corpus_eval_setup/prompts/
  - prompt_phase_0_001.md through prompt_phase_5_001.md
- **Intermediate Documentation**: @docs/initiatives/in_corpus_eval_setup/intermediate/
  - Phase-specific decisions, testing, and handoff documents

---
**Document Status**: Summary  
**Last Updated**: 2025-01-XX  
**Author**: Documentation Agent  
**Reviewers**: TBD

