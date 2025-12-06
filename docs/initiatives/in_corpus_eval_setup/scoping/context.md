# Context — In-Corpus Evaluation Dataset Generation System

## Purpose
The purpose of this document is to provide the context for generating a PRD, RFC, and TODO for the **In-Corpus Evaluation Dataset Generation System** within the broader platform.  
This system is a specialized component that generates evaluation datasets from Azure AI Search indexed documents, focusing on "In-Corpus" questions that can be answered from the indexed documents.

The system **integrates with existing RAG and evaluation systems** but operates as a standalone dataset generation tool.

---

## Scope

### In Scope
This system has **three main components**, each with its own scope:

- **Query Generator (`query_generator.py`)** → Generates diverse queries from Azure AI Search embeddings.  
- **Dataset Generator (`generate_eval_dataset.py`)** → Creates complete evaluation datasets with all required fields.  
- **Output Generator (`generate_eval_outputs.py`)** → Generates RAG outputs for evaluation dataset entries.

All components run **entirely locally**, using existing RAG pipeline components and Supabase for prompt storage.

#### 📝 **1. Query Generator (`query_generator.py`)**  
Generates diverse, effective queries from indexed documents.

0. **Sample Chunks**
   - Sample chunks from Azure AI Search index
   - Use embeddings to identify diverse content areas

1. **LLM Query Generation**
   - Use LLM to generate diverse queries based on chunk content
   - Ensure all queries are "In-Corpus" (answerable from indexed documents)
   - Generate queries that are effective for RAG evaluation

2. **Metadata Collection**
   - Link queries to source chunks for ground truth retrieval
   - Include document IDs and generation method
   - Save to `eval_inputs.json`

**Output of Query Generator:**  
A JSON file with diverse queries and metadata linking to source chunks.

#### 🔧 **2. Dataset Generator (`generate_eval_dataset.py`)**  
Creates complete evaluation datasets from eval_inputs.json.

0. **Load Inputs**
   - Load `eval_inputs.json` from same directory

1. **Retrieval Query Generation**
   - Generate retrieval_query (LLM call if sanitization needed, otherwise same as input)
   - Use retrieval_query for ground truth context retrieval

2. **Context Retrieval**
   - Retrieve context using retrieval_query (ground truth for BEIR)
   - Store retrieved chunks with similarity scores

3. **Prompt Loading**
   - Query Supabase for system_prompt (prompt_type with name="{type}_system")
   - Query Supabase for structured_prompt (prompt_type="RAG-enabled", live=TRUE)

4. **RAG Output Generation**
   - Generate output using RAG pipeline
   - Use system_prompt and structured_prompt

5. **BEIR Metrics Computation**
   - Compute BEIR metrics using ground truth chunk IDs from metadata
   - Use existing `compute_beir_metrics()` function

6. **Dataset Assembly**
   - Initialize LLM/human evaluation fields as null
   - Assemble complete evaluation dataset
   - Save to `eval_dataset.json`

**Output of Dataset Generator:**  
A complete evaluation dataset with all required fields for comprehensive RAG evaluation.

#### 🎯 **3. Output Generator (`generate_eval_outputs.py`)**  
Generates RAG outputs for evaluation dataset entries.

0. **Load Dataset**
   - Load `eval_dataset.json` from same directory

1. **Output Generation**
   - For each entry, generate output using:
     - `system_prompt` and `structured_prompt` from dataset
     - `retrieval_query` for retrieval
     - Retrieved context
   - Use existing RAG pipeline components

2. **Dataset Update**
   - Update `output` field in dataset
   - Save updated dataset

**Output of Output Generator:**  
Updated evaluation dataset with RAG outputs for all entries.

### Out of Scope
(Not to be mentioned or implemented in PRD/RFC/TODO)

#### Other Components
- Dashboard for viewing evaluation datasets  
- Real-time evaluation during RAG queries  
- Out-of-corpus query generation  
- Automated human evaluation collection  
- Evaluation dataset versioning system  
- Multi-document evaluation dataset generation

#### Advanced Features
- Query quality scoring or ranking
- Automated query diversity optimization
- Query generation from multiple document types
- Interactive query editing or refinement

---

## Adjacent Systems

### RAG System
- **Location**: `backend/rag_eval/services/rag/`
- **Key Components**:
  - `pipeline.py` - Main RAG orchestration (`run_rag()`)
  - `search.py` - Azure AI Search retrieval (`retrieve_chunks()`)
  - `generation.py` - LLM answer generation (`generate_answer()`)
  - `embeddings.py` - Query embedding generation
- **Integration**: Dataset generator uses RAG pipeline components for output generation

### Evaluation System
- **Location**: `backend/rag_eval/services/evaluator/`
- **Key Components**:
  - `orchestrator.py` - Evaluation pipeline orchestrator
  - `beir_metrics.py` - BEIR metrics computation
  - `meta_eval.py` - Meta-evaluator
- **Integration**: Generated datasets compatible with existing evaluation system

### Supabase Prompts
- **Location**: `infra/supabase/migrations/`
- **Key Tables**:
  - `prompts` - Prompt template storage with prompt_type, name, version, live fields
- **Integration**: Dataset generator queries Supabase for system and structured prompts

### Azure AI Search
- **Location**: Azure cloud service
- **Key Features**:
  - Vector search index with embeddings
  - Chunk storage with metadata
- **Integration**: Query generator samples chunks from index

---

## Design Principles

### **1. Automation**
All dataset generation steps are automated, requiring minimal manual intervention.

### **2. Completeness**
Generated datasets include all required fields for comprehensive evaluation.

### **3. Ground Truth**
Ground truth retrieval context included for accurate BEIR metrics computation.

### **4. Integration**
Seamless integration with existing RAG and evaluation systems without modification.

### **5. Flexibility**
Support for different evaluation scenarios via configuration and prompt selection.

---

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

---

## Key Data Structures

### eval_inputs.json
```json
[
  {
    "input": "string - query for RAG evaluation",
    "metadata": {
      "source_chunk_ids": ["chunk_1", "chunk_2"],
      "document_id": "doc_123",
      "generation_method": "ai_generated"
    }
  }
]
```

### eval_dataset.json
```json
[
  {
    "id": "eval_001",
    "content": {
      "input": "string - input for RAG",
      "retrieval_query": "string - sanitized/changed query",
      "context": {
        "retrieved_chunks": [...]
      },
      "system_prompt": "string - system prompt",
      "structured_prompt": "string - structured prompt",
      "output": "string - RAG output",
      "beir_metrics": {...},
      "llm_correctness": null,
      "llm_hallucination": null,
      "llm_risk_direction": null,
      "llm_risk_magnitude": null,
      "human_correctness": null,
      "human_hallucination": null,
      "human_risk_direction": null,
      "human_risk_magnitude": null
    }
  }
]
```

---

## Terminology Update

This system also includes a codebase-wide terminology update:
- `risk_impact` → `risk_magnitude` throughout codebase
- Affects interfaces, evaluator modules, database schemas, and all references

---
**Date**: 2025-01-XX  
**Status**: Planning  
**Related**: [PRD001.md](./PRD001.md), [RFC001.md](./RFC001.md), [TODO001.md](./TODO001.md)

