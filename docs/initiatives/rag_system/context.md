# Context — RAG System

## Purpose
The purpose of this document is to provide the minimum necessary context for generating a PRD, RFC, and TODO for the **RAG (Retrieval-Augmented Generation) system** within the broader platform.  
This RAG subsystem is the foundational component: it ingests documents, indexes them, retrieves relevant context, constructs prompts, queries an LLM, and logs results.

The documentation the agent produces must describe the RAG system **as a standalone module** with clean boundaries and no dependency on evaluator, meta-evaluator, or dashboard logic.

---

## Scope

### In Scope
The RAG system must implement the following functionality:

1. **Document Ingestion**  
   - Load a single document from Azure Blob Storage.  
   - Extract plaintext.  
   - Apply deterministic, fixed-size chunking.

2. **Embedding Generation**  
   - Use Azure AI Foundry embedding models to embed chunks and queries.

3. **Indexing & Retrieval**  
   - Store embeddings in Azure AI Search.  
   - Retrieve top-k relevant chunks for a user query.

4. **Prompt Construction**  
   - Use local prompt templates (e.g., `prompt_v1.md`, `prompt_v2.md`).  
   - Assemble context + query into LLM-ready prompts.

5. **LLM Generation**  
   - Use Azure AI Foundry to generate an answer from the constructed prompt.

6. **Structured Logging**  
   - Persist query logs, retrieval logs, and model responses to Supabase Postgres.  
   - Include chunk IDs, similarity scores, prompt version, and timestamps.

7. **API Surface**  
   - Expose a single FastAPI endpoint:  
     - `POST /query` → returns RAG answer + metadata.

### Out of Scope
(Not to be mentioned or implemented in PRD/RFC/TODO)

- LLM-as-judge evaluator  
- Meta-evaluation or drift analyses  
- Observability dashboards  
- Multi-document ingestion  
- Reranking, hybrid retrieval, reranker models  
- Async pipelines, task queues, streaming  
- Authentication or multi-user features  
- Any advanced retrieval or optimization strategy  

The documentation must explicitly avoid scope creep into these areas.

---

## Goals

The RAG system must accomplish:

1. **Correctness**  
   - Given a query, the system retrieves appropriate context and produces grounded answers.

2. **Reproducibility**  
   - Chunking, embeddings, and retrieval are deterministic given the same inputs.

3. **Traceability**  
   - Every step (query, retrieval, generation) is logged and versioned in Postgres.

4. **Modularity**  
   - Each subsystem is isolated:
     - ingestion  
     - chunking  
     - embeddings  
     - search  
     - generation  
     - logging  

5. **Minimal API + Minimal Dependencies**  
   - Only one public API route.
   - No unnecessary abstractions.

6. **Extendability**  
   - The design should allow:
     - new prompt versions  
     - new embedding models  
     - new indexing strategies  
   - But none should be implemented now.

---

## Required Interfaces

### Python API
The RAG system must expose:

```python
def run_rag(query: str, prompt_version: str) -> dict:
    """
    Executes the RAG pipeline end-to-end.
    Returns structured data containing:
    - final answer
    - retrieved chunks
    - prompt version
    - metadata for logging
    """
````

### FastAPI

Must expose:

```
POST /query
```

Body:

```json
{ "query": "...", "prompt_version": "v1" }
```

---

## Storage

The RAG system uses:

* **Azure Blob Storage** → document source
* **Azure AI Search** → embedding index + retrieval
* **Supabase Postgres** → logs (queries, retrieved chunks, model answers)

The documentation must identify the DB tables clearly but avoid evaluator/meta-eval schemas.

---

## Constraints

The RAG system must:

* use deterministic chunking,
* never implement multiple ingestion sources,
* avoid async complexities,
* use small, transparent components,
* minimize cloud usage for cost-friendliness,
* remain runnable locally with mock clients.
