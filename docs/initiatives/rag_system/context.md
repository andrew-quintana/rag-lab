# Context — RAG System

## Purpose
The purpose of this document is to provide the context for generating a PRD, RFC, and TODO for the **RAG (Retrieval-Augmented Generation) system** within the broader platform.  
This RAG subsystem is the foundational component: it ingests documents, indexes them, retrieves relevant context, constructs prompts, queries an LLM, and logs results.
The RAG system **is a standalone module** with clean boundaries and no dependency on the evaluator, meta-evaluator, or dashboard logic.

---

## Scope

### In Scope
This system has **two distinct pipelines**, each with its own scope:

- **Upload Pipeline (`POST /upload`)** → Processes documents and builds the retrieval index.  
- **Query Pipeline (`POST /query`)** → Performs RAG retrieval + LLM generation.

Both pipelines run **entirely on the same server**, serialized, with no queues or background workers.

#### 📄 **1. Upload Pipeline (`POST /upload`)**  
Ingests a document and prepares it for retrieval.

0. **Document Upload**
   - Frontend provides a simple GUI to upload a document.
   - Backend receives the file directly (no storage queues).
   - File may be persisted to Azure Blob Storage, but all processing uses in-memory bytes.

1. **OCR & Text Extraction**
   - The same server immediately begins processing the uploaded file.
   - Document bytes are passed to **Azure Document Intelligence**.
   - Extract:
     - OCR text  
     - Layout  
     - Tables  
     - Block/line segmentation  
   - Output: unified extracted text.

2. **Chunking**
   - Locally chunk the extracted text using the app's chunking module.
   - Smart chunking heuristics (size-based, sentence-based, hybrid) handled locally.

3. **Embedding Generation**
   - Use **Azure AI Foundry embedding models** to embed each chunk.
   - Output: list of `(chunk, embedding, metadata)` objects.

4. **Indexing**
   - Store chunk embeddings + metadata as documents in **Azure AI Search**.
   - These become searchable vector documents for retrieval.

5. **Structured Logging**
   - Upload-related events are *not* sent to Supabase.
   - Log internal pipeline stats locally for debugging, for now.

**Output of Upload Pipeline:**  
A fully indexed document ready for retrieval during `/query`.

#### 🔎 **2. Query Pipeline (`POST /query`)**  
Retrieves relevant context and generates an answer.

0. **Receive Query**
   - Client calls `POST /query` with a natural-language question.

1. **Query Embedding**
   - Use Azure AI Foundry to embed the query into vector space.

2. **Retrieval**
   - Query Azure AI Search using vector similarity.
   - Retrieve top-k relevant chunks and associated metadata.

3. **Prompt Construction**
   - Load prompt templates from Supabase `prompt_versions` table by version name (e.g., "v1", "v2").
   - Assemble:
     - user query  
     - retrieved context  
     - system instructions from template
   - Produce final LLM-ready prompt.

4. **LLM Generation**
   - Use Azure AI Foundry (GPT-4o-mini or GPT-4o) to generate a grounded answer.

5. **Structured Logging (Supabase)**
   - Log:
     - query  
     - retrieved chunks  
     - model output  
     - evaluation scores (if present)  
     - metadata for observability  
   - Supabase is **only used for evaluation, meta-evaluation, and telemetry**.

6. **Return Response**
   - Respond with:
     - final answer  
     - retrieved context  
     - vector scores  
     - latency metrics  

### Out of Scope
(Not to be mentioned or implemented in PRD/RFC/TODO)

#### Other Components
- LLM-as-judge evaluator  
- Meta-evaluation or drift analyses  
- Observability dashboards  
- Multi-document ingestion  
- Reranking, hybrid retrieval, reranker models  
- Async pipelines, task queues, streaming  
- Authentication or multi-user features  
- Any advanced retrieval or optimization strategy

#### Failure Recovery and Robustness
The upload pipeline should have simple recovery like retry strategies for api calls and error reporting, but should be scaled down to a simple, serialized pipeline at first.
- Azure Storage Queues
- Modular step interfaces: Independent Azure Functions for each processing step
- Expand retry strategy with dead-letter storage, persistent retry queues, task resubmission, etc.
- Idempotency
- Logs in Azure Application Insights wiht distributed tracing and correlation IDs

The documentation must explicitly avoid scope creep into these areas.

---

## Goals

The RAG system must accomplish:

1. **Correctness**  
   - Given a query, the system retrieves appropriate context and produces grounded answers.

2. **Reproducibility**  
   - Chunking, embeddings, and retrieval are deterministic given the same inputs.

3. **Traceability**  
   - RAG pipeline operations are tracked internally; evaluation and meta-evaluation data is logged to Supabase Postgres.

4. **Modularity**  
   - Each subsystem is isolated, modular (in the serialized, local process):
     - ingestion  
     - chunking  
     - embeddings  
     - search  
     - generation (sub-components in scope are also isolated, modular)
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

## External Components

The RAG system uses:

* **Azure Blob Storage** → holds raw documents
* **Azure Document Intelligence** → parses documents (OCR, table extraction, text segmentation)
* **Azure AI Foundry** → performs chunking and generates embeddings for chunks and queries, and generates LLM answers
* **Azure AI Search** → vector index where chunks (with Azure AI Foundry embeddings) are stored as searchable documents

**Note:** Supabase Postgres is used exclusively for evaluation, meta-evaluation, and telemetry data for the observability dashboard. RAG pipeline operations are not logged to Supabase.

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
