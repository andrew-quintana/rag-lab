# context.md — RAG Ingestion Worker–Queue Architecture (Azure)

## 1. Purpose

This document defines how to convert the existing synchronous RAG ingestion flow into a **worker–queue-based pipeline** on Azure.

Goals:

- Keep the existing service modules (`ingestion.py`, `chunking.py`, `embeddings.py`, `search.py`, `storage.py`, `supabase_storage.py`, `logging.py`, etc.) as-is and **wrap them in workers**.   
- Introduce **queues + workers per stage** for modularity and failure isolation.  
- Respect **Azure Document Intelligence free-tier constraints** (2 pages per request) while maintaining constant throughput. :contentReference[oaicite:1]{index=1}  
- Keep the query-time RAG pipeline (`pipeline.py`) unchanged; this refactor is about **document ingestion** only. :contentReference[oaicite:2]{index=2}  

---

## 2. Existing Components (Code-Level)

These already exist and should not be reimplemented, only orchestrated by workers:

### 2.1 Document Ingestion (Parsing)

- **Module:** `ingestion.py`
- **Key function:** `extract_text_from_document(file_content: bytes, config) -> str`  
  - Uses **Azure Document Intelligence** (`DocumentIntelligenceClient`) with the `prebuilt-read` model. :contentReference[oaicite:3]{index=3}  
  - Includes helper logic to handle multi-page PDFs and a free-tier workaround that processes pages in **2-page batches** (`_batch_extract_pages`, `_extract_page_range`). :contentReference[oaicite:4]{index=4}  

### 2.2 Chunking

- **Module:** `chunking.py`
- **Key functions:**
  - `chunk_text_fixed_size(text, document_id, chunk_size, overlap) -> List[Chunk]`  
  - `chunk_text(text, config, document_id, chunk_size, overlap) -> List[Chunk]`  
- Provides **deterministic, fixed-size** chunking over raw text, with overlap and per-chunk metadata (start/end positions, `chunking_method`). :contentReference[oaicite:5]{index=5}  

### 2.3 Embeddings

- **Module:** `embeddings.py`
- **Key function:** `generate_embeddings(chunks: List[Chunk], config) -> List[List[float]]`  
  - Calls **Azure AI Foundry** (OpenAI-compatible) embedding API. :contentReference[oaicite:6]{index=6}  
  - Uses model like `"text-embedding-3-small"` from config.  
  - Handles batching, retry with exponential backoff, and dimension consistency checks. :contentReference[oaicite:7]{index=7}  

### 2.4 Vector Indexing + Retrieval

- **Module:** `search.py`
- **Key functions:**  
  - `index_chunks(chunks: List[Chunk], embeddings: List[List[float]], config) -> None`  
  - `retrieve_chunks(query, top_k, config) -> List[RetrievalResult]`  
- Manages **Azure AI Search** index:
  - Ensures index exists with correct schema + vector search configuration. :contentReference[oaicite:8]{index=8}  
  - Uploads documents in batches with retry and partial-failure handling. :contentReference[oaicite:9]{index=9}  

### 2.5 Storage

Two storage backends exist:

- **Azure Blob Storage** (`storage.py`): `upload_document_to_blob(...)`, etc. :contentReference[oaicite:10]{index=10}  
- **Supabase Storage** (`supabase_storage.py`): `upload_document_to_storage(...)`, `download_document_from_storage(...)`, etc. :contentReference[oaicite:11]{index=11}  

The worker–queue conversion assumes **uploaded files are already persisted** (Blob or Supabase) and referenced via a `document_id`.

### 2.6 Query-Time RAG Pipeline

- **Module:** `pipeline.py`
- **Key function:** `run_rag(query, prompt_version, config) -> ModelAnswer`  
  - Handles query embedding, retrieval, prompt construction, and LLM answer generation via Azure AI Foundry.   

This path stays intact; ingestion refactor should not break it.

---

## 3. Target Worker–Queue Architecture

### 3.1 High-Level Flow

Upload and ingestion should move from “one big synchronous call” to the following **pipeline**:

1. **Upload → Ingestion Queue**  
2. **Ingestion Worker**: Download file → Azure Document Intelligence → extracted text → update status → enqueue chunking job.  
3. **Chunking Worker**: Load extracted text → chunk → persist chunks → enqueue embedding job.  
4. **Embedding Worker**: Load chunks → Azure AI Foundry embeddings → persist embeddings → enqueue indexing job.  
5. **Indexing Worker**: Load chunks + embeddings → Azure AI Search `index_chunks` → update final status.   

Each stage uses **its own queue** for modularity and failure isolation.

### 3.2 Queues

Use **Azure Storage Queues** (backed by the same storage account as Blob) for simplicity.

Proposed queue names:

- `ingestion-uploads` — trigger ingestion worker
- `ingestion-chunking` — trigger chunking worker
- `ingestion-embeddings` — trigger embedding worker
- `ingestion-indexing` — trigger indexing worker
- (Optional) `ingestion-dead-letter` — for poison messages / unrecoverable errors

Each queue message is small and references existing persisted data via IDs, not raw bytes.

### 3.3 Message Schema

Use a consistent JSON-like payload structure across queues:

```json
{
  "document_id": "doc_123",
  "source_storage": "azure_blob|supabase",
  "filename": "policy.pdf",
  "attempt": 1,
  "stage": "uploaded|parsed|chunked|embedded|indexed",
  "metadata": {
    "tenant_id": "tenant_abc",
    "user_id": "user_123",
    "mime_type": "application/pdf"
  }
}
````

* **`document_id`**: primary key used throughout pipeline.
* **`source_storage`**: tells the worker whether to call `download_document_from_storage` (Supabase) or Blob APIs.
* **`stage`**: current processing stage for debugging & metrics.
* **`attempt`**: incremented on retries; used to decide when to dead-letter.

### 3.4 Workers

Each worker is a separate deployment (e.g., separate Azure Function or container) with a single responsibility.

#### 3.4.1 Ingestion Worker (Parsing)

* **Trigger:** `ingestion-uploads` queue message

* **Steps:**

  1. Resolve file location from `document_id` + `source_storage`.
  2. Download file (Supabase or Blob).
  3. Call `extract_text_from_document(file_content, config)` to get raw text. 

     * This internally handles 2-page batching and free-tier limits.
  4. Persist extracted text (e.g., in DB or storage keyed by `document_id`).
  5. Update `documents` table / metadata: `status = 'parsed'`, `parsed_at = now()`.
  6. Enqueue message to **`ingestion-chunking`** with same `document_id` and `stage = "parsed"`.

* **Failure behavior:**

  * Retries with exponential backoff for transient Azure errors (network, rate limits).
  * After N attempts, send to `ingestion-dead-letter` and mark `status = 'failed_parsing'`.

#### 3.4.2 Chunking Worker

* **Trigger:** `ingestion-chunking` queue message

* **Steps:**

  1. Load extracted text for `document_id` from storage/DB.
  2. Call `chunk_text(...)` with deterministic fixed-size config. 
  3. Persist chunks (e.g., in a `chunks` table or document store) with `document_id` and `chunk_id`.
  4. Update `documents.status = 'chunked'`.
  5. Enqueue message to **`ingestion-embeddings`** (typically a single message per document, not per chunk).

* **Failure behavior:**

  * Most failures will be data/validation issues; fail fast, mark `failed_chunking`, optionally dead-letter.

#### 3.4.3 Embedding Worker

* **Trigger:** `ingestion-embeddings` queue message

* **Steps:**

  1. Fetch all chunks for `document_id`.
  2. Call `generate_embeddings(chunks, config)` (handles batching + retries against Azure AI Foundry). 
  3. Persist embeddings alongside chunks or in a separate table keyed by `chunk_id`.
  4. Update `documents.status = 'embedded'`.
  5. Enqueue message to **`ingestion-indexing`**.

* **Failure behavior:**

  * Retry transient Azure Foundry failures with backoff (the function already has a retry wrapper).
  * On repeated failures, mark `failed_embeddings` and dead-letter.

#### 3.4.4 Indexing Worker

* **Trigger:** `ingestion-indexing` queue message

* **Steps:**

  1. Retrieve chunks + embeddings for `document_id`.
  2. Call `index_chunks(chunks, embeddings, config)` to push to Azure AI Search. 
  3. Handle partial failures (some docs failing to index) using the existing logic.
  4. Update `documents.status = 'indexed'`, `indexed_at = now()`.

* **Failure behavior:**

  * Retry Azure Search transient errors via existing `_retry_with_backoff`. 
  * After N attempts, mark `failed_indexing` and dead-letter.

---

## 4. Document Status & Idempotency

Maintain a `documents` table (Supabase or Azure DB) with a simple state machine:

* `uploaded`
* `parsed`
* `chunked`
* `embedded`
* `indexed`
* `failed_parsing`
* `failed_chunking`
* `failed_embeddings`
* `failed_indexing`

Workers should be **idempotent**:

* If a worker sees that the `documents.status` is already equal to or beyond its stage, it should **no-op** and log a warning.
* Chunking worker should be safe if called twice (either overwrite or detect existing chunks for `document_id`).
* Indexing can be repeated safely: `upload_documents` in Azure Search is already idempotent per document ID. 

---

## 5. Azure Resources to Add to the Resource Group

This section enumerates the Azure products needed to support the worker–queue architecture, assuming Supabase continues to host your relational DB + storage for some parts.

### 5.1 Core Compute / Orchestration

* **Azure Functions** (Consumption Plan)

  * 1 Function App hosting **four functions**:

    * `ingestion-worker` (queue trigger: `ingestion-uploads`)
    * `chunking-worker` (queue trigger: `ingestion-chunking`)
    * `embedding-worker` (queue trigger: `ingestion-embeddings`)
    * `indexing-worker` (queue trigger: `ingestion-indexing`)

*(Alternative: Azure Container Apps with queue-listening services, but Functions are leanest.)*

### 5.2 Storage & Queues

* **Azure Storage Account**

  * **Blob containers** (if you use Azure for raw document storage):

    * e.g., `documents`
  * **Queue Storage**:

    * `ingestion-uploads`
    * `ingestion-chunking`
    * `ingestion-embeddings`
    * `ingestion-indexing`
    * `ingestion-dead-letter` (optional, but recommended)

### 5.3 Cognitive & AI Services

* **Azure AI Document Intelligence** (or Cognitive Services resource with Document Intelligence enabled)

  * Used by `ingestion.py` via `DocumentIntelligenceClient` and `prebuilt-read` model. 

* **Azure AI Foundry (or Azure OpenAI in your subscription)**

  * Deployed model: `"text-embedding-3-small"` for embeddings. 
  * (Optionally) a chat/completions model for answer generation (already used by `generation.py`).

* **Azure AI Search**

  * Single Search service with:

    * Index defined and managed via `_ensure_index_exists` in `search.py`. 

### 5.4 Observability & Secrets (Recommended)

* **Azure Application Insights**

  * For metrics, traces, and log centralization from all workers.

* **Azure Key Vault**

  * Store sensitive config:

    * Azure Document Intelligence keys
    * Azure AI Foundry API keys
    * Azure AI Search keys
    * (Optionally) Supabase URL + key

You can keep `Config.from_env()` but wire env vars from Key Vault via Function App configuration.

---

## 6. Migration Plan (High-Level)

1. **Introduce queues and workers** *without* turning off the current synchronous ingestion path.
2. For new uploads, prefer enqueuing a message into `ingestion-uploads` instead of calling the whole pipeline synchronously.
3. Gradually move UI / API endpoints to rely on document `status` and show users ingestion progress.
4. Once stable, remove or deprecate any direct “do everything in one request” ingestion paths.

---

## 7. Non-Goals

* No change to **query-time RAG** logic (`run_rag` and related retrieval/generation code).
* No change to Supabase schema beyond possibly adding `status`/timestamp columns to `documents`.
* No attempt to optimize for multi-tenant or cross-region beyond using configurable `tenant_id` in metadata.
