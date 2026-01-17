# Async Batch Processing Refactor

**Date**: 2026-01-16  
**Status**: 📋 **PLANNING**  
**Goal**: Enable parallel processing where each batch flows through the pipeline independently

---

## Current Architecture (Sequential)

### Current Flow
```
Upload → Ingestion (ALL batches) → Chunking (entire doc) → Embedding (all chunks) → Indexing (all chunks)
```

**Problems**:
- ❌ Must wait for ALL 116 batches to complete before chunking starts
- ❌ No parallelization - each stage waits for previous to complete
- ❌ Timeout issues - ingestion takes too long
- ❌ No incremental progress - can't see chunks until all batches done

---

## Desired Architecture (Parallel/Streaming)

### Desired Flow
```
Batch 1: Ingestion → Chunking → Embedding → Indexing (starts immediately)
Batch 2: Ingestion → Chunking → Embedding → Indexing (parallel)
Batch 3: Ingestion → Chunking → Embedding → Indexing (parallel)
...
Batch N: Ingestion → Chunking → Embedding → Indexing (parallel)
```

**Benefits**:
- ✅ Each batch flows through pipeline independently
- ✅ Chunking can start as soon as batch 1 is extracted
- ✅ Embedding can start as soon as batch 1 is chunked
- ✅ Indexing can start as soon as batch 1 is embedded
- ✅ True parallelization across batches
- ✅ Faster time-to-first-chunk
- ✅ Better resource utilization

---

## Refactoring Approach

### Option 1: Batch-Level Messages (Recommended)

**Concept**: Each batch becomes a "batch document" that flows through the pipeline

**Message Structure**:
```python
{
    "document_id": "doc-123",
    "batch_index": 0,  # NEW: Which batch this is
    "batch_text": "...",  # NEW: Text for this batch only
    "start_page": 1,
    "end_page": 2,
    "stage": "parsed",  # or "chunked", "embedded"
    "source_storage": "supabase",
    "filename": "doc.pdf"
}
```

**Flow**:
1. **Ingestion**: Extract batch → Enqueue to chunking (per batch)
2. **Chunking**: Chunk batch text → Enqueue to embedding (per batch)
3. **Embedding**: Embed batch chunks → Enqueue to indexing (per batch)
4. **Indexing**: Index batch chunks → Mark batch complete

**Document Completion Tracking**:
- Track which batches are at which stage
- Document status = "indexed" when ALL batches are indexed
- Metadata tracks: `batches_indexed: {0: true, 1: true, ...}`

### Option 2: Incremental Merging

**Concept**: Keep current architecture but enqueue each batch as it completes

**Flow**:
1. **Ingestion**: Extract batch → Append to extracted_text → Enqueue to chunking (if first batch or enough text)
2. **Chunking**: Process available text → Create chunks → Enqueue to embedding
3. **Embedding**: Process available chunks → Enqueue to indexing
4. **Indexing**: Process available chunks → Mark complete

**Problems**:
- ❌ Chunking still needs full text (can't chunk partial text properly)
- ❌ Overlap between batches becomes complex
- ❌ Less clean separation

---

## Recommended: Option 1 - Batch-Level Messages

### Architecture Changes

#### 1. Ingestion Worker Changes

**Current**:
```python
# Process all batches
for batch in batches:
    extract_text(batch)
    merge_to_accumulator()

# After ALL batches done
persist_extracted_text(all_text)
update_status("parsed")
enqueue_to_chunking()
```

**New**:
```python
# Process batches one at a time
for batch in batches:
    batch_text = extract_text(batch)
    
    # Enqueue THIS batch to chunking immediately
    enqueue_batch_to_chunking(
        document_id=document_id,
        batch_index=batch_index,
        batch_text=batch_text,
        start_page=start_page,
        end_page=end_page
    )
    
    # Update progress
    update_batch_progress(document_id, batch_index, "extracted")
```

#### 2. Chunking Worker Changes

**Current**:
```python
# Load entire extracted text
extracted_text = load_extracted_text(document_id)
chunks = chunk_text(extracted_text)  # Chunks entire document
persist_chunks(document_id, chunks)
```

**New**:
```python
# Process single batch
batch_text = message.batch_text  # From message, not DB
batch_index = message.batch_index

# Chunk this batch's text
chunks = chunk_text(batch_text, document_id=document_id)

# Add batch metadata to chunks
for chunk in chunks:
    chunk.metadata["batch_index"] = batch_index
    chunk.metadata["start_page"] = message.start_page
    chunk.metadata["end_page"] = message.end_page

# Persist chunks for this batch
persist_chunks(document_id, chunks)

# Enqueue to embedding (per batch)
enqueue_batch_to_embedding(
    document_id=document_id,
    batch_index=batch_index,
    chunk_ids=[chunk.chunk_id for chunk in chunks]
)
```

#### 3. Embedding Worker Changes

**Current**:
```python
# Load all chunks
chunks = load_chunks(document_id)
embeddings = generate_embeddings(chunks)
persist_embeddings(document_id, chunks, embeddings)
```

**New**:
```python
# Process chunks for this batch
chunk_ids = message.chunk_ids  # From message
chunks = load_chunks_by_ids(document_id, chunk_ids)

# Generate embeddings for this batch
embeddings = generate_embeddings(chunks)
persist_embeddings(document_id, chunks, embeddings)

# Enqueue to indexing (per batch)
enqueue_batch_to_indexing(
    document_id=document_id,
    batch_index=message.batch_index,
    chunk_ids=chunk_ids
)
```

#### 4. Indexing Worker Changes

**Current**:
```python
# Load all chunks with embeddings
chunks = load_chunks(document_id)
embeddings = load_embeddings(document_id)
index_chunks(chunks, embeddings)
update_status("indexed")
```

**New**:
```python
# Process chunks for this batch
chunk_ids = message.chunk_ids
chunks = load_chunks_by_ids(document_id, chunk_ids)
embeddings = load_embeddings_by_ids(document_id, chunk_ids)

# Index this batch's chunks
index_chunks(chunks, embeddings)

# Mark batch as indexed
update_batch_progress(document_id, message.batch_index, "indexed")

# Check if ALL batches are indexed
if all_batches_indexed(document_id):
    update_document_status(document_id, "indexed")
```

---

## Implementation Plan

### Phase 1: Message Structure Changes

1. **Update QueueMessage**:
   ```python
   class QueueMessage:
       document_id: str
       batch_index: Optional[int] = None  # NEW
       batch_text: Optional[str] = None  # NEW (for chunking)
       chunk_ids: Optional[List[str]] = None  # NEW (for embedding/indexing)
       start_page: Optional[int] = None  # NEW
       end_page: Optional[int] = None  # NEW
       # ... existing fields
   ```

2. **Update ProcessingStage**:
   ```python
   class ProcessingStage:
       UPLOADED = "uploaded"
       BATCH_EXTRACTED = "batch_extracted"  # NEW
       BATCH_CHUNKED = "batch_chunked"  # NEW
       BATCH_EMBEDDED = "batch_embedded"  # NEW
       BATCH_INDEXED = "batch_indexed"  # NEW
       # Keep existing for backward compatibility
   ```

### Phase 2: Ingestion Worker Refactor

**File**: `backend/src/services/workers/ingestion_worker.py`

**Changes**:
1. After each batch extraction, enqueue to chunking immediately
2. Remove "merge all batches" step
3. Remove "persist extracted text" step (or make it optional for backward compat)
4. Track batch progress in metadata

**Key Code Change**:
```python
# After extracting batch
batch_text = extract_text_from_batch(batch_pdf, page_range, config)

# Enqueue THIS batch to chunking immediately
batch_message = QueueMessage(
    document_id=document_id,
    batch_index=batch_index,
    batch_text=batch_text,  # Include text in message
    start_page=start_page,
    end_page=end_page,
    stage=ProcessingStage.BATCH_EXTRACTED,
    source_storage=source_storage,
    filename=filename,
    attempt=1,
    metadata={"batch_size": batch_size}
)
enqueue_message("ingestion-chunking", batch_message, config)

# Update batch progress
update_batch_progress(document_id, batch_index, "extracted")
```

### Phase 3: Chunking Worker Refactor

**File**: `backend/src/services/workers/chunking_worker.py`

**Changes**:
1. Accept `batch_text` from message (not load from DB)
2. Chunk only this batch's text
3. Add batch metadata to chunks
4. Enqueue to embedding per batch

**Key Code Change**:
```python
# Get batch text from message
if message.batch_text:
    batch_text = message.batch_text
    batch_index = message.batch_index
else:
    # Fallback: load from DB (backward compatibility)
    extracted_text = load_extracted_text(document_id)
    batch_text = extracted_text
    batch_index = None

# Chunk this batch
chunks = chunk_text(batch_text, document_id=document_id, ...)

# Add batch metadata
for chunk in chunks:
    if batch_index is not None:
        chunk.metadata["batch_index"] = batch_index
        chunk.metadata["start_page"] = message.start_page
        chunk.metadata["end_page"] = message.end_page

# Persist chunks
persist_chunks(document_id, chunks, config)

# Enqueue to embedding (per batch)
embedding_message = QueueMessage(
    document_id=document_id,
    batch_index=batch_index,
    chunk_ids=[chunk.chunk_id for chunk in chunks],
    stage=ProcessingStage.BATCH_CHUNKED,
    ...
)
enqueue_message("ingestion-embeddings", embedding_message, config)
```

### Phase 4: Embedding Worker Refactor

**File**: `backend/src/services/workers/embedding_worker.py`

**Changes**:
1. Accept `chunk_ids` from message
2. Load only chunks for this batch
3. Generate embeddings for this batch
4. Enqueue to indexing per batch

**Key Code Change**:
```python
# Get chunk IDs from message
if message.chunk_ids:
    chunk_ids = message.chunk_ids
    chunks = load_chunks_by_ids(document_id, chunk_ids)
else:
    # Fallback: load all chunks (backward compatibility)
    chunks = load_chunks(document_id)

# Generate embeddings
embeddings = generate_embeddings(chunks, config)

# Persist embeddings
persist_embeddings(document_id, chunks, embeddings, config)

# Enqueue to indexing (per batch)
indexing_message = QueueMessage(
    document_id=document_id,
    batch_index=message.batch_index,
    chunk_ids=chunk_ids,
    stage=ProcessingStage.BATCH_EMBEDDED,
    ...
)
enqueue_message("ingestion-indexing", indexing_message, config)
```

### Phase 5: Indexing Worker Refactor

**File**: `backend/src/services/workers/indexing_worker.py`

**Changes**:
1. Accept `chunk_ids` from message
2. Load only chunks for this batch
3. Index this batch's chunks
4. Check if all batches complete

**Key Code Change**:
```python
# Get chunk IDs from message
chunk_ids = message.chunk_ids
chunks = load_chunks_by_ids(document_id, chunk_ids)
embeddings = load_embeddings_by_ids(document_id, chunk_ids)

# Index this batch
index_chunks(chunks, embeddings, config)

# Mark batch as indexed
update_batch_progress(document_id, message.batch_index, "indexed")

# Check if all batches complete
if all_batches_indexed(document_id):
    update_document_status(document_id, "indexed", timestamp_field="indexed_at", config=config)
```

### Phase 6: Persistence Layer Changes

**New Functions Needed**:

1. **Batch Progress Tracking**:
   ```python
   def update_batch_progress(
       document_id: str,
       batch_index: int,
       stage: str,  # "extracted", "chunked", "embedded", "indexed"
       config
   ) -> None:
       """Update batch progress in metadata"""
       # Update documents.metadata->ingestion->batches_progress
   ```

2. **Load Chunks by IDs**:
   ```python
   def load_chunks_by_ids(
       document_id: str,
       chunk_ids: List[str],
       config
   ) -> List[Chunk]:
       """Load specific chunks by their IDs"""
   ```

3. **Check Completion**:
   ```python
   def all_batches_indexed(document_id: str, config) -> bool:
       """Check if all batches for document are indexed"""
       # Check metadata->ingestion->batches_indexed
   ```

---

## Database Schema Changes

### Metadata Structure

**Current**:
```json
{
  "ingestion": {
    "num_pages": 232,
    "batches_completed": {"0": true, "1": true},
    "parsing_status": "in_progress"
  }
}
```

**New**:
```json
{
  "ingestion": {
    "num_pages": 232,
    "num_batches_total": 116,
    "batches_progress": {
      "0": {"extracted": true, "chunked": true, "embedded": true, "indexed": true},
      "1": {"extracted": true, "chunked": true, "embedded": false, "indexed": false},
      ...
    },
    "parsing_status": "in_progress",
    "chunking_status": "in_progress",
    "embedding_status": "in_progress",
    "indexing_status": "in_progress"
  }
}
```

---

## Benefits

### Performance
- ✅ **Faster time-to-first-chunk**: Chunking starts as soon as batch 1 is extracted
- ✅ **True parallelization**: Multiple batches processing simultaneously
- ✅ **Better resource utilization**: All workers can be busy simultaneously
- ✅ **No timeout issues**: Each batch is small and fast

### Reliability
- ✅ **Finer-grained retries**: Retry individual batches, not entire document
- ✅ **Partial completion**: Can see progress as batches complete
- ✅ **Better error isolation**: One batch failure doesn't block others

### Observability
- ✅ **Real-time progress**: See batches completing in real-time
- ✅ **Better monitoring**: Track each batch through pipeline
- ✅ **Easier debugging**: Know exactly which batch failed

---

## Migration Strategy

### Step 1: Add Backward Compatibility
- Support both old (full document) and new (per-batch) message formats
- Workers check message format and handle accordingly

### Step 2: Deploy New Code
- Deploy updated workers that support both formats
- Old documents continue to work

### Step 3: Update Ingestion
- Update ingestion worker to send per-batch messages
- New documents use new format

### Step 4: Remove Old Format (Optional)
- After validation, remove backward compatibility code

---

## Testing Plan

1. **Unit Tests**: Test each worker with batch-level messages
2. **Integration Tests**: Test full pipeline with batch-level flow
3. **E2E Test**: Upload large document, verify batches process in parallel
4. **Performance Test**: Compare sequential vs parallel processing times

---

## Estimated Impact

### Time Savings
- **Current**: 20-30 minutes for 232-page document (sequential)
- **New**: ~5-10 minutes (parallel batches)
- **Improvement**: 2-3x faster

### Resource Utilization
- **Current**: One worker active at a time
- **New**: All 4 workers can be active simultaneously
- **Improvement**: 4x better utilization

---

**Next Steps**: 
1. Review and approve approach
2. Implement Phase 1 (message structure)
3. Implement Phase 2-5 (worker refactors)
4. Test with small document
5. Test with large document (232 pages)
6. Deploy and monitor
