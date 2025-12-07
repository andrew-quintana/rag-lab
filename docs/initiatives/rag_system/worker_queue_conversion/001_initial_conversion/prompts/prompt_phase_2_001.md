# Phase 2 Prompt — Queue Infrastructure

## Context

This prompt guides the implementation of **Phase 2: Queue Infrastructure** for the RAG Ingestion Worker–Queue Architecture Conversion. This phase implements Azure Storage Queue client utilities and message schema validation for the worker–queue architecture.

**Related Documents:**
- @docs/initiatives/rag_system/worker_queue_conversion/scoping/PRD001.md - Product requirements
- @docs/initiatives/rag_system/worker_queue_conversion/scoping/RFC001.md - Technical design (Queue Infrastructure section)
- @docs/initiatives/rag_system/worker_queue_conversion/scoping/TODO001.md - Implementation tasks (Phase 2 section - check off tasks as completed)
- @docs/initiatives/rag_system/worker_queue_conversion/scoping/context.md - Project context

## Objectives

1. **Azure Storage Queues**: Create and configure Azure Storage Queues
2. **Message Schema**: Implement message schema validation and serialization
3. **Queue Client Utilities**: Implement queue operations (enqueue, dequeue, peek, delete, dead-letter)
4. **Testing**: Create comprehensive unit tests for queue operations

## Execution Requirements

### Checklist Modifications
- **REQUIRED**: Check off completed tasks in @docs/initiatives/rag_system/worker_queue_conversion/scoping/TODO001.md Phase 2 section as you complete them
- Update phase status when all validation requirements are met

### Validation
- **REQUIRED**: All unit tests for Phase 2 must pass before proceeding to Phase 3
- **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/workers/test_queue_client.py -v`
- **REQUIRED**: Test coverage must meet minimum 80% for queue_client.py module
- **REQUIRED**: All test assertions must pass (no failures, no errors)

### Documentation
- **REQUIRED**: Create `phase_2_decisions.md` if any decisions are made that aren't in PRD/RFC
- **REQUIRED**: Create `phase_2_testing.md` documenting testing results
- **REQUIRED**: Create `phase_2_handoff.md` summarizing what's needed for Phase 3

## Key References

### Queue Infrastructure
- RFC001.md Queue Infrastructure section - Queue names and configuration
- RFC001.md Message Schema section - Message structure and field descriptions

### Azure Resources
- Azure Storage Account configuration
- Azure Storage Queue SDK documentation

### Data Structures
- `QueueMessage` dataclass with fields:
  - `document_id: str`
  - `source_storage: str` (enum: "azure_blob" | "supabase")
  - `filename: str`
  - `attempt: int`
  - `stage: str` (enum: "uploaded" | "parsed" | "chunked" | "embedded" | "indexed")
  - `metadata: Dict[str, Any]` (optional)

## Phase 2 Tasks

### Azure Storage Queues Setup
1. Create Azure Storage Queues (via Azure Portal or Infrastructure as Code):
   - `ingestion-uploads`
   - `ingestion-chunking`
   - `ingestion-embeddings`
   - `ingestion-indexing`
   - `ingestion-dead-letter` (optional but recommended)
2. Install Azure Storage Queue SDK: `pip install azure-storage-queue`
3. Verify queue creation and access

### Core Implementation
1. Create `rag_eval/services/workers/queue_client.py` module
2. Implement message schema:
   - Define `QueueMessage` dataclass
   - Implement `validate_message(message: dict) -> QueueMessage`
   - Implement `serialize_message(message: QueueMessage) -> str`
   - Implement `deserialize_message(message_str: str) -> QueueMessage`
3. Implement queue client utilities:
   - `enqueue_message(queue_name: str, message: QueueMessage, config) -> None`
   - `dequeue_message(queue_name: str, config) -> Optional[QueueMessage]`
   - `peek_message(queue_name: str, config) -> Optional[QueueMessage]`
   - `delete_message(queue_name: str, message_id: str, pop_receipt: str, config) -> None`
   - `send_to_dead_letter(queue_name: str, message: QueueMessage, reason: str, config) -> None`
   - `get_queue_length(queue_name: str, config) -> int`

### Testing
1. Create test file: `backend/tests/components/workers/test_queue_client.py`
2. Test message schema validation (valid/invalid schemas)
3. Test queue client operations (enqueue, dequeue, peek, delete)
4. Test error handling (queue not found, connection failures)
5. Test message serialization/deserialization
6. Test dead-letter queue handling
7. Test retry logic for transient failures
8. Test message visibility timeout handling
9. Test queue length monitoring

### Documentation
1. Add docstrings to all functions
2. Document message schema and field descriptions
3. Document queue configuration and setup
4. Document error handling and retry strategies

## Success Criteria

- [ ] Azure Storage Queues created and configured
- [ ] Message schema validation implemented
- [ ] All queue client utilities implemented
- [ ] All unit tests pass (minimum 80% coverage)
- [ ] All Phase 2 tasks in TODO001.md checked off
- [ ] Phase 2 handoff document created

## Important Notes

- **Queue Names**: Use exact queue names from RFC001.md
- **Message Schema**: Follow exact schema structure from RFC001.md Message Schema section
- **Error Handling**: Implement retry logic for transient Azure errors
- **Dead-Letter Queue**: Recommended for poison messages and unrecoverable errors

## Blockers

- **BLOCKER**: Phase 3 cannot proceed until Phase 2 validation complete
- **BLOCKER**: Azure Storage Account must be configured before queue creation

## Next Phase

After completing Phase 2, proceed to **Phase 3: Worker Implementation** using @docs/initiatives/rag_system/worker_queue_conversion/prompts/prompt_phase_3_001.md

