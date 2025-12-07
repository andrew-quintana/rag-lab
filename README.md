# RAG Evaluation Platform

A comprehensive platform for building, testing, and evaluating Retrieval-Augmented Generation (RAG) systems. Upload documents, ask questions, and get detailed metrics on answer quality, hallucination detection, and retrieval performance.

## What is This?

The RAG Evaluation Platform is a complete toolkit for RAG system development and evaluation. It provides:

- **A working RAG pipeline** - Upload documents, process them into searchable chunks, and answer questions using retrieved context
- **Comprehensive evaluation** - Measure answer correctness, detect hallucinations, assess risk, and evaluate retrieval quality
- **Full observability** - Track every query, retrieval, and evaluation result for analysis

Perfect for AI engineers experimenting with RAG systems, testing different retrieval strategies, prompt versions, or embedding models.

## What Can You Do?

### 1. Build and Test RAG Systems

**Upload Documents**
- Upload PDFs and other documents
- Automatic text extraction and chunking
- Vector embeddings generated and indexed in Azure AI Search
- Documents ready for querying in seconds
- Works with both Free and Standard Azure Document Intelligence tiers (automatic batch processing for Free tier)

**Ask Questions**
- Submit natural language queries
- System retrieves relevant document chunks
- LLM generates answers grounded in retrieved context
- Full trace of retrieval and generation process

**Experiment with Prompts**
- Store multiple prompt versions in the database
- Test different prompt strategies (v1, v2, etc.)
- No code changes needed - just update database records

### 2. Evaluate Answer Quality

**Correctness Evaluation**
- Compare model answers to reference answers
- Binary classification: correct or incorrect
- Understand when your system gets answers right

**Hallucination Detection**
- Detect when answers contain information not in retrieved context
- Strict grounding analysis against evidence only
- Identify potential misinformation risks

**Risk Assessment**
- Classify risk direction: care avoidance vs. unexpected cost
- Calculate impact magnitude (0-3 scale)
- Understand real-world consequences of errors

### 3. Measure Retrieval Performance

**BEIR-Style Metrics**
- **Recall@k**: How many relevant passages were found?
- **Precision@k**: How many retrieved passages were relevant?
- **nDCG@k**: Normalized discounted cumulative gain for ranking quality

**Judge Reliability**
- Meta-evaluator validates judge accuracy
- Understand when evaluation judgments are trustworthy
- Catch judge errors and inconsistencies
- **Bias-corrected accuracy estimates**: Corrects for imperfect judge specificity and sensitivity using the Rogan-Gladen adjustment method (inspired by [Lee et al., 2024](https://arxiv.org/pdf/2511.21140))
- **Confidence intervals**: Accounts for uncertainty from both test and calibration datasets

## How to Use It

### Quick Start

1. **Install Dependencies**

   ```bash
   # Backend
   cd backend
   pip install -r requirements.txt
   
   # Frontend (optional)
   cd ../frontend
   npm install
   ```

2. **Configure Environment**

   Create a `.env.local` file in the project root with your credentials. See `docs/setup/environment_variables.md` for a complete template.

   Minimum required variables:
   ```bash
   # Supabase (local or cloud)
   SUPABASE_URL=...
   SUPABASE_KEY=...
   DATABASE_URL=...

   # Azure Services
   AZURE_AI_FOUNDRY_ENDPOINT=...
   AZURE_AI_FOUNDRY_API_KEY=...
   AZURE_SEARCH_ENDPOINT=...
   AZURE_SEARCH_API_KEY=...
   AZURE_SEARCH_INDEX_NAME=...
   AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=...
   AZURE_DOCUMENT_INTELLIGENCE_API_KEY=...
   AZURE_BLOB_CONNECTION_STRING=...
   AZURE_BLOB_CONTAINER_NAME=...

   # Azure Storage Queues (for worker-queue architecture)
   AZURE_STORAGE_QUEUES_CONNECTION_STRING=...
   ```

3. **Start Supabase Locally**

   ```bash
   cd infra/supabase
   supabase start
   ```

   Copy the connection details from the output to your `.env.local` file.

4. **Start the Platform**

   ```bash
   # From project root
   make dev
   ```

   This starts:
   - Backend API at `http://localhost:8000`
   - Frontend dashboard at `http://localhost:5173` (if installed)
   - Supabase services (Postgres, API, Studio)

### Using the API

#### Upload a Document

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@your-document.pdf"
```

Response includes:
- Document ID
- Number of chunks created
- Processing statistics

#### Query the RAG System

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "text": "What is the copay for specialist visits?",
    "prompt_version": "v1"
  }'
```

Response includes:
- Generated answer
- Query ID for tracking
- Prompt version used

#### List Documents

```bash
curl http://localhost:8000/api/documents
```

#### Get Evaluation Metrics

```bash
curl http://localhost:8000/api/metrics
```

#### Run Meta-Evaluation

```bash
curl -X POST http://localhost:8000/api/meta_eval \
  -H "Content-Type: application/json" \
  -d '{
    "query_ids": ["query_123", "query_456"]
  }'
```

### Using Python Directly

You can also use the platform programmatically:

```python
from rag_eval.core.config import Config
from rag_eval.services.rag.pipeline import run_rag
from rag_eval.core.models import Query

# Load configuration
config = Config.from_env()

# Create a query
query = Query(text="What is the deductible?")

# Run RAG pipeline
answer = run_rag(query, prompt_version="v1", config=config)

print(f"Answer: {answer.text}")
print(f"Query ID: {answer.query_id}")
```

### Development Commands

From the project root:

- `make dev` - Start all services
- `make stop` - Stop all services
- `make restart` - Restart all services
- `make logs` - View logs from all processes
- `make backend` - Start backend only
- `make frontend` - Start frontend only
- `make reset-db` - Reset database and run migrations

## Architecture

The platform consists of four main subsystems:

1. **RAG System** - Document ingestion, chunking, embedding, vector search, and answer generation
2. **Evaluation System** - LLM-as-judge evaluation with correctness, hallucination detection, and risk assessment
3. **Meta-Evaluation** - Judge reliability validation
4. **Observability Dashboard** - Web interface for metrics and results (optional)

All systems are modular and can be used independently or together.

## Tech Stack

- **Backend**: Python 3.10+, FastAPI, Supabase Postgres
- **Frontend**: TypeScript, React, Vite (optional)
- **Infrastructure**: Supabase (local or cloud), Overmind (process management)
- **Azure Services**: 
  - AI Foundry (LLMs, embeddings)
  - AI Search (vector retrieval)
  - Document Intelligence (text extraction)

## Project Status

### ✅ RAG System - Complete
- 183 tests passing
- 88% code coverage
- All components tested and validated
- Production-ready

### ✅ Evaluation System - Complete
- 172+ tests passing
- 97%+ code coverage
- Full LLM-as-judge implementation
- Meta-evaluation and BEIR metrics
- Bias-corrected accuracy estimates with confidence intervals

## Testing

Run the test suite:

```bash
cd backend
source venv/bin/activate
pytest tests/ -v
```

With coverage report:

```bash
pytest tests/ --cov=rag_eval --cov-report=html
```

## Documentation

Comprehensive documentation is available in `docs/initiatives/`:

- **RAG System**: `docs/initiatives/rag_system/` - Complete implementation guide
- **Evaluation System**: `docs/initiatives/eval_system/` - Evaluation framework details
- **Initial Setup**: `docs/initiatives/initial_setup/` - Original scoping document

## Design Principles

1. **Modularity** - Each subsystem is independent with clean interfaces
2. **Testability** - Comprehensive test coverage (>80% achieved)
3. **Traceability** - Every operation is logged for analysis
4. **Reproducibility** - Deterministic behavior for reliable experimentation
5. **Correctness First** - All components validated with extensive testing

## References

**Bias Correction for LLM-as-a-Judge Evaluations**

The meta-evaluation system includes bias correction for LLM-as-a-Judge evaluations, implementing the method described in:

> Lee, C., Zeng, T., Jeong, J., Sohn, J., & Lee, K. (2024). How to Correctly Report LLM-as-a-Judge Evaluations. *arXiv preprint arXiv:2511.21140*.  
> https://arxiv.org/pdf/2511.21140

This approach uses the Rogan-Gladen adjustment to correct for imperfect judge specificity (q₀) and sensitivity (q₁), providing unbiased accuracy estimates and confidence intervals that account for uncertainty from both test and calibration datasets. The implementation is integrated into the meta-evaluation process and automatically computes bias-corrected accuracy estimates as part of the evaluation pipeline.

## License

[Add license here]
