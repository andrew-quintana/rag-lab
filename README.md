# RAG Evaluation Platform

A minimal but correct RAG evaluation platform for assessing retrieval-augmented generation systems using Azure services and Supabase Postgres.

## Architecture

The platform consists of four independent but interoperable subsystems:

1. **RAG Pipeline (Python)** - Ingestion → chunking → embeddings → Azure AI Search retrieval → answer generation → trace logging
2. **Evaluator (Python)** - LLM-as-judge scoring of grounding, relevance, hallucination
3. **Meta-Evaluation (Python)** - Version-to-version performance comparison; judge stability/delta analysis
4. **Observability Dashboard (TypeScript)** - Lightweight client visualizing RAG/eval/meta-eval metrics

## Tech Stack

- **Backend**: Python 3.10+, FastAPI, Supabase Postgres
- **Frontend**: TypeScript, React, Vite
- **Infrastructure**: Supabase (local), Overmind (process management)
- **Azure Services**: AI Foundry (LLMs, embeddings), AI Search (vector retrieval), Blob Storage

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Docker and Docker Compose
- Supabase CLI: `brew install supabase/tap/supabase` (macOS)
- Overmind: `brew install overmind` (macOS) or see [Overmind docs](https://github.com/DarthSim/overmind)

### Setup

1. **Clone and install dependencies:**

   ```bash
   # Backend
   cd backend
   pip install -r requirements.txt
   
   # Frontend
   cd ../frontend
   npm install
   ```

2. **Configure environment:**

   ```bash
   # Copy environment template
   cp infra/dev.env.example .env
   
   # Edit .env with your Azure credentials and Supabase connection details
   ```

3. **Start Supabase locally:**

   ```bash
   cd infra/supabase
   supabase start
   ```

   Note the connection details from the output and update your `.env` file.

4. **Start all services:**

   ```bash
   cd infra
   make dev
   ```

   This starts:
   - Supabase (Postgres, API, Studio)
   - Backend API (http://localhost:8000)
   - Frontend dashboard (http://localhost:5173)

### Development Commands

From the `infra/` directory:

- `make dev` - Start all services via Overmind
- `make stop` - Stop all services
- `make restart` - Restart all services
- `make logs` - View logs from all processes
- `make supabase-start` - Start Supabase only
- `make supabase-stop` - Stop Supabase only
- `make reset-db` - Reset database and run migrations
- `make backend` - Start backend only
- `make frontend` - Start frontend only

## Project Structure

```
rag-evaluator/
├── backend/              # Python FastAPI backend
│   ├── rag_eval/        # Main package
│   │   ├── core/        # Shared primitives
│   │   ├── db/          # Database layer
│   │   ├── services/    # Business logic (RAG, evaluator, meta-eval)
│   │   ├── api/         # FastAPI routes
│   │   ├── prompts/     # Prompt templates
│   │   └── utils/       # Utilities
│   └── tests/           # Test suite
├── frontend/            # TypeScript React dashboard
│   ├── src/
│   │   ├── components/ # React components
│   │   ├── api/        # Backend API client
│   │   └── lib/        # Utilities
├── infra/              # Infrastructure and tooling
│   ├── supabase/       # Supabase config and migrations
│   ├── docker/         # Dockerfiles (optional)
│   └── Makefile        # Development automation
├── docs/               # Documentation
└── .cursor/            # Cursor IDE rules and context
```

## API Endpoints

- `POST /api/query` - Process a query through the RAG pipeline
- `GET /api/metrics` - Get aggregated metrics for dashboard
- `POST /api/meta_eval` - Trigger version-to-version comparison
- `GET /health` - Health check

## Design Principles

1. **Modularity** - Each subsystem is independent with clean interfaces
2. **Minimalism** - Only components required for minimal working version
3. **Traceability** - Every RAG step produces logs saved to Supabase
4. **Reproducibility** - System runs deterministically with identical inputs
5. **Strict Layering** - FastAPI → Services → DB, no cross-contamination

## Development Phase

This project is in **rapid prototyping** mode, focusing on:
- Correctness
- Deterministic behavior
- Minimal API surface area
- Fast iteration
- Simple deployment-free tooling

See `docs/initiatives/initial_setup/context.md` for the full scoping document.

## License

[Add license here]

