# RAG Evaluation Platform — Codebase Scoping Document

## Purpose of This Document
This document defines the **initial codebase structure**, **module boundaries**, **interfaces**, **environment layout**, and **scaffolding requirements** for the Azure-based RAG Evaluation Platform.

It is written to enforce:
- a modular Python backend,
- a clean TypeScript frontend,
- a reproducible Supabase Postgres environment,
- strict separation of concerns,
- and a maintainable architecture that supports rapid prototyping.

This scoping document must be completed **before** implementation begins and serves as the source of truth for PRD, RFC, and TODO development.

---

## High-Level System Overview

The platform consists of four independent but interoperable subsystems:

1. **RAG Pipeline (Python)**  
   Ingestion → chunking → embeddings → Azure AI Search retrieval → answer generation → trace logging → result packaging.

2. **Evaluator (Python)**  
   LLM-as-judge scoring of grounding, relevance, hallucination, persisted for comparison.

3. **Meta-Evaluation (Python)**  
   Version-to-version performance comparison; judge stability/delta analysis; drift detection.

4. **Observability Dashboard (TypeScript)**  
   Lightweight client consuming backend `/metrics`, visualizing RAG/eval/meta-eval metrics.

Each subsystem must be isolated but consistent, sharing database storage, prompt versioning, and simple contracts.

---

## Design Principles

### **1. Modularity**
Each subsystem is an independent module under `backend/rag_eval/services/`, with clean interfaces and no cross-layer contamination.

### **2. Minimalism**
The project prioritizes correctness and clarity over abstraction.  
Only the components required for a minimal working version are included.

### **3. Traceability**
Every RAG step must produce logs saved to Supabase Postgres:
- queries,
- retrieval contexts,
- model answers,
- evaluator judgments,
- meta-eval summaries.

### **4. Reproducibility**
The system must run deterministically given identical inputs, prompt versions, and chunking logic.

### **5. Strict Layering**
- FastAPI contains *no business logic*.  
- Services contain *no database connection code*.  
- DB layer contains *no Azure calls*.  
- Frontend consumes API only.

---

## Development Phase
This phase is **rapid prototyping**, focusing on:
- correctness,
- deterministic behavior,
- minimal API surface area,
- fast iteration,
- and simple deployment-free tooling.

Out-of-scope features (auth, scaling, workers, async tasks, UI complexity, advanced chunkers, fine-grained access control, multiple models) must not be implemented during this phase.

---

## Codebase Structure (Insert Final Structure Here)

Paste your finalized directory structure in this section.

```
rag-eval-platform/
│
├── backend/
│   ├── rag_eval/                         # Python package
│   │   ├── __init__.py
│   │   │
│   │   ├── core/                         # Shared primitives across modules
│   │   │   ├── config.py                 # environment + settings loader
│   │   │   ├── logging.py                # structured logging helpers
│   │   │   ├── exceptions.py             # custom error types
│   │   │   └── interfaces.py             # common interface definitions
│   │   │
│   │   ├── db/                           # Supabase Postgres integration
│   │   │   ├── __init__.py
│   │   │   ├── connection.py             # DB connection + pool
│   │   │   ├── queries.py                # SQL wrappers for inserts/selects
│   │   │   └── models.py                 # (optional) lightweight dataclasses
│   │   │
│   │   ├── services/                     # Business logic modules
│   │   │   ├── rag/                      # RAG pipeline engine
│   │   │   │   ├── __init__.py
│   │   │   │   ├── ingestion.py
│   │   │   │   ├── chunking.py
│   │   │   │   ├── embeddings.py
│   │   │   │   ├── search.py
│   │   │   │   ├── generation.py
│   │   │   │   ├── logging.py
│   │   │   │   └── pipeline.py          # run_rag()
│   │   │   │
│   │   │   ├── evaluator/                # LLM-as-judge subsystem
│   │   │   │   ├── __init__.py
│   │   │   │   ├── evaluator.py
│   │   │   │   ├── scoring.py
│   │   │   │   └── judge_prompt.md
│   │   │   │
│   │   │   ├── meta_eval/                # Version drift + judge stability
│   │   │   │   ├── __init__.py
│   │   │   │   ├── compare_versions.py
│   │   │   │   ├── summarize.py
│   │   │   │   └── drift.py
│   │   │
│   │   ├── api/                          # FastAPI layer
│   │   │   ├── __init__.py
│   │   │   ├── routes/
│   │   │   │   ├── query.py              # POST /query
│   │   │   │   ├── metrics.py            # GET /metrics
│   │   │   │   └── meta_eval.py          # POST /meta_eval
│   │   │   └── main.py                   # FastAPI entrypoint
│   │   │
│   │   ├── prompts/
│   │   │   ├── prompt_v1.md
│   │   │   └── prompt_v2.md
│   │   │
│   │   └── utils/
│   │       ├── file.py
│   │       ├── timing.py
│   │       └── ids.py
│   │
│   ├── tests/
│   │   ├── test_rag_pipeline.py
│   │   ├── test_db.py
│   │   └── test_api.py
│   │
│   ├── requirements.txt
│   └── pyproject.toml                     # optional but recommended
│
├── frontend/                               # TypeScript dashboard (Next.js or Vite)
│   ├── src/
│   │   ├── components/
│   │   ├── pages/ (or /app/ routes)
│   │   ├── lib/
│   │   └── api/                            # fetches /metrics JSON
│   ├── public/
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts or next.config.js
│
├── infra/
│   ├── supabase/
│   │   ├── migrations/
│   │   │   └── 0001_init.sql
│   │   ├── seed/
│   │   │   └── demo_data.sql
│   │   ├── config.toml
│   │   └── README.md
│   │
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml (optional for local dev)
│   │
│   ├── Makefile
│   └── dev.env.example
│
├── docs/
│   ├── 01_scoping_document.md
│   ├── 02_PRD.md
│   ├── 03_RFC.md
│   └── 04_TODO.md
│
├── .cursor/
│   ├── rules/
│   │   ├── context.md
│   │   ├── state_of_development.md
│   │   ├── architecture_rules.md
│   │   └── scoping_document.md
│   └── config.json
│
└── README.md

```

---

## Backend Architecture Breakdown

### **1. Core Layer (`rag_eval/core/`)**

Shared primitives that multiple modules depend on:

* configuration loading (env vars),
* structured logging helpers,
* consistent exceptions,
* shared interface/type definitions.

### **2. Database Layer (`rag_eval/db/`)**

Contains:

* database connection logic (Supabase Postgres),
* centralized SQL queries,
* optional lightweight dataclasses for records.

No business logic or Azure logic belongs here.

### **3. Services Layer (`rag_eval/services/`)**

#### **a. RAG Service (`services/rag/`)**

Contains:

* ingestion (Azure Blob),
* chunking,
* embeddings (Azure Foundry),
* search (Azure AI Search),
* generation (Azure Foundry),
* logging,
* and the `run_rag()` pipeline function.

#### **b. Evaluator Service (`services/evaluator/`)**

Contains:

* judge prompt template,
* evaluator logic,
* scoring utilities.

#### **c. Meta-Evaluation Service (`services/meta_eval/`)**

Contains:

* version comparison logic,
* deltas computation,
* judge stability & drift metrics,
* summary-writing logic.

---

## API Layer

The API layer (`rag_eval/api/`) contains:

* FastAPI initialization (`main.py`),
* routers grouped by subsystem (`routes/`),
* no direct business logic.

Expected endpoints:

* `POST /query` → calls RAG pipeline,
* `POST /meta_eval` → triggers version comparison,
* `GET /metrics` → returns Supabase-backed dashboard metrics.

---

## Frontend Architecture

A TypeScript (Next.js or Vite) dashboard under `/frontend` consumes the backend `/metrics` endpoint.

Restrictions:

* single-page dashboard for metrics visualization,
* minimal UI,
* no routing complexity unless required,
* no auth or user management.

---

## Infrastructure Architecture

`infra/` holds all environment and devops tooling.

### **Supabase Directory**

Contains:

* migrations,
* seeds,
* config.toml,
* documentation for starting the local instance.

### **Docker Directory (Optional)**

Holds Dockerfiles or Compose definitions for future integration.

### **Makefile**

For:

* `make dev`,
* `make supabase-start`,
* `make reset-db`,
* `make backend`,
* and basic automation.

### **dev.env.example**

Documents all required env vars.

---

## Required Scaffolding Before Implementation

### ✔ Backend package structure created

### ✔ FastAPI app can start (even with placeholder routes)

### ✔ Supabase local instance starts successfully

### ✔ `schema.sql` initialized with required tables

### ✔ Directory structure for services, db, core, prompts established

### ✔ `.env.example` fully populated

### ✔ README includes high-level architecture

### ✔ Cursor rules and design docs in `.cursor/rules/`

---

## Success Criteria for Codebase Setup

1. `backend/` is a valid Python package with no import errors.
2. FastAPI boots in development mode with placeholder routes.
3. Supabase migrations run without modification.
4. The directory structure matches this scoping document.
5. Each module has clear boundaries and empty-but-documented entrypoints.
6. Frontend project compiles with `npm run dev` or `pnpm dev`.
7. The entire repo runs in "empty skeleton mode" without errors.

When these criteria are met, functional implementation may begin.

---

## Meta-Rule

If a feature or file does not directly support:

* RAG correctness,
* evaluator correctness,
* meta-eval correctness,
* or dashboard observability,

it must be postponed to a later phase.