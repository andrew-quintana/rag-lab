# CONTEXT.md — RAG Evaluation Harness (Notebook-driven + FAISS)

## 1. Overview
This project provides a lean, notebook-driven RAG evaluation stack operating on local files with FAISS indexing. The goal is to provide a simple yet comprehensive evaluation framework using:

1. **LLM-as-Judge** (primary evaluator)  
2. **Meta-Evaluator** (evaluates the judge)  
3. **BEIR-style Retrieval Metrics** (recall@k, nDCG, etc.)  
4. **Notebook-based workflow** for interactive evaluation and analysis
5. **Local file storage** replacing Azure/Supabase dependencies

This context will be fed into a documentation agent to generate PRD, RFC, TODO list, and implementation prompts.

---

## 2. Goals
- Create a **lean evaluation harness** to test RAG system variations with local storage.
- Provide **quantitative metrics** for:
  - Retrieval performance (BEIR)
  - Output correctness (LLM-as-Judge)
  - Judge reliability (Meta-Evaluator)
  - Insurance risk analysis (care avoidance vs unexpected cost)

- Stay **notebook-driven and simple**:
  - Jupyter notebooks for each phase (setup, indexing, retrieval eval, agent eval, analysis)
  - Local file-based storage (parquet, FAISS, JSONL)
  - Minimal external dependencies

---

## 3. Stakeholders
- **Primary user:** AI engineers experimenting with RAG evaluation using local datasets.
- **Builder:** Documentation agent and implementer.
- **Consumer of this file:** Documentation agent that will produce PRD/RFC/TODO.

---

## 4. Functional Requirements

### 4.1 LLM-as-Judge (Preserved from rag_evaluator)

- Architecture: **Deterministic Python script orchestrator** that calls LLM nodes sequentially
- Input:  
  - Query
  - Retrieved context (chunks)  
  - Model answer  
  - Reference answer  
- Process:
  1. **Correctness Classification Step**:  
     - Direct comparison: is the model answer correct or incorrect?
     - Returns: `correctness_binary` (true/false)
  
  2. **Hallucination Classification Step**:  
     - Grounding analysis: does the model answer contain information not supported by retrieved evidence?
     - Returns: `hallucination_binary` (true/false)
  
  3. **Risk Direction Classification Step** (if deviation detected):  
     - Evaluates entire RAG pipeline as black box
     - Analyzes whether deviation overestimates or underestimates costs
     - Returns: `risk_direction` with values:
       - **-1 = Care Avoidance Risk**: Model overestimated cost, potentially dissuading care
       - **+1 = Unexpected Cost Risk**: Model underestimated cost, potentially encouraging unaffordable care
       - **0 = No clear direction**
  
  4. **Risk Impact Calculation Step** (if deviation detected):  
     - Determines magnitude of real-world impact across time/money/steps
     - Returns: `risk_impact` (discrete values: 0, 1, 2, or 3)

- Output schema preserved from rag_evaluator:
  ```json
  {
    "correctness_binary": true | false,
    "hallucination_binary": true | false,
    "risk_direction": -1 | 0 | 1,
    "risk_impact": 0 | 1 | 2 | 3,
    "reasoning": string,
    "failure_mode": string
  }
  ```

### 4.2 Local Storage Architecture

- **Data structure**: 
  ```
  raglab/
    data/
      corpus.parquet          # Source documents
      tasks.jsonl            # Evaluation examples
      agent_tasks.jsonl      # Agent-specific tasks
    artifacts/
      embeddings.npy         # Document embeddings
      faiss.index           # FAISS search index
      docstore.parquet      # Chunk ID -> text mapping
    runs/
      {timestamp}_eval_run/
        config.yaml         # Run configuration
        outputs.jsonl       # Evaluation results
        metrics.json        # Summary metrics
        traces.jsonl        # Execution traces
  ```

- **Storage components**:
  - `src/io.py`: Load/save utilities for datasets and artifacts
  - `src/index.py`: FAISS indexing and retrieval
  - `src/eval.py`: Evaluation orchestration and result logging

### 4.3 Notebook Workflow

1. **00_setup.ipynb**: Environment setup and data loading
2. **01_ingest_and_index.ipynb**: Document chunking, embedding, and FAISS indexing
3. **02_retrieval_eval.ipynb**: BEIR-style retrieval evaluation
4. **03_agent_eval.ipynb**: End-to-end RAG agent evaluation with LLM-as-Judge
5. **04_analysis.ipynb**: Results analysis and visualization

### 4.4 Meta-Evaluator (Deterministic Validation)

- Implementation: **Deterministic Python function** (no LLM calls)
- Purpose: Validate LLM-as-Judge verdicts against ground truth
- Validation checks:
  - correctness_binary: Compare model answer to reference answer
  - hallucination_binary: Check if model answer is grounded in retrieved chunks
  - risk_direction: Validate risk direction against ground truth
  - risk_impact: Validate impact magnitude against ground truth
- Return: judge_correct/judge_incorrect with optional explanation

### 4.5 BEIR-style Retrieval Metrics

- Compute: Recall@k, Precision@k, nDCG@k
- Use evaluation examples with ground truth chunk IDs
- Store results in structured format for analysis

### 4.6 Insurance Risk Semantics (Preserved)

- Failure modes tied to insurance dimensions:
  - Copay, Coinsurance, Deductible
  - Out-of-pocket max
  - Eligibility / pre-authorization errors
- Risk direction semantics:
  - Care avoidance vs unexpected cost
  - System-level deviation analysis (not just hallucination)

---

## 5. Non-Functional Requirements

### 5.1 Stack Constraints
- **Python** for all evaluation components
- **FAISS** for vector similarity search
- **Jupyter notebooks** for interactive workflow
- **Local files** for all storage (no cloud dependencies)

### 5.2 Code Simplicity
- Everything runs **locally** with minimal dependencies
- Clear modular boundaries:
  - chunking (src/chunking.py)
  - indexing (src/index.py) 
  - evaluation (src/eval.py)
  - I/O utilities (src/io.py)
  - agent wrappers (src/agent.py)

### 5.3 Reproducibility
- Deterministic chunking and indexing
- Versioned evaluation runs with full configuration tracking
- All artifacts saved with timestamps and metadata

---

## 6. Milestones (Phased)

### Phase 1 — Core Infrastructure
- Implement src/ modules (io, index, eval, agent)
- Create notebook templates
- Establish data/artifacts/runs directory structure

### Phase 2 — Evaluation Components  
- Port and adapt judge/meta-evaluator from rag_evaluator
- Implement BEIR metrics computation
- Create evaluation orchestration pipeline

### Phase 3 — Notebook Workflow
- Develop interactive notebooks for each evaluation phase
- Add example datasets and sample runs
- Create analysis and visualization capabilities

### Phase 4 — Documentation and Examples
- Complete README with usage instructions
- Add example evaluation runs
- Document evaluation strategy and insurance risk semantics

---

## 7. Integration Points

- Embedding function: `Callable[[List[str]], np.ndarray]`
- Generation function: `Callable[[str, List[str]], str]`
- LLM function: `Callable[[str, float, int], str]`
- Retrieval API: `retrieve(query, k) -> List[RetrievalResult]`
- Evaluation API: `evaluate_dataset(examples, config) -> List[EvaluationResult]`

---

## 8. Key Adaptations from rag_evaluator

### Removed Dependencies
- Azure AI Search → FAISS
- Supabase → Local parquet files
- FastAPI → Jupyter notebooks
- Azure Functions → Local Python scripts

### Preserved Core Logic
- LLM-as-Judge evaluation strategy
- Insurance risk semantics and failure modes
- BEIR-style retrieval metrics
- Meta-evaluation and bias correction
- System-level deviation analysis

### Enhanced for Local Use
- File-based run management
- Notebook-driven interactive workflow
- Simplified deployment and setup
- Clear modular architecture