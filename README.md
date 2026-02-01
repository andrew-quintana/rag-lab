# RAGLab: Lean RAG Evaluation Framework

RAGLab is a notebook-driven RAG evaluation framework that provides comprehensive evaluation of retrieval-augmented generation systems using local file storage and FAISS indexing. Migrated and adapted from the rag_evaluator platform, RAGLab preserves sophisticated evaluation methodologies while eliminating cloud dependencies.

## Overview

RAGLab provides a complete evaluation stack for RAG systems with:
- **LLM-as-Judge** evaluation with insurance risk semantics
- **BEIR-style retrieval metrics** (Recall@K, Precision@K, nDCG@K)
- **Meta-evaluation** for judge reliability assessment
- **Notebook-driven workflow** for interactive evaluation
- **Local file storage** with no cloud dependencies

## Repository Structure

```
raglab/
├── notebooks/                    # Interactive evaluation workflow
│   ├── 00_setup.ipynb           # Environment setup and configuration
│   ├── 01_ingest_and_index.ipynb # Document chunking and FAISS indexing
│   ├── 02_retrieval_eval.ipynb   # BEIR-style retrieval evaluation
│   ├── 03_agent_eval.ipynb       # Complete RAG agent evaluation
│   └── 04_analysis.ipynb         # Results analysis and visualization
├── src/                          # Core evaluation modules
│   ├── core/                     # Core components
│   │   ├── interfaces.py         # Common data structures
│   │   ├── io.py                 # File I/O utilities
│   │   └── chunking.py           # Text chunking algorithms
│   ├── evaluation/               # Evaluation components
│   │   ├── beir_metrics.py       # BEIR evaluation metrics
│   │   ├── judge.py              # LLM-as-Judge orchestrator
│   │   ├── meta_evaluator.py     # Meta-evaluation components
│   │   └── *_evaluator.py        # Individual evaluators
│   ├── indexing/                 # Indexing and retrieval
│   │   └── index.py              # FAISS indexing and retrieval
│   ├── eval.py                   # Main evaluation pipeline
│   └── agent.py                  # Agent wrappers
├── data/                         # Source datasets
│   ├── corpus.parquet            # Document corpus
│   ├── tasks.jsonl               # Evaluation tasks
│   └── agent_tasks.jsonl         # Agent-specific tasks
├── artifacts/                    # Generated artifacts
│   ├── embeddings.npy            # Document embeddings
│   ├── faiss.index               # FAISS search index
│   └── docstore.parquet          # Chunk ID to text mapping
├── runs/                         # Timestamped evaluation results
│   └── YYYY-MM-DD_HHMM_run_name/
│       ├── config.yaml           # Run configuration
│       ├── outputs.jsonl         # Detailed results
│       ├── metrics.json          # Summary metrics
│       └── traces.jsonl          # Execution traces
├── docs/                         # Documentation and planning
│   ├── initiatives/              # Structured documentation
│   └── templates/                # Documentation templates
└── prompts/                      # Evaluation prompts
    └── evaluation/               # LLM-as-Judge prompts
```

**Where things live:** Data (`corpus.parquet`, `tasks.jsonl`, `agent_tasks.jsonl`) goes in `data/`. After ingest/index, artifacts (`embeddings.npy`, `faiss.index`, `docstore.parquet`) go in `artifacts/`. Each evaluation run creates `runs/<timestamp>_<name>/` with `config.yaml`, `outputs.jsonl`, `metrics.json`, `traces.jsonl`. Documentation: `docs/initiatives/` (e.g. `eval_harness/scoping/` for context, PRD001, RFC001, TODO001, summary, fracas) and `docs/templates/` (CONTEXT, PRD, RFC, TODO). LLM-as-Judge prompts can be loaded from disk by passing `prompts_dir` (e.g. `prompts/evaluation`) to the pipeline or `run_evaluation()`.

## Quick Start

### 1. Setup

```bash
# Install dependencies
pip install faiss-cpu pandas numpy matplotlib seaborn pyyaml

# Optional: Install specific embedding/LLM providers
pip install openai anthropic sentence-transformers
```

### 2. Configure Your Providers

Edit the embedding and LLM functions in the notebooks to use your preferred providers:

```python
# Example: OpenAI embeddings
def your_embedding_function(texts: list) -> np.ndarray:
    import openai
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=texts
    )
    return np.array([d.embedding for d in response.data])

# Example: OpenAI LLM
def your_llm_function(prompt: str, temperature: float = 0.1, max_tokens: int = 500) -> str:
    import openai
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

# Then import the modules:
from src.core.interfaces import EvaluationExample
from src.evaluation.beir_metrics import compute_beir_metrics
from src.indexing.index import RAGRetriever
from src.eval import run_evaluation
```

### 3. Run the Evaluation Workflow

Execute the notebooks in sequence:

1. **00_setup.ipynb** - Set up environment and create sample data
2. **01_ingest_and_index.ipynb** - Chunk documents and build FAISS index
3. **02_retrieval_eval.ipynb** - Evaluate retrieval performance with BEIR metrics
4. **03_agent_eval.ipynb** - Run complete RAG agent evaluation with LLM-as-Judge
5. **04_analysis.ipynb** - Analyze results and generate insights

## Evaluation Components

### LLM-as-Judge Evaluation

RAGLab implements a sophisticated multi-stage judge evaluation:

- **Correctness Classification**: Binary assessment of answer correctness
- **Hallucination Detection**: Grounding analysis against retrieved context
- **Risk Direction Assessment**: Insurance-specific risk classification
  - `-1`: Care avoidance risk (overestimated cost)
  - `+1`: Unexpected cost risk (underestimated cost) 
  - `0`: No clear risk direction
- **Risk Impact Calculation**: Discrete severity levels (0-3)

### BEIR-Style Retrieval Metrics

Standard information retrieval evaluation:
- **Recall@K**: Fraction of relevant documents retrieved
- **Precision@K**: Fraction of retrieved documents that are relevant
- **nDCG@K**: Normalized discounted cumulative gain

### Meta-Evaluation

Deterministic validation of LLM-as-Judge outputs:
- Judge reliability assessment
- Bias detection and correction
- Ground truth comparison

### Insurance Risk Semantics

Specialized evaluation for healthcare/insurance domains:
- Care avoidance vs unexpected cost analysis
- Cost estimation accuracy assessment
- Risk impact quantification

## Key Features

### Notebook-Driven Workflow
- Interactive development and analysis
- Step-by-step evaluation pipeline
- Rich visualizations and insights
- Easy experimentation and iteration

### Local-First Architecture
- No cloud dependencies required
- FAISS for efficient vector similarity search
- File-based storage (Parquet, JSONL, YAML)
- Complete data ownership and privacy

### Modular Design
- Clean separation of concerns
- Easy customization and extension
- Reusable components
- Clear interfaces between modules

### Comprehensive Evaluation
- Multi-faceted assessment (retrieval + generation + reliability)
- Domain-specific risk analysis
- Performance trend tracking
- Detailed failure analysis

## Advanced Usage

### Custom Evaluation Tasks

Create your own evaluation dataset:

```python
# data/custom_tasks.jsonl
{
  "example_id": "custom_001",
  "question": "Your evaluation question",
  "reference_answer": "Expected answer",
  "ground_truth_chunk_ids": ["chunk_1", "chunk_3"],
  "beir_failure_scale_factor": 1.0
}
```

### Custom Metrics

Extend the evaluation framework:

```python
from src.core.interfaces import EvaluationResult

def custom_metric(result: EvaluationResult) -> float:
    # Your custom evaluation logic
    return score

# Add to evaluation pipeline
```

### Batch Evaluation

Run evaluations with different retrieval_k or prompts_dir:

```python
from src.eval import run_evaluation

results, run_dir = run_evaluation(
    examples=examples,
    retriever=retriever,
    generator_function=your_generator_fn,
    llm_function=your_llm_fn,
    run_name="my_run",
    retrieval_k=5,
    prompts_dir="prompts/evaluation",
)
# Results and metrics are in run_dir (e.g. runs/2025-01-31_1200_my_run/)
```

## Documentation

### Evaluation Strategy
- LLM-as-Judge methodology preserved from rag_evaluator
- Insurance risk semantics and failure modes
- System-level deviation analysis approach
- Meta-evaluation and bias correction techniques

### Architecture Decisions
- Local storage rationale and trade-offs
- FAISS vs cloud search comparison
- Notebook workflow benefits
- Modular component design

### Initiative Documentation
See `docs/initiatives/eval_harness/` for comprehensive planning documents following the structured documentation methodology.

## Migration from rag_evaluator

RAGLab preserves the core evaluation methodology while simplifying deployment:

### What's Preserved
- ✅ Complete LLM-as-Judge evaluation pipeline
- ✅ Insurance risk semantics and analysis
- ✅ BEIR-style retrieval metrics
- ✅ Meta-evaluation and bias correction
- ✅ Sophisticated evaluation orchestration

### What's Changed
- 🔄 Azure AI Search → FAISS local indexing
- 🔄 Supabase → Parquet/JSONL file storage
- 🔄 FastAPI → Jupyter notebook workflow
- 🔄 Cloud deployment → Local execution
- 🔄 Complex setup → Minimal dependencies

## Contributing

1. Follow the existing code structure and interfaces
2. Add comprehensive docstrings and type hints
3. Update relevant notebooks when adding features
4. Follow the initiative documentation process in `docs/`
5. Test changes with multiple evaluation scenarios

## License

[Your License Here]

## Support

For questions, issues, or contributions:
1. Check existing evaluation runs in `runs/`
2. Review notebook outputs for debugging
3. Consult `docs/initiatives/` for methodology details
4. Open issues with specific evaluation scenarios