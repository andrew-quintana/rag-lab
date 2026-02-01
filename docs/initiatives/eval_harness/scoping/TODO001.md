# TODO 001 — RAG Evaluation Harness Implementation

## Done
- [x] Repo layout: notebooks/, src/, data/, artifacts/, runs/, docs/, prompts/
- [x] interfaces.py (Query, Chunk, RetrievalResult, JudgeEvaluationResult, MetaEvaluationResult, BEIRMetricsResult)
- [x] io.py (DataLoader, RunManager; .npy, .parquet, .jsonl, run dirs)
- [x] index.py (EmbeddingProvider, LocalFAISSIndex, RAGRetriever; FAISS + docstore)
- [x] chunking.py (fixed-size; output to docstore via pipeline)
- [x] beir_metrics.py, judge (evaluate_answer_with_judge), meta_evaluator (meta_evaluate_judge + MetaEvaluator)
- [x] eval.py (RAGEvaluationPipeline, JudgeEvaluator, prompts_dir; write to runs/)
- [x] docs/initiatives/eval_harness/scoping (context, PRD001, RFC001, TODO001), summary
- [x] docs/templates (CONTEXT, PRD, RFC, TODO), _shared
- [x] README (layout, where data/artifacts/runs/docs live, how to run notebooks 00–04)

## Optional / Follow-up
- [ ] Add placeholder correctness_prompt.md, hallucination_prompt.md, risk_direction_prompt.md, risk_impact_prompt.md in prompts/evaluation/ (defaults in evaluators used if missing)
- [ ] Example corpus.parquet and tasks.jsonl in data/
- [ ] Notebooks 00–04 filled with minimal runnable cells
