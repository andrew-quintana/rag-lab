# CONTEXT.md — RAG Evaluation MVP (LLM-as-Judge + BEIR + Meta-Eval)

## 1. Overview
This project provides an experimental evaluation stack for RAG systems operating on insurance-related documents. The goal is to measure how retrieval changes and prompt changes affect downstream performance using:

1. **LLM-as-Judge** (primary evaluator)  
2. **Meta-Evaluator** (evaluates the judge)  
3. **BEIR-style Retrieval Metrics** (recall@k, nDCG, etc.)  
4. **A small fixture-based evaluation dataset** derived from:  
   `backend/tests/fixtures/sample_documents/healthguard_select_ppo_plan.pdf`

This context will be fed into a documentation agent to generate PRD, RFC, TODO list, and implementation prompts.

---

## 2. Goals
- Create an **experimental harness** to test RAG system variations (retrieval method, chunking, generation prompt).
- Provide **quantitative metrics** for:
  - Retrieval performance (BEIR)
  - Output correctness (LLM-as-Judge)
  - Judge reliability (Meta-Evaluator)

- Stay **MVP-simple**:
  - No dashboard (deferred)
  - Minimal dataset (one document → synthetic QA pairs)
  - Azure stack only where necessary

---

## 3. Stakeholders
- **Primary user:** AI engineers experimenting with retrieval and generation performance using their own corpora.
- **Builder:** You (solo).
- **Consumer of this file:** documentation agent that will produce PRD/RFC/TODO.

---

## 4. Functional Requirements

### 4.1 LLM-as-Judge
- Model: **GPT-4o mini** (Azure Foundry)
- Input:  
  - Retrieved context  
  - Model answer  
  - Reference answer  
- Output:
  - **hallucination_impact**: integer 0–3  
    - `0`: No hallucination  
    - `1`: Low impact  
    - `2`: Medium impact  
    - `3`: High impact  
  - If impact > 0 → classify **failure mode** (e.g., cost misstatement, omitted deductible, incorrect coverage rule)

### 4.2 Hallucination Impact Calculation
- Based on **expected financial cost** of the hallucination.
- Impact computed relative to a **default U.S. average income** (set as default; configurable).
- Failure modes tied to **typical insurance cost categories**:
  - Copay  
  - Coinsurance  
  - Deductible  
  - Out-of-pocket max  
  - Eligibility / pre-authorization errors

### 4.3 Meta-Evaluator
- Model: **GPT-4o** (Azure Foundry)
- Purpose:
  - Evaluate whether the LLM-as-Judge’s verdict is itself reasonable.
  - Return:
    - judge_correct / judge_incorrect (binary)
    - optional explanation (short)

### 4.4 BEIR-style Retrieval Metrics
- Compute:
  - Recall@k
  - Precision@k
  - nDCG@k
- Use synthetic QA pairs generated from the fixture document.

### 4.5 Evaluation Dataset
- Built from:  
  `backend/tests/fixtures/sample_documents/healthguard_select_ppo_plan.pdf`
- Contains:
  - synthetic questions  
  - gold reference answers  
  - ground-truth chunks or paragraphs for retrieval relevance

---

## 5. Non-Functional Requirements

### 5.1 Stack Constraints
- **Python** for all evaluation components.
- **Azure Foundry** for all LLM calls.
- **No dashboards** in MVP (deferred).

### 5.2 Code Simplicity
- Everything should run **locally** and be easy to reason about.
- Minimal dependencies.
- Clear boundaries:
  - retriever
  - RAG answer generator
  - judge
  - meta-judge
  - BEIR evaluator

### 5.3 Reproducibility
- Deterministic fixtures.
- All example inputs should be inside `backend/tests/fixtures/...`.

---

## 6. Milestones (Phased)

### Phase 1 — Setup & Planning
- Review existing codebase.
- Define directory layout for:
  - evaluation scripts  
  - judge/meta-judge pipelines  
  - BEIR metrics

### Phase 2 — Dataset Construction
- Generate synthetic QA pairs from the fixture PDF.
- Annotate ground-truth passages for retrieval evaluation.

### Phase 3 — LLM-as-Judge Implementation
- Implement grounding + hallucination impact classifier.
- Implement failure mode tagging.

### Phase 4 — Meta-Evaluator
- Implement correctness check for judge outputs.
- Add minimal explanations.

### Phase 5 — BEIR Metrics Integration
- Implement retrieval scoring using generated dataset.
- Compute top-k metrics.

### Phase 6 — Integration & Tests
- Full end-to-end pipeline test:
  - Retrieval → RAG generation → LLM-as-Judge → Meta-Eval → Metrics
- Add unit tests for each layer.

---

## 7. Key Edge Cases & Open Questions

### Edge Cases
- Retrieval returns zero relevant passages.
- Judge incorrectly marks grounded answers as hallucinations.
- High-variance outputs from GPT-4o mini causing inconsistent judgments.
- Cost hallucinations where financial impact cannot be computed.

### Open Questions
- Should failure modes eventually map to a numeric risk score for analytics?
- Should we include confidence intervals for judge outputs (multiple samples)?
- Should the evaluation dataset grow beyond a single PDF in later versions?

---

## 8. Integration Points
- Retrieval component must expose:
  - API: `retrieve(query, k)` returning list of (chunk, score)
- Generation component must expose:
  - API: `generate_answer(query, retrieved_context)`
- Judge and meta-judge should be callable functions:
  - `evaluate_answer(...)`
  - `meta_evaluate_judge(...)`
- BEIR evaluator should accept:
  - queries  
  - ground-truth passage IDs  
  - retrieved rankings