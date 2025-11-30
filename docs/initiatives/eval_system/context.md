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
- Create an **experimental harness** to test RAG system variations (retrieval method and generation prompt).
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

- Architecture: **Deterministic Python script orchestrator** that calls LLM nodes sequentially
- Input:  
  - Query
  - Retrieved context (chunks)  
  - Model answer  
  - Reference answer  
- Process:
  1. **Correctness Classification Step**:  
     - The deterministic script calls a dedicated **correctness LLM-node** to directly compare model answer to gold reference answer.
     - This node performs direct comparison: is the model answer correct or incorrect?
     - Returns: `correctness_binary` (true/false)
  
  2. **Hallucination Classification Step**:  
     - The deterministic script calls a dedicated **hallucination LLM-node** to classify whether hallucination is present.
     - This node performs strict *grounding* analysis: does the model answer contain any information not supported by the retrieved evidence? (Note: reference answer is NOT used)
     - Returns: `hallucination_binary` (true/false)
  
  3. **Cost Type Classification Step** (if `hallucination_binary: true`):  
     - The deterministic script calls a dedicated **hallucination_cost LLM-node** (GPT-4o-mini, Azure Foundry) to classify the type of cost impact.
     - This node analyzes:
       - Model answer vs retrieved chunks (for ground-truth context)
       - Whether the hallucination overestimates or underestimates costs relative to the retrieved chunks (Note: reference answer is NOT used)
     - Returns: `hallucination_cost` with values:
       - **-1 = Opportunity Cost**: The user may have been dissuaded from seeking care because the model answer overestimated the cost (in time/money/steps), causing them to miss the opportunity to get care.
       - **+1 = Resource Cost**: The user may have been persuaded to pursue care because the model answer underestimated the cost (in time/money/steps), and it ended up costing them those resources.
     - This classification is made for all hallucinations (both quantitative and non-quantitative).
  
  4. **Impact Calculation Step** (if `hallucination_binary: true`):  
     - The deterministic script calls a dedicated **hallucination_impact LLM-node** (GPT-4o-mini, Azure Foundry) to determine the magnitude of impact.
     - This node handles mixed resource types (time/money/steps) and requires LLM reasoning to assess relative impact across these different dimensions.
     - Inputs:
       - Cost in time/money/steps from the model answer
       - Actual cost in time/money/steps from the retrieved chunks (ground truth)
     - Process:
       - Analyzes the difference between model answer cost and actual cost from chunks
       - Considers the mixed resource types (time, money, steps) and their relative importance
       - Computes **hallucination_impact** (range [0, 3]) representing the magnitude of real-world consequence
     - Output:
       - **hallucination_impact**: Numeric scaling factor [0, 3]
     - hallucination_impact serves as a scaling factor that, combined with hallucination_cost (-1 or +1), determines total real-world consequence.

- Output schema:
  ```json
  {
    "correctness_binary": true | false,
    "hallucination_binary": true | false,
    "hallucination_cost": -1 | 1,         // -1 = opportunity cost (overestimated, dissuaded from care), +1 = resource cost (underestimated, persuaded to care)
    "hallucination_impact": number,       // [0, 3], from hallucination_impact LLM-node
    "reasoning": string,                  // Reasoning trace constructed from LLM node outputs, including correctness classification, cost type classification (via hallucination_cost LLM-node) and impact calculation (via hallucination_impact LLM-node) steps
    "failure_mode": string                // Optional: e.g., "cost misstatement", "omitted deductible", "incorrect coverage rule"
  }
  ```

### 4.2 Correctness Classification

- **correctness LLM-node** (GPT-4o-mini, Azure Foundry):
  - **Invoked by deterministic script orchestrator** (always called).
  - Inputs:
    - Query
    - Model answer
    - Gold reference answer
  - Process:
    - Directly compares model answer to gold reference answer
    - Assesses whether model answer is correct or incorrect
  - Output:
    - **correctness_binary**: true | false

### 4.3 Hallucination Cost Classification and Impact Calculation

- **hallucination_cost LLM-node** (GPT-4o-mini, Azure Foundry):
  - **Invoked by deterministic script orchestrator** when `hallucination_binary: true`.
  - Inputs:
    - Model answer
    - Retrieved chunks (for ground-truth context)
    - Note: Reference answer is NOT used
  - Process:
    - Analyzes whether the hallucination overestimates or underestimates costs (time/money/steps) relative to the retrieved chunks (ground truth)
    - Determines the type of cost impact based on the direction of the misstatement
  - Output:
    - **hallucination_cost**: Classification value
      - **-1 = Opportunity Cost**: Model answer overestimated cost (time/money/steps), potentially dissuading user from seeking care and causing them to miss the opportunity.
      - **+1 = Resource Cost**: Model answer underestimated cost (time/money/steps), potentially persuading user to pursue care that ended up costing them those resources.
  - This classification is made for all hallucinations (both quantitative and non-quantitative).

- **hallucination_impact LLM-node** (GPT-4o-mini, Azure Foundry):
  - **Invoked by deterministic script orchestrator** when `hallucination_binary: true`.
  - **Rationale**: Uses an LLM node (rather than a deterministic function) because it must handle mixed resource types (time, money, steps) and assess their relative importance and impact, which requires nuanced reasoning.
  - Inputs:
    - Cost in time/money/steps from the model answer
    - Actual cost in time/money/steps from the retrieved chunks (ground truth)
  - Process:
    - Analyzes the difference between model answer cost and actual cost from chunks
    - Considers the mixed resource types (time, money, steps) and their relative importance
    - Assesses the magnitude of real-world consequence across these different dimensions
    - Computes **hallucination_impact** (range [0, 3]) representing the scaling factor for impact
  - Output:
    - **hallucination_impact**: Numeric scaling factor [0, 3] representing the magnitude of real-world consequence
  - This node is used for all hallucinations (both quantitative and non-quantitative), as it can handle both numeric and qualitative assessments of impact.

- Failure modes remain tied to key insurance dimensions:
  - Copay  
  - Coinsurance  
  - Deductible  
  - Out-of-pocket max  
  - Eligibility / pre-authorization errors

### 4.4 Meta-Evaluator (Deterministic Validation)

- Implementation: **Deterministic Python function** (no LLM calls)
- Purpose:
  - Deterministically evaluate whether the LLM-as-Judge's verdicts are correct
  - Compare judge output against ground truth (reference answer, retrieved chunks)
  - Use rule-based validation logic to check:
    - correctness_binary: Compare model answer to reference answer
    - hallucination_binary: Check if model answer is grounded in retrieved chunks
    - hallucination_cost: Validate cost direction against ground truth
    - hallucination_impact: Validate impact magnitude against ground truth
  - Return:
    - judge_correct / judge_incorrect (binary)
    - Optional, concise explanation (deterministic, not LLM-generated)

### 4.5 BEIR-style Retrieval Metrics

- Compute:
  - Recall@k
  - Precision@k
  - nDCG@k
- Use synthetic QA pairs created from the main fixture document.
- For each evaluation example, compute and add the **beir_failure_scale_factor** field for context-aware hallucination impact judging.

### 4.6 Evaluation Dataset (Development Task)

- **Development Task**: Coding agent manually creates 5 validation samples (not an automated function)
- Source: `backend/tests/fixtures/sample_documents/healthguard_select_ppo_plan.pdf`
- **Note**: Full evaluation dataset will be created by humans later; this is only for system validation
- Each of the 5 samples includes:
  - Question
  - Gold reference answer
  - Ground-truth chunks/paragraphs for retrieval relevance
  - **beir_failure_scale_factor** (float, representing retrieval challenge/severity for that QA pair)
- Stored as: `backend/tests/fixtures/evaluation_dataset/validation_dataset.json`

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
  - judge (deterministic script orchestrator)
  - hallucination LLM-node (binary classification)
  - hallucination_cost LLM-node (cost type classification: -1 or +1)
  - hallucination_impact LLM-node (impact magnitude calculation: 0-3)
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
- Annotate ground-truth passages for retrieval evaluation with query pairs per @https://github.com/beir-cellar/beir/wiki/Load-your-custom-dataset.

### Phase 3 — LLM-as-Judge Implementation
- Implement deterministic Python script orchestrator for LLM-as-Judge (sequential calls with conditional branching).
- Implement separate hallucination LLM-node for binary classification.
- Implement separate hallucination_cost LLM-node (GPT-4o-mini) for cost type classification (-1 for opportunity cost, +1 for resource cost).
- Implement separate hallucination_impact LLM-node (GPT-4o-mini) for impact magnitude calculation (0-3), handling mixed resource types (time/money/steps).
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
- Quantitative hallucinations where financial cost difference cannot be computed from chunks.
- Ambiguous cases where it's unclear if cost was overestimated (opportunity cost) or underestimated (resource cost).

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