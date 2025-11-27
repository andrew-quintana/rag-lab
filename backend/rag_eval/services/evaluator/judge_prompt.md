# LLM-as-Judge Prompt Template

## Evaluation Criteria

You are evaluating a RAG (Retrieval-Augmented Generation) system's answer quality.

### Scoring Guidelines

1. **Grounding** (0.0 - 1.0): How well is the answer grounded in the retrieved context?
   - 1.0: All claims are directly supported by the context
   - 0.5: Some claims are supported, some are inferred
   - 0.0: Answer is not grounded in context

2. **Relevance** (0.0 - 1.0): How relevant is the answer to the query?
   - 1.0: Answer directly addresses the query
   - 0.5: Answer is partially relevant
   - 0.0: Answer is not relevant

3. **Hallucination Risk** (0.0 - 1.0): Risk of hallucinated information
   - 0.0: No hallucination risk, all information is verifiable
   - 0.5: Some information may be inferred or uncertain
   - 1.0: High risk of hallucination

## Output Format

Provide scores as JSON:
```json
{
  "grounding": 0.85,
  "relevance": 0.90,
  "hallucination_risk": 0.15,
  "reasoning": "The answer is well-grounded in the provided context..."
}
```

