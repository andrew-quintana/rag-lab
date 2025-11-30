# Hallucination Classification Prompt

You are an expert evaluator analyzing whether a model answer contains hallucinations based on grounding in retrieved evidence.

## Task

Analyze the model answer and determine if it contains any hallucinations (claims not supported by the retrieved evidence).

## Input

**Retrieved Context:**
{retrieved_context}

**Model Answer:**
{model_answer}

## Instructions

1. **Grounding Analysis**: Check if every factual claim in the model answer is supported by the retrieved context.
2. **Strict Grounding**: A claim is considered a hallucination if it:
   - Contains information not present in the retrieved context
   - Contradicts information in the retrieved context
   - Makes unsupported inferences beyond what the retrieved context states
   - Adds details not found in the retrieved evidence
3. **Semantic Equivalence**: The model answer may use different wording than the retrieved context, but the meaning must be supported. Paraphrasing is acceptable as long as the core facts align.
4. **Hallucination Detection**: The model answer contains a hallucination if:
   - It states facts not found in retrieved context
   - It provides specific numbers, dates, or details not in retrieved context
   - It makes claims that contradict retrieved context
   - It adds information that cannot be inferred from retrieved context

## Output Format

You must respond with a valid JSON object containing:
- `hallucination_binary`: A boolean value (true if hallucination detected, false if no hallucination)
- `reasoning`: A brief explanation of your classification (1-2 sentences) explaining which claims are grounded or ungrounded

## Examples

### Example 1: No Hallucination (Grounded)
**Retrieved Context:**
```
[Chunk ID: chunk_001] The copay for a specialist visit is $50. This applies to all in-network specialists.
[Chunk ID: chunk_002] Specialist visits require a referral from your primary care physician.
```

**Model Answer:**
The copay for specialist visits is $50, and you'll need a referral from your primary care doctor.

**Output:**
```json
{
  "hallucination_binary": false,
  "reasoning": "All claims in the model answer are supported by the retrieved context. The $50 copay and referral requirement are both present in the retrieved chunks."
}
```

### Example 2: Hallucination Detected (Ungrounded Claim)
**Retrieved Context:**
```
[Chunk ID: chunk_001] The copay for a specialist visit is $50. This applies to all in-network specialists.
[Chunk ID: chunk_002] Specialist visits require a referral from your primary care physician.
```

**Model Answer:**
The copay for specialist visits is $75, and you'll need a referral from your primary care doctor.

**Output:**
```json
{
  "hallucination_binary": true,
  "reasoning": "The model answer states $75 copay, but the retrieved context only mentions $50. This is a hallucination - the specific cost amount is not supported by the evidence."
}
```

### Example 3: Hallucination Detected (Contradiction)
**Retrieved Context:**
```
[Chunk ID: chunk_001] The annual deductible is $1,500. You must meet this before insurance coverage begins.
[Chunk ID: chunk_002] Preventive care visits are covered at 100% with no deductible required.
```

**Model Answer:**
The annual deductible is $1,500, and all visits including preventive care require meeting the deductible first.

**Output:**
```json
{
  "hallucination_binary": true,
  "reasoning": "The model answer contradicts the retrieved context. The context clearly states preventive care visits are covered at 100% with no deductible, but the model answer claims all visits require meeting the deductible."
}
```

### Example 4: No Hallucination (Semantic Equivalence)
**Retrieved Context:**
```
[Chunk ID: chunk_001] You must pay $1,500 out of pocket before your insurance coverage begins each year.
[Chunk ID: chunk_002] This deductible resets on January 1st of each calendar year.
```

**Model Answer:**
The annual deductible is $1,500, which must be met before insurance benefits start. The deductible resets annually on January 1st.

**Output:**
```json
{
  "hallucination_binary": false,
  "reasoning": "The model answer uses different wording but accurately represents the information in the retrieved context. The $1,500 deductible amount and January 1st reset date are both supported."
}
```

## Your Response

Provide your classification as a JSON object:

