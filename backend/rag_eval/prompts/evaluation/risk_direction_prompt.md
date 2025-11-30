# System-Level Risk Direction Classification Prompt

You are an expert evaluator classifying the risk direction of system-level deviations from the gold answer.

## Task

Analyze the model answer and determine the risk direction of any deviations present. Deviations can originate from multiple sources in the RAG pipeline: retrieval misses, incomplete context, ambiguous phrasing, context ordering issues, prompting biases, model reasoning errors, or hallucinations. Your task is to evaluate the entire RAG system as a black box and classify the risk direction of any deviation, regardless of its origin.

## Risk Classification

System-level deviations can have two types of risks:

1. **Care Avoidance Risk (-1)**: The model **overestimated** the cost, which would **dissuade** the user from seeking care. This happens when the model states a cost that is higher than what the retrieved context indicates, potentially preventing the user from accessing needed services.

2. **Unexpected Cost Risk (+1)**: The model **underestimated** the cost, which would **persuade** the user to pursue care. This happens when the model states a cost that is lower than what the retrieved context indicates, potentially leading the user to incur unexpected expenses.

## Input

**Retrieved Context:**
{retrieved_context}

**Model Answer:**
{model_answer}

## Instructions

1. **System-Level Analysis**: Compare the cost information stated in the model answer with the cost information in the retrieved context. Remember that deviations can result from:
   - Retrieval misses (relevant context not retrieved)
   - Incomplete context (partial information retrieved)
   - Context ordering issues (important information buried)
   - Ambiguous phrasing in prompts
   - Prompting biases
   - Model reasoning errors
   - Hallucinations (unsupported claims)
   
   Your analysis should focus on the deviation itself, not its specific origin.

2. **Risk Direction**: Determine if the model overestimated (care avoidance risk, -1) or underestimated (unexpected cost risk, +1) the cost relative to the retrieved context.

3. **Quantitative Costs**: For quantitative cost deviations (e.g., "$500" vs "$300"), determine which direction the error points:
   - If model says $500 but context says $300 → Overestimated → Care Avoidance Risk (-1)
   - If model says $300 but context says $500 → Underestimated → Unexpected Cost Risk (+1)

4. **Non-Quantitative Costs**: For non-quantitative cost deviations (e.g., "covered" vs "not covered", "required" vs "optional"), determine the risk direction:
   - If model says cost is higher/more restrictive than context → Care Avoidance Risk (-1)
   - If model says cost is lower/less restrictive than context → Unexpected Cost Risk (+1)

5. **Ambiguous Cases**: If the risk direction is truly ambiguous or cannot be determined, classify as Unexpected Cost Risk (+1) as a conservative default (assuming we want to err on the side of caution for user protection).

## Output Format

You must respond with a valid JSON object containing:
- `risk_direction`: An integer value (-1 for care avoidance risk, +1 for unexpected cost risk)
- `reasoning`: A brief explanation of your classification (1-2 sentences) explaining the risk direction analysis

## Examples

### Example 1: Care Avoidance Risk (-1) - Overestimated Cost
**Retrieved Context:**
```
[Chunk ID: chunk_001] The copay for a specialist visit is $50. This applies to all in-network specialists.
[Chunk ID: chunk_002] Specialist visits require a referral from your primary care physician.
```

**Model Answer:**
The copay for specialist visits is $120, and you'll need a referral from your primary care doctor.

**Output:**
```json
{
  "risk_direction": -1,
  "reasoning": "The model overestimated the copay as $120 when the retrieved context states $50. This significant overestimation would likely dissuade the user from seeking care, causing them to believe the appointment is much more expensive than it actually is, and potentially preventing them from accessing needed services. This represents a care avoidance risk."
}
```

### Example 2: Unexpected Cost Risk (+1) - Underestimated Cost
**Retrieved Context:**
```
[Chunk ID: chunk_001] The annual deductible is $1,500. You must meet this before insurance coverage begins.
[Chunk ID: chunk_002] Preventive care visits are covered at 100% with no deductible required.
```

**Model Answer:**
The annual deductible is $1,000, and all visits require meeting the deductible first.

**Output:**
```json
{
  "risk_direction": +1,
  "reasoning": "The model underestimated the deductible as $1,000 when the retrieved context states $1,500. Additionally, the model incorrectly states all visits require the deductible, when preventive care is exempt. This underestimation would persuade the user to pursue care with incorrect cost expectations, representing an unexpected cost risk."
}
```

### Example 3: Unexpected Cost Risk (+1) - Underestimated Restriction (Retrieval Miss)
**Retrieved Context:**
```
[Chunk ID: chunk_001] Prior authorization is required for all specialist visits. This process takes 3-5 business days.
[Chunk ID: chunk_002] Emergency visits do not require prior authorization.
```

**Model Answer:**
Specialist visits do not require prior authorization and can be scheduled immediately.

**Output:**
```json
{
  "risk_direction": +1,
  "reasoning": "The model incorrectly states that prior authorization is not required, when the retrieved context clearly states it is required. This deviation (which could result from retrieval misses, incomplete context, or other system-level issues) underestimates restrictions and would persuade the user to pursue care without proper authorization, potentially leading to denied claims and unexpected costs, representing an unexpected cost risk."
}
```

### Example 4: Care Avoidance Risk (-1) - Overestimated Restriction
**Retrieved Context:**
```
[Chunk ID: chunk_001] Preventive care visits are covered at 100% with no copay or deductible.
[Chunk ID: chunk_002] Annual wellness visits are included in preventive care coverage.
```

**Model Answer:**
Preventive care visits require a $75 copay and count toward your annual deductible.

**Output:**
```json
{
  "risk_direction": -1,
  "reasoning": "The model overestimated the cost burden by stating a $75 copay and deductible requirement when the retrieved context states preventive care is covered at 100% with no copay or deductible. This overestimation would dissuade the user from seeking preventive care, representing a care avoidance risk."
}
```

### Example 5: Unexpected Cost Risk (+1) - Ambiguous Case (Default)
**Retrieved Context:**
```
[Chunk ID: chunk_001] The out-of-pocket maximum varies by plan tier.
[Chunk ID: chunk_002] Contact your insurance provider for specific cost information.
```

**Model Answer:**
The out-of-pocket maximum is $1,000 for all plans.

**Output:**
```json
{
  "risk_direction": +1,
  "reasoning": "The model stated the out-of-pocket maximum is $1,000 for all plans, despite the retrieved context indicating this amount varies by plan and the exact number is not given. By underestimating the possible cost (could be higher than $1,000), the user may choose services that end up costing them more than $1,000. This underestimation results in an unexpected cost risk to the user, so this is classified as Unexpected Cost Risk (+1)."
}
```

## Your Response

Provide your classification as a JSON object:
