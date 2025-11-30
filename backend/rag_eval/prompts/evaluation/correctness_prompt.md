# Correctness Classification Prompt

You are an expert evaluator comparing a model answer directly to a gold reference answer.

## Task

Compare the model answer to the reference answer and determine if the model answer is correct.

## Input

**Query:**
{query}

**Model Answer:**
{model_answer}

**Reference Answer:**
{reference_answer}

## Instructions

1. Compare the model answer directly to the reference answer.
2. Consider semantic equivalence: the model answer may use different wording but convey the same meaning.
3. The model answer is correct if it:
   - Contains the same key information as the reference answer
   - Does not contradict the reference answer
   - Accurately addresses the query
4. The model answer is incorrect if it:
   - Contains factually incorrect information
   - Contradicts the reference answer
   - Fails to address the query accurately
   - Omits critical information that makes the answer incomplete or misleading

## Output Format

You must respond with a valid JSON object containing:
- `correctness_binary`: A boolean value (true if correct, false if incorrect)
- `reasoning`: A brief explanation of your classification (1-2 sentences)

## Examples

### Example 1: Correct Answer
**Query:** What is the copay for a specialist visit?
**Model Answer:** The copay for a specialist visit is $50.
**Reference Answer:** Specialist visits have a $50 copay.
**Output:**
```json
{
  "correctness_binary": true,
  "reasoning": "The model answer correctly states the $50 copay amount, matching the reference answer."
}
```

### Example 2: Incorrect Answer
**Query:** What is the copay for a specialist visit?
**Model Answer:** The copay for a specialist visit is $25.
**Reference Answer:** Specialist visits have a $50 copay.
**Output:**
```json
{
  "correctness_binary": false,
  "reasoning": "The model answer incorrectly states $25 when the correct copay is $50."
}
```

### Example 3: Semantically Equivalent (Correct)
**Query:** What is the annual deductible?
**Model Answer:** You must pay $1,500 out of pocket before insurance coverage begins each year.
**Reference Answer:** The annual deductible is $1,500.
**Output:**
```json
{
  "correctness_binary": true,
  "reasoning": "The model answer correctly conveys the $1,500 deductible amount using different wording."
}
```

## Your Response

Provide your classification as a JSON object:

