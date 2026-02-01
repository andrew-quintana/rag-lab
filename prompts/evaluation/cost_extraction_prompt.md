# Cost Extraction Prompt

You are an expert parser extracting cost information (time, money, steps) from unstructured text.

## Task

Extract structured cost information from natural language text. Identify time-based costs, money-based costs, and step-based costs.

## Input

**Text:**
{text}

## Instructions

1. **Time Extraction**: Identify time-based costs in various formats:
   - "2 hours", "30 minutes", "1 day", "3 weeks", "45 minutes"
   - "hourly", "daily", "weekly" (if specific amounts mentioned)
   - Relative time: "next day", "within 24 hours" (if specific)

2. **Money Extraction**: Identify money-based costs in various formats:
   - "$500", "500 dollars", "500.00", "$1,500.00"
   - "five hundred dollars", "one thousand dollars" (written numbers)
   - Currency symbols: "$", "USD", "dollars"

3. **Steps Extraction**: Identify step-based costs:
   - "3 steps", "step 3", "third step", "5 steps required"
   - Sequential processes: "first step", "second step", etc. (if specific count)

4. **Optional Fields**: All cost fields are optional. If a cost type is not present in the text, omit that field from the output.

5. **Format Flexibility**: Handle various formats and expressions:
   - Numbers can be written as digits or words
   - Time can be expressed in different units (hours, minutes, days, weeks)
   - Money can include or exclude currency symbols
   - Steps can be expressed as numbers or ordinals

6. **Missing Information**: If no cost information is present in the text, return only the reasoning field explaining why no costs were found.

## Output Format

You must respond with a valid JSON object containing:
- `time`: Optional string or number representing time cost (e.g., "2 hours", "30 minutes", 2.5)
- `money`: Optional string or number representing monetary cost (e.g., "$500", 500.0, "500 dollars")
- `steps`: Optional integer or string representing step count (e.g., 3, "3 steps")
- `reasoning`: Required string explaining what cost information was found (or why none was found)

**Important**: Only include fields that are actually present in the text. Do not include fields with null or empty values.

## Examples

### Example 1: Time and Money Costs
**Text:**
The procedure takes 2 hours and costs $500.

**Output:**
```json
{
  "time": "2 hours",
  "money": "$500",
  "reasoning": "Found time cost: 2 hours. Found money cost: $500."
}
```

### Example 2: Money Only
**Text:**
The copay for specialist visits is 500 dollars.

**Output:**
```json
{
  "money": "500 dollars",
  "reasoning": "Found money cost: 500 dollars. No time or step costs mentioned."
}
```

### Example 3: Steps Only
**Text:**
The process requires 3 steps to complete.

**Output:**
```json
{
  "steps": 3,
  "reasoning": "Found step cost: 3 steps. No time or money costs mentioned."
}
```

### Example 4: Mixed Costs
**Text:**
You'll need to complete 5 steps, which takes about 30 minutes, and the total cost is $1,500.00.

**Output:**
```json
{
  "time": "30 minutes",
  "money": "$1,500.00",
  "steps": 5,
  "reasoning": "Found time cost: 30 minutes. Found money cost: $1,500.00. Found step cost: 5 steps."
}
```

### Example 5: No Cost Information
**Text:**
The insurance plan covers preventive care visits.

**Output:**
```json
{
  "reasoning": "No cost information found in the text. The text discusses coverage but does not mention time, money, or step costs."
}
```

### Example 6: Ambiguous Time Expression
**Text:**
The appointment should be scheduled within 24 hours.

**Output:**
```json
{
  "time": "24 hours",
  "reasoning": "Found time reference: within 24 hours. No money or step costs mentioned."
}
```

### Example 7: Written Numbers
**Text:**
The deductible is one thousand five hundred dollars, and the process takes two hours.

**Output:**
```json
{
  "time": "2 hours",
  "money": "$1,500",
  "reasoning": "Found time cost: two hours (converted to 2 hours). Found money cost: one thousand five hundred dollars (converted to $1,500)."
}
```

### Example 8: Step Ordinals
**Text:**
First, you need to complete step one. Then proceed to step two and step three.

**Output:**
```json
{
  "steps": 3,
  "reasoning": "Found step references: step one, step two, step three (total: 3 steps). No time or money costs mentioned."
}
```

## Your Response

Provide your cost extraction as a JSON object:

