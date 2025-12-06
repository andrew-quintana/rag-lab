# Phase 11: JSON/JSONB Storage Design

## Overview

Phase 11 will implement logging and persistence for evaluation results using JSON/JSONB storage in PostgreSQL. This approach provides flexibility and extensibility for storing evaluation metrics without requiring schema changes for new metric types.

## Design Rationale

### Why JSON/JSONB?

1. **Flexibility**: New evaluation metrics can be added without schema migrations
2. **Extensibility**: Supports nested structures (e.g., `JudgePerformanceMetrics` with nested `JudgeMetricScores`)
3. **Query Capability**: PostgreSQL JSONB supports efficient querying and indexing
4. **Future-Proof**: Easy to add new fields or metrics without breaking existing code

### Database Schema

**Table**: `evaluation_results`

```sql
CREATE TABLE evaluation_results (
    result_id VARCHAR(255) PRIMARY KEY,
    example_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    judge_output JSONB NOT NULL,
    meta_eval_output JSONB NOT NULL,
    beir_metrics JSONB NOT NULL,
    judge_performance_metrics JSONB,  -- NULLABLE, optional
    metadata JSONB  -- NULLABLE, for additional flexible data
);

-- Indexes for efficient queries
CREATE INDEX idx_evaluation_results_example_id ON evaluation_results(example_id);
CREATE INDEX idx_evaluation_results_timestamp ON evaluation_results(timestamp);
CREATE INDEX idx_evaluation_results_judge_output_gin ON evaluation_results USING GIN(judge_output);
CREATE INDEX idx_evaluation_results_meta_eval_output_gin ON evaluation_results USING GIN(meta_eval_output);
```

## JSON Structure

### JudgeEvaluationResult (judge_output)
```json
{
  "correctness_binary": true,
  "hallucination_binary": false,
  "risk_direction": -1,
  "risk_impact": 2,
  "reasoning": "Model answer matches reference...",
  "failure_mode": null
}
```

### MetaEvaluationResult (meta_eval_output)
```json
{
  "judge_correct": true,
  "explanation": "All validations passed...",
  "ground_truth_correctness": true,
  "ground_truth_hallucination": false,
  "ground_truth_risk_direction": -1,
  "ground_truth_risk_impact": 2
}
```

### BEIRMetricsResult (beir_metrics)
```json
{
  "recall_at_k": 0.8,
  "precision_at_k": 0.6,
  "ndcg_at_k": 0.75
}
```

### JudgePerformanceMetrics (judge_performance_metrics, optional)
```json
{
  "correctness": {
    "precision": 0.95,
    "recall": 0.92,
    "f1_score": 0.935,
    "true_positives": 19,
    "true_negatives": 1,
    "false_positives": 1,
    "false_negatives": 1,
    "total_samples": 22
  },
  "hallucination": {
    "precision": 0.88,
    "recall": 0.85,
    "f1_score": 0.865,
    "true_positives": 17,
    "true_negatives": 3,
    "false_positives": 2,
    "false_negatives": 3,
    "total_samples": 25
  },
  "risk_direction": {
    "precision": 0.90,
    "recall": 0.87,
    "f1_score": 0.885,
    "true_positives": 13,
    "true_negatives": 5,
    "false_positives": 1,
    "false_negatives": 2,
    "total_samples": 21
  },
  "risk_impact": {
    "precision": 0.85,
    "recall": 0.82,
    "f1_score": 0.835,
    "true_positives": 14,
    "true_negatives": 4,
    "false_positives": 2,
    "false_negatives": 3,
    "total_samples": 23
  }
}
```

## Implementation Details

### JSON Serialization

1. **Custom Encoder**: Create JSON encoder that handles:
   - Dataclasses → dict
   - datetime objects → ISO format strings
   - Optional fields → null or omitted
   - Nested structures → recursive serialization

2. **Helper Functions**:
   ```python
   def serialize_evaluation_result(result: EvaluationResult) -> Dict[str, Any]:
       """Serialize EvaluationResult to JSON-compatible dict"""
       
   def serialize_judge_output(judge: JudgeEvaluationResult) -> Dict[str, Any]:
       """Serialize JudgeEvaluationResult to JSON"""
       
   def serialize_meta_eval(meta: MetaEvaluationResult) -> Dict[str, Any]:
       """Serialize MetaEvaluationResult to JSON"""
   ```

### JSON Deserialization

1. **Custom Decoder**: Create JSON decoder that handles:
   - Dict → dataclass reconstruction
   - ISO format strings → datetime objects
   - null values → None
   - Nested structures → recursive deserialization

2. **Helper Functions**:
   ```python
   def deserialize_evaluation_result(data: Dict[str, Any]) -> EvaluationResult:
       """Deserialize JSON dict to EvaluationResult"""
   ```

## Query Examples

### Query by Example ID
```sql
SELECT * FROM evaluation_results 
WHERE example_id = 'val_001';
```

### Query by Judge Correctness
```sql
SELECT * FROM evaluation_results 
WHERE meta_eval_output->>'judge_correct' = 'true';
```

### Query by Correctness Precision (if metrics logged)
```sql
SELECT * FROM evaluation_results 
WHERE (judge_performance_metrics->'correctness'->>'precision')::float > 0.9;
```

### Query with JSON Path
```sql
SELECT 
    example_id,
    judge_output->>'correctness_binary' as correctness,
    beir_metrics->>'recall_at_k' as recall
FROM evaluation_results
WHERE timestamp > NOW() - INTERVAL '7 days';
```

## Benefits

1. **No Schema Changes**: Add new metrics without migrations
2. **Flexible Queries**: Use JSONB operators for complex queries
3. **Efficient Indexing**: GIN indexes support fast JSON queries
4. **Easy Integration**: Simple serialization/deserialization
5. **Future-Proof**: Can evolve without breaking changes

## Migration File

**Location**: `infra/supabase/migrations/0011_add_evaluation_results_table.sql`

The migration will:
1. Create `evaluation_results` table with JSONB columns
2. Create indexes for efficient queries
3. Include comments documenting JSON structure

## Testing Considerations

1. **Serialization Tests**: Verify all fields serialize correctly
2. **Deserialization Tests**: Verify round-trip (serialize → deserialize → compare)
3. **JSONB Query Tests**: Test various JSONB query patterns
4. **Index Tests**: Verify indexes improve query performance
5. **Migration Tests**: Verify migration runs successfully

## Phase 11 Implementation

This JSON/JSONB storage approach will be implemented in Phase 11 as part of the logging and persistence functionality. All evaluation metrics will be stored as JSONB, providing maximum flexibility for future enhancements.

