# Evaluation Agent Configuration

Agent-specific rules and constraints for agents working on evaluation pipelines and metrics.

## Scope

This configuration applies to agents:
- Implementing evaluation metrics (BEIR, judge, meta-evaluation)
- Modifying evaluation pipelines and workflows
- Creating or updating evaluation datasets
- Analyzing evaluation results and generating insights

## Evaluation-Specific Constraints

### BEIR Metrics Requirements
- Must support standard BEIR metrics: Recall@K, Precision@K, nDCG@K
- Handle ground truth relevance judgments correctly
- Support configurable K values for metrics calculation
- Ensure metrics are computed consistently across different retrievers

### Judge Evaluation Standards
- Implement 4-stage evaluation process:
  1. Correctness classification (binary)
  2. Hallucination detection (binary)  
  3. Risk direction assessment (-1, 0, 1)
  4. Risk impact calculation (0-3)
- Support insurance risk semantics (care avoidance vs unexpected cost)
- Parse LLM responses robustly with fallback handling
- Generate meaningful reasoning explanations

### Meta-Evaluation Requirements  
- Validate judge outputs against ground truth when available
- Detect systematic biases in judge evaluation
- Provide explanations for meta-evaluation decisions
- Support judge reliability assessment

## Data Handling Rules

### Dataset Management
- Use standardized data formats: Parquet for structured data, JSONL for evaluation results
- Maintain data lineage and provenance information
- Support incremental evaluation and result updates
- Handle missing or incomplete data gracefully

### Result Storage
- Create timestamped run directories in `runs/`
- Save configuration as `config.yaml` in run directory
- Store detailed results as `outputs.jsonl`
- Generate summary metrics as `metrics.json`
- Maintain execution traces in `traces.jsonl`

### Artifact Management
- Store embeddings as `.npy` files in `artifacts/`
- Save FAISS indices as `.index` files
- Maintain document store as Parquet in `artifacts/docstore.parquet`
- Version artifacts when schemas change

## Pipeline Orchestration

### Evaluation Workflow
1. **Setup**: Load configuration and initialize components
2. **Retrieval**: Execute retrieval and compute BEIR metrics
3. **Generation**: Generate answers using RAG pipeline
4. **Judgment**: Evaluate answers using LLM-as-Judge
5. **Meta-Evaluation**: Validate judge outputs
6. **Analysis**: Compute summary statistics and insights

### Error Handling
- Gracefully handle component failures without stopping entire evaluation
- Log detailed error information for debugging
- Generate error-marked results for failed examples
- Provide partial results when possible

### Progress Tracking
- Report progress every 10 examples during evaluation
- Save intermediate results to prevent data loss
- Provide time estimates and performance metrics
- Log component-specific timing information

## Metrics and Analysis

### Statistical Requirements
- Compute confidence intervals for key metrics
- Support statistical significance testing
- Handle different sample sizes appropriately
- Provide distribution analysis for risk metrics

### Comparative Analysis
- Support side-by-side comparison of different components
- Generate performance matrices for multiple configurations
- Highlight statistically significant differences
- Provide recommendations based on evaluation results

### Visualization Support
- Generate data suitable for plotting and visualization
- Support trend analysis over multiple evaluation runs
- Provide summary statistics in human-readable format
- Export results in formats suitable for further analysis

## Quality Assurance

### Validation Checks
- Verify evaluation examples have required fields
- Validate ground truth data consistency
- Check for data leakage between training and evaluation
- Ensure reproducibility through configuration saving

### Performance Standards
- Evaluation should complete within reasonable time bounds
- Memory usage should scale appropriately with dataset size
- Component failures should not crash entire evaluation
- Results should be deterministic given same inputs and configuration

### Documentation Requirements
- Document evaluation methodology and assumptions
- Explain metric calculations and interpretations
- Provide examples of configuration usage
- Include troubleshooting guides for common issues

## Security and Privacy

### Data Protection
- Never log or expose sensitive information from evaluation data
- Handle personal information according to privacy requirements
- Secure storage of evaluation artifacts and results
- Appropriate access controls for sensitive datasets

### Model Safety
- Validate that judge evaluations don't reveal sensitive training data
- Ensure generated content doesn't contain inappropriate material
- Monitor for potential bias in evaluation judgments
- Report concerning patterns in evaluation results

## Integration Requirements

### Component Compatibility
- Ensure evaluation works with all registered component types
- Support both legacy function-based and registry-based components
- Handle version differences gracefully
- Provide clear error messages for incompatible configurations

### External Dependencies
- Minimize dependencies on external services
- Support offline evaluation when possible
- Handle API rate limits and failures gracefully
- Provide fallback options for external service outages

### Notebook Integration
- Ensure evaluation functions work properly in Jupyter notebooks
- Support interactive progress reporting
- Provide visualization-friendly output formats
- Handle notebook-specific execution environments