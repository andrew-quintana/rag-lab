# Component Registry & Version Control

RAGLab supports multiple implementations of each component type through a registry system, enabling easy comparative evaluation and experimentation. This document explains how to use the component registry and test different implementations.

## Overview

The component registry allows you to:
- Register multiple implementations of each component type (judges, chunkers, embedders, retrievers, generators)
- Easily switch between implementations for comparative evaluation
- Test new implementations side-by-side with existing ones
- Track metadata about each implementation for reproducibility

## Component Types

### Supported Components

1. **Judges**: LLM-as-Judge evaluation implementations
2. **Chunkers**: Text chunking strategies
3. **Embedders**: Text embedding models
4. **Retrievers**: Document retrieval systems
5. **Generators**: Text generation models

## Using the Registry

### Basic Usage

```python
from src.core.registry import registry, list_all_components

# List all available implementations
components = list_all_components()
print(components)
# Output: {
#   'judges': ['llm_judge'],
#   'chunkers': ['simple_chunker', 'fixed_chunker'], 
#   'embedders': ['openai_embedder', 'sentence_transformer_embedder'],
#   'retrievers': ['faiss_retriever'],
#   'generators': ['openai_generator', 'simple_rag_generator']
# }

# Get a specific implementation
judge_class = registry.get('judges', 'llm_judge')
chunker_class = registry.get('chunkers', 'simple_chunker')
```

### Running Evaluation with Registry Components

```python
from src.eval import run_evaluation
from src.core.interfaces import EvaluationExample

# Example: Compare different judges
examples = [...]  # Your evaluation examples

# Run with LLM judge
results1, run_dir1 = run_evaluation(
    examples=examples,
    judge_name='llm_judge',
    retriever_name='faiss_retriever',
    generator_name='openai_generator',
    llm_function=your_llm_function,
    component_configs={
        'judge': {'temperature': 0.1},
        'retriever': {'index_path': 'artifacts/faiss.index'},
        'generator': {'model_name': 'gpt-4'}
    },
    run_name='llm_judge_comparison'
)

# Run with different chunker
results2, run_dir2 = run_evaluation(
    examples=examples,
    judge_name='llm_judge',
    chunker_name='fixed_chunker',
    retriever_name='faiss_retriever',
    generator_name='simple_rag_generator',
    llm_function=your_llm_function,
    component_configs={
        'chunker': {'chunk_size': 256, 'overlap_size': 32},
        'generator': {'approach': 'context_injection'}
    },
    run_name='fixed_chunker_comparison'
)
```

## Creating New Implementations

### Step 1: Create Implementation

Choose the appropriate subdirectory and create your implementation:

```
src/evaluation/
├── judges/          # Judge implementations
├── chunkers/        # Chunker implementations  
├── embedders/       # Embedder implementations
├── retrievers/      # Retriever implementations
└── generators/      # Generator implementations
```

### Step 2: Inherit from Base Class

```python
# Example: Custom judge implementation
# src/evaluation/judges/custom_judge.py

from ..base.base_judge import MultiStageJudge
from ...core.registry import register_judge

@register_judge(
    name="custom_judge",
    description="Custom multi-stage judge with domain-specific logic",
    methodology="Custom approach"
)
class CustomJudge(MultiStageJudge):
    def __init__(self, llm_function, prompts, custom_param=None, **config):
        super().__init__(**config)
        self.llm_function = llm_function
        self.prompts = prompts
        self.custom_param = custom_param
    
    def get_name(self) -> str:
        return "custom_judge"
    
    def get_description(self) -> str:
        return f"Custom judge with param: {self.custom_param}"
    
    def evaluate_correctness(self, question, reference_answer, generated_answer):
        # Your custom correctness evaluation logic
        return True  # placeholder
    
    # Implement other required methods...
```

### Step 3: Register and Test

```python
# Import to register the component
from src.evaluation.judges.custom_judge import CustomJudge

# Verify registration
from src.core.registry import registry
print(registry.list_implementations('judges'))
# Should include 'custom_judge'

# Test the implementation
results, run_dir = run_evaluation(
    examples=test_examples,
    judge_name='custom_judge',
    component_configs={
        'judge': {'custom_param': 'test_value'}
    },
    run_name='custom_judge_test'
)
```

## Component Implementation Examples

### Custom Chunker

```python
# src/evaluation/chunkers/semantic_chunker.py

from ..base.base_chunker import BaseChunker
from ...core.registry import register_chunker

@register_chunker(
    name="semantic_chunker",
    description="Semantic-aware text chunking",
    method="semantic_similarity"
)
class SemanticChunker(BaseChunker):
    def __init__(self, similarity_threshold=0.7, **config):
        super().__init__(**config)
        self.similarity_threshold = similarity_threshold
    
    def get_name(self) -> str:
        return "semantic_chunker"
    
    def get_description(self) -> str:
        return f"Semantic chunker (threshold={self.similarity_threshold})"
    
    def chunk_text(self, text, document_id=None, **kwargs):
        # Your semantic chunking implementation
        chunks = []  # Implementation here
        return chunks
```

### Custom Embedder

```python
# src/evaluation/embedders/custom_embedder.py

from ..base.base_embedder import BaseEmbedder
from ...core.registry import register_embedder

@register_embedder(
    name="custom_embedder",
    description="Custom embedding model",
    provider="Custom"
)
class CustomEmbedder(BaseEmbedder):
    def __init__(self, model_path, **config):
        super().__init__(**config)
        self.model_path = model_path
        # Load your custom model
    
    def get_name(self) -> str:
        return "custom_embedder"
    
    def get_description(self) -> str:
        return f"Custom embedder from {self.model_path}"
    
    def embed_texts(self, texts, **kwargs):
        # Your embedding implementation
        embeddings = []  # Implementation here
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        return 768  # Your embedding dimension
```

## Testing and Comparison Workflows

### A/B Testing Different Components

```python
# Test different chunking strategies
chunker_tests = [
    ('simple_chunker', {'max_chunk_size': 500}),
    ('fixed_chunker', {'chunk_size': 512}),
    ('semantic_chunker', {'similarity_threshold': 0.8})
]

results = {}
for chunker_name, config in chunker_tests:
    result, run_dir = run_evaluation(
        examples=test_examples,
        chunker_name=chunker_name,
        component_configs={'chunker': config},
        run_name=f'chunker_test_{chunker_name}'
    )
    results[chunker_name] = (result, run_dir)

# Compare results
for name, (result, run_dir) in results.items():
    print(f"\\n=== {name} ===")
    print_evaluation_summary(run_dir)
```

### Batch Evaluation Across Components

```python
# Compare all combinations
import itertools

judges = ['llm_judge']
chunkers = ['simple_chunker', 'fixed_chunker']
generators = ['openai_generator', 'simple_rag_generator']

results_matrix = {}

for judge, chunker, generator in itertools.product(judges, chunkers, generators):
    combination_name = f"{judge}_{chunker}_{generator}"
    
    result, run_dir = run_evaluation(
        examples=test_examples,
        judge_name=judge,
        chunker_name=chunker,  
        generator_name=generator,
        run_name=f'matrix_test_{combination_name}'
    )
    
    results_matrix[combination_name] = {
        'results': result,
        'run_dir': run_dir,
        'components': {'judge': judge, 'chunker': chunker, 'generator': generator}
    }

# Analyze results matrix
best_combo = max(results_matrix.items(), 
                key=lambda x: x[1]['results'][0].beir_metrics.recall_at_k)
print(f"Best combination: {best_combo[0]}")
```

### Performance Benchmarking

```python
import time

def benchmark_component(component_type, component_name, test_data, iterations=10):
    \"\"\"Benchmark component performance\"\"\"
    times = []
    
    for _ in range(iterations):
        start_time = time.time()
        
        # Run evaluation with component
        run_evaluation(
            examples=test_data[:5],  # Small subset for speed
            **{f'{component_type}_name': component_name},
            run_name=f'benchmark_{component_name}'
        )
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        'mean_time': sum(times) / len(times),
        'min_time': min(times),
        'max_time': max(times)
    }

# Benchmark different chunkers
chunker_benchmarks = {}
for chunker in ['simple_chunker', 'fixed_chunker']:
    chunker_benchmarks[chunker] = benchmark_component('chunker', chunker, test_examples)

print("Chunker Performance:")
for name, stats in chunker_benchmarks.items():
    print(f"{name}: {stats['mean_time']:.2f}s avg")
```

## Registry Metadata and Comparison

### Viewing Component Information

```python
# Get detailed component information
metadata = registry.get_metadata('judges', 'llm_judge')
print(metadata)

# Compare all implementations of a type
comparison = registry.compare_implementations('chunkers')
for name, info in comparison.items():
    print(f"{name}: {info['description']} (method: {info.get('method', 'unknown')})")
```

### Configuration Management

```python
# Save successful configurations
best_config = {
    'judge': {'name': 'llm_judge', 'config': {'temperature': 0.1}},
    'chunker': {'name': 'simple_chunker', 'config': {'max_chunk_size': 400}},
    'embedder': {'name': 'openai_embedder', 'config': {'model_name': 'text-embedding-ada-002'}},
    'retriever': {'name': 'faiss_retriever', 'config': {'index_path': 'artifacts/faiss.index'}},
    'generator': {'name': 'openai_generator', 'config': {'model_name': 'gpt-4'}}
}

# Save to file for reproducibility
import yaml
with open('best_config.yaml', 'w') as f:
    yaml.dump(best_config, f)

# Load and use saved configuration
with open('best_config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

component_configs = {k: v['config'] for k, v in config.items()}
component_names = {f"{k}_name": v['name'] for k, v in config.items()}

results, run_dir = run_evaluation(
    examples=examples,
    **component_names,
    component_configs=component_configs,
    run_name='best_config_reproduction'
)
```

## Best Practices

### 1. Version Control for Components

- Tag component implementations with versions
- Document changes between versions
- Keep backward compatibility when possible

### 2. Testing New Implementations

- Start with small test datasets
- Compare against baseline implementations
- Document performance characteristics and use cases

### 3. Component Isolation

- Each implementation should be self-contained
- Minimize dependencies between components
- Use clear, descriptive names and metadata

### 4. Reproducibility

- Save exact component configurations used in experiments
- Document model versions and parameters
- Include component metadata in evaluation results

### 5. Error Handling

- Implement graceful error handling in components
- Provide meaningful error messages
- Include validation of inputs and configurations

## Troubleshooting

### Common Issues

1. **Component not found**: Ensure the component file is imported
2. **Configuration errors**: Validate config parameters in component `__init__`
3. **Registry conflicts**: Use unique names for different implementations
4. **Import errors**: Check that all dependencies are available

### Debugging

```python
# Debug component registration
print("Registered components:", list_all_components())

# Test component instantiation
try:
    component_class = registry.get('judges', 'custom_judge')
    instance = component_class(llm_function=test_llm, prompts={})
    print(f"Successfully created: {instance.get_description()}")
except Exception as e:
    print(f"Error creating component: {e}")

# Validate component interface
from src.evaluation.base.base_judge import BaseJudge
print(f"Is valid judge: {isinstance(instance, BaseJudge)}")
```

This registry system enables systematic evaluation of different RAG component combinations, making it easy to find optimal configurations for your specific use case.