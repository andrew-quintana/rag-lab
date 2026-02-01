# Component Development Guide

Detailed rules and workflows for developing new RAGLab components.

## Development Workflow

### 1. Component Planning
- Identify component type (judge, chunker, embedder, retriever, generator)
- Define implementation approach and unique characteristics
- Choose descriptive name following `{approach}_{component_type}` pattern
- Document expected configuration parameters and metadata

### 2. Implementation Steps

1. **Create Implementation File**
   - Use appropriate subdirectory: `src/evaluation/{component_type}s/`
   - Inherit from corresponding base class in `src/evaluation/base/`
   - Import registry decorator: `from ...core.registry import register_{component_type}`

2. **Implement Required Interface**
   - Override all abstract methods from base class
   - Implement `get_name()` and `get_description()` methods
   - Add proper type hints and docstrings
   - Handle configuration via `**config` parameter

3. **Register Component**
   ```python
   @register_{component_type}(
       name="descriptive_name",
       description="Clear description of approach",
       key_metadata="relevant_info"
   )
   class YourComponent(BaseComponent):
       # Implementation
   ```

### 3. Testing Protocol

1. **Unit Testing**: Validate individual methods work correctly
2. **Integration Testing**: Test with small evaluation dataset  
3. **Comparative Testing**: Compare against existing implementations
4. **Performance Testing**: Benchmark execution time and memory usage
5. **Error Testing**: Verify graceful handling of invalid inputs

### 4. Documentation Requirements

- Comprehensive docstrings for all public methods
- Clear parameter descriptions with types
- Usage examples in component file or separate example
- Performance characteristics and recommended use cases
- Configuration parameter documentation

## Component-Specific Guidelines

### Judges
- Must support 4-stage evaluation: correctness, hallucination, risk direction, risk impact
- Implement proper response parsing for LLM outputs
- Handle edge cases (empty responses, parsing failures)
- Support insurance risk semantics (care avoidance vs unexpected cost)

### Chunkers
- Return `Chunk` objects with proper metadata
- Handle empty or very short texts gracefully
- Support configurable chunk sizes and overlap
- Maintain document relationships in chunk IDs

### Embedders  
- Return numpy arrays with consistent dimensions
- Support batch processing for efficiency
- Handle empty text lists appropriately
- Document embedding dimensions and model characteristics

### Retrievers
- Return `RetrievalResult` objects with scores and rankings
- Support configurable retrieval count (k parameter)
- Implement proper similarity scoring
- Handle index initialization and document addition

### Generators
- Return `GenerationResult` objects with metadata
- Support context chunk integration
- Handle prompt formatting consistently
- Track token usage and model information

## Quality Checklist

Before registering a new component:

- [ ] Inherits from appropriate base class
- [ ] Implements all required abstract methods
- [ ] Includes proper type hints throughout
- [ ] Has comprehensive docstrings
- [ ] Handles errors gracefully with meaningful messages
- [ ] Supports configuration via `**config` parameter
- [ ] Includes descriptive metadata in registration
- [ ] Passes unit and integration tests
- [ ] Performance is acceptable for evaluation workflows
- [ ] No security vulnerabilities (secrets exposure, etc.)

## Registry Best Practices

### Naming Conventions
- Use snake_case for component names
- Include approach or key characteristic in name
- Avoid generic names like "default" or "basic"
- Be descriptive but concise

### Metadata Documentation
- Provide clear, informative descriptions
- Include key configuration parameters
- Document performance characteristics
- Specify use cases or recommended scenarios

### Version Management
- Consider versioning for significant changes
- Maintain backward compatibility when possible
- Document breaking changes clearly
- Provide migration guidance for updates

## Common Patterns

### Configuration Handling
```python
def __init__(self, param1=default1, param2=default2, **config):
    super().__init__(**config)
    self.param1 = param1
    self.param2 = param2
    # Store config for inspection
    self.config.update({'param1': param1, 'param2': param2})
```

### Error Handling
```python
def process(self, input_data):
    try:
        # Main processing logic
        result = self._do_processing(input_data)
        return result
    except Exception as e:
        logger.error(f"Error in {self.get_name()}: {e}")
        # Return appropriate error response or re-raise
        raise RuntimeError(f"Component {self.get_name()} failed: {e}")
```

### Resource Management
```python
def __init__(self, model_path, **config):
    super().__init__(**config)
    self.model = None
    self._load_model(model_path)

def _load_model(self, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    # Load model safely
```

## Performance Guidelines

- **Batch Processing**: Support batch operations for efficiency
- **Lazy Loading**: Load expensive resources only when needed
- **Caching**: Cache expensive computations when appropriate
- **Memory Management**: Clean up resources properly
- **Progress Tracking**: Provide progress feedback for long operations

## Troubleshooting

### Common Issues
1. **Registration Failures**: Check import statements and decorator usage
2. **Interface Errors**: Verify all abstract methods are implemented
3. **Configuration Issues**: Validate config parameters in `__init__`
4. **Import Errors**: Ensure all dependencies are available
5. **Type Errors**: Add proper type hints and validate inputs

### Debugging Steps
1. Test component instantiation independently
2. Validate with minimal test cases
3. Check error logs for specific failure points
4. Use registry inspection tools to verify registration
5. Compare with working implementations for patterns