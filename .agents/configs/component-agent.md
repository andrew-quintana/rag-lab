# Component Agent Configuration

Agent-specific rules and constraints for agents developing RAGLab components.

## Scope

This configuration applies to agents:
- Creating new component implementations (judges, chunkers, embedders, retrievers, generators)
- Modifying existing component interfaces and base classes
- Working with the component registry system
- Implementing component-specific features and optimizations

## Component Development Standards

### Interface Compliance
- All components must inherit from appropriate base class in `src/evaluation/base/`
- Implement all abstract methods defined in base class
- Follow method signatures exactly as specified in base class
- Return appropriate data types as defined in core interfaces

### Registry Integration
- Use `@register_{component_type}` decorator for all new components
- Provide meaningful name and description in registration
- Include relevant metadata for component discovery and comparison
- Ensure registration happens at import time

### Configuration Management
- Accept configuration via `**config` parameter in `__init__`
- Store configuration for later inspection via `get_config()`
- Validate configuration parameters in constructor
- Provide sensible defaults for optional parameters

## Component-Specific Rules

### Judge Components
```python
# Required interface compliance
class YourJudge(BaseJudge):
    def evaluate(self, question, reference_answer, generated_answer, context_chunks):
        # Must return JudgeEvaluationResult
        pass
    
    def get_name(self) -> str:
        # Must return unique identifier
        pass
    
    def get_description(self) -> str:
        # Must return human-readable description
        pass
```

**Constraints:**
- Must handle insurance risk semantics correctly
- Support 4-stage evaluation when extending MultiStageJudge
- Parse LLM responses robustly with error handling
- Generate meaningful reasoning text

### Chunker Components
```python
class YourChunker(BaseChunker):
    def chunk_text(self, text, document_id=None, **kwargs) -> List[Chunk]:
        # Must return list of Chunk objects
        pass
```

**Constraints:**
- Handle empty or very short texts gracefully
- Generate unique chunk IDs with document relationships
- Include relevant metadata in chunks
- Support configurable chunk sizes and overlap

### Embedder Components
```python
class YourEmbedder(BaseEmbedder):
    def embed_texts(self, texts: List[str], **kwargs) -> np.ndarray:
        # Must return numpy array with shape (len(texts), embedding_dim)
        pass
    
    def get_embedding_dimension(self) -> int:
        # Must return consistent dimension
        pass
```

**Constraints:**
- Return consistent embedding dimensions
- Handle empty text lists appropriately
- Support batch processing for efficiency
- Use numpy arrays for all embedding outputs

### Retriever Components
```python
class YourRetriever(BaseRetriever):
    def retrieve(self, query: str, k: int = 10, **kwargs) -> List[RetrievalResult]:
        # Must return sorted list by relevance score
        pass
```

**Constraints:**
- Return results sorted by decreasing relevance score
- Support configurable retrieval count (k parameter)
- Include proper scoring and ranking information
- Handle index initialization and updates

### Generator Components
```python
class YourGenerator(BaseGenerator):
    def generate(self, prompt: str, context_chunks=None, **kwargs) -> GenerationResult:
        # Must return GenerationResult with response and metadata
        pass
```

**Constraints:**
- Integrate context chunks appropriately
- Track token usage and model information
- Support both prompt-based and chat-based generation
- Handle generation failures gracefully

## Quality Standards

### Error Handling
```python
def component_method(self, input_data):
    try:
        if not input_data:
            raise ValueError(f"{self.get_name()}: Input data cannot be empty")
        
        result = self._process(input_data)
        return result
    except Exception as e:
        logger.error(f"Error in {self.get_name()}: {e}")
        # Either handle gracefully or re-raise with context
        raise ComponentError(f"Component {self.get_name()} failed: {e}") from e
```

### Performance Requirements
- Components should complete typical operations within reasonable time
- Memory usage should be proportional to input size
- Support batch processing when applicable
- Avoid blocking operations without timeout

### Logging and Debugging
- Use appropriate logging levels (DEBUG for detailed info, INFO for progress, ERROR for failures)
- Include component name in log messages for identification
- Log configuration and important state changes
- Avoid logging sensitive information

## Security Requirements

### Input Validation
```python
def process_input(self, user_input):
    if not isinstance(user_input, str):
        raise TypeError("Input must be string")
    
    if len(user_input) > MAX_INPUT_SIZE:
        raise ValueError("Input exceeds maximum size")
    
    # Additional validation as needed
    return self._safe_process(user_input)
```

### Resource Management
- Clean up resources properly in destructors or context managers
- Avoid loading large models unnecessarily
- Implement proper timeout handling for external calls
- Validate file paths and prevent directory traversal

### Data Protection
- Never log or store sensitive information
- Sanitize inputs that might contain personal data
- Use secure communication for external API calls
- Handle authentication credentials securely

## Testing Requirements

### Unit Testing
```python
def test_component_basic_functionality():
    component = YourComponent(param1="test", param2=42)
    
    # Test basic interface compliance
    assert component.get_name() == "expected_name"
    assert component.get_description()
    assert component.get_config()['param1'] == "test"
    
    # Test core functionality
    result = component.main_method(test_input)
    assert isinstance(result, ExpectedType)
    assert result.validate()
```

### Integration Testing
```python
def test_component_integration():
    # Test with evaluation pipeline
    pipeline = RAGEvaluationPipeline(
        component_name='your_component',
        component_configs={'component': {'param': 'value'}}
    )
    
    results = pipeline.evaluate_dataset(small_test_dataset)
    assert len(results) > 0
    assert all(r.validate() for r in results)
```

### Error Testing
```python
def test_component_error_handling():
    component = YourComponent()
    
    # Test empty input
    with pytest.raises(ValueError):
        component.process("")
    
    # Test invalid input
    with pytest.raises(TypeError):
        component.process(None)
```

## Documentation Standards

### Docstring Requirements
```python
def component_method(self, param1: str, param2: int = 10) -> ResultType:
    """
    Brief description of what the method does.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter with default
        
    Returns:
        Description of return value and its structure
        
    Raises:
        ValueError: When param1 is empty
        TypeError: When param2 is not an integer
        
    Example:
        >>> component = YourComponent()
        >>> result = component.component_method("test", 5)
        >>> print(result.value)
        
    Note:
        Any additional important information about usage or behavior.
    """
```

### Component Documentation
- Include usage examples in component file or separate example
- Document configuration parameters and their effects
- Explain performance characteristics and limitations
- Provide troubleshooting guidance for common issues

## Deployment and Versioning

### Version Management
- Consider semantic versioning for significant component changes
- Document breaking changes and migration paths
- Maintain backward compatibility when possible
- Provide deprecation warnings for removed features

### Configuration Compatibility
- Support old configuration formats when possible
- Validate configuration parameters and provide helpful error messages
- Document configuration schema changes
- Provide migration tools for configuration updates