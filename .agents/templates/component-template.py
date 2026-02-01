"""Template for creating new RAGLab components"""

from typing import List, Dict, Any, Optional
from ..base.base_{component_type} import Base{ComponentType}
from ...core.interfaces import {ResultType}  # Replace with appropriate result type
from ...core.registry import register_{component_type}


@register_{component_type}(
    name="{component_name}",
    description="{component_description}",
    # Add relevant metadata
    approach="{approach_name}",
    provider="{provider_name}",  # If applicable
    version="1.0.0"
)
class {ComponentClassName}(Base{ComponentType}):
    """
    {Detailed component description}
    
    This component implements {approach description} for {use case}.
    
    Example:
        >>> component = {ComponentClassName}(param1="value", param2=42)
        >>> result = component.main_method(input_data)
        >>> print(result)
    
    Attributes:
        param1: Description of parameter 1
        param2: Description of parameter 2
        config: Configuration dictionary
    """
    
    def __init__(
        self, 
        param1: str,
        param2: int = 10,
        **config
    ):
        """
        Initialize {component_type} with configuration
        
        Args:
            param1: Required parameter description
            param2: Optional parameter with default value
            **config: Additional configuration parameters
            
        Raises:
            ValueError: If param1 is empty or invalid
            TypeError: If param2 is not an integer
        """
        super().__init__(**config)
        
        # Validate parameters
        if not param1 or not isinstance(param1, str):
            raise ValueError("param1 must be a non-empty string")
        
        if not isinstance(param2, int) or param2 <= 0:
            raise TypeError("param2 must be a positive integer")
        
        # Store configuration
        self.param1 = param1
        self.param2 = param2
        
        # Initialize component state
        self._initialized = False
        self._initialize_component()
    
    def get_name(self) -> str:
        """Get the name/identifier of this {component_type} implementation"""
        return "{component_name}"
    
    def get_description(self) -> str:
        """Get a human-readable description of this {component_type}"""
        return f"{component_description} (param1={self.param1}, param2={self.param2})"
    
    # Main component methods - implement based on component type
    
    def main_method(
        self, 
        input_data: {InputType},
        **kwargs
    ) -> {ResultType}:
        """
        Main processing method for this component
        
        Args:
            input_data: Input data to process
            **kwargs: Additional processing parameters
            
        Returns:
            Processed result of appropriate type
            
        Raises:
            ValueError: If input_data is invalid
            RuntimeError: If processing fails
        """
        try:
            # Input validation
            self._validate_input(input_data)
            
            # Main processing logic
            result = self._process_data(input_data, **kwargs)
            
            # Validate and return result
            self._validate_result(result)
            return result
            
        except Exception as e:
            logger.error(f"Error in {self.get_name()}: {e}")
            raise RuntimeError(f"Component {self.get_name()} failed: {e}") from e
    
    # Private helper methods
    
    def _initialize_component(self):
        """Initialize component-specific resources"""
        try:
            # Component initialization logic
            # e.g., load models, connect to services, etc.
            self._initialized = True
            logger.info(f"Initialized {self.get_name()}")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.get_name()}: {e}")
            raise RuntimeError(f"Component initialization failed: {e}") from e
    
    def _validate_input(self, input_data: {InputType}):
        """Validate input data"""
        if input_data is None:
            raise ValueError("Input data cannot be None")
        
        # Add specific validation logic
        if hasattr(input_data, '__len__') and len(input_data) == 0:
            raise ValueError("Input data cannot be empty")
    
    def _process_data(self, input_data: {InputType}, **kwargs) -> {ResultType}:
        """
        Core processing logic - implement based on component type
        
        This is where the main component functionality should be implemented.
        """
        # Example processing logic - replace with actual implementation
        processed_result = {ResultType}(
            # Initialize result fields based on component type
            # For example:
            # response=f"Processed: {input_data}",
            # metadata={'component': self.get_name(), 'param1': self.param1}
        )
        
        return processed_result
    
    def _validate_result(self, result: {ResultType}):
        """Validate output result"""
        if result is None:
            raise RuntimeError("Component produced None result")
        
        # Add result-specific validation
        # For example, check required fields are populated
        
    def get_component_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about this component
        
        Returns:
            Dictionary with component information
        """
        return {
            'name': self.get_name(),
            'description': self.get_description(),
            'config': self.get_config(),
            'initialized': self._initialized,
            'type': '{component_type}',
            'version': '1.0.0'
        }


# Example usage and testing
if __name__ == "__main__":
    # Basic component testing
    try:
        component = {ComponentClassName}(param1="test", param2=5)
        print(f"Created component: {component.get_description()}")
        
        # Test with sample data
        # sample_input = {create appropriate sample input}
        # result = component.main_method(sample_input)
        # print(f"Result: {result}")
        
    except Exception as e:
        print(f"Component test failed: {e}")


# Template replacement instructions:
# 1. Replace {component_type} with: judge, chunker, embedder, retriever, generator
# 2. Replace {ComponentType} with: Judge, Chunker, Embedder, Retriever, Generator  
# 3. Replace {ResultType} with appropriate result type from core.interfaces
# 4. Replace {InputType} with expected input type
# 5. Replace {component_name} with your component's registry name
# 6. Replace {ComponentClassName} with your class name
# 7. Replace {component_description} with description text
# 8. Replace {approach_name} and {provider_name} with relevant metadata
# 9. Implement component-specific logic in _process_data method
# 10. Add component-specific validation and configuration as needed