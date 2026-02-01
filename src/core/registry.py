"""Component registry system for managing multiple implementations"""

from typing import Dict, Type, Any, List, Optional
from abc import ABC, abstractmethod
import inspect
from pathlib import Path


class ComponentRegistry:
    """Registry for managing multiple implementations of components"""
    
    def __init__(self):
        self._components = {
            'judges': {},
            'chunkers': {},
            'embedders': {},
            'retrievers': {},
            'generators': {}
        }
    
    def register(self, component_type: str, name: str, implementation: Type, **metadata):
        """Register a component implementation"""
        if component_type not in self._components:
            raise ValueError(f"Unknown component type: {component_type}")
        
        self._components[component_type][name] = {
            'implementation': implementation,
            'metadata': metadata
        }
    
    def get(self, component_type: str, name: str) -> Type:
        """Get a component implementation by type and name"""
        if component_type not in self._components:
            raise ValueError(f"Unknown component type: {component_type}")
        
        if name not in self._components[component_type]:
            available = list(self._components[component_type].keys())
            raise ValueError(f"Unknown {component_type} implementation: {name}. Available: {available}")
        
        return self._components[component_type][name]['implementation']
    
    def list_implementations(self, component_type: str) -> List[str]:
        """List all available implementations for a component type"""
        return list(self._components[component_type].keys())
    
    def get_metadata(self, component_type: str, name: str) -> Dict[str, Any]:
        """Get metadata for a specific implementation"""
        component = self._components[component_type][name]
        return component['metadata']
    
    def compare_implementations(self, component_type: str) -> Dict[str, Dict[str, Any]]:
        """Get comparison data for all implementations of a type"""
        implementations = {}
        for name, data in self._components[component_type].items():
            implementations[name] = {
                'class': data['implementation'].__name__,
                'module': data['implementation'].__module__,
                **data['metadata']
            }
        return implementations


# Global registry instance
registry = ComponentRegistry()


def register_component(component_type: str, name: str, **metadata):
    """Decorator to register a component implementation"""
    def decorator(cls):
        registry.register(component_type, name, cls, **metadata)
        return cls
    return decorator


# Convenience functions
def register_judge(name: str, **metadata):
    return register_component('judges', name, **metadata)

def register_chunker(name: str, **metadata):
    return register_component('chunkers', name, **metadata)

def register_embedder(name: str, **metadata):
    return register_component('embedders', name, **metadata)

def register_retriever(name: str, **metadata):
    return register_component('retrievers', name, **metadata)

def register_generator(name: str, **metadata):
    return register_component('generators', name, **metadata)


def get_judge(name: str):
    """Get a judge implementation"""
    return registry.get('judges', name)

def get_chunker(name: str):
    """Get a chunker implementation"""
    return registry.get('chunkers', name)

def get_embedder(name: str):
    """Get an embedder implementation"""
    return registry.get('embedders', name)

def get_retriever(name: str):
    """Get a retriever implementation"""
    return registry.get('retrievers', name)

def get_generator(name: str):
    """Get a generator implementation"""
    return registry.get('generators', name)


def list_all_components() -> Dict[str, List[str]]:
    """List all registered components"""
    return {
        component_type: registry.list_implementations(component_type)
        for component_type in ['judges', 'chunkers', 'embedders', 'retrievers', 'generators']
    }