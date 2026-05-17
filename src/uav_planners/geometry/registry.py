"""Generator registry for pluggable algorithms."""

from typing import Dict, Type, Callable
from .base_generator import BaseGeometryGenerator
import importlib
import pkgutil


# Global registry mapping generator names to classes
GENERATOR_REGISTRY: Dict[str, Type[BaseGeometryGenerator]] = {}

# def auto_discover():
#     """自动发现并注册所有生成器模块"""
#     for module_info in pkgutil.iter_modules(generators.__path__):
#         importlib.import_module(f"generators.{module_info.name}")

def register_generator(name: str) -> Callable:
    """Decorator to register a geometry generator.
    
    Usage:
        @register_generator("boustrophedon")
        class BoustrophedonGenerator(BaseGeometryGenerator):
            ...
    
    Args:
        name: Unique identifier for the generator
        
    Returns:
        Decorator function that registers the class
        
    Raises:
        ValueError: If name is already registered
    """
    def decorator(cls: Type[BaseGeometryGenerator]) -> Type[BaseGeometryGenerator]:
        if name in GENERATOR_REGISTRY:
            raise ValueError(
                f"Generator '{name}' is already registered. "
                f"Existing: {GENERATOR_REGISTRY[name].__name__}"
            )
        
        # Verify the class is a valid generator
        if not issubclass(cls, BaseGeometryGenerator):
            raise TypeError(
                f"Class {cls.__name__} must inherit from BaseGeometryGenerator"
            )
        
        GENERATOR_REGISTRY[name] = cls
        return cls
    
    return decorator


def get_generator(name: str) -> Type[BaseGeometryGenerator]:
    """Get a generator class by name.
    
    Args:
        name: Registered generator name
        
    Returns:
        Generator class
        
    Raises:
        ValueError: If generator name is not registered
        
    Example:
        GeneratorClass = get_generator("boustrophedon")
        generator = GeneratorClass()
        waypoints = generator.generate(pointcloud, camera, config)
    """
    if name not in GENERATOR_REGISTRY:
        available = list(GENERATOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown generator: '{name}'. "
            f"Available: {available}"
        )
    return GENERATOR_REGISTRY[name]


def list_generators() -> list:
    """List all registered generator names.
    
    Returns:
        List of registered generator names
    """
    return list(GENERATOR_REGISTRY.keys())


def unregister_generator(name: str) -> None:
    """Remove a generator from the registry.
    
    Mainly used for testing purposes.
    
    Args:
        name: Generator name to remove
    """
    if name in GENERATOR_REGISTRY:
        del GENERATOR_REGISTRY[name]
