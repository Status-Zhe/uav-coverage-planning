"""Geometry generation layer (Stage 1 of pipeline)."""

from .base_generator import BaseGeometryGenerator, GeneratorConfig
from .registry import register_generator, get_generator, GENERATOR_REGISTRY
from .facade_analyzer import (
    FacadeAnalyzer,
    CorridorDetector,
    RecessedAreaDetector,
    merge_adjacent_footprints,
)

__all__ = [
    "BaseGeometryGenerator",
    "GeneratorConfig",
    "register_generator",
    "get_generator",
    "GENERATOR_REGISTRY",
    "FacadeAnalyzer",
    "CorridorDetector",
    "RecessedAreaDetector",
    "merge_adjacent_footprints",
]
