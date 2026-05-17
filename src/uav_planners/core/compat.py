"""Backward compatibility module.

This module provides aliases to new core config classes for
gradual migration. New code should import from core.config directly.
"""

# Re-export from core.config for backward compatibility
from .config import (
    GeneratorConfig,
    FlightConfig,
    CoverageConfig,
    ViewpointConfig,
    TransitionConfig,
    RegionConfig,
    Result,
    ResultStatus,
)

# Also support importing from geometry.base_generator
# (legacy import path)
from .config import GeneratorConfig as BaseGeneratorConfig

__all__ = [
    'GeneratorConfig',
    'FlightConfig',
    'CoverageConfig',
    'ViewpointConfig',
    'TransitionConfig',
    'RegionConfig',
    'Result',
    'ResultStatus',
    'BaseGeneratorConfig',
]
