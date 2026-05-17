"""Common interfaces and base classes for module standardization.

This module provides consistent interfaces across all components
following the single responsibility principle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, TypeVar, Generic, Any, Dict, Optional
from enum import Enum


# Generic type variable for result types
T = TypeVar('T')


class ResultStatus(Enum):
    """Standard result status for all operations."""
    SUCCESS = "success"
    PARTIAL = "partial"    # Completed with warnings
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Result(Generic[T]):
    """Standard result wrapper for all operations.
    
    Provides consistent error handling and status reporting.
    
    Attributes:
        status: Operation status
        data: Result data (if successful)
        message: Human-readable message
        warnings: List of warning messages
        errors: List of error messages
    """
    status: ResultStatus = ResultStatus.SUCCESS
    data: Optional[T] = None
    message: str = ""
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def is_success(self) -> bool:
        """Check if operation was successful."""
        return self.status == ResultStatus.SUCCESS
    
    @property
    def is_partial(self) -> bool:
        """Check if operation completed with warnings."""
        return self.status == ResultStatus.PARTIAL
    
    @property
    def is_failed(self) -> bool:
        """Check if operation failed."""
        return self.status == ResultStatus.FAILED
    
    @classmethod
    def ok(cls, data: T = None, message: str = "") -> Result[T]:
        """Create successful result."""
        return cls(status=ResultStatus.SUCCESS, data=data, message=message)
    
    @classmethod
    def partial(cls, data: T = None, message: str = "", warnings: List[str] = None) -> Result[T]:
        """Create partial success result with warnings."""
        return cls(
            status=ResultStatus.PARTIAL,
            data=data,
            message=message,
            warnings=warnings or [],
        )
    
    @classmethod
    def fail(cls, message: str = "", errors: List[str] = None) -> Result[T]:
        """Create failed result."""
        return cls(
            status=ResultStatus.FAILED,
            message=message,
            errors=errors or [],
        )
    
    def add_warning(self, warning: str) -> Result[T]:
        """Add a warning message."""
        self.warnings.append(warning)
        if self.status == ResultStatus.SUCCESS:
            self.status = ResultStatus.PARTIAL
        return self
    
    def add_error(self, error: str) -> Result[T]:
        """Add an error message."""
        self.errors.append(error)
        self.status = ResultStatus.FAILED
        return self


@dataclass
class BaseResult:
    """Base class for result objects with standard fields.
    
    Inherited by all result types to ensure consistency.
    """
    waypoint_count: int = 0
    """Number of waypoints generated."""
    
    processing_time_ms: float = 0.0
    """Processing time in milliseconds."""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""


class BaseGenerator(ABC):
    """Abstract base class for all generators.
    
    All generators (waypoint, path, trajectory) must follow
    this interface for consistency.
    
    Example:
        class WaypointGenerator(BaseGenerator):
            def generate(self, input_data) -> Result[List[Waypoint]]:
                ...
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return generator name for registry and logging."""
        pass
    
    @property
    def version(self) -> str:
        """Return generator version. Override for versioning."""
        return "1.0.0"
    
    @abstractmethod
    def validate(self, *args, **kwargs) -> Result:
        """Validate input parameters.
        
        Returns:
            Result with validation status and any errors
        """
        pass
    
    @abstractmethod
    def generate(self, *args, **kwargs) -> Result:
        """Generate output.
        
        Returns:
            Result with generated data
        """
        pass


class BaseChecker(ABC):
    """Abstract base class for all checkers/validators.
    
    Example:
        class CollisionChecker(BaseChecker):
            def check(self, position) -> Result[bool]:
                ...
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return checker name."""
        pass
    
    @abstractmethod
    def check(self, *args, **kwargs) -> Result:
        """Perform check.
        
        Returns:
            Result with check status
        """
        pass
    
    def batch_check(self, items: List[Any]) -> List[Result]:
        """Batch check multiple items.
        
        Default implementation iterates and checks each item.
        Override for optimized batch processing.
        """
        return [self.check(item) for item in items]


class BaseOptimizer(ABC):
    """Abstract base class for all optimizers.
    
    Example:
        class PathOptimizer(BaseOptimizer):
            def optimize(self, path) -> Result[Path]:
                ...
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return optimizer name."""
        pass
    
    @property
    def max_iterations(self) -> int:
        """Maximum optimization iterations. Default: 100."""
        return 100
    
    @property
    def convergence_threshold(self) -> float:
        """Convergence threshold. Default: 0.001."""
        return 0.001
    
    @abstractmethod
    def optimize(self, *args, **kwargs) -> Result:
        """Perform optimization.
        
        Returns:
            Result with optimized data
        """
        pass
    
    def reset(self) -> None:
        """Reset optimizer state for new optimization run."""
        pass


class ModuleRegistry:
    """Registry for modules following singleton pattern.
    
    Provides centralized registration and lookup for generators,
    checkers, and optimizers.
    """
    
    _instance: Optional[ModuleRegistry] = None
    _generators: Dict[str, Any] = {}
    _checkers: Dict[str, Any] = {}
    _optimizers: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register_generator(self, name: str, generator: Any) -> None:
        """Register a generator."""
        self._generators[name] = generator
    
    def get_generator(self, name: str) -> Optional[Any]:
        """Get a registered generator."""
        return self._generators.get(name)
    
    def list_generators(self) -> List[str]:
        """List all registered generators."""
        return list(self._generators.keys())
    
    def register_checker(self, name: str, checker: Any) -> None:
        """Register a checker."""
        self._checkers[name] = checker
    
    def get_checker(self, name: str) -> Optional[Any]:
        """Get a registered checker."""
        return self._checkers.get(name)
    
    def register_optimizer(self, name: str, optimizer: Any) -> None:
        """Register an optimizer."""
        self._optimizers[name] = optimizer
    
    def get_optimizer(self, name: str) -> Optional[Any]:
        """Get a registered optimizer."""
        return self._optimizers.get(name)


# Convenience function for getting registry
def get_registry() -> ModuleRegistry:
    """Get the global module registry."""
    return ModuleRegistry()
