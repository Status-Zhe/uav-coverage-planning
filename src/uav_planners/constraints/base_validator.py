"""Base class for constraint validators."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass
from ..models.waypoint import Waypoint


@dataclass
class ValidationResult:
    """Result of constraint validation."""
    valid: bool
    errors: List[str] = None
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.details is None:
            self.details = {}


class BaseValidator(ABC):
    """Abstract base class for constraint validators."""
    
    @abstractmethod
    def validate(self, waypoints: List[Waypoint]) -> List[bool]:
        """Validate waypoints against constraints.
        
        Args:
            waypoints: Waypoints to validate
            
        Returns:
            List of validation results (True = valid)
        """
        pass
