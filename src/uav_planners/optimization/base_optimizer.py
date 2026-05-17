"""Base class for optimizers."""

from abc import ABC, abstractmethod
from typing import List
from ..models.waypoint import Waypoint


class BaseOptimizer(ABC):
    """Abstract base class for trajectory optimizers."""
    
    @abstractmethod
    def optimize(self, waypoints: List[Waypoint]) -> List[Waypoint]:
        """Optimize waypoint sequence.
        
        Args:
            waypoints: Input waypoints
            
        Returns:
            Optimized waypoints
        """
        pass
