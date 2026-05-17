"""Enhanced facade analyzer for complex building structures.

This module provides specialized handling for:
- Complex facades (凹凸立面)
- Corridors/bridges between buildings (连廊)
- Multi-level overlapping areas
- Hollow and recessed structures
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.ops import unary_union, linemerge

logger = logging.getLogger(__name__)


class CorridorDetector:
    """Detect corridors/bridges between building structures.
    
    Corridors are narrow connections between separate buildings that should
    be treated as part of the coverage area.
    """
    
    def __init__(
        self,
        min_corridor_width: float = 2.0,
        max_corridor_width: float = 15.0,
        min_corridor_length: float = 3.0,
    ):
        """Initialize corridor detector.
        
        Args:
            min_corridor_width: Minimum width to be considered a corridor
            max_corridor_width: Maximum width for corridor classification
            min_corridor_length: Minimum length for corridor detection
        """
        self.min_corridor_width = min_corridor_width
        self.max_corridor_width = max_corridor_width
        self.min_corridor_length = min_corridor_length
    
    def detect_corridors(
        self,
        footprints: List[Polygon],
        point_cloud: Optional[np.ndarray] = None,
    ) -> List[Polygon]:
        """Detect corridors between building footprints.
        
        Args:
            footprints: List of building footprint polygons
            point_cloud: Optional point cloud for verification
        
        Returns:
            List of corridor polygons detected
        """
        if len(footprints) < 2:
            return []
        
        corridors = []
        
        # Check gaps between adjacent footprints
        for i, fp1 in enumerate(footprints):
            for fp2 in footprints[i + 1:]:
                corridor = self._find_corridor_between(fp1, fp2)
                if corridor is not None:
                    corridors.append(corridor)
        
        return corridors
    
    def _find_corridor_between(
        self,
        fp1: Polygon,
        fp2: Polygon,
    ) -> Optional[Polygon]:
        """Find corridor connection between two footprints."""
        # Get bounding boxes
        min_dist = self._min_distance_between(fp1, fp2)
        
        if min_dist > self.max_corridor_width:
            return None
        
        if min_dist < self.min_corridor_width:
            return None  # Too close, not a corridor
        
        # Try to extract corridor shape
        corridor_polygon = self._extract_corridor_polygon(fp1, fp2, min_dist)
        
        if corridor_polygon is not None:
            area = corridor_polygon.area
            length = self._estimate_length(corridor_polygon)
            
            if area > 0 and length >= self.min_corridor_length:
                return corridor_polygon
        
        return None
    
    def _min_distance_between(self, fp1: Polygon, fp2: Polygon) -> float:
        """Calculate minimum distance between two polygons."""
        return fp1.distance(fp2)
    
    def _extract_corridor_polygon(
        self,
        fp1: Polygon,
        fp2: Polygon,
        distance: float,
    ) -> Optional[Polygon]:
        """Extract corridor polygon from gap between buildings."""
        # Use buffer to expand footprints and find intersection
        buffer_dist = distance / 2
        
        expanded1 = fp1.buffer(buffer_dist)
        expanded2 = fp2.buffer(buffer_dist)
        
        # Find the overlap region that connects both
        intersection = expanded1.intersection(expanded2)
        
        if intersection.is_empty or intersection.area < 1.0:
            return None
        
        # Simplify and return
        if isinstance(intersection, Polygon):
            return intersection.simplify(0.1, preserve_topology=True)
        elif isinstance(intersection, MultiPolygon):
            return max(intersection.geoms, key=lambda p: p.area)
        
        return None
    
    def _estimate_length(self, polygon: Polygon) -> float:
        """Estimate the length of a corridor polygon."""
        minx, miny, maxx, maxy = polygon.bounds
        return max(maxx - minx, maxy - miny)


class RecessedAreaDetector:
    """Detect and handle recessed areas in building facades.
    
    Recessed areas are concave regions that need special handling
    for proper coverage planning.
    """
    
    def __init__(
        self,
        min_recess_depth: float = 1.0,
        min_recess_area: float = 4.0,
    ):
        """Initialize recessed area detector.
        
        Args:
            min_recess_depth: Minimum depth to be considered a recess
            min_recess_area: Minimum area for recess classification
        """
        self.min_recess_depth = min_recess_depth
        self.min_recess_area = min_recess_area
    
    def detect_recesses(
        self,
        footprint: Polygon,
        building_outline: Polygon,
    ) -> List[Polygon]:
        """Detect recessed areas in a building footprint.
        
        Args:
            footprint: Inner building footprint
            building_outline: Outer building outline
        
        Returns:
            List of recess polygons
        """
        if building_outline.contains(footprint):
            # Find the difference between outer and inner
            diff = building_outline.difference(footprint)
            
            if isinstance(diff, Polygon):
                return [diff] if diff.area >= self.min_recess_area else []
            elif isinstance(diff, MultiPolygon):
                return [p for p in diff.geoms if p.area >= self.min_recess_area]
        
        return []
    
    def smooth_concave_edges(
        self,
        polygon: Polygon,
        max_concave_angle: float = 120.0,
    ) -> Polygon:
        """Smooth concave edges to improve coverage.
        
        Args:
            polygon: Input polygon with possible concave edges
            max_concave_angle: Maximum angle for concave corner (degrees)
        
        Returns:
            Smoothed polygon
        """
        # Simplified smoothing by buffer operations
        # Positive buffer inflates, negative deflates
        # Small oscillation smooths sharp corners
        
        smoothed = polygon.buffer(0.1).buffer(-0.1)
        
        if smoothed.is_empty or not smoothed.is_valid:
            return polygon
        
        return smoothed


class FacadeAnalyzer:
    """Main facade analyzer coordinating all facade analysis components.
    
    Provides unified interface for complex facade structure analysis.
    """
    
    def __init__(
        self,
        corridor_config: Optional[dict] = None,
        recess_config: Optional[dict] = None,
    ):
        """Initialize facade analyzer.
        
        Args:
            corridor_config: Configuration for corridor detection
            recess_config: Configuration for recessed area detection
        """
        self.corridor_config = corridor_config or {}
        self.recess_config = recess_config or {}
        
        self.corridor_detector = CorridorDetector(
            min_corridor_width=self.corridor_config.get("min_width", 2.0),
            max_corridor_width=self.corridor_config.get("max_width", 15.0),
            min_corridor_length=self.corridor_config.get("min_length", 3.0),
        )
        
        self.recess_detector = RecessedAreaDetector(
            min_recess_depth=self.recess_config.get("min_depth", 1.0),
            min_recess_area=self.recess_config.get("min_area", 4.0),
        )
    
    def analyze_complex_facade(
        self,
        footprints: List[Polygon],
        point_cloud: Optional[np.ndarray] = None,
    ) -> dict:
        """Analyze complex facade structures.
        
        Args:
            footprints: List of building footprint polygons
            point_cloud: Point cloud for verification
        
        Returns:
            Dictionary with analysis results:
                - corridors: Detected corridor polygons
                - recesses: Detected recessed areas
                - merged_footprint: Combined footprint including corridors
        """
        result = {
            "corridors": [],
            "recesses": [],
            "merged_footprint": None,
        }
        
        if not footprints:
            return result
        
        # Detect corridors
        corridors = self.corridor_detector.detect_corridors(footprints, point_cloud)
        result["corridors"] = corridors
        
        # Merge footprints with corridors for unified coverage
        all_polygons = list(footprints) + corridors
        
        if len(all_polygons) > 1:
            try:
                merged = unary_union(all_polygons)
                if isinstance(merged, Polygon):
                    result["merged_footprint"] = merged
                elif isinstance(merged, MultiPolygon):
                    result["merged_footprint"] = max(merged.geoms, key=lambda p: p.area)
            except Exception as e:
                logger.warning(f"Failed to merge footprints: {e}")
                result["merged_footprint"] = footprints[0]
        else:
            result["merged_footprint"] = footprints[0]
        
        return result
    
    def enhance_coverage_footprint(
        self,
        footprint: Polygon,
        buffer_distance: float = 0.5,
    ) -> Polygon:
        """Enhance footprint for better coverage planning.
        
        Applies smoothing and buffering to handle complex geometries.
        
        Args:
            footprint: Input footprint polygon
            buffer_distance: Buffer distance for smoothing
        
        Returns:
            Enhanced footprint polygon
        """
        if footprint.is_empty or not footprint.is_valid:
            return footprint
        
        # Apply small buffer to smooth edges
        enhanced = footprint.buffer(buffer_distance).buffer(-buffer_distance)
        
        if enhanced.is_empty or not enhanced.is_valid:
            return footprint
        
        return enhanced


def merge_adjacent_footprints(
    footprints: List[Polygon],
    merge_distance: float = 1.0,
) -> List[Polygon]:
    """Merge adjacent footprints based on distance threshold.
    
    Args:
        footprints: List of footprint polygons
        merge_distance: Distance threshold for merging
    
    Returns:
        List of merged footprints
    """
    if len(footprints) <= 1:
        return footprints
    
    # Buffer and merge
    buffered = [fp.buffer(merge_distance) for fp in footprints]
    merged = unary_union(buffered)
    
    if isinstance(merged, Polygon):
        # Deflate back to original size
        result = merged.buffer(-merge_distance)
        return [result] if result.is_valid and result.area > 0 else footprints
    elif isinstance(merged, MultiPolygon):
        result = [p.buffer(-merge_distance) for p in merged.geoms]
        return [p for p in result if p.is_valid and p.area > 0]
    
    return footprints
