"""Camera model for UAV coverage planning."""

from dataclasses import dataclass
import math


@dataclass
class Camera:
    """Camera specification for coverage calculation.
    
    Attributes:
        focal_length_mm: Focal length in millimeters
        sensor_width_mm: Sensor width in millimeters
        sensor_height_mm: Sensor height in millimeters
        resolution_x: Image width in pixels
        resolution_y: Image height in pixels
    """
    focal_length_mm: float
    sensor_width_mm: float
    sensor_height_mm: float
    resolution_x: int
    resolution_y: int
    
    @property
    def fov_horizontal_deg(self) -> float:
        """Calculate horizontal field of view in degrees."""
        return 2 * math.degrees(math.atan(
            self.sensor_width_mm / (2 * self.focal_length_mm)
        ))
    
    @property
    def fov_vertical_deg(self) -> float:
        """Calculate vertical field of view in degrees."""
        return 2 * math.degrees(math.atan(
            self.sensor_height_mm / (2 * self.focal_length_mm)
        ))
    
    def gsd_at_altitude(self, altitude_m: float) -> float:
        """Calculate ground sampling distance (meters per pixel) at given altitude.
        
        Args:
            altitude_m: Altitude above ground in meters
            
        Returns:
            Ground sampling distance in meters per pixel
        """
        # Convert mm to m
        sensor_width_m = self.sensor_width_mm / 1000.0
        focal_length_m = self.focal_length_mm / 1000.0
        
        # GSD = (sensor_width * altitude) / (focal_length * resolution)
        return (sensor_width_m * altitude_m) / (focal_length_m * self.resolution_x)
