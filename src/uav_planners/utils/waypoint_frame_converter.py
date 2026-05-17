"""Waypoint coordinate frame conversion helpers for export."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def _ecef_to_geodetic(origin_ecef: np.ndarray) -> Tuple[float, float, float]:
    x, y, z = float(origin_ecef[0]), float(origin_ecef[1]), float(origin_ecef[2])
    lon = math.atan2(y, x)
    p = math.hypot(x, y)

    if p < 1e-9:
        lat = math.copysign(math.pi / 2.0, z)
        h = abs(z) - WGS84_A * (1.0 - WGS84_F)
        return lat, lon, h

    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(8):
        sin_lat = math.sin(lat)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        h = p / max(math.cos(lat), 1e-12) - n
        lat = math.atan2(z, p * (1.0 - WGS84_E2 * n / max(n + h, 1e-12)))

    sin_lat = math.sin(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    h = p / max(math.cos(lat), 1e-12) - n
    return lat, lon, h


def _enu_rotation_matrix(origin_ecef: np.ndarray) -> np.ndarray:
    lat, lon, _ = _ecef_to_geodetic(origin_ecef)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)

    return np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=np.float64,
    )


def enu_to_ecef(x: float, y: float, z: float, enu_origin_ecef: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Convert one ENU point to ECEF."""
    origin = np.asarray(enu_origin_ecef, dtype=np.float64)
    enu = np.array([x, y, z], dtype=np.float64)
    rot = _enu_rotation_matrix(origin)
    delta_ecef = enu @ rot
    ecef = origin + delta_ecef
    return float(ecef[0]), float(ecef[1]), float(ecef[2])


def enu_heading_to_ecef_heading(
    heading_deg: float,
    x_enu: float,
    y_enu: float,
    z_enu: float,
    enu_origin_ecef: Tuple[float, float, float],
) -> float:
    """Convert ENU heading to heading in local tangent frame at converted ECEF point.

    The returned value remains a yaw in degrees (0=north, 90=east), but evaluated
    at the destination geodetic tangent frame.
    """
    heading_rad = math.radians(float(heading_deg))
    direction_enu = np.array(
        [math.sin(heading_rad), math.cos(heading_rad), 0.0],
        dtype=np.float64,
    )

    origin = np.asarray(enu_origin_ecef, dtype=np.float64)
    rot_origin = _enu_rotation_matrix(origin)

    base_enu = np.array([x_enu, y_enu, z_enu], dtype=np.float64)
    base_ecef = origin + (base_enu @ rot_origin)

    step_m = 1.0
    next_enu = base_enu + direction_enu * step_m
    next_ecef = origin + (next_enu @ rot_origin)

    delta_ecef = next_ecef - base_ecef

    local_rot = _enu_rotation_matrix(base_ecef)
    delta_local_enu = delta_ecef @ local_rot.T

    east = float(delta_local_enu[0])
    north = float(delta_local_enu[1])
    return float(math.degrees(math.atan2(east, north)))


def can_convert_enu_to_ecef(metadata: Dict) -> bool:
    """Check if export metadata has enough context for ENU->ECEF conversion."""
    planning_frame = str(metadata.get("planning_frame", "")).lower()
    source_frame = str(metadata.get("source_frame", "")).lower()
    origin = metadata.get("enu_origin_ecef", None)
    if planning_frame != "enu":
        return False
    if source_frame != "ecef":
        return False
    if origin is None:
        return False
    if not isinstance(origin, (list, tuple)) or len(origin) != 3:
        return False
    return True


def maybe_convert_waypoint_pose(
    x: float,
    y: float,
    z: float,
    heading_deg: float,
    metadata: Dict,
) -> Optional[Tuple[float, float, float, float]]:
    """Convert waypoint pose from ENU to ECEF based on export metadata."""
    if not can_convert_enu_to_ecef(metadata):
        return None

    origin = tuple(float(value) for value in metadata["enu_origin_ecef"])
    x_ecef, y_ecef, z_ecef = enu_to_ecef(x, y, z, origin)
    heading_ecef = enu_heading_to_ecef_heading(heading_deg, x, y, z, origin)
    return x_ecef, y_ecef, z_ecef, heading_ecef
