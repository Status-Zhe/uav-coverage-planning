"""Camera-based conversion helpers for oblique distance and GSD."""

from __future__ import annotations

from ..models.camera import Camera


def _validate_camera(camera: Camera) -> None:
    if camera.focal_length_mm <= 0:
        raise ValueError("camera.focal_length_mm must be > 0")
    if camera.sensor_width_mm <= 0:
        raise ValueError("camera.sensor_width_mm must be > 0")
    if camera.sensor_height_mm <= 0:
        raise ValueError("camera.sensor_height_mm must be > 0")
    if camera.resolution_x <= 0:
        raise ValueError("camera.resolution_x must be > 0")
    if camera.resolution_y <= 0:
        raise ValueError("camera.resolution_y must be > 0")


def oblique_dst_srf_to_gsd(camera: Camera, oblique_dst_srf_m: float) -> float:
    """Convert oblique surface distance (m) to conservative GSD (m/pixel).

    Conservative strategy uses both image axes and returns the worse one:
    - gsd_x = (sensor_width * distance) / (focal_length * resolution_x)
    - gsd_y = (sensor_height * distance) / (focal_length * resolution_y)
    - gsd = max(gsd_x, gsd_y)

    Args:
        camera: Camera intrinsics model.
        oblique_dst_srf_m: Distance from camera center to target surface in meters.

    Returns:
        Ground sampling distance in meters/pixel.
    """
    _validate_camera(camera)
    if oblique_dst_srf_m <= 0:
        raise ValueError("oblique_dst_srf_m must be > 0")

    sensor_width_m = camera.sensor_width_mm / 1000.0
    sensor_height_m = camera.sensor_height_mm / 1000.0
    focal_length_m = camera.focal_length_mm / 1000.0
    gsd_x = (sensor_width_m * oblique_dst_srf_m) / (focal_length_m * camera.resolution_x)
    gsd_y = (sensor_height_m * oblique_dst_srf_m) / (focal_length_m * camera.resolution_y)
    return max(gsd_x, gsd_y)


def gsd_to_global_distance_m(camera: Camera, gsd_m_per_pixel: float) -> float:
    """Convert target GSD (m/pixel) to conservative global distance (m).

    Conservative strategy requires both axes to satisfy target GSD, so
    returns the smaller of axis-limited distances:
    - dist_x = (gsd * focal_length * resolution_x) / sensor_width
    - dist_y = (gsd * focal_length * resolution_y) / sensor_height
    - distance = min(dist_x, dist_y)

    Args:
        camera: Camera intrinsics model.
        gsd_m_per_pixel: Target ground sampling distance in meters/pixel.

    Returns:
        Required global stand-off distance in meters.
    """
    _validate_camera(camera)
    if gsd_m_per_pixel <= 0:
        raise ValueError("gsd_m_per_pixel must be > 0")

    sensor_width_m = camera.sensor_width_mm / 1000.0
    sensor_height_m = camera.sensor_height_mm / 1000.0
    focal_length_m = camera.focal_length_mm / 1000.0
    dist_x = (gsd_m_per_pixel * focal_length_m * camera.resolution_x) / sensor_width_m
    dist_y = (gsd_m_per_pixel * focal_length_m * camera.resolution_y) / sensor_height_m
    return min(dist_x, dist_y)


def gsd_to_oblique_dst_srf(camera: Camera, gsd_m_per_pixel: float) -> float:
    """Backward-compatible alias for gsd_to_global_distance_m."""
    return gsd_to_global_distance_m(camera, gsd_m_per_pixel)
