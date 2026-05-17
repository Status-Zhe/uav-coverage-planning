"""3D Tiles (.json + .pnts) loader utilities."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..models.pointcloud import PointCloud


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def _tiles_transform_to_matrix(transform: Optional[List[float]]) -> np.ndarray:
    if not transform:
        return np.eye(4, dtype=float)
    matrix = np.array(transform, dtype=float).reshape((4, 4), order="F")
    return matrix


def _parse_pnts_file(file_path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    raw = file_path.read_bytes()
    if len(raw) < 28:
        raise ValueError(f"Invalid PNTS file (too small): {file_path}")

    magic = raw[0:4]
    if magic != b"pnts":
        raise ValueError(f"Invalid PNTS magic in {file_path}: {magic}")

    _, byte_length, ft_json_len, ft_bin_len, bt_json_len, bt_bin_len = struct.unpack("<6I", raw[4:28])
    if byte_length > len(raw):
        raise ValueError(f"Invalid PNTS byteLength in {file_path}")

    offset = 28
    ft_json_bytes = raw[offset: offset + ft_json_len]
    offset += ft_json_len
    ft_bin = raw[offset: offset + ft_bin_len]
    offset += ft_bin_len
    _ = raw[offset: offset + bt_json_len]
    offset += bt_json_len
    _ = raw[offset: offset + bt_bin_len]

    ft_json_text = ft_json_bytes.decode("utf-8", errors="ignore").strip(" \t\n\r\x00")
    ft: Dict = json.loads(ft_json_text) if ft_json_text else {}

    points_len = int(ft.get("POINTS_LENGTH", 0))
    if points_len <= 0:
        return np.empty((0, 3), dtype=np.float64), None

    positions = None
    if "POSITION" in ft:
        pos_offset = int(ft["POSITION"].get("byteOffset", 0))
        required = pos_offset + points_len * 3 * 4
        if required > len(ft_bin):
            raise ValueError(f"POSITION buffer overflow in {file_path}")
        positions = np.frombuffer(ft_bin[pos_offset:pos_offset + points_len * 3 * 4], dtype="<f4").reshape((-1, 3)).astype(np.float64)
    elif "POSITION_QUANTIZED" in ft:
        q_offset = int(ft["POSITION_QUANTIZED"].get("byteOffset", 0))
        required = q_offset + points_len * 3 * 2
        if required > len(ft_bin):
            raise ValueError(f"POSITION_QUANTIZED buffer overflow in {file_path}")
        q_positions = np.frombuffer(ft_bin[q_offset:q_offset + points_len * 3 * 2], dtype="<u2").reshape((-1, 3)).astype(np.float64)
        quantized_volume_scale = np.array(ft.get("QUANTIZED_VOLUME_SCALE", [1.0, 1.0, 1.0]), dtype=np.float64)
        quantized_volume_offset = np.array(ft.get("QUANTIZED_VOLUME_OFFSET", [0.0, 0.0, 0.0]), dtype=np.float64)
        positions = quantized_volume_offset + (q_positions / 65535.0) * quantized_volume_scale

    if positions is None:
        return np.empty((0, 3), dtype=np.float64), None

    rtc_center = np.array(ft.get("RTC_CENTER", [0.0, 0.0, 0.0]), dtype=np.float64)
    if rtc_center.size == 3:
        positions = positions + rtc_center

    colors = None
    if "RGB" in ft:
        rgb_offset = int(ft["RGB"].get("byteOffset", 0))
        required = rgb_offset + points_len * 3
        if required <= len(ft_bin):
            colors = np.frombuffer(ft_bin[rgb_offset:rgb_offset + points_len * 3], dtype=np.uint8).reshape((-1, 3)).copy()

    return positions, colors


def _iter_content_nodes(node: Dict, depth: int = 0) -> Iterable[Tuple[Dict, int]]:
    yield node, depth
    for child in node.get("children", []):
        yield from _iter_content_nodes(child, depth + 1)


def _infer_world_crs(points: np.ndarray) -> str:
    if points.shape[0] == 0:
        return "unknown"
    norms = np.linalg.norm(points, axis=1)
    mean_norm = float(np.mean(norms))
    if 5_500_000.0 <= mean_norm <= 7_500_000.0:
        return "ecef"
    return "unknown"


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


def _ecef_to_enu(points_ecef: np.ndarray, origin_ecef: np.ndarray) -> np.ndarray:
    lat, lon, _ = _ecef_to_geodetic(origin_ecef)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)

    rot = np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=np.float64,
    )

    delta = points_ecef - origin_ecef.reshape(1, 3)
    return delta @ rot.T


def load_pointcloud_from_tileset(
    tileset_path: str,
    tiles_max_points: int = 800000,
    tiles_lod_max: Optional[int] = None,
    tiles_bbox: Optional[Tuple[float, float, float, float, float, float]] = None,
    tiles_output_frame: str = "world",
    tiles_input_crs: str = "auto",
    tiles_enu_origin_ecef: Optional[Tuple[float, float, float]] = None,
) -> PointCloud:
    tileset_file = Path(tileset_path)
    root_dir = tileset_file.parent

    with tileset_file.open("r", encoding="utf-8") as handle:
        tileset = json.load(handle)

    root = tileset.get("root", {})
    root_transform = _tiles_transform_to_matrix(root.get("transform"))

    all_points: List[np.ndarray] = []
    all_colors: List[np.ndarray] = []

    for node, depth in _iter_content_nodes(root, depth=0):
        if tiles_lod_max is not None and depth > tiles_lod_max:
            continue

        content = node.get("content")
        if not content:
            continue
        uri = content.get("uri") or content.get("url")
        if not uri:
            continue
        if not str(uri).lower().endswith(".pnts"):
            continue

        file_path = root_dir / uri
        if not file_path.exists():
            continue

        points, colors = _parse_pnts_file(file_path)
        if points.shape[0] == 0:
            continue

        node_transform = _tiles_transform_to_matrix(node.get("transform"))
        transform = root_transform @ node_transform
        homogeneous = np.hstack([points, np.ones((points.shape[0], 1), dtype=np.float64)])
        world = (homogeneous @ transform.T)[:, :3]

        if tiles_bbox is not None:
            min_x, max_x, min_y, max_y, min_z, max_z = tiles_bbox
            mask = (
                (world[:, 0] >= min_x)
                & (world[:, 0] <= max_x)
                & (world[:, 1] >= min_y)
                & (world[:, 1] <= max_y)
                & (world[:, 2] >= min_z)
                & (world[:, 2] <= max_z)
            )
            world = world[mask]
            if colors is not None:
                colors = colors[mask]

        if world.shape[0] == 0:
            continue

        all_points.append(world)
        if colors is not None and colors.shape[0] == world.shape[0]:
            all_colors.append(colors)

    if not all_points:
        raise ValueError(f"No PNTS points loaded from tileset: {tileset_path}")

    points = np.vstack(all_points)
    colors = np.vstack(all_colors) if all_colors else None

    if points.shape[0] > tiles_max_points:
        rng = np.random.default_rng(42)
        indices = rng.choice(points.shape[0], size=tiles_max_points, replace=False)
        points = points[indices]
        if colors is not None and colors.shape[0] == len(indices):
            colors = colors[indices]
        elif colors is not None and colors.shape[0] == np.vstack(all_points).shape[0]:
            colors = colors[indices]
        else:
            colors = None

    if tiles_output_frame not in {"world", "enu"}:
        raise ValueError("tiles_output_frame must be one of: 'world', 'enu'")

    if tiles_input_crs not in {"auto", "ecef"}:
        raise ValueError("tiles_input_crs must be one of: 'auto', 'ecef'")

    coordinate_frame = "world"
    enu_origin_ecef: Optional[Tuple[float, float, float]] = None
    if tiles_output_frame == "enu":
        resolved_crs = tiles_input_crs if tiles_input_crs != "auto" else _infer_world_crs(points)
        if resolved_crs != "ecef":
            raise ValueError(
                "Unable to convert tileset world coordinates to ENU: input CRS is not ECEF. "
                "Set tiles_input_crs='ecef' when world is ECEF."
            )

        if tiles_enu_origin_ecef is None:
            origin = np.mean(points, axis=0)
        else:
            origin = np.array(tiles_enu_origin_ecef, dtype=np.float64)
            if origin.shape != (3,):
                raise ValueError("tiles_enu_origin_ecef must be a 3-tuple (x, y, z)")

        points = _ecef_to_enu(points, origin)
        coordinate_frame = "enu"
        enu_origin_ecef = (float(origin[0]), float(origin[1]), float(origin[2]))

    return PointCloud(
        points=points,
        colors=colors,
        source_file=str(tileset_path),
        coordinate_frame=coordinate_frame,
        enu_origin_ecef=enu_origin_ecef,
    )


def export_tileset_to_ply(
    tileset_path: str,
    output_ply: str,
    tiles_max_points: int = 800000,
    tiles_lod_max: Optional[int] = None,
    tiles_bbox: Optional[Tuple[float, float, float, float, float, float]] = None,
    write_ascii: bool = True,
    coord_frame: str = "centroid",
    tiles_input_crs: str = "auto",
    enu_origin_ecef: Optional[Tuple[float, float, float]] = None,
) -> str:
    if coord_frame not in {"world", "first_point", "centroid", "enu"}:
        raise ValueError("coord_frame must be one of: 'world', 'first_point', 'centroid', 'enu'")

    tiles_output_frame = "enu" if coord_frame == "enu" else "world"
    pointcloud = load_pointcloud_from_tileset(
        tileset_path=tileset_path,
        tiles_max_points=tiles_max_points,
        tiles_lod_max=tiles_lod_max,
        tiles_bbox=tiles_bbox,
        tiles_output_frame=tiles_output_frame,
        tiles_input_crs=tiles_input_crs,
        tiles_enu_origin_ecef=enu_origin_ecef,
    )

    try:
        import open3d as o3d
    except ImportError as error:
        raise ImportError("open3d is required to export PLY") from error

    output_path = Path(output_ply)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    points = pointcloud.points

    if points.shape[0] > 0 and coord_frame in {"first_point", "centroid"}:
        if coord_frame == "first_point":
            origin = points[0]
        else:
            origin = np.mean(points, axis=0)
        points = points - origin

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if pointcloud.colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(pointcloud.colors.astype(np.float64) / 255.0)

    if not o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=write_ascii):
        raise RuntimeError(f"Failed to write PLY: {output_ply}")

    return str(output_path)
