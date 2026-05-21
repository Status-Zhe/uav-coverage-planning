# -*- coding: utf-8 -*-
# ------------------------------------------------------------ #
# Description: path_generator（UAV 覆盖规划）后端调用接口
# 与 path_editor.api 对齐：参数/返回值为 JSON 可序列化结构，统一 plan / plan_from_file（合并参数用 dict 浅拷贝）。
# ------------------------------------------------------------ #

from __future__ import annotations

import logging
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.utils.response_util import path_generator_plan_fail

# 将 path_generator/src 加入路径，便于在未 pip install 时导入 uav_planners
_PG_DIR = Path(__file__).resolve().parent
_SRC = _PG_DIR / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from uav_planners import CoveragePlanner, MissionConfig  # noqa: E402
from uav_planners.models import Camera  # noqa: E402
from uav_planners.core.planner_result import PlannerResult  # noqa: E402

_DEFAULT_CAMERA: Dict[str, Any] = {
    "focal_length_mm": 24.0,
    "sensor_width_mm": 36.0,
    "sensor_height_mm": 24.0,
    "resolution_x": 6000,
    "resolution_y": 4000,
}

# 非 MissionConfig 字段、仅用于 plan_from_file / HTTP 的键
_IGNORED_PARAM_KEYS = frozenset({"input_path"})

_logger = logging.getLogger(__name__)

# 日志摘要：长几何序列只记长度与首尾，避免刷屏
_LONG_XY_KEYS = frozenset({"coverage_area_polygon_xy"})
_LONG_XYZ_KEYS = frozenset({"template_base_vertices_enu", "template_top_vertices_enu"})


def _summarize_generator_mission_dict(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """供 plan_coverage 与 path_editor_bridge 复用：将 mission/参数字典压成可日志化的小 dict。"""
    if not params:
        return {}
    out: Dict[str, Any] = {}
    for key in sorted(params.keys(), key=lambda k: str(k)):
        val = params[key]
        if key == "pointcloud_path" and val is not None:
            out[key] = Path(str(val)).name
            continue
        if key == "camera" and val is not None:
            if hasattr(val, "focal_length_mm"):
                out[key] = {
                    "focal_length_mm": getattr(val, "focal_length_mm", None),
                    "resolution_x": getattr(val, "resolution_x", None),
                    "resolution_y": getattr(val, "resolution_y", None),
                }
            elif isinstance(val, dict):
                out[key] = {
                    k: val.get(k)
                    for k in ("focal_length_mm", "resolution_x", "resolution_y")
                    if k in val
                }
            else:
                out[key] = str(type(val).__name__)
            continue
        if key in _LONG_XY_KEYS and isinstance(val, (list, tuple)):
            n = len(val)
            if n == 0:
                out[key] = {"len": 0}
            else:
                out[key] = {"len": n, "first": val[0], "last": val[-1]}
            continue
        if key in _LONG_XYZ_KEYS and isinstance(val, (list, tuple)):
            n = len(val)
            if n == 0:
                out[key] = {"len": 0}
            else:
                out[key] = {"len": n, "first": val[0], "last": val[-1]}
            continue
        if key == "aabb_transform" and val is not None:
            arr = np.asarray(val)
            out[key] = f"ndarray_shape_{tuple(arr.shape)}"
            continue
        if isinstance(val, np.ndarray):
            out[key] = f"ndarray_shape_{tuple(val.shape)}"
            continue
        if isinstance(val, (list, tuple)) and len(val) > 12:
            out[key] = {"len": len(val), "first": val[0], "last": val[-1]}
            continue
        if isinstance(val, str) and len(val) > 120:
            out[key] = val[:117] + "..."
            continue
        if isinstance(val, dict) and len(val) > 8:
            out[key] = {"keys": sorted(val.keys())[:20], "n_keys": len(val)}
            continue
        out[key] = val
    return out


def _coverage_report_log_snippet(report: Any) -> Dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {}
    snippet: Dict[str, Any] = {}
    for k in sorted(report.keys(), key=lambda x: str(x)):
        v = report[k]
        if isinstance(v, (bool, int, float)) or (isinstance(v, str) and len(v) < 64):
            snippet[k] = v
        if len(snippet) >= 2:
            break
    return snippet


def _waypoints_xyz_head_tail(waypoints: Any, head: int = 2, tail: int = 2) -> Dict[str, Any]:
    if not isinstance(waypoints, list) or not waypoints:
        return {}
    pts: List[Dict[str, Any]] = []

    def _xyz(w: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(w, dict):
            return None
        if "x" not in w or "y" not in w or "z" not in w:
            return None
        return {"x": w.get("x"), "y": w.get("y"), "z": w.get("z")}

    for w in waypoints[:head]:
        p = _xyz(w)
        if p:
            pts.append(p)
    out: Dict[str, Any] = {"head": pts}
    tail_pts: List[Dict[str, Any]] = []
    if tail and len(waypoints) > head:
        for w in waypoints[-tail:]:
            p = _xyz(w)
            if p:
                tail_pts.append(p)
    if tail_pts:
        out["tail"] = tail_pts
    return out


def _try_resolve_data_path(path_str: str) -> str:
    """相对路径时尝试 WORK_DIR / 项目根，与常见后端数据目录对齐。"""
    p = Path(path_str).expanduser()
    if p.is_file():
        return str(p.resolve())
    if p.is_absolute():
        return str(p)
    try:
        import config as cfg  # type: ignore

        work = Path(cfg.WORK_DIR) / path_str
        if work.is_file():
            return str(work.resolve())
        base = Path(cfg.BASE_DIR) / path_str
        if base.is_file():
            return str(base.resolve())
    except Exception:
        pass
    return str(p)


def _coerce_field(name: str, value: Any) -> Any:
    if value is None:
        return None
    if name == "aabb_transform":
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape != (4, 4):
            raise ValueError("aabb_transform 须为 4x4 矩阵（可嵌套列表）")
        return arr
    if name in (
        "coverage_area_rect_xy",
        "coverage_area_polygon_xy",
        "aabb_center",
        "aabb_size",
        "tiles_enu_origin_ecef",
        "tiles_bbox",
    ):
        if isinstance(value, (list, tuple)):
            if name == "coverage_area_polygon_xy":
                return tuple((float(p[0]), float(p[1])) for p in value)
            return tuple(float(x) for x in value)
    if name in (
        "facade_plane_origin_enu",
        "facade_plane_along_unit_enu",
        "facade_plane_up_unit_enu",
        "facade_plane_normal_unit_enu",
    ):
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return tuple(float(x) for x in value)
    if name in ("facade_plane_width", "facade_plane_height", "facade_scan_gimbal_pitch_deg"):
        return float(value)
    if name in ("template_base_vertices_enu", "template_top_vertices_enu"):
        if isinstance(value, (list, tuple)):
            return tuple(
                (float(p[0]), float(p[1]), float(p[2]))
                for p in value
                if isinstance(p, (list, tuple)) and len(p) >= 3
            )
    if name == "template_prism_height":
        return float(value)
    if name in (
        "takeoff_ingress_x",
        "takeoff_ingress_y",
        "takeoff_ingress_z",
        "takeoff_ingress_heading_deg",
    ):
        if value is None:
            return None
        return float(value)
    if name == "camera" and isinstance(value, dict):
        return Camera(**{k: v for k, v in value.items() if k in {f.name for f in fields(Camera)}})
    return value


def mission_config_from_params(params: Optional[Dict[str, Any]]) -> MissionConfig:
    """由字典构造 MissionConfig；未知键忽略。"""
    raw = dict(params or {})
    for k in _IGNORED_PARAM_KEYS:
        raw.pop(k, None)

    cam_raw = raw.pop("camera", None)
    if cam_raw is None:
        camera = Camera(**_DEFAULT_CAMERA)
    elif isinstance(cam_raw, Camera):
        camera = cam_raw
    elif isinstance(cam_raw, dict):
        merged_cam = {**_DEFAULT_CAMERA, **cam_raw}
        camera = Camera(**merged_cam)
    else:
        raise ValueError("camera 须为 dict 或 Camera")

    pointcloud_path = raw.pop("pointcloud_path", None)
    mission_field_names = {f.name for f in fields(MissionConfig)}
    kwargs: Dict[str, Any] = {"pointcloud_path": pointcloud_path, "camera": camera}

    for key, val in raw.items():
        if key not in mission_field_names:
            continue
        kwargs[key] = _coerce_field(key, val)

    return MissionConfig(**kwargs)


def _sanitize_for_json(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    return str(obj)


def planner_result_to_dict(result: PlannerResult) -> Dict[str, Any]:
    """PlannerResult → 与 path_editor 风格一致的业务字典（含 waypoints / count / error）。

    每个航点含 x,y,z（米）。水平分量 x,y **不是** WGS84 经纬度；uav_planners 模型按 ENU 口径命名，
    实际帧随任务而定：region-only（如 boustrophedon）下与 ``coverage_area_rect_xy`` 同为
    投影平面上的米制水平坐标；点云/倾斜类管线中多为点云局部或 ENU 等水平米制轴。
    ``region_horizontal_frame=facade_plane`` 且默认 ``waypoint_export_frame=planning`` 时，
    导出 JSON 的 x,y,z 为立面局部 (沿宽、沿高、法向)，与 ``MissionConfig.facade_plane_*_enu`` 及
    bridge 回投公式一致。facade 题型使用 boustrophedon 时，规划器内部航迹已为立面局部坐标，
    不再先做 ENU；旧版 oblique facade 则为 ENU，导出时再投影到立面局部。
    z 为相对规划所用垂直基准的高度（米，见 MissionConfig / 规划器实现）。``frame`` 字段为历史兼容的粗标签。

    ``waypoints`` / ``sparse_waypoints`` 与 :meth:`PlannerResult.export_json` 中密集航点使用同一套
    帧转换（``metadata.waypoint_export_frame`` 等）与字段集合，见 :meth:`PlannerResult.to_waypoint_dicts`。
    """
    conv = result.to_waypoint_dicts()
    waypoints_out = conv["waypoints"]
    sparse_out = conv["sparse_waypoints"]

    meta = _sanitize_for_json(
        {**(result.metadata or {}), "export_frame_applied": conv["export_frame_applied"]}
    )
    report = _sanitize_for_json(dict(result.coverage_report or {}))

    return {
        "waypoints": waypoints_out,
        "count": len(waypoints_out),
        "sparse_waypoints": sparse_out,
        "keypoint_count": result.keypoint_count,
        "total_waypoints": result.total_waypoints,
        "coverage_report": report,
        "metadata": meta,
        "frame": "enu",
        "error": None,
        "input_geojson": None,
        "input_kml": None,
        "input_kml_json": None,
    }


def plan_coverage(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """执行覆盖规划（点云 / 3D Tiles / region-only）。

    返回 ``waypoints`` 中 x,y,z 为米制轨迹坐标；勿将 x,y 理解为「地图平面 (x,y)」或经纬度，
    含义见 :func:`planner_result_to_dict` 说明。
    """
    _logger.info("plan_coverage input summary: %s", _summarize_generator_mission_dict(params))
    try:
        config = mission_config_from_params(params)
        planner = CoveragePlanner(config)
        result = planner.plan()
        out = planner_result_to_dict(result)
        meta = out.get("metadata") if isinstance(out.get("metadata"), dict) else {}
        sparse = out.get("sparse_waypoints") or []
        wps = out.get("waypoints") or []
        cov_snip = _coverage_report_log_snippet(out.get("coverage_report"))
        _logger.info(
            "plan_coverage output: count=%s keypoint_count=%s sparse_count=%s "
            "metadata_keys=%s waypoint_export_frame=%s export_frame_applied=%s "
            "coverage_report_snippet=%s xyz_sample=%s",
            out.get("count"),
            out.get("keypoint_count"),
            len(sparse) if isinstance(sparse, list) else None,
            sorted(meta.keys())[:24] if isinstance(meta, dict) else None,
            meta.get("waypoint_export_frame"),
            meta.get("export_frame_applied"),
            cov_snip,
            _waypoints_xyz_head_tail(wps),
        )
        return out
    except Exception as e:
        _logger.warning("plan_coverage failed: %s", str(e))
        return path_generator_plan_fail(str(e))


def plan(
    route_type: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """统一入口（与 path_editor.api.plan 对齐）。

    Args:
        route_type: 仅支持 ``"coverage"``（覆盖规划）。
        params: MissionConfig 可序列化字段，见 README / MissionConfig。
    """
    if route_type != "coverage":
        _logger.warning(
            "plan rejected: unsupported route_type=%s (path_generator only coverage)",
            route_type,
        )
        return path_generator_plan_fail(
            f"不支持的航线类型: {route_type}（path_generator 仅支持 coverage）"
        )
    return plan_coverage(params)


def plan_from_file(
    input_path: str,
    route_type: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """文件路径作为点云或 tileset.json 输入（与 path_editor 的 input_path 用法对齐）。

    将 ``input_path`` 合并进参数：写入 ``pointcloud_path``，由 MissionConfig 自动识别
    ``.json``（tileset）与 ``.ply`` / ``.pcd``（点云）。
    """
    if route_type != "coverage":
        _logger.warning(
            "plan_from_file rejected: unsupported route_type=%s",
            route_type,
        )
        return path_generator_plan_fail(
            f"不支持的航线类型: {route_type}（path_generator 仅支持 coverage）"
        )
    merged = dict(params or {})
    resolved = _try_resolve_data_path(input_path)
    merged["pointcloud_path"] = resolved
    return plan_coverage(merged)
