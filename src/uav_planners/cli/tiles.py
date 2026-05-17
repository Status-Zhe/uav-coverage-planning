"""CLI for 3D Tiles related tools."""

from __future__ import annotations

import argparse

from ..io.tileset_loader import export_tileset_to_ply


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uav-planners", description="UAV planners CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert-tiles", help="Convert 3D Tiles pointcloud tileset to PLY")
    convert_parser.add_argument("--tileset", required=True, help="Path to tileset.json")
    convert_parser.add_argument("--out", required=True, help="Output .ply file path")
    convert_parser.add_argument("--max-points", type=int, default=800000, help="Maximum number of points to keep")
    convert_parser.add_argument("--lod-max", type=int, default=None, help="Maximum tileset traversal depth")
    convert_parser.add_argument(
        "--coord-frame",
        choices=["centroid", "first_point", "world", "enu"],
        default="centroid",
        help="Output coordinate frame: centroid/first_point local frame, world frame, or ENU frame",
    )
    convert_parser.add_argument(
        "--input-crs",
        choices=["auto", "ecef"],
        default="auto",
        help="Input world CRS for 3D Tiles. Required as ECEF when --coord-frame enu",
    )
    convert_parser.add_argument(
        "--enu-origin-ecef",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional ENU origin in ECEF meters. Default: centroid of final point set",
    )
    convert_parser.add_argument(
        "--bbox",
        nargs=6,
        type=float,
        default=None,
        metavar=("MIN_X", "MAX_X", "MIN_Y", "MAX_Y", "MIN_Z", "MAX_Z"),
        help="Optional spatial crop box",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "convert-tiles":
        output_path = export_tileset_to_ply(
            tileset_path=args.tileset,
            output_ply=args.out,
            tiles_max_points=args.max_points,
            tiles_lod_max=args.lod_max,
            tiles_bbox=tuple(args.bbox) if args.bbox else None,
            coord_frame=args.coord_frame,
            tiles_input_crs=args.input_crs,
            enu_origin_ecef=tuple(args.enu_origin_ecef) if args.enu_origin_ecef else None,
        )
        print(f"Exported PLY: {output_path}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
