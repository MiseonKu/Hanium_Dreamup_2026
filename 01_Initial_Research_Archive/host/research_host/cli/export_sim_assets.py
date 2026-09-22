from __future__ import annotations

import argparse
from pathlib import Path

from ..sim.exporters import write_sim_assets


ALL_FORMATS = ("urdf", "mjcf", "rviz", "ros2_launch", "isaac_manifest")


def _format_list(value: str) -> list[str]:
    formats = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(formats) - set(ALL_FORMATS))
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown formats: " + ", ".join(unknown)
        )
    return formats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export simulator-ready baseline assets for SO-101 research."
    )
    parser.add_argument("--robot-model", default="long", choices=("long", "short"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("simulation") / "generated",
    )
    parser.add_argument(
        "--formats",
        type=_format_list,
        default=list(ALL_FORMATS),
        help="Comma-separated subset: " + ",".join(ALL_FORMATS),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    written = write_sim_assets(
        args.output,
        robot_model=args.robot_model,
        formats=args.formats,
    )
    print(f"Simulation assets for {args.robot_model}:")
    for key, path in sorted(written.items()):
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
