from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .exporters import processed_row_to_joint_state


class OptionalDependencyError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    t_common_s: float
    joint_state_rad: dict[str, float]


def build_replay_frames(rows: Sequence[Mapping[str, object]]) -> list[ReplayFrame]:
    return [
        ReplayFrame(
            t_common_s=_float(row.get("t_common_s")),
            joint_state_rad=processed_row_to_joint_state(row),
        )
        for row in rows
    ]


def summarize_replay_frames(frames: Sequence[ReplayFrame]) -> dict[str, object]:
    if not frames:
        return {"frame_count": 0, "duration_s": 0.0, "joint_names": []}
    return {
        "frame_count": len(frames),
        "duration_s": frames[-1].t_common_s - frames[0].t_common_s,
        "joint_names": list(frames[0].joint_state_rad),
    }


def require_optional_module(module_name: str, install_hint: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            f"Optional dependency {module_name!r} is not installed. {install_hint}"
        ) from exc


def run_mujoco_replay(
    rows: Sequence[Mapping[str, object]],
    *,
    model_path: str,
) -> dict[str, object]:
    require_optional_module("mujoco", "Install MuJoCo Python bindings to run replay.")
    frames = build_replay_frames(rows)
    summary = summarize_replay_frames(frames)
    return {**summary, "model_path": model_path, "backend": "mujoco"}


def run_ros2_joint_state_replay(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    require_optional_module("rclpy", "Install ROS 2 rclpy to publish JointState messages.")
    frames = build_replay_frames(rows)
    return {**summarize_replay_frames(frames), "backend": "ros2_joint_states"}


def _float(value: object, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    return float(value)
