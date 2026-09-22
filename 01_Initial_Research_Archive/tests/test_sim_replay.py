from __future__ import annotations

import sys
import types

import pytest

from research_host.sim.replay import (
    OptionalDependencyError,
    build_replay_frames,
    run_mujoco_replay,
    run_ros2_joint_state_replay,
    summarize_replay_frames,
)


ROWS = [
    {
        "t_common_s": "0.00",
        "q_encoder_rad_1": "0.1",
        "q_encoder_rad_2": "0.2",
        "q_encoder_rad_3": "0.3",
    },
    {
        "t_common_s": "0.02",
        "q_encoder_rad_1": "0.2",
        "q_encoder_rad_2": "0.3",
        "q_encoder_rad_3": "0.4",
    },
]


def test_replay_frame_summary() -> None:
    frames = build_replay_frames(ROWS)

    assert frames[0].joint_state_rad["joint_1"] == 0.1
    assert summarize_replay_frames(frames) == {
        "frame_count": 2,
        "duration_s": 0.02,
        "joint_names": ["joint_1", "joint_2", "joint_3"],
    }


def test_mujoco_replay_reports_missing_dependency(monkeypatch) -> None:
    def missing_import(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("research_host.sim.replay.importlib.import_module", missing_import)

    with pytest.raises(OptionalDependencyError):
        run_mujoco_replay(ROWS, model_path="model.xml")


def test_ros2_replay_mocked_dependency(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "rclpy", types.ModuleType("rclpy"))

    summary = run_ros2_joint_state_replay(ROWS)

    assert summary["backend"] == "ros2_joint_states"
    assert summary["frame_count"] == 2
