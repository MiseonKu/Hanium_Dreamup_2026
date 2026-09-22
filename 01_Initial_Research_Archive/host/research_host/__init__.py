"""Offline-first research support package for SO-101 IMU-array experiments."""

from .geometry import ROBOT_MODELS, RobotModel
from .types import TrialMetadata

__all__ = ["ROBOT_MODELS", "RobotModel", "TrialMetadata"]
