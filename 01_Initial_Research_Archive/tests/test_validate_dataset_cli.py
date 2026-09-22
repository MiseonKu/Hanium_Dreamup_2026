from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_validate_dataset_accepts_optional_contract_files() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "host")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "research_host.cli.validate_dataset",
            "--metadata",
            "data/samples/raw/trial_0001_long_C2_imu_array_only_T001_rep01_2026-05-01_metadata.json",
            "--processed",
            "data/samples/processed/trial_0001_long_C2_imu_array_only_T001_rep01_2026-05-01_processed.csv",
            "--trial-plan",
            "data/samples/trial_plan.csv",
            "--trial-plan-metadata",
            "data/samples/trial_plan_metadata.json",
            "--optimization-report",
            "data/samples/analysis/optimization_report.json",
            "--sim-manifest",
            "simulation/generated/so101_long_manifest.json",
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Dataset contract OK" in result.stdout


def test_validate_calibration_cli_reports_missing_files() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "host")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "research_host.cli.validate_calibration",
            "--calibration-dir",
            "calibration",
            "--templates",
            "calibration/templates",
            "--json",
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "missing calibration file" in result.stdout
