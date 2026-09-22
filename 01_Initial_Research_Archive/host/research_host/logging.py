from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .types import TrialMetadata


PROCESSED_FIELDNAMES = [
    "loop_index",
    "t_common_s",
    "P_hat_encoder_base_mm_x",
    "P_hat_encoder_base_mm_y",
    "P_hat_encoder_base_mm_z",
    "P_hat_imu_base_mm_x",
    "P_hat_imu_base_mm_y",
    "P_hat_imu_base_mm_z",
    "P_hat_fusion_base_mm_x",
    "P_hat_fusion_base_mm_y",
    "P_hat_fusion_base_mm_z",
    "P_actual_base_mm_x",
    "P_actual_base_mm_y",
    "P_actual_base_mm_z",
    "q_imu_world_w",
    "q_imu_world_x",
    "q_imu_world_y",
    "q_imu_world_z",
    "q_encoder_rad_1",
    "q_encoder_rad_2",
    "q_encoder_rad_3",
    "latency_encoder_s",
    "latency_imu_s",
    "latency_tracker_s",
    "quality_flags",
]


ANALYSIS_FIELDNAMES = [
    "trial_id",
    "condition_id",
    "robot_model",
    "target_id",
    "position_rmse_mm",
    "shape_rmse_mm",
    "settling_time_s",
    "transient_rms_mm",
    "residual_vibration_rms_mm",
    "dropped_sample_ratio",
]


class TrialLogger:
    def __init__(self, root_dir: str | Path, metadata: TrialMetadata) -> None:
        self.root_dir = Path(root_dir)
        self.metadata = metadata

    @property
    def stem(self) -> str:
        return self.metadata.filename_stem

    def metadata_path(self) -> Path:
        return self.root_dir / "raw" / f"{self.stem}_metadata.json"

    def raw_csv_path(self, stream_name: str) -> Path:
        return self.root_dir / "raw" / f"{self.stem}_{stream_name}.csv"

    def processed_csv_path(self) -> Path:
        return self.root_dir / "processed" / f"{self.stem}_processed.csv"

    def analysis_csv_path(self) -> Path:
        return self.root_dir / "analysis" / f"{self.stem}_analysis.csv"

    def write_metadata_json(self, extra: Mapping[str, Any] | None = None) -> Path:
        path = self.metadata_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.metadata.to_jsonable()
        if extra:
            payload.update(dict(extra))
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def write_raw_rows(
        self,
        stream_name: str,
        rows: Iterable[Mapping[str, object]],
        *,
        fieldnames: list[str],
    ) -> Path:
        path = self.raw_csv_path(stream_name)
        _write_csv(path, rows, fieldnames=fieldnames)
        return path

    def write_processed_rows(self, rows: Iterable[Mapping[str, object]]) -> Path:
        path = self.processed_csv_path()
        _write_csv(path, rows, fieldnames=PROCESSED_FIELDNAMES)
        return path

    def write_analysis_summary(self, row: Mapping[str, object]) -> Path:
        path = self.analysis_csv_path()
        _write_csv(path, [row], fieldnames=ANALYSIS_FIELDNAMES)
        return path


def _write_csv(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    *,
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_csv_dicts(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
