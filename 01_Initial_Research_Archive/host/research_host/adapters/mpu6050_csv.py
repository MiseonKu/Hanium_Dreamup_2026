from __future__ import annotations

import csv
from pathlib import Path

from ..types import ImuRawSample


class MPU6050CsvReader:
    def __init__(
        self,
        path: str | Path,
        *,
        imu_id: str,
        link_id: str,
        imu_position_id: str,
    ) -> None:
        self.path = Path(path)
        self.imu_id = imu_id
        self.link_id = link_id
        self.imu_position_id = imu_position_id

    def samples(self) -> list[ImuRawSample]:
        rows: list[ImuRawSample] = []
        with self.path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                t_request = _read_float(row, "t_request_s", "time_s", "t_s")
                t_receive = _read_float(row, "t_receive_s", "time_s", "t_s")
                rows.append(
                    ImuRawSample(
                        imu_id=self.imu_id,
                        link_id=self.link_id,
                        imu_position_id=self.imu_position_id,
                        t_request_s=t_request,
                        t_receive_s=t_receive,
                        t_imu_corrected_s=_read_float(
                            row, "t_imu_corrected_s", "time_s", "t_s"
                        ),
                        acc_raw_m_s2=(
                            _read_float(row, "acc_x_m_s2", "ax_mps2", "ax"),
                            _read_float(row, "acc_y_m_s2", "ay_mps2", "ay"),
                            _read_float(row, "acc_z_m_s2", "az_mps2", "az"),
                        ),
                        gyro_raw_rad_s=(
                            _read_float(row, "gyro_x_rad_s", "gx_rad_s", "gx"),
                            _read_float(row, "gyro_y_rad_s", "gy_rad_s", "gy"),
                            _read_float(row, "gyro_z_rad_s", "gz_rad_s", "gz"),
                        ),
                        dropped_sample_flag=_read_bool(row, "dropped_sample_flag"),
                        delayed_sample_flag=_read_bool(row, "delayed_sample_flag"),
                    )
                )
        return rows


def _read_float(row: dict[str, str], *names: str) -> float:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return float(value)
    return 0.0


def _read_bool(row: dict[str, str], name: str) -> bool:
    return str(row.get(name, "")).strip().lower() in {"1", "true", "yes"}
