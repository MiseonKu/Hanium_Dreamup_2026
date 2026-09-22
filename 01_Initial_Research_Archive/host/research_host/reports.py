from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from .quality import QualitySummary, summarize_quality


METRIC_COLUMNS = (
    "position_rmse_mm",
    "residual_vibration_rms_mm",
    "settling_time_s",
    "dropped_sample_ratio",
)


def read_analysis_rows(paths: Sequence[str | Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def generate_condition_report(
    analysis_rows: Sequence[Mapping[str, object]],
    *,
    quality_summary: QualitySummary | None = None,
) -> str:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in analysis_rows:
        grouped[str(row.get("condition_id", "unknown"))].append(row)

    lines = [
        "# SO-101 Condition Comparison",
        "",
        "Descriptive summary only. Do not treat this report as an inferential statistical conclusion.",
        "",
        "| condition_id | trials | mean_position_rmse_mm | mean_residual_vibration_rms_mm | mean_settling_time_s | mean_dropped_sample_ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition_id in sorted(grouped):
        rows = grouped[condition_id]
        lines.append(
            "| "
            + " | ".join(
                [
                    condition_id,
                    str(len(rows)),
                    _mean_cell(rows, "position_rmse_mm"),
                    _mean_cell(rows, "residual_vibration_rms_mm"),
                    _mean_cell(rows, "settling_time_s"),
                    _mean_cell(rows, "dropped_sample_ratio"),
                ]
            )
            + " |"
        )

    if quality_summary is not None:
        lines.extend(
            [
                "",
                "## Quality Flags",
                "",
                f"- Processed rows: {quality_summary.row_count}",
                f"- Max timestamp gap s: {quality_summary.max_timestamp_gap_s:.6f}",
                "",
                "| flag | count |",
                "|---|---:|",
            ]
        )
        for flag, count in sorted(quality_summary.flag_counts.items()):
            lines.append(f"| {flag} | {count} |")

    return "\n".join(lines) + "\n"


def summarize_processed_files(paths: Sequence[str | Path]) -> QualitySummary:
    rows: list[dict[str, str]] = []
    for path in paths:
        with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
            rows.extend(csv.DictReader(handle))
    return summarize_quality(rows)


def _mean_cell(rows: Sequence[Mapping[str, object]], key: str) -> str:
    values = [_try_float(row.get(key)) for row in rows]
    numeric = [value for value in values if value is not None]
    if not numeric:
        return ""
    return f"{sum(numeric) / len(numeric):.6f}"


def _try_float(value: object) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
