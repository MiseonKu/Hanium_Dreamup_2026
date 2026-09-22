from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class QualitySummary:
    row_count: int
    flag_counts: dict[str, int]
    dropped_sample_ratio: float
    tracker_occlusion_count: int
    max_timestamp_gap_s: float

    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)


def summarize_quality(rows: Sequence[Mapping[str, object]]) -> QualitySummary:
    flag_counts: Counter[str] = Counter()
    times = [_float(row.get("t_common_s")) for row in rows if row.get("t_common_s") not in ("", None)]
    for row in rows:
        for flag in split_quality_flags(str(row.get("quality_flags", "ok"))):
            flag_counts[flag] += 1
    row_count = len(rows)
    dropped = flag_counts.get("dropped_sample_flag", 0)
    return QualitySummary(
        row_count=row_count,
        flag_counts=dict(sorted(flag_counts.items())),
        dropped_sample_ratio=0.0 if row_count == 0 else dropped / row_count,
        tracker_occlusion_count=flag_counts.get("tracker_occlusion_flag", 0),
        max_timestamp_gap_s=max_timestamp_gap_s(times),
    )


def split_quality_flags(flags: str) -> list[str]:
    parts = [part.strip() for part in flags.split("|") if part.strip()]
    return parts or ["ok"]


def max_timestamp_gap_s(times_s: Sequence[float]) -> float:
    if len(times_s) < 2:
        return 0.0
    return max(current - previous for previous, current in zip(times_s, times_s[1:]))


def load_quality_thresholds(path: str | Path) -> dict[str, float | bool]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    thresholds = payload.get("quality_default", payload)
    return dict(thresholds)


def quality_issues(
    summary: QualitySummary,
    thresholds: Mapping[str, float | bool],
) -> list[str]:
    issues: list[str] = []
    max_dropped = float(thresholds.get("max_dropped_sample_ratio", 1.0))
    if summary.dropped_sample_ratio > max_dropped:
        issues.append(
            f"dropped_sample_ratio={summary.dropped_sample_ratio:.6f} > {max_dropped:.6f}"
        )

    max_gap = thresholds.get("max_timestamp_gap_s")
    if max_gap is not None and summary.max_timestamp_gap_s > float(max_gap):
        issues.append(
            f"max_timestamp_gap_s={summary.max_timestamp_gap_s:.6f} > {float(max_gap):.6f}"
        )

    occlusion_allowed = bool(thresholds.get("tracker_occlusion_allowed", True))
    if not occlusion_allowed and summary.tracker_occlusion_count > 0:
        issues.append(
            f"tracker_occlusion_flag count={summary.tracker_occlusion_count} but occlusion is disabled"
        )
    return issues


def _float(value: object, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    return float(value)
