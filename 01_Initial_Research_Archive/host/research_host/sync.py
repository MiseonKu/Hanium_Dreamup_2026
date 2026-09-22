from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TimestampCorrection:
    t_request_s: float
    t_receive_s: float
    latency_s: float
    offset_s: float

    @property
    def t_estimated_sample_s(self) -> float:
        return estimate_sample_time(self.t_request_s, self.t_receive_s)

    @property
    def t_corrected_s(self) -> float:
        return correct_sample_time(
            self.t_request_s,
            self.t_receive_s,
            latency_s=self.latency_s,
            offset_s=self.offset_s,
        )


def estimate_sample_time(t_request_s: float, t_receive_s: float) -> float:
    return (float(t_request_s) + float(t_receive_s)) * 0.5


def correct_sample_time(
    t_request_s: float,
    t_receive_s: float,
    *,
    latency_s: float = 0.0,
    offset_s: float = 0.0,
) -> float:
    return estimate_sample_time(t_request_s, t_receive_s) - float(latency_s) - float(offset_s)


def compute_jitter_s(t_receive_s: list[float], nominal_period_s: float) -> list[float]:
    if not t_receive_s:
        return []
    start = t_receive_s[0]
    return [
        receive_time - (start + (index * nominal_period_s))
        for index, receive_time in enumerate(t_receive_s)
    ]


def dropped_sample_flags(
    t_receive_s: list[float],
    *,
    max_gap_s: float,
) -> list[bool]:
    if not t_receive_s:
        return []
    flags = [False]
    for previous, current in zip(t_receive_s, t_receive_s[1:]):
        flags.append((current - previous) > max_gap_s)
    return flags


def resample_scalar_series(
    source_times_s: list[float],
    source_values: list[float],
    target_times_s: list[float],
    *,
    method: str = "linear",
) -> list[float]:
    if len(source_times_s) != len(source_values):
        raise ValueError("source_times_s and source_values must have the same length")
    if not source_times_s:
        return []
    if method not in {"linear", "zero_order_hold"}:
        raise ValueError("method must be linear or zero_order_hold")

    values: list[float] = []
    source_index = 0
    for target in target_times_s:
        while (
            source_index < len(source_times_s) - 2
            and source_times_s[source_index + 1] < target
        ):
            source_index += 1

        if target <= source_times_s[0]:
            values.append(source_values[0])
            continue
        if target >= source_times_s[-1]:
            values.append(source_values[-1])
            continue

        t0 = source_times_s[source_index]
        t1 = source_times_s[source_index + 1]
        v0 = source_values[source_index]
        v1 = source_values[source_index + 1]
        if method == "zero_order_hold" or t1 == t0:
            values.append(v0)
            continue
        fraction = (target - t0) / (t1 - t0)
        values.append(v0 + ((v1 - v0) * fraction))
    return values


def compute_offset_by_correlation(
    reference_values: list[float],
    shifted_values: list[float],
    *,
    sample_period_s: float,
    max_lag_samples: int,
) -> float:
    if len(reference_values) != len(shifted_values):
        raise ValueError("signals must have the same length")
    if not reference_values:
        return 0.0

    best_lag = 0
    best_score = float("-inf")
    for lag in range(-max_lag_samples, max_lag_samples + 1):
        pairs: list[tuple[float, float]] = []
        for index, reference in enumerate(reference_values):
            shifted_index = index + lag
            if 0 <= shifted_index < len(shifted_values):
                pairs.append((reference, shifted_values[shifted_index]))
        if not pairs:
            continue
        score = sum(left * right for left, right in pairs)
        if score > best_score:
            best_score = score
            best_lag = lag
    return best_lag * sample_period_s
