"""Per-target experiment state machine (master spec section 19.1).

    MOVE_TO_TARGET -> WAIT_ENCODER_ARRIVAL -> ADAPTIVE_SETTLING_MONITOR ->
    STABLE_CANDIDATE -> TIP_IMU_STABLE_CONFIRMED -> MEASUREMENT_WINDOW ->
    MEASUREMENT_ACCEPTED -> NEXT_TARGET

Plain stdlib `Enum` + an explicit if/elif transition table -- no state-
machine library, no framework (ponytail: this is small enough that a
framework would be pure overhead).

Every transition publishes an `experiment_state_changed` event via
`core.event_bus.EventBus` (if one was supplied) and the current state is
exposed via the cheap `current_state` property, so other collectors can
stamp it into their own records' `experiment_state` field
(`core/schemas.py`'s `base_record`) without touching this class's
internals.

Raw data collection is never gated by this state machine -- master spec:
"stable detector가 실패해도 raw data는 전부 남긴다" (even if the stable
detector fails, all raw data stays). This class only decides what counts as
an *accepted measurement*; it has no handle on any collector/writer and
cannot stop one.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

# Needed for the `__main__` self-check at the bottom to run standalone.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.event_bus import EventBus  # noqa: E402

from experiment.adaptive_settling_detector import (
    AdaptiveSettlingDetector,
    TipSample,
    stability_record_from_evaluation,
)


class ExperimentState(Enum):
    MOVE_TO_TARGET = "MOVE_TO_TARGET"
    WAIT_ENCODER_ARRIVAL = "WAIT_ENCODER_ARRIVAL"
    ADAPTIVE_SETTLING_MONITOR = "ADAPTIVE_SETTLING_MONITOR"
    STABLE_CANDIDATE = "STABLE_CANDIDATE"
    TIP_IMU_STABLE_CONFIRMED = "TIP_IMU_STABLE_CONFIRMED"
    MEASUREMENT_WINDOW = "MEASUREMENT_WINDOW"
    MEASUREMENT_ACCEPTED = "MEASUREMENT_ACCEPTED"
    NEXT_TARGET = "NEXT_TARGET"


@dataclass
class EncoderGateConfig:
    """WAIT_ENCODER_ARRIVAL coarse-arrival gate thresholds -- config-driven
    per master spec ("Threshold는 config로 분리한다"), reasonable-but-
    UNVALIDATED defaults."""

    epsilon_q_rad: float = 0.02          # ~1.1 deg per-joint position tolerance
    epsilon_q_dot_rad_s: float = 0.05    # per-joint velocity tolerance
    dwell_s: float = 0.15                # both conditions must hold this long


class EncoderArrivalGate:
    """All joints: |q_encoder - q_cmd| < epsilon_q AND |q_encoder_dot| <
    epsilon_q_dot, sustained for `dwell_s`."""

    def __init__(self, config: EncoderGateConfig):
        self.config = config
        self._good_since_ns: Optional[int] = None
        self.arrived: bool = False

    def reset(self) -> None:
        self._good_since_ns = None
        self.arrived = False

    def update(
        self,
        now_ns: int,
        q_encoder_rad: Sequence[float],
        q_cmd_rad: Sequence[float],
        q_encoder_dot_rad_s: Sequence[float],
    ) -> bool:
        # `zip` would silently truncate to the shortest sequence, and `all`
        # over an empty sequence is True -- either way a mismatched or
        # empty joint vector would declare "arrived" without having checked
        # the missing joints. This rig is 3-axis (shoulder_pan/
        # shoulder_lift/wrist_flex) while some older call sites still pass
        # 5-element arrays, so the mismatch is real, not hypothetical.
        n = len(q_encoder_rad)
        if n == 0 or len(q_cmd_rad) != n or len(q_encoder_dot_rad_s) != n:
            raise ValueError(
                "EncoderArrivalGate.update needs three non-empty equal-length joint "
                f"vectors, got {n}/{len(q_cmd_rad)}/{len(q_encoder_dot_rad_s)}"
            )
        if any(
            v is None
            for v in (*q_encoder_rad, *q_cmd_rad, *q_encoder_dot_rad_s)
        ):
            # unmapped axis (firmware logical_deg=NA) or a first-sample
            # velocity that is still unknown -- unknown is not arrival;
            # never treat a null as satisfying the tolerance.
            self._good_since_ns = None
            self.arrived = False
            return False

        within_position = all(
            abs(qe - qc) < self.config.epsilon_q_rad for qe, qc in zip(q_encoder_rad, q_cmd_rad)
        )
        within_velocity = all(abs(qd) < self.config.epsilon_q_dot_rad_s for qd in q_encoder_dot_rad_s)
        ok = within_position and within_velocity

        if ok:
            if self._good_since_ns is None:
                self._good_since_ns = now_ns
            if (now_ns - self._good_since_ns) / 1e9 >= self.config.dwell_s:
                self.arrived = True
        else:
            self._good_since_ns = None
            self.arrived = False
        return self.arrived


class ExperimentStateMachine:
    def __init__(
        self,
        detector: AdaptiveSettlingDetector,
        encoder_gate: EncoderArrivalGate,
        event_bus: Optional[EventBus] = None,
        measurement_window_s: Optional[float] = None,
        settle_timeout_s: Optional[float] = 15.0,
    ):
        self._detector = detector
        self._encoder_gate = encoder_gate
        self._bus = event_bus
        self._measurement_window_s = (
            measurement_window_s if measurement_window_s is not None else detector.config.measurement_window_s
        )
        # reasonable-but-UNVALIDATED default (same status as every
        # threshold in adaptive_settling_detector.SettlingConfig -- see
        # that module's docstring), None disables the timeout entirely.
        self.settle_timeout_s = settle_timeout_s

        self._state = ExperimentState.MOVE_TO_TARGET
        self._encoder_arrival_ns: Optional[int] = None
        self._measurement_window_start_ns: Optional[int] = None
        self.settle_time_from_encoder_s: Optional[float] = None
        self.last_evaluation = None
        self._motion_onset_emitted = False

    @property
    def current_state(self) -> str:
        """Cheap read of the current state -- other collectors stamp this
        into their own records' `experiment_state` field."""
        return self._state.value

    def _set_state(self, new_state: ExperimentState, now_ns: int, **event_details) -> None:
        if new_state == self._state:
            return
        old = self._state
        self._state = new_state
        if self._bus is not None:
            self._bus.publish(
                "experiment_state_changed",
                {"from": old.value, "to": new_state.value, "t_pc_ns": now_ns, **event_details},
            )

    def start_new_target(self, now_ns: int, target_id: Optional[str] = None) -> None:
        """Reset everything for a fresh trial and enter WAIT_ENCODER_ARRIVAL.
        `MOVE_TO_TARGET` is not itself an actionable state in this task
        (no hardware is ever moved here) -- it's a one-tick pass-through so
        the enum/event trail stays complete."""
        if self._bus is not None:
            self._bus.publish("target_selected", {"target_id": target_id, "t_pc_ns": now_ns})
        self._encoder_gate.reset()
        self._detector.reset()
        self._encoder_arrival_ns = None
        self._measurement_window_start_ns = None
        self.settle_time_from_encoder_s = None
        self.last_evaluation = None
        self._motion_onset_emitted = False
        self._state = ExperimentState.MOVE_TO_TARGET  # no event for the implicit starting state
        self._set_state(ExperimentState.WAIT_ENCODER_ARRIVAL, now_ns, target_id=target_id)

    def command_sent(self, now_ns: int, **details) -> None:
        """Call right after the actual move command is written to
        hardware (e.g. the ESP32 D2/D3/D5 legacy interface) -- publishes
        the COMMAND_SENT event. No state change: this state machine has no
        handle on any collector/writer and MOVE_TO_TARGET already
        pass-throughs immediately (see start_new_target's docstring)."""
        if self._bus is not None:
            self._bus.publish("command_sent", {"t_pc_ns": now_ns, **details})

    def update_encoder(
        self,
        now_ns: int,
        q_encoder_rad: Sequence[float],
        q_cmd_rad: Sequence[float],
        q_encoder_dot_rad_s: Sequence[float],
    ) -> str:
        if self._state != ExperimentState.WAIT_ENCODER_ARRIVAL:
            return self._state.value
        if not self._motion_onset_emitted:
            # ponytail: no independent motion-onset detector exists (that
            # would be new detection math, out of scope here) -- the first
            # encoder update after a target is selected is treated as
            # onset. Ceiling: this fires even if the arm was already at
            # the target (zero real motion) or the first update lands well
            # after actual motion started; upgrade to a real
            # delta-position-vs-baseline threshold if that ever matters in
            # practice.
            self._motion_onset_emitted = True
            if self._bus is not None:
                self._bus.publish(
                    "encoder_motion_onset",
                    {"t_pc_ns": now_ns, "heuristic": "first_encoder_update_after_target_selected"},
                )
        arrived = self._encoder_gate.update(now_ns, q_encoder_rad, q_cmd_rad, q_encoder_dot_rad_s)
        if arrived:
            self._encoder_arrival_ns = now_ns
            self._set_state(ExperimentState.ADAPTIVE_SETTLING_MONITOR, now_ns)
        return self._state.value

    def update_tip_imu(self, now_ns: int, tip_sample: Optional[TipSample], **guard_kwargs) -> str:
        """Advance the settling monitor / measurement window. `guard_kwargs`
        is forwarded to `AdaptiveSettlingDetector.evaluate` as-is (
        `base_gyro_rad_s`, `is_clock_sync_valid`, `queues`, ... )."""
        active_states = (
            ExperimentState.ADAPTIVE_SETTLING_MONITOR,
            ExperimentState.STABLE_CANDIDATE,
            ExperimentState.TIP_IMU_STABLE_CONFIRMED,
            ExperimentState.MEASUREMENT_WINDOW,
        )
        if self._state not in active_states:
            return self._state.value

        if tip_sample is not None:
            self._detector.add_tip_sample(tip_sample)
        evaluation = self._detector.evaluate(
            now_ns, encoder_settled=self._encoder_arrival_ns is not None, **guard_kwargs
        )
        self.last_evaluation = evaluation

        if (
            self.settle_timeout_s is not None
            and self._encoder_arrival_ns is not None
            and self._state != ExperimentState.MEASUREMENT_WINDOW
            and (now_ns - self._encoder_arrival_ns) / 1e9 >= self.settle_timeout_s
        ):
            # Never reached stable within the timeout -- abandon this
            # trial's measurement and move on. Raw data collection is
            # never gated by this (master spec: raw streams keep recording
            # regardless of detector state) -- this only decides the trial
            # produces no MEASUREMENT_ACCEPTED.
            self._set_state(ExperimentState.NEXT_TARGET, now_ns, reason="settle_timeout")
            return self._state.value

        if self._state == ExperimentState.MEASUREMENT_WINDOW:
            if not evaluation.overall_pass:
                # Discard and go back to monitoring -- never abort the whole
                # trial, never touch raw data collection (master spec: raw
                # streams keep recording regardless of detector state).
                self._measurement_window_start_ns = None
                self._set_state(
                    ExperimentState.ADAPTIVE_SETTLING_MONITOR, now_ns, reason="stability_broke_in_window"
                )
                return self._state.value
            elapsed = (now_ns - self._measurement_window_start_ns) / 1e9
            if elapsed >= self._measurement_window_s:
                self._set_state(ExperimentState.MEASUREMENT_ACCEPTED, now_ns)
            return self._state.value

        if self._state == ExperimentState.TIP_IMU_STABLE_CONFIRMED:
            # Held for at least one tick (a real, observable state -- not a
            # same-call pass-through) before opening the measurement window,
            # so a caller polling `current_state` can actually see it.
            if not evaluation.overall_pass:
                self._set_state(
                    ExperimentState.ADAPTIVE_SETTLING_MONITOR, now_ns, reason="broke_before_window_open"
                )
            else:
                self._measurement_window_start_ns = now_ns
                self._set_state(
                    ExperimentState.MEASUREMENT_WINDOW,
                    now_ns,
                    settle_time_from_encoder_s=self.settle_time_from_encoder_s,
                )
        elif evaluation.is_confirmed:
            self._set_state(ExperimentState.TIP_IMU_STABLE_CONFIRMED, now_ns)
            if self._encoder_arrival_ns is not None:
                self.settle_time_from_encoder_s = (now_ns - self._encoder_arrival_ns) / 1e9
        elif evaluation.is_candidate:
            self._set_state(ExperimentState.STABLE_CANDIDATE, now_ns)
        else:
            self._set_state(ExperimentState.ADAPTIVE_SETTLING_MONITOR, now_ns)
        return self._state.value

    def advance_to_next_target(self, now_ns: int) -> str:
        """Explicit caller-driven step from MEASUREMENT_ACCEPTED ->
        NEXT_TARGET (kept separate from the automatic transitions above --
        dispatching the next target is an external decision, e.g. the
        session controller consulting `safety_gate.py` first)."""
        if self._state != ExperimentState.MEASUREMENT_ACCEPTED:
            return self._state.value
        self._set_state(ExperimentState.NEXT_TARGET, now_ns)
        return self._state.value

    def build_stability_record(self, base: dict) -> Optional[dict]:
        """Convenience: build a `core.schemas.stability_record` from the
        most recent evaluation, stamped with the current state. `base` must
        already have `schema=SCHEMA_STABILITY, stream=STREAM_STABILITY`."""
        if self.last_evaluation is None:
            return None
        return stability_record_from_evaluation(base, self.last_evaluation, settling_state=self.current_state)


def translate_transition_to_event_types(payload: dict) -> list[str]:
    """Map one `experiment_state_changed` event payload (`{"from", "to",
    "t_pc_ns", ...}`, as published by `_set_state` above) to zero or more
    of master spec section 10's exact event-type strings. Kept as a pure
    function (not a method) so `session_controller.py` can drive it off
    the EventBus without this module depending on that one, and so it's
    trivially unit-testable on its own -- see `_self_check` below.

    `TARGET_SELECTED`/`COMMAND_SENT`/`ENCODER_MOTION_ONSET` are NOT
    produced here: those are published directly (bus event names
    `target_selected`/`command_sent`/`encoder_motion_onset`, not
    `experiment_state_changed`) from `start_new_target`/`command_sent`/
    `update_encoder` above -- session_controller.py subscribes to all
    three by name directly, 1:1, no translation needed.
    """
    to = payload.get("to")
    from_ = payload.get("from")
    reason = payload.get("reason")

    if to == ExperimentState.ADAPTIVE_SETTLING_MONITOR.value:
        if from_ == ExperimentState.WAIT_ENCODER_ARRIVAL.value:
            return ["ENCODER_ARRIVAL", "SETTLING_MONITOR_START"]
        if from_ == ExperimentState.STABLE_CANDIDATE.value:
            return ["STABLE_CANDIDATE_RESET"]
        if from_ in (ExperimentState.TIP_IMU_STABLE_CONFIRMED.value, ExperimentState.MEASUREMENT_WINDOW.value):
            return ["STABILITY_LOST"]
        return []
    if to == ExperimentState.STABLE_CANDIDATE.value:
        return ["STABLE_CANDIDATE"]
    if to == ExperimentState.TIP_IMU_STABLE_CONFIRMED.value:
        return ["TIP_IMU_STABLE_CONFIRMED"]
    if to == ExperimentState.MEASUREMENT_WINDOW.value:
        return ["MEASUREMENT_WINDOW_START"]
    if to == ExperimentState.MEASUREMENT_ACCEPTED.value:
        return ["MEASUREMENT_ACCEPTED"]
    if to == ExperimentState.NEXT_TARGET.value:
        if reason == "settle_timeout":
            return ["SETTLE_TIMEOUT", "TRIAL_COMPLETE"]
        return ["TRIAL_COMPLETE"]
    return []


def _self_check() -> None:
    from experiment.adaptive_settling_detector import SettlingConfig

    config = SettlingConfig(window_s=0.1, candidate_duration_s=0.05, confirmed_duration_s=0.1, measurement_window_s=0.1)
    detector = AdaptiveSettlingDetector(config)
    gate = EncoderArrivalGate(EncoderGateConfig(dwell_s=0.05))
    bus = EventBus()
    state_events = []
    other_events = []
    bus.subscribe("experiment_state_changed", lambda et, payload: state_events.append(payload))
    for name in ("target_selected", "command_sent", "encoder_motion_onset"):
        bus.subscribe(name, lambda et, payload: other_events.append(et))
    sm = ExperimentStateMachine(detector, gate, event_bus=bus)

    now = 0
    period_ns = int(1e9 / config.tip_output_rate_hz)
    sm.start_new_target(now, target_id="T001")
    assert sm.current_state == "WAIT_ENCODER_ARRIVAL"
    assert other_events == ["target_selected"]
    sm.command_sent(now)
    assert other_events == ["target_selected", "command_sent"]

    for _ in range(10):
        now += period_ns
        sm.update_encoder(now, [0.0] * 3, [0.0] * 3, [0.0] * 3)
    assert sm.current_state == "ADAPTIVE_SETTLING_MONITOR", sm.current_state
    assert other_events.count("encoder_motion_onset") == 1, "must fire exactly once per target, not every tick"

    reached_accepted = False
    for i in range(200):
        now += period_ns
        sample = TipSample(
            t_pc_rx_ns=now,
            gyro_rads=(0.0, 0.0, 0.0),
            accel_mps2=(0.0, 0.0, 9.81),
            euler_rad=(0.0, 0.0, 0.0),
            raw_hex=f"{i:08x}",
        )
        sm.update_tip_imu(now, sample)
        if sm.current_state == "MEASUREMENT_ACCEPTED":
            reached_accepted = True
            break
    assert reached_accepted, sm.current_state
    to_values = [p["to"] for p in state_events]
    assert "TIP_IMU_STABLE_CONFIRMED" in to_values
    assert sm.settle_time_from_encoder_s is not None and sm.settle_time_from_encoder_s >= 0.0

    # translate_transition_to_event_types against the real transitions this
    # run actually produced -- section 10's event flow, end to end.
    mapped = [t for p in state_events for t in translate_transition_to_event_types(p)]
    for expected in ("ENCODER_ARRIVAL", "SETTLING_MONITOR_START", "TIP_IMU_STABLE_CONFIRMED", "MEASUREMENT_WINDOW_START", "MEASUREMENT_ACCEPTED"):
        assert expected in mapped, (expected, mapped)

    sm.advance_to_next_target(now)
    assert sm.current_state == "NEXT_TARGET"
    assert translate_transition_to_event_types({"to": "NEXT_TARGET", "from": "MEASUREMENT_ACCEPTED"}) == ["TRIAL_COMPLETE"]
    assert translate_transition_to_event_types(
        {"to": "NEXT_TARGET", "from": "ADAPTIVE_SETTLING_MONITOR", "reason": "settle_timeout"}
    ) == ["SETTLE_TIMEOUT", "TRIAL_COMPLETE"]
    assert translate_transition_to_event_types(
        {"to": "ADAPTIVE_SETTLING_MONITOR", "from": "STABLE_CANDIDATE"}
    ) == ["STABLE_CANDIDATE_RESET"]
    assert translate_transition_to_event_types(
        {"to": "ADAPTIVE_SETTLING_MONITOR", "from": "MEASUREMENT_WINDOW", "reason": "stability_broke_in_window"}
    ) == ["STABILITY_LOST"]

    # settle_timeout: a target that never stabilizes must give up and
    # advance on its own, emitting SETTLE_TIMEOUT + TRIAL_COMPLETE.
    detector2 = AdaptiveSettlingDetector(config)
    gate2 = EncoderArrivalGate(EncoderGateConfig(dwell_s=0.05))
    bus2 = EventBus()
    timeout_events = []
    bus2.subscribe("experiment_state_changed", lambda et, payload: timeout_events.append(payload))
    sm2 = ExperimentStateMachine(detector2, gate2, event_bus=bus2, settle_timeout_s=0.05)
    now2 = 0
    sm2.start_new_target(now2)
    for _ in range(10):
        now2 += period_ns
        sm2.update_encoder(now2, [0.0] * 3, [0.0] * 3, [0.0] * 3)
    assert sm2.current_state == "ADAPTIVE_SETTLING_MONITOR"
    # never-settling tip stream (large constant gyro -- always fails the threshold)
    for i in range(200):
        now2 += period_ns
        sample = TipSample(
            t_pc_rx_ns=now2,
            gyro_rads=(1.0, 0.0, 0.0),
            accel_mps2=(0.0, 0.0, 9.81),
            euler_rad=(0.0, 0.0, 0.0),
            raw_hex=f"{i:08x}",
        )
        sm2.update_tip_imu(now2, sample)
        if sm2.current_state == "NEXT_TARGET":
            break
    assert sm2.current_state == "NEXT_TARGET", sm2.current_state
    assert timeout_events[-1]["reason"] == "settle_timeout"
    assert translate_transition_to_event_types(timeout_events[-1]) == ["SETTLE_TIMEOUT", "TRIAL_COMPLETE"]

    print("experiment_state_machine.py self-check OK")


if __name__ == "__main__":
    _self_check()
