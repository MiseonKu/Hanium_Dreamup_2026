"""Composable fault injectors for the mocks in this package.

Each `with_*` function takes a mock (`MockWT901CImu`, `MockESP32Encoder`, or
`MockViveTracker` -- or another `with_*`-wrapped mock) and returns a thin
wrapper exposing the SAME "produce next output" method name
(`next_packet_bytes` / `next_line` / `next_sample`), so wrappers compose:

    m = with_disconnect(with_corrupted_checksum(MockWT901CImu()))
    m.next_packet_bytes()   # corrupted checksum sometimes, may raise ConnectionError

Every other attribute (`set_moving`, `sequence`, `set_target`, ...) passes
through untouched to the innermost mock via `__getattr__`.

Master spec section 23 / 19.3 fault types covered: corrupted checksum,
random-byte noise (resync), duplicate packets, reordered packets, simulated
disconnect, and frozen/stale sensor values.
"""
from __future__ import annotations

import random
import re
from typing import Any, Callable, Optional

_PRODUCE_METHODS = ("next_packet_bytes", "next_line", "next_sample")


def _find_produce_name(mock: Any) -> str:
    for name in _PRODUCE_METHODS:
        if hasattr(mock, name):
            return name
    raise TypeError(f"mock {mock!r} exposes none of {_PRODUCE_METHODS}")


def _require_bytes_mock(mock: Any, name: str, fn_name: str) -> None:
    if name != "next_packet_bytes":
        raise TypeError(f"{fn_name} only applies to byte-stream mocks (next_packet_bytes)")


class _Wrapped:
    """Forwards everything to `inner` except `produce_name`, which is
    overridden with `call_fn`."""

    def __init__(self, inner: Any, produce_name: str, call_fn: Callable[[], Any]):
        self._inner = inner
        self._produce_name = produce_name
        setattr(self, produce_name, call_fn)

    def __getattr__(self, item):
        return getattr(self._inner, item)


def with_corrupted_checksum(mock: Any, every_n: int = 10) -> Any:
    """Every `every_n`-th packet has its checksum byte XORed with 0xFF, so
    `sum(pkt[:10]) & 0xFF == pkt[10]` no longer holds (guaranteed mismatch,
    since XOR with a nonzero value always changes the byte)."""
    name = _find_produce_name(mock)
    _require_bytes_mock(mock, name, "with_corrupted_checksum")
    produce = getattr(mock, name)
    state = {"n": 0}

    def call():
        state["n"] += 1
        pkt = produce()
        if state["n"] % every_n == 0:
            pkt = bytearray(pkt)
            pkt[-1] ^= 0xFF
            pkt = bytes(pkt)
        return pkt

    return _Wrapped(mock, name, call)


def with_random_byte_noise(
    mock: Any, every_n: int = 20, num_bytes: int = 3, seed: Optional[int] = None
) -> Any:
    """Every `every_n`-th call, prepends `num_bytes` of random junk before
    the real packet, simulating line noise a parser must resync past."""
    name = _find_produce_name(mock)
    _require_bytes_mock(mock, name, "with_random_byte_noise")
    produce = getattr(mock, name)
    rng = random.Random(seed)
    state = {"n": 0}

    def call():
        state["n"] += 1
        pkt = produce()
        if state["n"] % every_n == 0:
            noise = bytes(rng.randrange(256) for _ in range(num_bytes))
            pkt = noise + pkt
        return pkt

    return _Wrapped(mock, name, call)


def with_duplicate_packets(mock: Any, rate: float = 0.05, seed: Optional[int] = None) -> Any:
    """With probability `rate`, re-emit the exact previous output instead of
    a fresh one (byte/line/dict-identical duplicate, including whatever
    sequence number it already carried)."""
    name = _find_produce_name(mock)
    produce = getattr(mock, name)
    rng = random.Random(seed)
    state = {"last": None}

    def call():
        if state["last"] is not None and rng.random() < rate:
            return state["last"]
        val = produce()
        state["last"] = val
        return val

    return _Wrapped(mock, name, call)


def with_reordered_packets(mock: Any, window: int = 3, seed: Optional[int] = None) -> Any:
    """Buffers `window` outputs and drains them out of order (shuffled),
    refilling from the underlying mock as each is consumed -- emits packets
    out of the sequence order they were generated in."""
    name = _find_produce_name(mock)
    produce = getattr(mock, name)
    rng = random.Random(seed)
    buf = [produce() for _ in range(window)]
    rng.shuffle(buf)

    def call():
        out = buf.pop(0)
        buf.append(produce())
        if rng.random() < 0.3:
            rng.shuffle(buf)
        return out

    return _Wrapped(mock, name, call)


def with_disconnect(mock: Any, after_n: int = 50, duration_calls: int = 20) -> Any:
    """After `after_n` calls, raises `ConnectionError` for the next
    `duration_calls` calls (simulated dropped link), then resumes normally."""
    name = _find_produce_name(mock)
    produce = getattr(mock, name)
    state = {"n": 0}

    def call():
        state["n"] += 1
        if after_n < state["n"] <= after_n + duration_calls:
            raise ConnectionError(f"mock sensor disconnected (call {state['n']})")
        return produce()

    return _Wrapped(mock, name, call)


def with_frozen_values(mock: Any, after_n: int = 100) -> Any:
    """After `after_n` calls, the underlying mock keeps running (its own
    sequence/timestamp counters keep advancing normally) but the physical
    reading is pinned to whatever it was at freeze time -- the specific
    "frozen/stale sensor" failure mode (master spec 19.3): metadata looks
    live, the signal itself never changes again.
    """
    name = _find_produce_name(mock)
    produce = getattr(mock, name)
    state = {"n": 0, "frozen": None}

    if name == "next_packet_bytes":
        frozen_by_type: dict = {}

        def call():
            pkt = produce()
            state["n"] += 1
            if state["n"] <= after_n:
                return pkt
            ftype = pkt[1]
            if ftype not in frozen_by_type:
                frozen_by_type[ftype] = pkt
            return frozen_by_type[ftype]

    elif name == "next_line":
        def call():
            line = produce()
            state["n"] += 1
            if state["n"] <= after_n:
                return line
            if state["frozen"] is None:
                state["frozen"] = {
                    "pos": re.search(r"pos=(\S+)", line).group(1),
                    "deg": re.search(r"deg=(\S+)", line).group(1),
                }
            line = re.sub(r"pos=\S+", "pos=" + state["frozen"]["pos"], line)
            line = re.sub(r"deg=\S+", "deg=" + state["frozen"]["deg"], line)
            return line

    else:  # next_sample -> dict (Vive)
        _FROZEN_KEYS = ("position_m", "quaternion_xyzw", "pose_matrix_3x4")

        def call():
            sample = produce()
            state["n"] += 1
            if state["n"] <= after_n:
                return sample
            if state["frozen"] is None:
                state["frozen"] = {k: sample[k] for k in _FROZEN_KEYS}
            sample = dict(sample)
            sample.update(state["frozen"])
            return sample

    return _Wrapped(mock, name, call)


def _self_check() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mock_imu import MockWT901CImu

    imu = MockWT901CImu(seed=1)
    corrupted = with_corrupted_checksum(imu, every_n=3)
    bad_seen = False
    for _ in range(9):
        pkt = corrupted.next_packet_bytes()
        ok = (sum(pkt[:10]) & 0xFF) == pkt[10]
        if not ok:
            bad_seen = True
    assert bad_seen, "with_corrupted_checksum never produced an invalid checksum"

    disc = with_disconnect(MockWT901CImu(seed=2), after_n=2, duration_calls=2)
    outcomes = []
    for _ in range(5):
        try:
            disc.next_packet_bytes()
            outcomes.append("ok")
        except ConnectionError:
            outcomes.append("down")
    assert outcomes == ["ok", "ok", "down", "down", "ok"], outcomes

    from mock_esp32 import MockESP32Encoder

    enc = MockESP32Encoder(seed=3)
    enc.set_target([10.0, 20.0, 30.0, 40.0, 50.0])
    frozen = with_frozen_values(enc, after_n=2)
    lines = [frozen.next_line() for _ in range(6)]
    degs = [ln.split("deg=")[1].split(" ")[0] for ln in lines]
    assert degs[0] != degs[1], "pre-freeze samples must still be moving"
    assert degs[2] == degs[3] == degs[4] == degs[5], "deg= must freeze after after_n calls"
    seqs = [int(re.search(r"seq=(\d+)", ln).group(1)) for ln in lines]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), "seq must keep advancing while frozen"

    print("fault_injector.py self-check OK")


if __name__ == "__main__":
    _self_check()
