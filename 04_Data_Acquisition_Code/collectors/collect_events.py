"""Event-logger collector: thin persistence adapter, not a logging
framework.

`log_event(...)` builds a `core.schemas.event_record` and writes it through
a `core.jsonl_writer.JsonlWriter` for the `events` stream. Meant to be
called directly by other components (session controller, health
supervisor, other collectors reporting connect/disconnect), and/or wired to
a shared `core.event_bus.EventBus` via `wire_event_type()` so anything that
`publish()`es an event of interest gets persisted automatically.

ponytail: intentionally small -- just base_record/event_record ->
JsonlWriter.put(). No batching, no severity levels, no filtering, no
wildcard subscription (EventBus itself doesn't support that either, see its
own docstring). Add those only if a real need shows up.
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Any, Optional

# Same pattern as the sibling collectors -- needed for the `__main__`
# self-check at the bottom to run standalone.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import schemas  # noqa: E402
from core.event_bus import EventBus  # noqa: E402
from core.jsonl_writer import JsonlWriter  # noqa: E402


class EventCollector:
    def __init__(
        self,
        session_id: str,
        segment_id: int,
        output_dir: str,
        *,
        event_bus: Optional[EventBus] = None,
        flush_interval_s: float = 0.4,
        fsync_interval_s: float = 1.5,
    ) -> None:
        self.session_id = session_id
        self.segment_id = segment_id
        self.event_bus = event_bus
        self.writer = JsonlWriter(
            output_dir,
            schemas.STREAM_EVENT,
            segment_id,
            flush_interval_s=flush_interval_s,
            fsync_interval_s=fsync_interval_s,
        )
        self._lock = threading.Lock()
        self._host_seq = 0

    def wire_event_type(self, event_type: str) -> None:
        """Subscribe so every `event_bus.publish(event_type, payload)`
        call gets persisted automatically. `payload["message"]` (if
        present) becomes the record's `message`; the rest of `payload`
        becomes `details`. Call once per `event_type` of interest -- there
        is no wildcard subscription (see `core/event_bus.py`)."""
        if self.event_bus is None:
            raise RuntimeError("no event_bus was given to this EventCollector")
        self.event_bus.subscribe(event_type, self._on_bus_event)

    def _on_bus_event(self, event_type: str, payload: dict) -> None:
        details = dict(payload)
        message = details.pop("message", None)
        self.log_event(event_type, message=message, details=details)

    def log_event(
        self,
        event_type: str,
        *,
        message: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        sensor_id: Optional[str] = None,
        **base_kwargs: Any,
    ) -> dict:
        """Build and persist one `event_record`. `**base_kwargs` forwards
        to `core.schemas.base_record` (e.g. `trial_id`, `target_id`,
        `experiment_state`) for callers that have that context."""
        with self._lock:
            self._host_seq += 1
            host_seq = self._host_seq

        base = schemas.base_record(
            schema=schemas.SCHEMA_EVENT,
            session_id=self.session_id,
            segment_id=self.segment_id,
            stream=schemas.STREAM_EVENT,
            sensor_id=sensor_id,
            host_seq=host_seq,
            t_pc_rx_ns=schemas.monotonic_ns(),
            valid=True,
            parse_status="ok",
            **base_kwargs,
        )
        record = schemas.event_record(base, event_type=event_type, message=message, details=details)
        self.writer.put(record)
        return record

    def close(self, clean: bool = True) -> None:
        self.writer.close(clean=clean)


def _self_check() -> None:
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        bus = EventBus()
        collector = EventCollector("session_selfcheck", 0, tmp, event_bus=bus)
        collector.wire_event_type("sensor_disconnected")

        rec1 = collector.log_event("session_started", message="hello")
        assert rec1["event_type"] == "session_started"
        assert rec1["message"] == "hello"
        assert rec1["host_seq"] == 1

        bus.publish("sensor_disconnected", {"sensor_id": "imu3_tip", "message": "timeout", "connection_epoch": 2})
        collector.close(clean=True)

        with open(collector.writer.final_path, encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh]
        assert len(records) == 2
        assert records[1]["event_type"] == "sensor_disconnected"
        assert records[1]["message"] == "timeout"
        assert records[1]["details"] == {"sensor_id": "imu3_tip", "connection_epoch": 2}
        json.dumps(records)  # every field must be JSON-serializable

    print("collect_events.py self-check OK")


if __name__ == "__main__":
    _self_check()
