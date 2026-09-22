"""Minimal in-process pub/sub for cross-component events.

Lets `session_controller.py` (built separately) wire components together
(health-state changes, sensor disconnects, target transitions) without
tight coupling.

ponytail: deliberately not a generic message broker -- synchronous
dispatch, no event queue/thread, no wildcard subscriptions, no
unsubscribe-by-id. Add those only if a real need shows up.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Callable

EventCallback = Callable[[str, dict], None]


class EventBus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[str, list[EventCallback]] = defaultdict(list)

    def subscribe(self, event_type: str, callback: EventCallback) -> None:
        with self._lock:
            self._subscribers[event_type].append(callback)

    def publish(self, event_type: str, payload: dict) -> None:
        """Synchronous dispatch to every subscriber of `event_type`, in
        subscription order. Callbacks are invoked outside the lock so a
        callback that subscribes/publishes again doesn't deadlock."""
        with self._lock:
            callbacks = list(self._subscribers.get(event_type, ()))
        for callback in callbacks:
            callback(event_type, payload)


def _self_check() -> None:
    bus = EventBus()
    received = []
    bus.subscribe("health_state_changed", lambda et, payload: received.append((et, payload)))
    bus.publish("health_state_changed", {"new_state": "WARN"})
    bus.publish("unrelated_event", {"x": 1})  # no subscriber -- must not raise
    assert received == [("health_state_changed", {"new_state": "WARN"})]
    print("event_bus.py self-check OK")


if __name__ == "__main__":
    _self_check()
