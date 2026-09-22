"""Graceful-shutdown signal handling (master spec section 16, step 1 of the
12-step sequence: set shutdown flag -> stop issuing new work -> drain ->
flush -> fsync -> close).

This module only owns the flag-setting. It does NOT implement the rest of
the sequence -- that's `session_controller.py`'s job (built separately),
reading `shutdown_requested` and acting on it.
"""
from __future__ import annotations

import contextlib
import signal
import threading

shutdown_requested = threading.Event()

_SIGNALS = [signal.SIGINT, signal.SIGTERM]
if hasattr(signal, "SIGBREAK"):  # Windows Ctrl+Break
    _SIGNALS.append(signal.SIGBREAK)


def _handler(signum, frame) -> None:  # noqa: ARG001 - signal handler signature
    shutdown_requested.set()


def install() -> None:
    """Wire SIGINT/SIGTERM (+SIGBREAK on Windows) to set
    `shutdown_requested` instead of raising/exiting immediately. Must be
    called from the main thread (a `signal.signal` requirement)."""
    for sig in _SIGNALS:
        signal.signal(sig, _handler)


@contextlib.contextmanager
def installed():
    """Context-manager form: installs on enter, restores whatever handlers
    were previously registered on exit. Yields `shutdown_requested`."""
    previous = {sig: signal.getsignal(sig) for sig in _SIGNALS}
    install()
    try:
        yield shutdown_requested
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _self_check() -> None:
    shutdown_requested.clear()
    install()
    assert not shutdown_requested.is_set()
    _handler(signal.SIGINT, None)
    assert shutdown_requested.is_set()
    shutdown_requested.clear()
    print("interrupt_handler.py self-check OK")


if __name__ == "__main__":
    _self_check()
