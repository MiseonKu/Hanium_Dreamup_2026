from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, TypeVar


T = TypeVar("T")


class SensorReader(Protocol[T]):
    def samples(self) -> Iterable[T]:
        """Yield normalized samples without binding the rest of the pipeline to IO."""


class CloseableReader(Protocol[T]):
    def samples(self) -> Iterable[T]:
        """Yield normalized samples."""

    def close(self) -> None:
        """Release any underlying resource."""
