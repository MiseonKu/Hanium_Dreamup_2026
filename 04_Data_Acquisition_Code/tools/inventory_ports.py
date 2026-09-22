"""CLI: python tools/inventory_ports.py

Thin wrapper around `core.port_discovery.list_ports()` -- prints device/
description/VID/PID/serial/manufacturer/location as a table (master spec
section 13).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.port_discovery import list_ports

_COLUMNS = ("device", "description", "vid", "pid", "serial_number", "manufacturer", "location")


def _cell(port: dict, col: str) -> str:
    value = port.get(col)
    if col in ("vid", "pid") and value is not None:
        return f"0x{value:04X}"
    return "" if value is None else str(value)


def format_table(ports: list[dict]) -> str:
    if not ports:
        return "(no serial ports found)"
    rows = [[_cell(p, c) for c in _COLUMNS] for p in ports]
    widths = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(_COLUMNS)]
    header = "  ".join(c.upper().ljust(w) for c, w in zip(_COLUMNS, widths))
    sep = "  ".join("-" * w for w in widths)
    lines = [header, sep]
    lines += ["  ".join(v.ljust(w) for v, w in zip(row, widths)) for row in rows]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ports = list_ports()
    print(format_table(ports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
