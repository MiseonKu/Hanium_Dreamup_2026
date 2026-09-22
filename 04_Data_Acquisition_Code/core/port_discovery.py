"""Serial port enumeration + sensor-map matching (master spec section 13).

No hardcoded COM port numbers anywhere in this module -- 07_WT901C's
`DEFAULT_PORTS = ("COM22", "COM21", "COM20", "COM18")` was flagged as a
machine-specific smell in `docs/07_WT901C_CODE_AUDIT.md`; this module
always enumerates live via pyserial instead. `tools/inventory_ports.py`
(built separately) is a thin CLI wrapper around `list_ports()`.
"""
from __future__ import annotations

from typing import Any, Optional

import serial.tools.list_ports


def list_ports() -> list[dict[str, Any]]:
    """Enumerate currently-attached serial devices."""
    ports = []
    for info in serial.tools.list_ports.comports():
        ports.append(
            {
                "device": info.device,
                "description": info.description,
                "vid": info.vid,
                "pid": info.pid,
                "serial_number": info.serial_number,
                "manufacturer": info.manufacturer,
                "location": info.location,
            }
        )
    return ports


def match_sensor_map(
    ports: list[dict[str, Any]], sensor_map: dict[str, dict[str, Any]]
) -> dict[str, Optional[dict[str, Any]]]:
    """Match discovered `ports` (as returned by `list_ports()`) against
    `sensor_map`, keyed by logical sensor id:

        sensor_map = {
            "imu3_tip": {"vid": 0x1A86, "pid": 0x7523, "serial_number": "ABC123"},
            "esp32_encoder": {"vid": 0x10C4, "pid": 0xEA60},
        }

    Any subset of vid/pid/serial_number may be given per logical id; unset
    keys are ignored (not required to match). Returns
    `{logical_id: matching_port_dict_or_None}`. A port already claimed by
    an earlier logical id in `sensor_map` is not reused (first match wins,
    in `sensor_map` iteration order) -- plain dict matching, no class
    hierarchy.
    """
    claimed_devices: set[str] = set()
    result: dict[str, Optional[dict[str, Any]]] = {}
    for logical_id, expected in sensor_map.items():
        match = None
        for port in ports:
            if port["device"] in claimed_devices:
                continue
            if _matches(port, expected):
                match = port
                break
        result[logical_id] = match
        if match is not None:
            claimed_devices.add(match["device"])
    return result


def _matches(port: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key in ("vid", "pid", "serial_number"):
        if key in expected and expected[key] is not None:
            if port.get(key) != expected[key]:
                return False
    return True


def _self_check() -> None:
    ports = [
        {"device": "COM3", "vid": 0x1A86, "pid": 0x7523, "serial_number": "AAA", "description": "", "manufacturer": "", "location": ""},
        {"device": "COM4", "vid": 0x10C4, "pid": 0xEA60, "serial_number": "BBB", "description": "", "manufacturer": "", "location": ""},
        {"device": "COM5", "vid": 0x10C4, "pid": 0xEA60, "serial_number": "CCC", "description": "", "manufacturer": "", "location": ""},
    ]
    sensor_map = {
        "imu3_tip": {"vid": 0x1A86, "pid": 0x7523},
        "esp32_encoder": {"vid": 0x10C4, "pid": 0xEA60, "serial_number": "CCC"},
        "not_connected": {"vid": 0xFFFF, "pid": 0xFFFF},
    }
    matched = match_sensor_map(ports, sensor_map)
    assert matched["imu3_tip"]["device"] == "COM3"
    assert matched["esp32_encoder"]["device"] == "COM5"  # serial_number narrows it to CCC
    assert matched["not_connected"] is None
    print("port_discovery.py self-check OK")


if __name__ == "__main__":
    _self_check()
