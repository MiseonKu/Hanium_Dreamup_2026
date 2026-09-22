# Windows USB-Serial Driver Setup

General guidance only. This document still does not link a specific driver
installer, deliberately — get it from the chip vendor's own site.

**Amendment (2026-08-05):** the original framing ("no WT901C-RS232 dongle,
ESP32 board, or their driver installers were ever obtained or tested here",
`HARDWARE STATUS: NOT_TESTED`) is stale. The hardware exists and enumerates:
4x WT901C on COM16-19 and an ESP32 on COM20 on the bench that produced
`sessions/real_motion_demo/`. What is still true is that **nobody recorded
which bridge chip or driver version each device uses** — the VID/PID/
description values were never captured into
`config/sensor_map.example.yaml`, which still holds only `REPLACE_ME_*`
placeholders. That is a documentation gap worth closing on the next
bring-up (see step 5 below), not an unavailability of hardware.

## Why this matters

Every sensor this package talks to (4x WT901C IMU, ESP32 encoder bridge)
shows up to Windows as a USB-serial (COM) port. If the right driver isn't
installed, the device either doesn't enumerate at all, or enumerates as an
"Unknown device" / with a yellow-bang error in Device Manager, and
`core/port_discovery.py`'s `list_ports()` (via `pyserial`) won't see it as
a usable COM port.

## Common USB-serial chips you're likely to encounter

Sensor USB-to-serial adapters (WT901C-RS232 dongles, many ESP32 dev boards)
typically use one of a small number of bridge chips. Which one is actually
on your hardware is something you must check yourself (silkscreen on the
board, or the exact device string Windows shows) -- do not assume without
checking:

- **CP210x** (Silicon Labs) -- common on many ESP32 dev boards.
- **CH340/CH341** (WCH) -- common on WT901C-RS232-to-USB dongles and some
  ESP32 dev boards. A correctly-installed CH340 device typically shows a
  description like `USB-SERIAL CH340 (COMxx)` with VID `0x1A86` in
  `tools/inventory_ports.py`'s output.
- **FTDI FT232** -- common on many general-purpose USB-serial boards.

Get the correct driver from the chip vendor's own site (Silicon Labs' CP210x
VCP driver page, WCH's CH340 driver page, FTDI's VCP driver page) -- this
document does not link a specific installer because none was downloaded or
verified as part of this task. Search for the chip name plus "Windows VCP
driver" from the vendor's own domain, not a third-party mirror.

## General setup steps

1. Identify the actual bridge chip on your device (see above).
2. Download and run that chip vendor's official Windows VCP (Virtual COM
   Port) driver installer.
3. Plug in the device.
4. Open Device Manager (`Win+X` -> Device Manager) and check under
   "Ports (COM & LPT)":
   - A correctly-installed device shows as e.g. `Silicon Labs CP210x USB to
     UART Bridge (COMxx)` or `USB-SERIAL CH340 (COMxx)`, with no warning
     icon.
   - A yellow warning triangle means the driver isn't installed or didn't
     match -- reinstall, or try a different USB cable/port (some cables are
     power-only).
5. Note the assigned COM port number and the VID/PID Windows reports (visible
   in the device's Properties -> Details -> Hardware Ids). You'll want these
   for `config/sensor_map.example.yaml`.

## Verifying enumeration with this package's own tooling

Once Device Manager shows the port cleanly, confirm this package's own
serial-enumeration code agrees, using
[`tools/inventory_ports.py`](../tools/inventory_ports.py) (a thin wrapper
around `core.port_discovery.list_ports()`):

```powershell
python tools/inventory_ports.py
```

This prints a table of every currently-enumerated serial device: `DEVICE`,
`DESCRIPTION`, `VID`, `PID`, `SERIAL_NUMBER`, `MANUFACTURER`, `LOCATION`.
Run it once with the sensor unplugged and once with it plugged in -- the new
row that appears is your sensor. Copy its `VID`/`PID`/`SERIAL_NUMBER`
values into `config/sensor_map.example.yaml` (or your real
`sensor_map.yaml`) for `core.port_discovery.match_sensor_map()` to use.

If the device does not appear in this table at all after driver
installation, it is a Windows/driver-level enumeration problem, not
something in this package's code -- `list_ports()` is a direct pass-through
to `serial.tools.list_ports.comports()` (pyserial), which itself is a
direct pass-through to the OS's device list.

## What this document does not cover

- Sensor-specific configuration (WT901C output rate, baud, full-scale range
  registers) -- that's a device-configuration step via the sensor
  manufacturer's own tool, separate from the Windows driver, and still not
  done. It is now a known blocker rather than a hypothetical one: the units
  emit ~10 Hz per packet type and their full-scale-range registers have
  never been read back. See `docs/SAMPLING_AND_UART_BUDGET.md` and
  `docs/HARDWARE_TEST_PLAN.md`.
- ESP32 firmware flashing -- see
  `../firmware/esp32_so101_raw_bridge/README.md`; this package flashes
  nothing.
- Linux/macOS driver setup -- this package's development and testing both
  happened on Windows; no non-Windows guidance is offered here.
