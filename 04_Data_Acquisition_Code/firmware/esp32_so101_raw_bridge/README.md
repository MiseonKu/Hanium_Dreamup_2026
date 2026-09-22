# esp32_so101_raw_bridge

## This is documentation, not firmware

There is no `.cpp`/`.h` source in this directory, deliberately
(`DECISIONS.md` #7). This folder documents the wire-protocol contract that
`collectors/collect_esp32_encoder.py` implements a PC-side client against.
It does not add, fork, or modify any ESP32 firmware.

## What actually runs on the ESP32

The real, already-flashed-to-hardware firmware target is
`02_SO-101_rtos_motor_driver/main` (PlatformIO `esp32dev`), **not**
`03_SO-101_control/main`. `DECISIONS.md` #3 documents the evidence for this:
`02_SO-101_rtos_motor_driver` was created and promoted to be the active
`main` after `03_SO-101_control/main`'s last real commit, and only 05's
firmware implements the `STREAM`/`ENCODER` ASCII telemetry commands this
bridge depends on.

This package's "bridge" is a PC-side serial client only. It:

- opens the ESP32's USB serial port (115200 baud, per
  `02_SO-101_rtos_motor_driver/README.md`),
- sends exactly one command on connect: `STREAM ON <period_ms>`,
- parses the resulting `STREAM ...` lines into
  `core.schemas.encoder_record` rows,
- and sends **nothing else**.

It never sends `TORQUE`, `INIT`, `ARM`, `MOVE_RAW`, `SET`, `FRAME`,
`CAL_AUTO`, `HOLD`, `STOP`, `ETOP`, `DISARM`, `SAVE`, `LOAD`, `LIMIT`,
`MAP`, `GAINS`, `MODE`, or `WATCHDOG` -- any command that could move a
motor, change calibration, or alter persisted firmware config. This is a
read-only telemetry subscription, consistent with root `AGENTS.md`'s
hardware-safety rules and master-spec section 1.5 (no real motor commands
without explicit user intent, which this task does not have).

## Implementation

The actual client code is
[`collectors/collect_esp32_encoder.py`](../../collectors/collect_esp32_encoder.py)
(`Esp32EncoderCollector` class). Its `start()` method is the entire
write-side of this bridge:

```python
def start(self) -> None:
    """Read-only telemetry subscription. NEVER send TORQUE/INIT/
    MOVE_RAW/CAL_AUTO or any other write/motion command from this
    collector..."""
    if hasattr(self.conn, "write"):
        self.conn.write(f"STREAM ON {self.period_ms}\n".encode("ascii"))
```

Everything after that is read-only: `read_record()` blocks for the next
`STREAM ...` line, parses it, and returns one `encoder_record`. See
[`serial_protocol.md`](serial_protocol.md) in this same directory for the
exact line format, and
[`docs/DATA_SCHEMA.md`](../../docs/DATA_SCHEMA.md) for the resulting record
schema.

## Why no firmware source lives here

Per `DECISIONS.md` #7: the master spec's recommended structure suggested new
ESP32 firmware source files (`encoder_streamer.*`, `clock_sync.*`,
`experiment_state_machine.*`, `watchdog.*`, `safety.*`). Writing those here
would have meant a second, parallel firmware implementation that:

- risks diverging from whatever is actually flashed to the board, since
  this task never flashes or verifies firmware against real hardware
  (master-spec section 1.5), and
- would duplicate a wire contract that `02_SO-101_rtos_motor_driver/main`
  already implements and documents.

If a future task needs ESP32-side changes not already covered by the
`STREAM`/`ENCODER` protocol (e.g. a dedicated clock-sync response frame or a
new watchdog behavior), that is new, explicitly-scoped firmware work against
`02_SO-101_rtos_motor_driver/main` -- not something this documentation-only
directory should grow into.

## Source of truth for the protocol itself

- `02_SO-101_rtos_motor_driver/README.md` (top-level "Compact Status
  Stream" section, and the full command list)
- `02_SO-101_rtos_motor_driver/main/README.md` ("Compact Status Stream"
  section, with the field-by-field description and worked example line)

Both were read in full to write [`serial_protocol.md`](serial_protocol.md);
nothing there was guessed.

## Hardware status

**`HARDWARE_NOT_TESTED`.** This bridge has never been run against a real
ESP32 running `02_SO-101_rtos_motor_driver/main` -- only against
`mocks/mock_esp32.py`'s `MockESP32Encoder`, which emits byte-for-byte the
same `STREAM ...` line format described in `serial_protocol.md`. See
`docs/HARDWARE_TEST_PLAN.md` for what real-hardware validation this would
need before being trusted.
