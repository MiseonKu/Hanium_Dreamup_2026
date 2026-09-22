# Serial Protocol: `STREAM` Telemetry (02_SO-101_rtos_motor_driver)

Source of truth: `02_SO-101_rtos_motor_driver/README.md` (top-level, "For
PC-side periodic logging..." section) and
`02_SO-101_rtos_motor_driver/main/README.md` ("Compact Status Stream"
section). Cross-checked byte-for-byte against
`collectors/collect_esp32_encoder.py`'s parser (`_LINE_RE`,
`parse_stream_line`), which is this protocol's only consumer in this
package, and against `mocks/mock_esp32.py`'s `MockESP32Encoder`, which
emits lines in this exact format for tests.

## Transport

- ASCII over USB serial, newline-terminated lines.
- Baud: `115200` (USB command serial; the servo bus itself runs at
  `1,000,000` baud internally and is not exposed over USB).
- No handshake beyond opening the port.

## Commands this bridge sends

Exactly one, on connect:

```text
STREAM ON <period_ms>
```

Example: `STREAM ON 100` (10 Hz). `collectors/collect_esp32_encoder.py`'s
`Esp32EncoderCollector.__init__` defaults `period_ms=100`.

`02_SO-101_rtos_motor_driver` also supports `STREAM OFF` and `STREAM
STATUS`, and a separate, more verbose `ENCODER ON [period_ms]` / `ENCODER
OFF` / `ENCODER STATUS` command family (one line per motor per tick,
documented in `02_SO-101_rtos_motor_driver/main/README.md`'s "Encoder
Logging" section). This bridge does not use `ENCODER` -- it only ever
subscribes to the compact `STREAM` format, per the 05 README's own
recommendation ("For PC-side periodic logging, use `STREAM ON 100` rather
than requesting full `STATUS` and verbose `ENCODER` lines").

This bridge never sends any other command (see `README.md` in this
directory for the full list of what it deliberately does not send).

## Line format received

One line per sample tick, fixed field order:

```text
STREAM t_ms=12345 seq=1 fresh=1 armed=1 cal=0 fault=0 fault_axis=-1 mode=PID enabled=0x1F pos=2225,2100,1988,1900,700 upd=12344,12344,12344,12344,12344 deg=80.00,35.00,77.00,155.00,50.00 target=80.00,35.00,77.00,155.00,50.00
```

Exact regex `collectors/collect_esp32_encoder.py` matches against
(`_LINE_RE`):

```text
^STREAM t_ms=(?P<t_ms>\d+) seq=(?P<seq>\d+) fresh=(?P<fresh>\d) armed=(?P<armed>\d) cal=(?P<cal>\d) fault=(?P<fault>\d) fault_axis=(?P<fault_axis>-?\d+) mode=(?P<mode>\S+) enabled=0x(?P<enabled>[0-9A-Fa-f]+) pos=(?P<pos>[\-\d,]+) upd=(?P<upd>[\d,]+) deg=(?P<deg>[\-\d.,]+) target=(?P<target>[\-\d.,]+)$
```

### Field-by-field

| Field | Type | Meaning |
|---|---|---|
| `t_ms` | uint | ESP32 stream timestamp since boot, milliseconds. Converted to microseconds for `t_esp_us`/`t_source` (`t_ms * 1000`) since `core.schemas.encoder_record` expects microseconds. |
| `seq` | uint | Stream sample number, assigned by the ESP32. Mapped to `source_seq`. |
| `fresh` | 0/1 | `1` when feedback was refreshed this tick (or the armed control loop is providing cached live feedback). |
| `armed` | 0/1 | Torque-enabled state. |
| `cal` | 0/1 | Calibration (`CAL_AUTO RUN`) in progress. |
| `fault` | 0/1 | A fault is latched. |
| `fault_axis` | int, `-1` or joint index | `-1` when no fault; otherwise the 0-based index of the faulted joint. |
| `mode` | string (`PD`\|`PID`) | Outer controller mode. |
| `enabled` | hex bitmask, `0x`-prefixed | One bit per joint, joint order below. `0x1F` = all 5 enabled. |
| `pos` | 5 comma-separated ints | Raw ST3215 encoder ticks, one per joint, fixed joint order. |
| `upd` | 5 comma-separated uints | Per-axis ESP32 feedback-update timestamp (ms since boot), fixed joint order. |
| `deg` | 5 comma-separated floats | Logical joint angle in degrees, or the sentinel `-999.00` when raw soft limits aren't calibrated yet (`mapping_ready=false` in `robot_config.h`/`rtos_motor_config.h`). A `-999.00` value is mapped to `None` (not `-999.0`) in `q_encoder_rad`, with `quality_flags: ["joint_limits_not_ready"]` added to the record. |
| `target` | 5 comma-separated floats | Current commanded logical target, degrees, fixed joint order. Always a real angle (no `-999.00` sentinel observed for this field). |

### Fixed joint order

`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, gripper` -- 05's axis
IDs 1 through 5, in that order, for every comma-separated field above
(`pos`, `upd`, `deg`, `target`) and every bit of `enabled` (bit 0 =
`shoulder_pan` ... bit 4 = `gripper`). This order is positional, not
labeled in the wire line itself -- both `collectors/collect_esp32_encoder.py`
and `mocks/mock_esp32.py` hardcode it as
`JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "gripper")`.
Do not reorder.

## Mapping to `core.schemas.encoder_record`

| Wire field(s) | Schema field |
|---|---|
| `t_ms * 1000` | `t_esp_us`, `t_source` |
| `seq` | `source_seq` |
| `pos` | `data.encoder_raw_tick` |
| `deg` (degrees -> radians, `-999.00` -> `None`) | `data.q_encoder_rad` |
| `target` (degrees -> radians) | `data.q_cmd_rad` |
| `armed`/`cal`/`fault` | `data.esp32_state` (`"fault"` > `"calibrating"` > `"armed"` > `"idle"`, priority in that order) |
| `fault` + `fault_axis` | `data.fault` (`"fault"`/`"ok"` per joint) |
| `enabled` bitmask | `data.motor_status` (`"enabled"`/`"disabled"` per joint) |
| unparsable line | record still written: `valid=False`, `parse_status="parse_error"`, `raw_line=<original text>` -- never silently dropped |

## What is deliberately NOT implemented here

- `ENCODER ON`/`ENCODER OFF`/`ENCODER STATUS` verbose per-motor format --
  documented above for context only; no parser for it exists in this
  package.
- Any write/motion/calibration command (`SET`, `FRAME`, `ARM`, `HOLD`,
  `STOP`, `ETOP`, `DISARM`, `CAL_AUTO`, `LIMIT`, `MAP`, `GAINS`, `MODE`,
  `WATCHDOG`, `SAVE`, `LOAD`, `TIMING`, `PINGALL`, `DETECT`, `INFO`,
  `FAULTRESET`) -- see `README.md` in this directory.

## Hardware status

`HARDWARE_NOT_TESTED`. This exact parser has only been exercised against
`mocks/mock_esp32.py`'s synthetic `STREAM` lines (see
`tests/test_collect_esp32_encoder.py`,
`tests/test_disconnect_reconnect.py`), never against a real ESP32 running
`02_SO-101_rtos_motor_driver/main`.
