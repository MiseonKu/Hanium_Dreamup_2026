"""Read the ESP32's STATE once. No motion: only CAL_AUTO LOAD (which just
reloads the mapping into RAM) and STATE are sent.

`CAL_AUTO LOAD` is not optional here. Opening the serial port toggles DTR/RTS
and reboots the board, which wipes the runtime raw->logical mapping even
though the calibration itself survives in servo EEPROM. Without the reload,
every axis reports `logical_deg=NA` and `mapping_ready=false`, which looks
alarmingly like a lost calibration when nothing is actually wrong.

Note the board reset does NOT change the servos' real torque state, so
`torque=false` here can coexist with an arm that is still firmly held --
see docs/FIRMWARE_MOTION_HAZARDS.md.
"""
import time

import serial

TICKS_PER_DEG = 4096.0 / 360.0

ser = serial.Serial("COM20", 115200, timeout=0.3)
time.sleep(1.8)
ser.reset_input_buffer()


def send(cmd, wait):
    ser.write((cmd + "\r\n").encode("ascii"))
    time.sleep(wait)
    raw = ser.read(8000)
    return "".join(chr(c) if 32 <= c < 127 or c in (9, 10, 13) else "." for c in raw)


send("CAL_AUTO LOAD", 2.0)
text = send("STATE", 1.0)
ser.close()

print(text)

# Physical angles the firmware does not report: logical_deg is a per-axis
# label scale, not degrees (docs/FK_CONVENTION_CORRECTION.md).
REFERENCE_RAW = {"shoulder_pan": 2055, "shoulder_lift": 2093, "wrist_flex": 1970}
print("--- position relative to the reference pose (physical degrees) ---")
for line in text.splitlines():
    if not line.startswith("axis="):
        continue
    fields = dict(kv.split("=", 1) for kv in line.split() if "=" in kv)
    axis = fields.get("axis")
    if axis not in REFERENCE_RAW:
        continue
    raw = int(fields["raw_pos"])
    rel = (raw - REFERENCE_RAW[axis]) / TICKS_PER_DEG
    print(f"  {axis:15s} raw={raw:5d}  {rel:+7.1f} deg from reference   "
          f"(firmware label logical_deg={fields.get('logical_deg')})")
