import serial
import time

ser = serial.Serial("COM20", 115200, timeout=0.3)
time.sleep(1.8)
ser.reset_input_buffer()


def send(cmd, wait=0.6):
    ser.write((cmd + "\r\n").encode("ascii"))
    time.sleep(wait)
    resp = ser.read(2000)
    text = "".join(chr(c) if 32 <= c < 127 or c in (9, 10, 13) else "." for c in resp)
    print(f">>> {cmd}\n{text}")


send("CONFIG SET FAULT_TEMP 120")
send("CONFIG SAVE")
send("CONFIG SHOW")

ser.close()
