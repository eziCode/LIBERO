import hid
import struct
import time
import numpy as np

VID = 1118
PID = 765

DEADZONE = 0.05
AXIS_MAX = 32768.0
TRIGGER_MAX = 1023.0

def apply_deadzone(value, deadzone=DEADZONE):
    if abs(value) < deadzone:
        return 0.0
    scaled = (abs(value) - deadzone) / (1.0 - deadzone)
    return np.sign(value) * scaled

def read_axis(data, offset):
    """Read signed 16-bit LE and normalize to -1.0..1.0."""
    raw = struct.unpack_from('<h', bytes(data), offset)[0]
    return max(-1.0, min(1.0, raw / AXIS_MAX))

# Find and open the controller
controller = hid.device()
controller.open(VID, PID)
controller.set_nonblocking(True)
print("Controller opened. Press buttons / move sticks to test.\n")

try:
    while True:
        data = controller.read(64)
        if data and len(data) >= 18 and data[0] == 0x20:
            # Buttons (byte 4)
            btn1 = data[4]
            a_btn = (btn1 >> 4) & 1
            b_btn = (btn1 >> 5) & 1
            x_btn = (btn1 >> 6) & 1
            y_btn = (btn1 >> 7) & 1
            back_btn = (btn1 >> 3) & 1
            menu_btn = (btn1 >> 2) & 1

            # Triggers (16-bit LE, 0-1023)
            lt = (data[6] | (data[7] << 8)) / TRIGGER_MAX
            rt = (data[8] | (data[9] << 8)) / TRIGGER_MAX

            # Joystick axes (signed 16-bit LE)
            lx = apply_deadzone(read_axis(data, 10))
            ly = apply_deadzone(read_axis(data, 12))
            rx = apply_deadzone(read_axis(data, 14))
            ry = apply_deadzone(read_axis(data, 16))

            # Build button list
            buttons = []
            if a_btn: buttons.append("A")
            if b_btn: buttons.append("B")
            if x_btn: buttons.append("X")
            if y_btn: buttons.append("Y")
            if back_btn: buttons.append("BACK")
            if menu_btn: buttons.append("MENU")

            # Only print when something is happening
            if buttons or lt > 0.05 or rt > 0.05 or lx or ly or rx or ry:
                print(f"LX:{lx:+.2f} LY:{ly:+.2f} RX:{rx:+.2f} RY:{ry:+.2f} "
                      f"| LT:{lt:.2f} RT:{rt:.2f} | {' '.join(buttons)}")
        time.sleep(0.01)
finally:
    controller.close()