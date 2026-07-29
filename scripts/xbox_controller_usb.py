# xbox_controller_hid.py
import hid
import numpy as np
import struct
import threading
import time

from robosuite.devices import Device
from robosuite.utils.transform_utils import rotation_matrix


class XboxControllerHID(Device):
    """
    Xbox controller wrapper using hidapi (GIP protocol, report ID 0x20).
    Uses the SpaceMouse pattern: joystick position = velocity command.
    Holding the stick at full deflection = constant speed.

    GIP Report Format (19 bytes):
        d[0]     = 0x20 report ID
        d[1]     = unused
        d[2]     = sequence counter
        d[3]     = data length (44)
        d[4]     = buttons byte 1:
                     bit 2 = Menu (Start)
                     bit 3 = View (Back/Select)
                     bit 4 = A
                     bit 5 = B
                     bit 6 = X
                     bit 7 = Y
        d[5]     = buttons byte 2
        d[6:8]   = Left Trigger  (16-bit LE, 0-1023)
        d[8:10]  = Right Trigger (16-bit LE, 0-1023)
        d[10:12] = Left Stick X  (signed 16-bit LE)
        d[12:14] = Left Stick Y  (signed 16-bit LE)
        d[14:16] = Right Stick X (signed 16-bit LE)
        d[16:18] = Right Stick Y (signed 16-bit LE)
    """

    DEADZONE = 0.15
    AXIS_MAX = 32768.0
    TRIGGER_MAX = 1023.0

    def __init__(self, pos_sensitivity=1.0, rot_sensitivity=1.0, vendor_id=1118, product_id=2834): # 2834
        self.vid = vendor_id
        self.pid = product_id
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.device = None
        self._enabled = False
        self._reset_state = 0

        # Current control values (updated by background thread)
        # [lx, ly, dz, rx, ry, 0]  → [pos_y, pos_x, pos_z, rot_pitch, rot_yaw, 0]
        self._control = np.zeros(6)
        self.grasp = False
        self._last_b_state = 0
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])

    def start_control(self):
        """Open the HID device and start the background reader thread."""
        # Stop any previous thread
        self._enabled = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)

        # Close previous device if open
        if self.device:
            try:
                self.device.close()
            except Exception:
                pass

        self.device = hid.device()
        self.device.open(self.vid, self.pid)
        self.device.set_nonblocking(True)

        self._control = np.zeros(6)
        self.grasp = False
        self._last_b_state = 0
        self._reset_state = 0
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self._enabled = True

        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

        print(f"[DEBUG] HID controller opened (VID={self.vid}, PID={self.pid})")

    def _run(self):
        """Background thread that continuously reads HID reports."""
        while self._enabled:
            try:
                # Drain HID buffer, use only the latest report
                latest_data = None
                while True:
                    data = self.device.read(64)
                    if data:
                        latest_data = data
                    else:
                        break

                if latest_data and len(latest_data) >= 18 and latest_data[0] == 0x20:
                    self._process_gip_report(latest_data)
            except (ValueError, OSError):
                break  # device was closed, exit thread

            time.sleep(0.001)

    def _process_gip_report(self, data):
        """Parse GIP report and update current control values."""
        # --- Buttons (byte 4) ---
        btn1 = data[4]
        a_btn = (btn1 >> 4) & 1
        b_btn = (btn1 >> 5) & 1
        y_btn = (btn1 >> 7) & 1
        back_btn = (btn1 >> 3) & 1

        # --- Triggers (16-bit LE) ---
        lt = (data[6] | (data[7] << 8)) / self.TRIGGER_MAX
        rt = (data[8] | (data[9] << 8)) / self.TRIGGER_MAX

        # --- Joystick axes (signed 16-bit LE → -1.0 to 1.0) ---
        lx = self._apply_deadzone(self._read_axis(data, 10))
        ly = self._apply_deadzone(self._read_axis(data, 12))
        rx = self._apply_deadzone(self._read_axis(data, 14))
        ry = self._apply_deadzone(self._read_axis(data, 16))

        # Z axis from Y/A buttons
        dz = float(y_btn - a_btn)

        # Store current control state (like SpaceMouse._control)
        # Mapping: LY→X (inverted), LX→Y, buttons→Z, RY→roll(X-axis), 0, RX→yaw(Z-axis)
        self._control = np.array([
            -ly,   # X (forward/back, inverted)
            lx,    # Y (left/right)
            dz,    # Z (up/down from buttons)
            rt - lt,   # roll: triggers → rotation around X axis (X-Z plane)
            -ry,   # pitch: right stick U/D → rotation around Y axis (Y-Z plane, inverted)
            -rx,   # yaw: right stick L/R → rotation around Z axis (X-Y plane, inverted)
        ])

        # Grasp: B button toggles
        if b_btn and not self._last_b_state:
            self.grasp = not self.grasp
        self._last_b_state = b_btn

        # Reset
        if back_btn:
            self._reset_state = 1

    def _read_axis(self, data, offset):
        """Read a signed 16-bit LE value and normalize to -1.0..1.0."""
        raw = struct.unpack_from('<h', bytes(data), offset)[0]
        return max(-1.0, min(1.0, raw / self.AXIS_MAX))

    def get_controller_state(self):
        """
        Returns current controller state (SpaceMouse pattern).
        Joystick position = velocity command. Holding at full deflection = constant speed.
        """
        # Scale by sensitivity (same as SpaceMouse: control * 0.005 * sensitivity)
        dpos = self._control[:3] * 0.005 * self.pos_sensitivity

        roll, pitch, yaw = self._control[3:] * 0.005 * self.rot_sensitivity

        # Update rotation matrix (same as SpaceMouse)
        drot1 = rotation_matrix(angle=-pitch, direction=[1.0, 0, 0], point=None)[:3, :3]
        drot2 = rotation_matrix(angle=roll, direction=[0, 1.0, 0], point=None)[:3, :3]
        drot3 = rotation_matrix(angle=yaw, direction=[0, 0, 1.0], point=None)[:3, :3]

        self.rotation = self.rotation.dot(drot1.dot(drot2.dot(drot3)))

        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=np.array([roll, pitch, yaw]),
            grasp=int(self.grasp),
            reset=self._reset_state,
        )

    def close(self):
        self._enabled = False
        if self.device:
            try:
                self.device.close()
            except Exception:
                pass
            print("[DEBUG] HID controller closed")

    def _apply_deadzone(self, value):
        """Apply deadzone with scaled output."""
        if abs(value) < self.DEADZONE:
            return 0.0
        sign = 1 if value > 0 else -1
        return sign * (abs(value) - self.DEADZONE) / (1.0 - self.DEADZONE)