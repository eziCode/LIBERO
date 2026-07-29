# xbox_controller_bluetooth.py
"""
Xbox One S / Series controller wrapper for Bluetooth HID (Report ID 0x01).

Bluetooth HID Report Format (17 bytes, Report ID 0x01):
    data[0]      = 0x01 report ID
    data[1:3]    = Left Stick X  (uint16 LE, 0-65535, center ~32768)
    data[3:5]    = Left Stick Y  (uint16 LE, 0-65535, center ~32768)
    data[5:7]    = Right Stick X (uint16 LE, 0-65535, center ~32768)
    data[7:9]    = Right Stick Y (uint16 LE, 0-65535, center ~32768)
    data[9:11]   = Left Trigger  (uint16 LE, 0-1023)
    data[11:13]  = Right Trigger (uint16 LE, 0-1023)
    data[13]     = D-pad hat (0=none, 1=N, 2=NE, 3=E, 4=SE, 5=S, 6=SW, 7=W, 8=NW)
    data[14]     = Buttons byte 1:
                     bit 0 = A
                     bit 1 = B
                     bit 3 = X
                     bit 4 = Y
                     bit 6 = LB
                     bit 7 = RB
    data[15]     = Buttons byte 2:
                     bit 0 = View (Back/Select)
                     bit 1 = Menu (Start)
                     bit 2 = Left Stick click (L3)
                     bit 3 = Right Stick click (R3)
    data[16]     = unused
"""
import hid
import numpy as np
import struct
import threading
import time

from robosuite.devices import Device
from robosuite.utils.transform_utils import rotation_matrix


class XboxControllerBluetooth(Device):
    """
    Xbox controller wrapper for Bluetooth HID connections.
    Uses the SpaceMouse pattern: joystick position = velocity command.
    Holding the stick at full deflection = constant speed.
    """

    DEADZONE = 0.15
    AXIS_CENTER = 32768.0
    AXIS_MAX = 32768.0  # distance from center to edge
    TRIGGER_MAX = 1023.0

    def __init__(self, pos_sensitivity=1.0, rot_sensitivity=1.0,
                 vendor_id=1118, product_id=2835,
                 accel_max=4.0, accel_ramp_time=0.1):
        self.vid = vendor_id
        self.pid = product_id
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.device = None
        self._enabled = False
        self._reset_state = 0

        # Acceleration: speed ramps from 1x to accel_max over accel_ramp_time seconds
        self.accel_max = accel_max          # max speed multiplier at full hold
        self.accel_ramp_time = accel_ramp_time  # seconds to reach max multiplier

        # Current control values (updated by background thread)
        self._control = np.zeros(6)
        self._deflect_start_time = None  # when stick was first deflected
        self.grasp = False
        self._last_b_state = 0
        self.rotation = np.array([[-1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, -1.0]])

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
        self._deflect_start_time = None
        self.grasp = False
        self._last_b_state = 0
        self._reset_state = 0
        self.rotation = np.array([[-1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, -1.0]])
        self._enabled = True

        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

        print(f"[DEBUG] Bluetooth Xbox controller opened (VID={self.vid}, PID={self.pid})")
        print(f"[DEBUG] Acceleration: {self.accel_max}x max over {self.accel_ramp_time}s ramp")

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

                if latest_data and len(latest_data) >= 16 and latest_data[0] == 0x01:
                    self._process_bt_report(latest_data)
            except (ValueError, OSError):
                break  # device was closed, exit thread

            time.sleep(0.001)

    def _process_bt_report(self, data):
        """Parse Bluetooth HID report and update current control values."""
        # --- Joystick axes (unsigned 16-bit LE, center ~32768) ---
        lx = self._apply_deadzone(self._read_unsigned_axis(data, 1))
        ly = self._apply_deadzone(self._read_unsigned_axis(data, 3))
        rx = self._apply_deadzone(self._read_unsigned_axis(data, 5))
        ry = self._apply_deadzone(self._read_unsigned_axis(data, 7))

        # --- Triggers (unsigned 16-bit LE, 0-1023) ---
        lt = struct.unpack_from('<H', bytes(data), 9)[0] / self.TRIGGER_MAX
        rt = struct.unpack_from('<H', bytes(data), 11)[0] / self.TRIGGER_MAX

        # --- Buttons (byte 14) ---
        btn1 = data[14] if len(data) > 14 else 0
        a_btn = (btn1 >> 0) & 1
        b_btn = (btn1 >> 1) & 1
        y_btn = (btn1 >> 4) & 1

        # --- Buttons byte 2 (byte 16) ---
        btn2 = data[16] if len(data) > 16 else 0
        back_btn = (btn2 >> 0) & 1

        # Z axis from Y/A buttons
        dz = float(y_btn - a_btn)

        # Track deflection time for acceleration
        any_stick_active = (lx != 0 or ly != 0 or rx != 0 or ry != 0 or dz != 0)
        if any_stick_active:
            if self._deflect_start_time is None:
                self._deflect_start_time = time.time()
        else:
            self._deflect_start_time = None

        # Store current control state
        # LY→X, LX→Y, buttons→Z, triggers→roll, RY→pitch, RX→yaw
        self._control = np.array([
            ly,        # X (forward/back)
            lx,        # Y (left/right)
            dz,        # Z (up/down from buttons)
            rt - lt,   # roll: triggers → rotation around X axis
            -ry,       # pitch: right stick U/D → rotation around Y axis (inverted)
            -rx,       # yaw: right stick L/R → rotation around Z axis (inverted)
        ])

        # Grasp: B button toggles
        if b_btn and not self._last_b_state:
            self.grasp = not self.grasp
        self._last_b_state = b_btn

        # Reset
        if back_btn:
            self._reset_state = 1

    def _read_unsigned_axis(self, data, offset):
        """Read unsigned 16-bit LE, convert to -1.0..1.0 centered at 32768."""
        raw = struct.unpack_from('<H', bytes(data), offset)[0]
        centered = raw - self.AXIS_CENTER
        return max(-1.0, min(1.0, centered / self.AXIS_MAX))

    def _get_accel_multiplier(self):
        """Compute acceleration multiplier based on how long the stick has been held.
        Ramps linearly from 1.0 to accel_max over accel_ramp_time seconds."""
        if self._deflect_start_time is None:
            return 1.0
        elapsed = time.time() - self._deflect_start_time
        t = min(1.0, elapsed / self.accel_ramp_time)
        return 1.0 + (self.accel_max - 1.0) * t

    def get_controller_state(self):
        """
        Returns current controller state (SpaceMouse pattern).
        Joystick position = velocity command. Holding the stick ramps up speed
        from base to accel_max over accel_ramp_time seconds.
        """
        accel = self._get_accel_multiplier()

        # Scale by sensitivity and acceleration
        dpos = self._control[:3] * 0.005 * self.pos_sensitivity * accel

        roll, pitch, yaw = self._control[3:] * 0.005 * self.rot_sensitivity * accel

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
            print("[DEBUG] Bluetooth Xbox controller closed")

    def _apply_deadzone(self, value):
        """Apply deadzone with scaled output."""
        if abs(value) < self.DEADZONE:
            return 0.0
        sign = 1 if value > 0 else -1
        return sign * (abs(value) - self.DEADZONE) / (1.0 - self.DEADZONE)

