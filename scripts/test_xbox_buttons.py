"""Press buttons on the Xbox controller to see which bytes change."""
import hid
import time

VID = 1118
PID = 765

controller = hid.device()
controller.open(VID, PID)
controller.set_nonblocking(True)
print("Controller opened. Press buttons one at a time...")
print("Press Ctrl+C to stop.\n")

prev_data = None

try:
    while True:
        data = controller.read(64)
        if data and len(data) >= 16 and data[0] == 0x01:
            # Only print when button/dpad bytes change (ignore stick drift)
            # Compare bytes 13-16 (dpad + buttons area)
            button_bytes = bytes(data[13:17]) if len(data) >= 17 else bytes(data[13:])
            
            if prev_data is None or button_bytes != prev_data:
                hex_all = ' '.join(f'{b:02x}' for b in data)
                # Highlight the button area
                b13 = data[13] if len(data) > 13 else 0
                b14 = data[14] if len(data) > 14 else 0
                b15 = data[15] if len(data) > 15 else 0
                b16 = data[16] if len(data) > 16 else 0
                
                print(f"byte13={b13:08b} (0x{b13:02x})  "
                      f"byte14={b14:08b} (0x{b14:02x})  "
                      f"byte15={b15:08b} (0x{b15:02x})  "
                      f"byte16={b16:08b} (0x{b16:02x})")
                prev_data = button_bytes
        time.sleep(0.005)
except KeyboardInterrupt:
    print("\nDone.")
finally:
    controller.close()
