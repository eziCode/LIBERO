"""Dump raw HID reports from the Xbox controller to figure out the Bluetooth report format."""
import hid
import time

VID = 1118
PID = 765

controller = hid.device()
controller.open(VID, PID)
controller.set_nonblocking(True)
print(f"Controller opened (VID={VID}, PID={PID}). Move sticks / press buttons...")
print("Press Ctrl+C to stop.\n")

seen_report_ids = set()

try:
    while True:
        data = controller.read(64)
        if data:
            report_id = data[0]
            length = len(data)
            hex_str = ' '.join(f'{b:02x}' for b in data)
            
            if report_id not in seen_report_ids:
                seen_report_ids.add(report_id)
                print(f"\n=== NEW REPORT ID: 0x{report_id:02x} (len={length}) ===")
            
            print(f"[0x{report_id:02x}] len={length:2d}: {hex_str}")
        time.sleep(0.01)
except KeyboardInterrupt:
    print("\n\nDone.")
finally:
    controller.close()
