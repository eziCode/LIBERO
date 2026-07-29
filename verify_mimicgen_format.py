import os
import h5py
from libero.libero import get_libero_path

def verify_dataset(filepath):
    print(f"\nVerifying: {filepath}")
    if not os.path.exists(filepath):
        print(f"  [ERROR] File not found!")
        return
        
    f = h5py.File(filepath, 'r')
    
    if "data/demo_0/obs" not in f:
        print("  [ERROR] 'obs' group is MISSING! Not in MimicGen format.")
        return
        
    obs = f["data/demo_0/obs"]
    keys = list(obs.keys())
    
    print("  [SUCCESS] 'obs' group found! Detected Observation Keys:")
    for key in keys:
        shape = obs[key].shape
        print(f"    - {key} {shape}")
        
    f.close()

if __name__ == "__main__":
    base_path = get_libero_path("datasets")
    f1 = os.path.join(base_path, "custom/shake_cup_demo.hdf5")
    f2 = os.path.join(base_path, "custom/upright_flipped_cup_demo.hdf5")
    
    verify_dataset(f1)
    verify_dataset(f2)
