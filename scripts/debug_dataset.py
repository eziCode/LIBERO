import h5py
import numpy as np
import json

def debug_dataset(path):
    print(f"Debugging dataset: {path}")
    with h5py.File(path, "r") as f:
        print("Global Attributes:")
        for attr in f["data"].attrs:
            val = f["data"].attrs[attr]
            if isinstance(val, str) and len(val) > 200:
                print(f"  {attr}: [Long String]")
            else:
                print(f"  {attr}: {val}")
        
        env_args = json.loads(f["data"].attrs["env_args"])
        print("\nEnv Args (env_kwargs):")
        for k, v in env_args["env_kwargs"].items():
            print(f"  {k}: {v}")
            
        demo = list(f["data"].keys())[0]
        print(f"\nStats for {demo}:")
        actions = f[f"data/{demo}/actions"][()]
        print(f"  Action shape: {actions.shape}")
        print(f"  Action range: [{np.min(actions):.4f}, {np.max(actions):.4f}]")
        print(f"  Action mean: {np.mean(actions):.4f}")
        
        obs_keys = list(f[f"data/{demo}/obs"].keys())
        print(f"\nObservation keys: {obs_keys}")
        for k in obs_keys:
            data = f[f"data/{demo}/obs/{k}"][()]
            print(f"  {k}: shape {data.shape}, dtype {data.dtype}, range [{np.min(data)}, {np.max(data)}]")

if __name__ == "__main__":
    debug_dataset("datasets/custom/shake_cup_demo.hdf5")
