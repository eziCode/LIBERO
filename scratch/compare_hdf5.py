import h5py
import sys

def print_hdf5_info(filepath):
    print(f"--- Info for {filepath} ---")
    try:
        with h5py.File(filepath, 'r') as f:
            print("Attributes on 'data':")
            if "data" not in f:
                print("  'data' group not found!")
                return
            
            for attr in f["data"].attrs.keys():
                try:
                    val = f["data"].attrs[attr]
                    if isinstance(val, str):
                        print(f"  {attr}: [string length {len(val)}]")
                    else:
                        print(f"  {attr}: type {type(val)}")
                except Exception as e:
                    print(f"  {attr}: Error reading value - {e}")
            
            demo_keys = [k for k in f["data"].keys() if k.startswith("demo_")]
            if not demo_keys:
                print("  No demos found.")
                return
                
            demo_0 = demo_keys[0]
            print(f"Attributes on 'data/{demo_0}':")
            for attr in f[f"data/{demo_0}"].attrs.keys():
                print(f"  {attr}: type {type(f[f'data/{demo_0}'].attrs[attr])}")
                
            print(f"Groups/Datasets in 'data/{demo_0}':")
            for k in f[f"data/{demo_0}"].keys():
                item = f[f"data/{demo_0}/{k}"]
                if isinstance(item, h5py.Group):
                    print(f"  {k} (Group)")
                    for sub_k in item.keys():
                        sub_item = item[sub_k]
                        if isinstance(sub_item, h5py.Dataset):
                            print(f"    {sub_k}: {sub_item.shape} {sub_item.dtype}")
                        else:
                            print(f"    {sub_k}: Group")
                elif isinstance(item, h5py.Dataset):
                    print(f"  {k} (Dataset): {item.shape} {item.dtype}")
                    
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print_hdf5_info(arg)
        print()
