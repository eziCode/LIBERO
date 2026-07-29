"""
Fix shake_cup_demo.hdf5 to match libero_90 format.

Changes made:
  1. Add missing 'tag' = 'libero-v1' attribute to the data/ group
  2. Remove extra 'bddl_file_content' attribute from data/ group (not present in reference)

Everything else (obs keys, demo keys, action dims, etc.) already matches.
The 'states' dimension (32) is correct for the shake_cup scene — it varies
across all libero_90 tasks (45–123) depending on the number of scene objects.
"""

import h5py
import shutil
import os

SRC = '/Users/ezraakresh/Documents/LIBERO/datasets/custom/shake_cup_demo.hdf5'
BACKUP = '/Users/ezraakresh/Documents/LIBERO/datasets/custom/shake_cup_demo.hdf5.bak'

# Make a backup first
if not os.path.exists(BACKUP):
    shutil.copy2(SRC, BACKUP)
    print(f"Backup created: {BACKUP}")
else:
    print(f"Backup already exists: {BACKUP}")

with h5py.File(SRC, 'r+') as f:
    data = f['data']

    # 1. Add missing 'tag' attribute
    if 'tag' not in data.attrs:
        data.attrs['tag'] = 'libero-v1'
        print("✓ Added: data.attrs['tag'] = 'libero-v1'")
    else:
        print(f"  'tag' already present: {data.attrs['tag']}")

    # 2. Remove extra 'bddl_file_content' attribute
    if 'bddl_file_content' in data.attrs:
        del data.attrs['bddl_file_content']
        print("✓ Removed: data.attrs['bddl_file_content']")
    else:
        print("  'bddl_file_content' not present, nothing to remove")

    # Verify final state
    print("\nFinal data/ attrs:", sorted(data.attrs.keys()))

print("\nDone. File updated in-place.")
