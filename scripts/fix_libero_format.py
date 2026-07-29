"""
Fix a custom LIBERO HDF5 file to match libero_90 format.

Usage: python fix_libero_format.py <path_to_hdf5>

Changes made:
  1. Add missing 'tag' = 'libero-v1' attribute to the data/ group
  2. Remove extra 'bddl_file_content' attribute from data/ group (not in reference)
"""

import h5py
import shutil
import os
import sys

if len(sys.argv) < 2:
    print("Usage: python fix_libero_format.py <path_to_hdf5>")
    sys.exit(1)

SRC = sys.argv[1]
BACKUP = SRC + '.bak'

if not os.path.exists(SRC):
    print(f"ERROR: File not found: {SRC}")
    sys.exit(1)

print(f"Processing: {SRC}")

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

    # Report final state
    print(f"\nFinal data/ attrs: {sorted(data.attrs.keys())}")
    print(f"Num demos: {data.attrs.get('num_demos', '?')}")
    print(f"Total steps: {data.attrs.get('total', '?')}")

    # Verify demo structure
    demos = sorted(data.keys())
    demo0 = demos[0]
    print(f"\nDemo count: {len(demos)}")
    print(f"Demo keys: {sorted(data[demo0].keys())}")
    print(f"Obs keys:  {sorted(data[f'{demo0}/obs'].keys())}")
    print(f"States dim: {data[f'{demo0}/states'].shape[1]}")
    print(f"Actions dim: {data[f'{demo0}/actions'].shape[1]}")

print("\nDone.")
