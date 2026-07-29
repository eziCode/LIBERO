#!/bin/bash
echo "Starting MimicGen format conversion for Custom Tasks..."
echo "====================================================="

echo "1/2: Processing Shaking Bottle (50 Demos)..."
PYTHONPATH=. ./myenv/bin/python scripts/create_dataset.py --demo-file /Users/ezraakresh/Documents/LIBERO/custom_tasks/shaking_bottle_50_demos/demo.hdf5 --use-camera-obs

echo "\n2/2: Processing Upright Flipped Cup (50 Demos)..."
PYTHONPATH=. ./myenv/bin/python scripts/create_dataset.py --demo-file /Users/ezraakresh/Documents/LIBERO/custom_tasks/upright_flipped_cup_50_demos/demo.hdf5 --use-camera-obs

echo "\n====================================================="
echo "Conversion complete! Checking output..."
PYTHONPATH=. ./myenv/bin/python verify_mimicgen_format.py
