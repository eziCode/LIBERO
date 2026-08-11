#!/usr/bin/env python3
"""Inspect a ManiSkill-to-MuJoCo / robomimic HDF5 dataset."""

import argparse
from pathlib import Path

import h5py
import numpy as np


def format_attrs(obj, max_length=240):
    if not obj.attrs:
        return "none"
    formatted = []
    for key, value in obj.attrs.items():
        rendered = repr(value)
        if len(rendered) > max_length:
            rendered = rendered[:max_length] + f"... <{len(rendered)} chars>"
        formatted.append(f"{key}={rendered}")
    return ", ".join(formatted)


def dataset_rows(dataset):
    return dataset.shape[0] if dataset.ndim else None


def inspect_file(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    print(f"File: {path}")
    print(f"Size: {path.stat().st_size / (1024 ** 2):.1f} MiB")

    with h5py.File(path, "r") as handle:
        print(f"Root attributes: {format_attrs(handle)}")
        if "data" not in handle:
            print("No /data group found. Root contents:")
            handle.visititems(
                lambda name, obj: print(
                    f"  /{name}: shape={obj.shape}, dtype={obj.dtype}"
                    if isinstance(obj, h5py.Dataset)
                    else f"  /{name}/"
                )
            )
            return

        data = handle["data"]
        demo_names = sorted(
            (name for name, obj in data.items() if isinstance(obj, h5py.Group)),
            key=lambda name: (
                0,
                int(name.split("_")[-1]),
            )
            if name.split("_")[-1].isdigit()
            else (1, name),
        )
        total = data.attrs.get("total")
        print(f"Data attributes: {format_attrs(data)}")
        print(f"Demos: {len(demo_names)}" + (f"; total transitions: {total}" if total is not None else ""))
        if not demo_names:
            return

        demo = data[demo_names[0]]
        print(f"\nFirst demo: /data/{demo_names[0]}")
        print(f"Attributes: {format_attrs(demo)}")

        datasets = {}

        def collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[name] = obj
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")

        demo.visititems(collect)

        expected_rows = dataset_rows(datasets["actions"]) if "actions" in datasets else None
        if expected_rows is not None:
            mismatches = []
            for name, dataset in datasets.items():
                rows = dataset_rows(dataset)
                if rows is not None and rows != expected_rows:
                    mismatches.append(f"{name}={rows}")
            print(f"\nTransition alignment: actions={expected_rows}", end="")
            print("; OK" if not mismatches else "; mismatches: " + ", ".join(mismatches))

        print("\nForce/torque fields:")
        wrench_names = [
            "obs/robot0_eef_force",
            "obs/robot0_eef_torque",
            "next_obs/robot0_eef_force",
            "next_obs/robot0_eef_torque",
        ]
        found = False
        for name in wrench_names:
            if name not in datasets:
                continue
            found = True
            values = datasets[name][...]
            norms = np.linalg.norm(values, axis=-1)
            print(
                f"  {name}: shape={values.shape}, finite={bool(np.isfinite(values).all())}, "
                f"norm min/mean/max={norms.min():.6g}/{norms.mean():.6g}/{norms.max():.6g}"
            )
        if not found:
            print("  none found")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="HDF5 file(s) to inspect")
    args = parser.parse_args()

    for index, path in enumerate(args.files):
        if index:
            print("\n" + "=" * 80 + "\n")
        inspect_file(path)


if __name__ == "__main__":
    main()
