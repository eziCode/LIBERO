#!/usr/bin/env python3
"""Rebuild MimicGen force/torque files as complete enriched datasets.

For every matching HDF5 filename, this script copies the entire original
MimicGen file and appends the reconstructed end-effector force and torque
datasets from the compact sidecar. The destination is replaced atomically only
after structural validation succeeds.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import h5py
import numpy as np


FORCE_KEY = "robot0_eef_force"
TORQUE_KEY = "robot0_eef_torque"
GRIPPER_CONTACT_KEY = "gripper_contact_force"
LEFT_GRIPPER_KEY = "left_gripper_force"
RIGHT_GRIPPER_KEY = "right_gripper_force"
WRENCH_KEYS = (
    FORCE_KEY,
    TORQUE_KEY,
    GRIPPER_CONTACT_KEY,
    LEFT_GRIPPER_KEY,
    RIGHT_GRIPPER_KEY,
)


def attributes_equal(left: h5py.AttributeManager, right: h5py.AttributeManager) -> bool:
    if set(left.keys()) != set(right.keys()):
        return False
    for key in left.keys():
        a = np.asarray(left[key])
        b = np.asarray(right[key])
        if a.shape != b.shape or not np.array_equal(a, b):
            return False
    return True


def original_structure(file: h5py.File) -> dict[str, tuple[tuple[int, ...], str]]:
    datasets: dict[str, tuple[tuple[int, ...], str]] = {}

    def collect(name: str, obj) -> None:
        if isinstance(obj, h5py.Dataset):
            datasets[name] = (obj.shape, obj.dtype.str)

    file.visititems(collect)
    return datasets


def validate_enriched(original_path: Path, enriched_path: Path) -> None:
    with h5py.File(original_path, "r") as original, h5py.File(enriched_path, "r") as enriched:
        expected = original_structure(original)
        actual = original_structure(enriched)
        missing = sorted(set(expected) - set(actual))
        changed = sorted(
            path for path, signature in expected.items() if actual.get(path) != signature
        )
        if missing or changed:
            raise RuntimeError(
                f"validation failed for {enriched_path.name}: "
                f"missing={missing[:5]}, changed={changed[:5]}"
            )
        if not attributes_equal(original.attrs, enriched.attrs):
            raise RuntimeError(f"root attributes changed in {enriched_path.name}")
        if not attributes_equal(original["data"].attrs, enriched["data"].attrs):
            raise RuntimeError(f"data attributes changed in {enriched_path.name}")

        original_demos = set(original["data"].keys())
        enriched_demos = set(enriched["data"].keys())
        if original_demos != enriched_demos:
            raise RuntimeError(f"demo keys differ in {enriched_path.name}")

        for demo_name in original_demos:
            original_demo = original[f"data/{demo_name}"]
            enriched_demo = enriched[f"data/{demo_name}"]
            if not attributes_equal(original_demo.attrs, enriched_demo.attrs):
                raise RuntimeError(
                    f"original demo attributes changed: {enriched_path.name}/{demo_name}"
                )
            samples = len(original_demo["actions"])
            obs = enriched_demo["obs"]
            for key in WRENCH_KEYS:
                expected_shape = (samples, 3) if key in (FORCE_KEY, TORQUE_KEY) else (samples, 1)
                if key not in obs or obs[key].shape != expected_shape:
                    raise RuntimeError(
                        f"invalid {key} in {enriched_path.name}/{demo_name}: "
                        f"{obs[key].shape if key in obs else 'missing'}"
                    )
                if not np.isfinite(obs[key][()]).all():
                    raise RuntimeError(
                        f"non-finite {key} values in {enriched_path.name}/{demo_name}"
                    )


def enrich_file(original_path: Path, sidecar_path: Path) -> None:
    temporary_path = sidecar_path.with_name(f".{sidecar_path.name}.enriching.tmp")
    if temporary_path.exists():
        temporary_path.unlink()

    print(f"Copying complete original: {original_path} -> {temporary_path}")
    shutil.copy2(original_path, temporary_path)

    try:
        with h5py.File(sidecar_path, "r") as sidecar, h5py.File(temporary_path, "a") as enriched:
            original_demos = set(enriched["data"].keys())
            sidecar_demos = set(sidecar["data"].keys())
            if original_demos != sidecar_demos:
                raise RuntimeError(
                    f"demo mismatch for {sidecar_path.name}: "
                    f"original={len(original_demos)}, force={len(sidecar_demos)}"
                )

            metadata = enriched.require_group("force_torque_metadata")
            for key, value in sidecar.attrs.items():
                metadata.attrs[key] = value
            metadata.attrs["storage"] = "data/<demo>/obs/{" + ",".join(WRENCH_KEYS) + "}"

            for index, demo_name in enumerate(sorted(original_demos), start=1):
                destination_obs = enriched[f"data/{demo_name}/obs"]
                source_obs = sidecar[f"data/{demo_name}/obs"]
                for key in WRENCH_KEYS:
                    if key in destination_obs:
                        del destination_obs[key]
                    sidecar.copy(source_obs[key], destination_obs, name=key)
                if index % 100 == 0 or index == len(original_demos):
                    print(f"  appended {index}/{len(original_demos)} demos")
            enriched.flush()

        print(f"Validating {temporary_path} ...")
        validate_enriched(original_path, temporary_path)
        os.replace(temporary_path, sidecar_path)
        print(f"Replaced with complete enriched dataset: {sidecar_path}")
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-dir",
        type=Path,
        default=Path("datasets/mimicgen"),
        help="Directory containing complete original MimicGen HDF5 files",
    )
    parser.add_argument(
        "--force-dir",
        type=Path,
        default=Path("datasets/mimicgen-force-torque"),
        help="Directory containing compact force/torque HDF5 sidecars",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sidecars = sorted(args.force_dir.glob("*.hdf5"))
    if not sidecars:
        raise FileNotFoundError(f"no HDF5 files found in {args.force_dir}")

    for sidecar_path in sidecars:
        original_path = args.original_dir / sidecar_path.name
        if not original_path.is_file():
            raise FileNotFoundError(original_path)
        enrich_file(original_path.resolve(), sidecar_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
