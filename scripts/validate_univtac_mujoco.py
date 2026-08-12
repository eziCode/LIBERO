#!/usr/bin/env python3
"""Validate a converted UniVTAC MuJoCo / robomimic HDF5 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import robosuite as suite

import univtac_mujoco_envs  # noqa: F401


def validate_demo(name: str, demo: h5py.Group) -> list[str]:
    errors: list[str] = []
    required = ("states", "actions", "rewards", "dones", "obs", "next_obs")
    for key in required:
        if key not in demo:
            errors.append(f"{name}: missing {key}")
    if errors:
        return errors
    length = len(demo["actions"])
    for key in ("states", "rewards", "dones"):
        if len(demo[key]) != length:
            errors.append(f"{name}: {key} has {len(demo[key])}, expected {length}")
    if demo["actions"].shape[1:] != (8,):
        errors.append(f"{name}: actions must have shape (T, 8)")
    for group_name in ("obs", "next_obs"):
        for key, dataset in demo[group_name].items():
            if len(dataset) != length:
                errors.append(f"{name}: {group_name}/{key} length mismatch")
            if dataset.dtype.kind in "fc" and not np.isfinite(dataset[:]).all():
                errors.append(f"{name}: {group_name}/{key} contains non-finite values")
    common = set(demo["obs"]) & set(demo["next_obs"])
    for key in common:
        if length > 1 and not np.array_equal(demo["obs"][key][1:], demo["next_obs"][key][:-1]):
            errors.append(f"{name}: obs/next_obs continuity failed for {key}")
    if not bool(demo["dones"][-1]):
        errors.append(f"{name}: final transition is not terminal")
    if np.any(np.asarray(demo["actions"]) < -1.00001) or np.any(np.asarray(demo["actions"]) > 1.00001):
        errors.append(f"{name}: action outside [-1, 1]")
    return errors


def restore_check(data: h5py.Group) -> list[str]:
    meta = json.loads(data.attrs["env_args"])
    kwargs = dict(meta["env_kwargs"])
    kwargs.update(has_renderer=False, has_offscreen_renderer=False, hard_reset=False)
    env = suite.make(meta["env_name"], **kwargs)
    errors: list[str] = []
    try:
        demo = data[sorted(data, key=lambda key: int(key.split("_")[-1]))[0]]
        env.reset_from_xml_string(demo.attrs["model_file"])
        env.sim.set_state_from_flattened(np.asarray(demo["states"][0]))
        env.sim.forward()
        for actor in ("prism", "slot"):
            expected = np.asarray(demo[f"obs/{actor}_pos"][0])
            actual = env.actor_pose(actor)[:3]
            if not np.allclose(actual, expected, atol=1e-6):
                errors.append(f"state restore {actor} mismatch: {actual} != {expected}")
    finally:
        env.close()
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    errors: list[str] = []
    with h5py.File(args.dataset, "r") as file:
        if "data" not in file:
            raise SystemExit("missing /data group")
        data = file["data"]
        demos = sorted(data, key=lambda key: int(key.split("_")[-1]))
        for name in demos:
            errors.extend(validate_demo(name, data[name]))
        errors.extend(restore_check(data))
        total = sum(len(data[name]["actions"]) for name in demos)
        if int(data.attrs.get("total", -1)) != total:
            errors.append("/data total attribute does not match transitions")
    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print(f"Validated {len(demos)} demos / {total} transitions in {args.dataset}")


if __name__ == "__main__":
    main()
