#!/usr/bin/env python3
"""Relabel successful MuJoCo joint rollouts with semantic OSC pose actions."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import h5py
import numpy as np
from robosuite.utils import transform_utils as T

from retarget_maniskill_to_mujoco import TASK_MAPPINGS, make_environment, output_env_metadata


def osc_actions(demo: h5py.Group) -> np.ndarray:
    """Return normalized OSC deltas that target each achieved next TCP pose."""
    current_pos = np.asarray(demo["obs/robot0_eef_pos"], dtype=np.float64)
    next_pos = np.asarray(demo["next_obs/robot0_eef_pos"], dtype=np.float64)
    current_quat = np.asarray(demo["obs/robot0_eef_quat"], dtype=np.float64)
    next_quat = np.asarray(demo["next_obs/robot0_eef_quat"], dtype=np.float64)
    joint_actions = np.asarray(demo["actions"], dtype=np.float32)
    translation = np.clip((next_pos - current_pos) / 0.05, -1.0, 1.0)
    rotation = np.stack([
        T.get_orientation_error(target, current) / 0.5
        for target, current in zip(next_quat, current_quat)
    ])
    rotation = np.clip(rotation, -1.0, 1.0)
    return np.c_[translation, rotation, joint_actions[:, -1]].astype(np.float32)


def convert(input_path: Path, output_path: Path) -> None:
    input_path, output_path = input_path.resolve(), output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    with h5py.File(input_path, "r") as source, h5py.File(temporary, "w") as destination:
        source.copy("data", destination)
        data = destination["data"]
        task_id = str(data.attrs["source_task"])
        mapping = TASK_MAPPINGS[task_id]
        env = make_environment(mapping.env_name, 20, controller="OSC_POSE")
        try:
            data.attrs["env_args"] = output_env_metadata(env, mapping.env_name, 20)
            data.attrs["env_meta"] = data.attrs["env_args"]
        finally:
            env.close()
        data.attrs["controller"] = "OSC_POSE"
        data.attrs["conversion_mode"] = "kinematic-osc-inverse-action-relabel"
        data.attrs["action_space"] = "normalized OSC_POSE delta (xyz+axis-angle) + gripper"
        data.attrs["action_dimensions"] = 7
        data.attrs["transition_dynamics_controller"] = "JOINT_POSITION"
        data.attrs["action_relabeling"] = (
            "pre_to_actual_post_tcp_pose_delta_under_OSC_scaling; gripper preserved"
        )
        data.attrs["joint_reference_file"] = str(input_path)
        for key in sorted(k for k in data if k.startswith("demo_")):
            demo = data[key]
            actions = osc_actions(demo)
            del demo["actions"]
            demo.create_dataset("actions", data=actions, compression="gzip")
            demo.attrs["action_relabeling"] = "kinematic_inverse_OSC_POSE"
    os.replace(temporary, output_path)
    print(f"Wrote OSC-relabeled dataset to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
