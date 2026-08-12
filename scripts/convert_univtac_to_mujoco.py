#!/usr/bin/env python3
"""Convert UniVTAC insert_HDMI episodes to robomimic-compatible MuJoCo HDF5."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import h5py
import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config

import univtac_mujoco_envs  # noqa: F401


ENV_NAME = "UniVTACMujocoInsertHDMI"
CONTROL_FREQ = 60  # Isaac Lab 120 Hz, source save_frequency=2.


def make_env():
    return suite.make(
        ENV_NAME,
        robots="Panda",
        controller_configs=load_controller_config(default_controller="JOINT_POSITION"),
        gripper_types="PandaGripper",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=False,
        control_freq=CONTROL_FREQ,
        hard_reset=False,
        ignore_done=True,
    )


def decode_rgb(value) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("invalid encoded RGB frame")
    return image[..., ::-1]


def normalized_joint_target(controller, target: np.ndarray, current: np.ndarray) -> np.ndarray:
    delta = np.asarray(target) - np.asarray(current)
    positive = delta / np.maximum(np.asarray(controller.output_max), 1e-12)
    negative = delta / np.maximum(np.abs(np.asarray(controller.output_min)), 1e-12)
    return np.clip(np.where(delta >= 0, positive, negative), -1, 1)


def set_frame(env, joints: np.ndarray, prism: np.ndarray, slot: np.ndarray, qvel: np.ndarray) -> None:
    robot = env.robots[0]
    env.sim.data.qpos[robot._ref_joint_pos_indexes] = joints[:7]
    env.sim.data.qvel[robot._ref_joint_vel_indexes] = qvel[:7]
    # UniVTAC / Isaac uses positive displacement for both fingers; MuJoCo's
    # second Panda finger has the opposite signed joint range.
    env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes] = (joints[7], -joints[8])
    env.sim.data.qvel[robot._ref_gripper_joint_vel_indexes] = (qvel[7], -qvel[8])
    env.set_actor_pose("prism", prism)
    env.set_actor_pose("slot", slot)
    env.sim.forward()


def numeric_obs(env) -> dict[str, np.ndarray]:
    result = {}
    for key, value in env._get_observations(force_update=True).items():
        array = np.asarray(value)
        if array.dtype.kind in "biuf" and array.ndim:
            result[key] = array.astype(np.float32, copy=False)
    result["robot0_eef_force"] = np.asarray(env.robots[0].ee_force, dtype=np.float32)
    result["robot0_eef_torque"] = np.asarray(env.robots[0].ee_torque, dtype=np.float32)
    return result


def source_obs(source: h5py.File, index: int) -> dict[str, np.ndarray]:
    return {
        "agentview_image": decode_rgb(source["observation/head/rgb"][index]),
        "robot0_eye_in_hand_image": decode_rgb(source["observation/wrist/rgb"][index]),
        "tactile_left_image": decode_rgb(source["tactile/left_gsmini/rgb"][index]),
        "tactile_right_image": decode_rgb(source["tactile/right_gsmini/rgb"][index]),
        "tactile_left_marker_image": decode_rgb(source["tactile/left_gsmini/rgb_marker"][index]),
        "tactile_right_marker_image": decode_rgb(source["tactile/right_gsmini/rgb_marker"][index]),
        "tactile_left_depth": np.asarray(source["tactile/left_gsmini/depth"][index], dtype=np.float32),
        "tactile_right_depth": np.asarray(source["tactile/right_gsmini/depth"][index], dtype=np.float32),
        "tactile_left_marker": np.asarray(source["tactile/left_gsmini/marker"][index], dtype=np.float32),
        "tactile_right_marker": np.asarray(source["tactile/right_gsmini/marker"][index], dtype=np.float32),
        "tactile_left_pose": np.asarray(source["tactile/left_gsmini/pose"][index], dtype=np.float32),
        "tactile_right_pose": np.asarray(source["tactile/right_gsmini/pose"][index], dtype=np.float32),
        "source_ee_pose": np.asarray(source["embodiment/ee"][index], dtype=np.float32),
    }


def merge_obs(env_obs: dict[str, np.ndarray], source: h5py.File, index: int):
    return {**env_obs, **source_obs(source, index)}


def write_obs(parent: h5py.Group, name: str, observations: list[dict[str, np.ndarray]]) -> None:
    group = parent.create_group(name)
    for key in sorted(observations[0]):
        values = np.stack([item[key] for item in observations])
        group.create_dataset(key, data=values, compression="gzip", compression_opts=4)


def model_xml_with_slot_pose(env, pose: np.ndarray) -> str:
    """Serialize the fixed socket at this episode's randomized source pose."""
    root = ET.fromstring(env.model.get_xml())
    body = root.find(".//body[@name='slot_main']")
    if body is None:
        raise ValueError("slot_main body missing from MuJoCo model")
    body.set("pos", " ".join(f"{value:.17g}" for value in pose[:3]))
    body.set("quat", " ".join(f"{value:.17g}" for value in pose[3:]))
    return ET.tostring(root, encoding="unicode")


def convert_episode(env, path: Path, demo: h5py.Group, metadata: dict) -> int:
    with h5py.File(path, "r") as source:
        joints = np.asarray(source["embodiment/joint"], dtype=np.float64)
        prism = np.asarray(source["actor/prism"], dtype=np.float64)
        slot = np.asarray(source["actor/slot"], dtype=np.float64)
        frames = len(joints)
        if frames < 2 or len(prism) != frames or len(slot) != frames:
            raise ValueError(f"unaligned source episode {path}")
        qvel = np.gradient(joints, 1 / CONTROL_FREQ, axis=0)
        env.reset()
        states, actions, obs, next_obs = [], [], [], []
        rewards = np.zeros(frames - 1, dtype=np.float32)
        dones = np.zeros(frames - 1, dtype=np.bool_)
        for index in range(frames):
            set_frame(env, joints[index], prism[index], slot[index], qvel[index])
            current = merge_obs(numeric_obs(env), source, index)
            if index < frames - 1:
                states.append(np.asarray(env.sim.get_state().flatten(), dtype=np.float64))
                obs.append(current)
                arm = normalized_joint_target(
                    env.robots[0].controller, joints[index + 1, :7], joints[index, :7]
                )
                gripper = 1.0 if np.mean(joints[index + 1, 7:9]) <= 0.02 else -1.0
                actions.append(np.r_[arm, gripper].astype(np.float32))
            if index > 0:
                next_obs.append(current)

        success = metadata.get("result") == "success"
        rewards[-1] = float(success)
        dones[-1] = True
        demo.create_dataset("states", data=np.asarray(states), compression="gzip")
        demo.create_dataset("actions", data=np.asarray(actions), compression="gzip")
        demo.create_dataset("source_joint_targets", data=joints[1:].astype(np.float32), compression="gzip")
        demo.create_dataset("rewards", data=rewards)
        demo.create_dataset("dones", data=dones)
        write_obs(demo, "obs", obs)
        write_obs(demo, "next_obs", next_obs)
        demo.attrs["num_samples"] = frames - 1
        demo.attrs["source_file"] = str(path)
        demo.attrs["source_metadata"] = json.dumps(metadata, sort_keys=True)
        demo.attrs["model_file"] = model_xml_with_slot_pose(env, slot[0])
        return frames - 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("datasets/UniVTAC/insert_HDMI/clean"))
    parser.add_argument("--output", type=Path, default=Path("datasets/UniVTAC/mujoco/insert_HDMI.hdf5"))
    parser.add_argument("--count", type=int)
    args = parser.parse_args()
    files = sorted(args.input_dir.glob("*.hdf5"), key=lambda path: int(path.stem))
    if args.count is not None:
        files = files[: args.count]
    metadata = json.loads((args.input_dir / "metadata.json").read_text())
    env = make_env()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    total = 0
    try:
        with h5py.File(temporary, "w") as output:
            data = output.create_group("data")
            env_args = {
                "env_name": ENV_NAME,
                "type": 1,
                "env_kwargs": {
                    "robots": ["Panda"],
                    "controller_configs": load_controller_config(default_controller="JOINT_POSITION"),
                    "gripper_types": "PandaGripper",
                    "control_freq": CONTROL_FREQ,
                    "use_camera_obs": False,
                    "use_object_obs": True,
                },
            }
            data.attrs["env_args"] = json.dumps(env_args, sort_keys=True)
            data.attrs["env_meta"] = data.attrs["env_args"]
            data.attrs["source_dataset"] = "UniVTAC"
            data.attrs["source_task"] = "insert_HDMI"
            data.attrs["conversion_mode"] = "recorded_states_mapped_to_mujoco"
            data.attrs["tactile_provenance"] = "original UniVTAC TacEx GelSight Mini"
            for index, path in enumerate(files):
                demo = data.create_group(f"demo_{index}")
                count = convert_episode(env, path, demo, metadata.get(path.stem, {}))
                total += count
                print(f"{path.name} -> demo_{index}: {count} transitions")
            data.attrs["total"] = total
            data.attrs["num_demos"] = len(files)
        os.replace(temporary, args.output)
    finally:
        env.close()
        if temporary.exists():
            temporary.unlink()
    print(f"Wrote {len(files)} demos / {total} transitions to {args.output}")


if __name__ == "__main__":
    main()
