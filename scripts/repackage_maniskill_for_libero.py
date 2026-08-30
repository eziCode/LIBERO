#!/usr/bin/env python3
"""Migrate converted ManiSkill MuJoCo datasets to the custom LIBERO schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import mujoco
import numpy as np
import robosuite as suite
from robosuite.utils import transform_utils as T
from robosuite.utils.binding_utils import MjRenderContextOffscreen

import maniskill_mujoco_envs  # noqa: F401
from scripts.retarget_maniskill_to_mujoco import make_environment  # noqa: E402
from scripts.validate_maniskill_osc_replay import reset_to_demo  # noqa: E402


def natural(name: str):
    stem, _, suffix = name.rpartition("_")
    return stem, int(suffix)


def make_env(data: h5py.Group):
    meta = json.loads(data.attrs["env_args"])
    return make_environment(meta["env_name"], 20, controller="OSC_POSE")


def finger_forces(env) -> tuple[float, float]:
    names = env.robots[0].gripper.contact_geoms
    geom_ids = [
        {env.sim.model.geom_name2id(name) for name in names if f"finger{side}" in name}
        for side in (1, 2)
    ]
    result = [0.0, 0.0]
    wrench = np.zeros(6)
    for index in range(env.sim.data.ncon):
        contact = env.sim.data.contact[index]
        for side, ids in enumerate(geom_ids):
            if contact.geom1 in ids or contact.geom2 in ids:
                mujoco.mj_contactForce(env.sim.model._model, env.sim.data._data, index, wrench)
                result[side] += abs(float(wrench[0]))
    return result[0], result[1]


def replace_dataset(group: h5py.Group, key: str, values: np.ndarray) -> None:
    temporary = f"__new_{key}"
    if temporary in group:
        del group[temporary]
    group.create_dataset(temporary, data=values, compression="gzip", shuffle=True)
    if key in group:
        del group[key]
    group.move(temporary, key)


def hard_link(group: h5py.Group, target: str, source: str) -> None:
    if target in group:
        return
    group[target] = group[source]


def canonical_actions(obs: h5py.Group, next_obs: h5py.Group, gripper: np.ndarray) -> np.ndarray:
    current_pos = np.asarray(obs["robot0_eef_pos"])
    next_pos = np.asarray(next_obs["robot0_eef_pos"])
    current_quat = np.asarray(obs["robot0_eef_quat"])
    next_quat = np.asarray(next_obs["robot0_eef_quat"])
    actions = np.empty((len(current_pos), 7), dtype=np.float32)
    actions[:, :3] = np.clip((next_pos - current_pos) / 0.05, -1, 1)
    for index, (q0, q1) in enumerate(zip(current_quat, next_quat)):
        error = T.get_orientation_error(q1, q0)
        actions[index, 3:6] = np.clip(error / 0.5, -1, 1)
    actions[:, 6] = gripper
    return actions


def migrate_demo(env, demo: h5py.Group) -> None:
    obs = demo["obs"]
    count = len(demo["states"])
    if "joint_states" not in obs or "left_gripper_force" not in obs:
        joint = np.empty((count, 7), np.float32)
        gripper = np.empty((count, 2), np.float32)
        ee_pos = np.empty((count, 3), np.float32)
        ee_ori = np.empty((count, 3), np.float32)
        left = np.empty((count, 1), np.float32)
        right = np.empty((count, 1), np.float32)
        for index, state in enumerate(demo["states"]):
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            robot = env.robots[0]
            joint[index] = robot._joint_positions
            gripper[index] = env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes]
            ee_pos[index] = robot._hand_pos
            ee_ori[index] = T.quat2axisangle(T.mat2quat(robot._hand_orn))
            left[index, 0], right[index, 0] = finger_forces(env)
        replace_dataset(obs, "joint_states", joint)
        replace_dataset(obs, "gripper_states", gripper)
        replace_dataset(obs, "ee_pos", ee_pos)
        replace_dataset(obs, "ee_ori", ee_ori)
        replace_dataset(obs, "ee_states", np.concatenate((ee_pos, ee_ori), axis=1))
        replace_dataset(obs, "left_gripper_force", left)
        replace_dataset(obs, "right_gripper_force", right)
        replace_dataset(demo, "robot_states", np.concatenate((joint, gripper), axis=1))
    hard_link(obs, "ee_force", "robot0_eef_force")
    hard_link(obs, "ee_torque", "robot0_eef_torque")
    hard_link(obs, "robot0_joint_pos", "joint_states")
    hard_link(obs, "object", "maniskill_mujoco_object-state")
    if "gripper_contact_force" not in obs:
        replace_dataset(
            obs, "gripper_contact_force",
            np.asarray(obs["left_gripper_force"], dtype=np.float32)
            + np.asarray(obs["right_gripper_force"], dtype=np.float32),
        )
    if "gripper_action" not in obs:
        replace_dataset(obs, "gripper_action", np.asarray(demo["actions"][:, -1:], dtype=np.float32))
    if "gripper_actions" not in demo:
        replace_dataset(demo, "gripper_actions", np.asarray(demo["actions"][:, -1:], dtype=np.float32))
    if "robot0_eef_vel_lin" not in obs:
        position = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        velocity = np.gradient(position, 0.05, axis=0) if len(position) > 1 else np.zeros_like(position)
        replace_dataset(obs, "robot0_eef_vel_lin", velocity)
    if "robot0_eef_vel_ang" not in obs:
        quaternion = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
        velocity = np.zeros((len(quaternion), 3), dtype=np.float64)
        if len(quaternion) > 1:
            velocity[:-1] = [
                T.get_orientation_error(q1, q0) * 20.0
                for q0, q1 in zip(quaternion[:-1], quaternion[1:])
            ]
            velocity[-1] = velocity[-2]
        replace_dataset(obs, "robot0_eef_vel_ang", velocity)
    if demo["actions"].shape[1] == 8:
        if "mujoco_joint_actions" not in demo:
            demo["mujoco_joint_actions"] = demo["actions"]
        old = np.asarray(demo["actions"])
        replace_dataset(demo, "actions", canonical_actions(obs, demo["next_obs"], old[:, -1]))
    demo.attrs["libero_schema_numeric_complete"] = True


def resize_nearest(images: np.ndarray, size: int) -> np.ndarray:
    source = images.shape[1]
    indices = np.linspace(0, source - 1, size).round().astype(np.int64)
    return images[:, indices][:, :, indices]


def add_images(env, demo: h5py.Group, size: int) -> None:
    obs = demo["obs"]
    required = ("agentview_rgb", "eye_in_hand_rgb", "agentview_image", "robot0_eye_in_hand_image")
    if demo.attrs.get("libero_schema_images_complete", False) and all(key in obs for key in required):
        return
    if env.sim._render_context_offscreen is None:
        env.sim.add_render_context(MjRenderContextOffscreen(env.sim, device_id=-1))
        env.sim.render(camera_name="agentview", width=size, height=size)
        context = env.sim._render_context_offscreen
        context.vopt.geomgroup[:] = 1
        context.vopt.geomgroup[0] = 0
    # Flattened MuJoCo state contains dynamic qpos / qvel, but not the poses of
    # fixed task bodies. Restore these from the recorded observation before
    # rendering so goals, holes, and receptacles match the rollout.
    for name in ("goal_site", "box_with_hole", "receptacle"):
        position_key, quaternion_key = f"{name}_pos", f"{name}_quat"
        if position_key in obs and quaternion_key in obs:
            position = np.asarray(obs[position_key][0])
            quaternion_xyzw = np.asarray(obs[quaternion_key][0])
            env.set_task_object_pose(
                name, np.r_[position, quaternion_xyzw[[3, 0, 1, 2]]]
            )
    datasets = {}
    for key in ("agentview_rgb", "eye_in_hand_rgb"):
        if key in obs:
            del obs[key]
        datasets[key] = obs.create_dataset(
            key, shape=(len(demo["states"]), size, size, 3), dtype=np.uint8,
            chunks=(1, size, size, 3), compression="gzip", compression_opts=4,
        )
    for index, state in enumerate(demo["states"]):
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        datasets["agentview_rgb"][index] = env.sim.render(camera_name="agentview", width=size, height=size)[::-1]
        datasets["eye_in_hand_rgb"][index] = env.sim.render(camera_name="robot0_eye_in_hand", width=size, height=size)[::-1]
    for key, source in (
        ("agentview_image", "agentview_rgb"),
        ("robot0_eye_in_hand_image", "eye_in_hand_rgb"),
    ):
        if key in obs:
            del obs[key]
        values = resize_nearest(np.asarray(obs[source]), 84)
        obs.create_dataset(
            key, data=values, chunks=(1, 84, 84, 3), compression="gzip", compression_opts=4,
        )
    demo.attrs["libero_schema_images_complete"] = True
    demo.attrs["rgb_image_size"] = size
    demo.attrs["rgb_image_convention"] = "uint8 RGB, vertically flipped from OpenGL"


def migrate(path: Path, render_images: bool, image_size: int, max_demos: int | None) -> None:
    print(f"Migrating {path}", flush=True)
    with h5py.File(path, "r+") as file:
        data = file["data"]
        env = make_env(data)
        try:
            env.reset()
            demos = sorted((key for key in data if key.startswith("demo_")), key=natural)
            if max_demos is not None:
                demos = demos[:max_demos]
            for number, name in enumerate(demos, 1):
                demo = data[name]
                # Episodes carry their own XML and fixed-body placement.
                reset_to_demo(env, demo)
                migrate_demo(env, demo)
                if render_images:
                    add_images(env, demo, image_size)
                if number % 25 == 0 or number == len(demos):
                    file.flush()
                    print(f"  [{number}/{len(demos)}] {name}", flush=True)
            data.attrs["libero_custom_tasks_schema"] = "v1"
            data.attrs["action_space"] = "OSC_POSE normalized delta + gripper"
            data.attrs["force_keys"] = "ee_force,ee_torque,left_gripper_force,right_gripper_force"
            metadata = json.loads(str(data.attrs["env_args"]))
            metadata["env_kwargs"]["controller_configs"]["type"] = "OSC_POSE"
            metadata["env_kwargs"].update(
                use_camera_obs=True,
                has_offscreen_renderer=True,
                camera_names=["robot0_eye_in_hand", "agentview"],
                camera_heights=128,
                camera_widths=128,
            )
            data.attrs["env_args"] = json.dumps(metadata)
            all_demos = sorted((key for key in data if key.startswith("demo_")), key=natural)
            data.attrs["num_demos"] = len(all_demos)
            data.attrs["total"] = sum(int(data[name].attrs["num_samples"]) for name in all_demos)
            file.attrs["completed_demos"] = len(all_demos)
            file.attrs["requested_demos"] = len(all_demos)
            file.attrs["processing_complete"] = True
            ft = file.require_group("force_torque_metadata")
            ft.attrs.update(
                format="maniskill_mujoco_force_torque_v1",
                force_units="N",
                torque_units="N*m",
                sensor_frame="Panda end-effector force/torque site",
                storage="data/<demo>/obs/robot0_eef_force and data/<demo>/obs/robot0_eef_torque",
                completed_demos=len(all_demos),
                requested_demos=len(all_demos),
                processing_complete=True,
            )
            file.flush()
        finally:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--render-images", action="store_true")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-demos", type=int)
    args = parser.parse_args()
    inputs = args.inputs or sorted(Path("datasets/ManiSkill_Demonstrations/mujoco").glob("*.hdf5"))
    for path in inputs:
        migrate(path, args.render_images, args.image_size, args.max_demos)


if __name__ == "__main__":
    main()
