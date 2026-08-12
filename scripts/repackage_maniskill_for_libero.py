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


def natural(name: str):
    stem, _, suffix = name.rpartition("_")
    return stem, int(suffix)


def make_env(data: h5py.Group):
    meta = json.loads(data.attrs["env_args"])
    kwargs = dict(meta["env_kwargs"])
    kwargs.update(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, hard_reset=False)
    return suite.make(meta["env_name"], **kwargs)


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
    if demo["actions"].shape[1] == 8:
        if "mujoco_joint_actions" not in demo:
            demo["mujoco_joint_actions"] = demo["actions"]
        old = np.asarray(demo["actions"])
        replace_dataset(demo, "actions", canonical_actions(obs, demo["next_obs"], old[:, -1]))
    demo.attrs["libero_schema_numeric_complete"] = True


def add_images(env, demo: h5py.Group, size: int) -> None:
    obs = demo["obs"]
    if demo.attrs.get("libero_schema_images_complete", False):
        return
    if env.sim._render_context_offscreen is None:
        env.sim.add_render_context(MjRenderContextOffscreen(env.sim, device_id=-1))
        env.sim.render(camera_name="agentview", width=size, height=size)
        context = env.sim._render_context_offscreen
        context.vopt.geomgroup[:] = 1
        context.vopt.geomgroup[0] = 0
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
    demo.attrs["libero_schema_images_complete"] = True


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
                migrate_demo(env, demo)
                if render_images:
                    add_images(env, demo, image_size)
                if number % 25 == 0 or number == len(demos):
                    file.flush()
                    print(f"  [{number}/{len(demos)}] {name}", flush=True)
            data.attrs["libero_custom_tasks_schema"] = "v1"
            data.attrs["action_space"] = "OSC_POSE normalized delta + gripper"
            data.attrs["force_keys"] = "ee_force,ee_torque,left_gripper_force,right_gripper_force"
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
