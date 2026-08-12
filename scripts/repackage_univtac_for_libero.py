#!/usr/bin/env python3
"""Migrate UniVTAC MuJoCo data to the custom LIBERO schema in place."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import h5py
import mujoco
import numpy as np
import robosuite as suite
from robosuite.utils import transform_utils as T

import univtac_mujoco_envs  # noqa: F401


def natural(name: str):
    prefix, _, suffix = name.rpartition("_")
    return prefix, int(suffix)


def replace(group: h5py.Group, key: str, values: np.ndarray, compression="gzip") -> None:
    temporary = f"__new_{key}"
    if temporary in group:
        del group[temporary]
    group.create_dataset(temporary, data=values, compression=compression, shuffle=True)
    if key in group:
        del group[key]
    group.move(temporary, key)


def finger_forces(env) -> tuple[float, float]:
    names = env.robots[0].gripper.contact_geoms
    ids = [
        {env.sim.model.geom_name2id(name) for name in names if f"finger{side}" in name}
        for side in (1, 2)
    ]
    values = [0.0, 0.0]
    wrench = np.zeros(6)
    for index in range(env.sim.data.ncon):
        contact = env.sim.data.contact[index]
        for side, geom_ids in enumerate(ids):
            if contact.geom1 in geom_ids or contact.geom2 in geom_ids:
                mujoco.mj_contactForce(env.sim.model._model, env.sim.data._data, index, wrench)
                values[side] += abs(float(wrench[0]))
    return values[0], values[1]


def socket_reaction(env) -> tuple[float, float]:
    """Return summed socket normal load and tangential insertion resistance."""
    plug = env.sim.model.geom_name2id("prism_collision")
    socket = {
        env.sim.model.geom_name2id(name)
        for name in env.sim.model.geom_names
        if name and name.startswith("slot_") and name != "slot_visual"
    }
    normal = tangential = 0.0
    wrench = np.zeros(6)
    for index in range(env.sim.data.ncon):
        contact = env.sim.data.contact[index]
        if not (
            (contact.geom1 == plug and contact.geom2 in socket)
            or (contact.geom2 == plug and contact.geom1 in socket)
        ):
            continue
        mujoco.mj_contactForce(env.sim.model._model, env.sim.data._data, index, wrench)
        normal += abs(float(wrench[0]))
        tangential += float(np.linalg.norm(wrench[1:3]))
    return normal, tangential


def simulated_wrench(env, demo: h5py.Group, joint_actions: np.ndarray):
    states = demo["states"]
    count = len(states)
    force = np.empty((count, 3), np.float32)
    torque = np.empty((count, 3), np.float32)
    left = np.empty((count, 1), np.float32)
    right = np.empty((count, 1), np.float32)
    joint = np.empty((count, 7), np.float32)
    gripper = np.empty((count, 2), np.float32)
    slot = np.r_[demo["obs/slot_pos"][0], demo["obs/slot_quat"][0][[3, 0, 1, 2]]]
    env.reset()
    env.set_actor_pose("slot", slot)
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()
    env.set_recorded_grasp(True)
    env.robots[0].controller.reset_goal()
    for index, action in enumerate(joint_actions):
        robot = env.robots[0]
        joint[index] = robot._joint_positions
        gripper[index] = env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes]
        env.step(action)
        normal_load, contact_resistance = socket_reaction(env)
        insertion_resistance = contact_resistance + env.insertion_spring_force
        # The connector is inserted along -world-z. Store the opposing socket
        # reaction along +z; before contact this is exactly zero.
        force[index] = (0.0, 0.0, insertion_resistance)
        torque[index] = env.robots[0].ee_torque
        # Opposing virtual GelSight pads share the socket's normal preload.
        left[index, 0] = 0.5 * normal_load
        right[index, 0] = 0.5 * normal_load
        if index + 1 < count:
            env.sim.set_state_from_flattened(states[index + 1])
            env.sim.forward()
            env.set_recorded_grasp(True)
    return force, torque, left, right, joint, gripper


def resized_images(dataset: h5py.Dataset, size: int) -> np.ndarray:
    result = np.empty((len(dataset), size, size, 3), np.uint8)
    for index, image in enumerate(dataset):
        result[index] = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return result


def libero_actions(obs: h5py.Group, next_obs: h5py.Group, gripper: np.ndarray) -> np.ndarray:
    pos = np.asarray(obs["robot0_eef_pos"])
    next_pos = np.asarray(next_obs["robot0_eef_pos"])
    quat = np.asarray(obs["robot0_eef_quat"])
    next_quat = np.asarray(next_obs["robot0_eef_quat"])
    result = np.empty((len(pos), 7), np.float32)
    result[:, :3] = np.clip((next_pos - pos) / 0.05, -1, 1)
    for index in range(len(pos)):
        result[index, 3:6] = np.clip(T.get_orientation_error(next_quat[index], quat[index]) / 0.5, -1, 1)
    result[:, 6] = gripper
    return result


def update_model_file(env, demo: h5py.Group) -> None:
    """Embed this demo's fixed slot pose and active recorded grasp in its XML."""
    slot = np.r_[demo["obs/slot_pos"][0], demo["obs/slot_quat"][0][[3, 0, 1, 2]]]
    env.reset()
    env.set_actor_pose("slot", slot)
    env.sim.set_state_from_flattened(demo["states"][0])
    env.sim.forward()
    env.set_recorded_grasp(True)
    root = ET.fromstring(env.model.get_xml())
    slot_body = root.find(".//body[@name='slot_main']")
    weld = root.find(".//weld[@name='univtac_recorded_grasp']")
    if slot_body is None or weld is None:
        raise ValueError("UniVTAC slot body or grasp weld missing from model XML")
    slot_body.set("pos", " ".join(map(str, slot[:3])))
    slot_body.set("quat", " ".join(map(str, slot[3:])))
    relative_pose = env.sim.model.eq_data[env.grasp_weld_id, 3:10]
    weld.set("relpose", " ".join(map(str, relative_pose)))
    weld.set("active", "true")
    demo.attrs["model_file"] = ET.tostring(root, encoding="unicode")


def migrate_demo(env, demo: h5py.Group, image_size: int) -> None:
    version = demo.attrs.get("libero_schema_version", 0)
    if version >= 10:
        return
    obs = demo["obs"]
    joint_actions = np.asarray(demo.get("mujoco_joint_actions", demo["actions"]))
    if joint_actions.shape[1] != 8:
        raise ValueError("preserved JOINT_POSITION actions are unavailable")
    force, torque, left, right, joint_states, gripper_states = simulated_wrench(
        env, demo, joint_actions
    )
    ee_pos = np.asarray(obs["robot0_eef_pos"], np.float32)
    quats = np.asarray(obs["robot0_eef_quat"])
    ee_ori = np.stack([T.quat2axisangle(quat) for quat in quats]).astype(np.float32)

    replace(obs, "joint_states", joint_states)
    replace(obs, "gripper_states", gripper_states)
    replace(obs, "ee_pos", ee_pos)
    replace(obs, "ee_ori", ee_ori)
    replace(obs, "ee_states", np.concatenate((ee_pos, ee_ori), axis=1))
    replace(obs, "ee_force", force)
    replace(obs, "ee_torque", torque)
    replace(obs, "left_gripper_force", left)
    replace(obs, "right_gripper_force", right)
    replace(obs, "agentview_rgb", resized_images(obs["agentview_image"], image_size))
    replace(obs, "eye_in_hand_rgb", resized_images(obs["robot0_eye_in_hand_image"], image_size))
    replace(demo, "robot_states", np.concatenate((joint_states, gripper_states), axis=1))
    if "mujoco_joint_actions" not in demo:
        demo["mujoco_joint_actions"] = demo["actions"]
    replace(demo, "actions", libero_actions(obs, demo["next_obs"], joint_actions[:, -1]))
    demo.attrs["libero_schema_complete"] = True
    update_model_file(env, demo)
    demo.attrs["libero_schema_version"] = 10
    demo.attrs["force_source"] = "action-driven MuJoCo replay, state re-anchored each transition"
    demo.attrs["gripper_force_model"] = "symmetric split of MuJoCo socket normal reaction"
    demo.attrs["ee_force_model"] = "MuJoCo socket tangential contact plus axial pin/detent spring reaction along +world-z"
    demo.attrs["insertion_spring_stiffness_N_per_m"] = 1400.0
    demo.attrs["force_units"] = "N"
    demo.attrs["torque_units"] = "N*m"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", type=Path, default=Path("datasets/UniVTAC/mujoco/insert_HDMI.hdf5"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-demos", type=int)
    args = parser.parse_args()
    with h5py.File(args.dataset, "r+") as file:
        data = file["data"]
        meta = json.loads(data.attrs["env_args"])
        kwargs = dict(meta["env_kwargs"])
        kwargs.update(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, hard_reset=False)
        env = suite.make(meta["env_name"], **kwargs)
        try:
            demos = sorted((name for name in data if name.startswith("demo_")), key=natural)
            if args.max_demos is not None:
                demos = demos[:args.max_demos]
            for number, name in enumerate(demos, 1):
                migrate_demo(env, data[name], args.image_size)
                file.flush()
                print(f"[{number}/{len(demos)}] {name}", flush=True)
            data.attrs["libero_custom_tasks_schema"] = "v1"
            data.attrs["force_source"] = "action-driven MuJoCo replay; simulated, not UniVTAC ground truth"
            data.attrs["insertion_force_model"] = "compliant socket tabs plus 1400 N/m axial pin/detent spring"
            data.attrs["action_space"] = "OSC_POSE normalized delta + gripper"
        finally:
            env.close()


if __name__ == "__main__":
    main()
