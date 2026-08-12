#!/usr/bin/env python3
"""Render UniVTAC MuJoCo trajectories with optional GelSight panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np
import robosuite as suite
from robosuite.utils.binding_utils import MjRenderContextOffscreen

import univtac_mujoco_envs  # noqa: F401


CAMERAS = {"agentview": "univtac_head", "sideview": "univtac_side"}


def contact_forces(env) -> tuple[float, float]:
    names = env.robots[0].gripper.contact_geoms
    ids = [
        {env.sim.model.geom_name2id(name) for name in names if f"finger{finger}" in name}
        for finger in (1, 2)
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


def tactile_overlay(
    scene: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    force: np.ndarray,
    torque: np.ndarray,
    contacts: tuple[float, float],
) -> np.ndarray:
    """Compose the MuJoCo view with original left/right GelSight observations."""
    size = scene.shape[0]
    panel_h = max(120, size // 3)
    tactile_w = panel_h * 4 // 3
    left = cv2.resize(left, (tactile_w, panel_h), interpolation=cv2.INTER_AREA)
    right = cv2.resize(right, (tactile_w, panel_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size + panel_h, size, 3), dtype=np.uint8)
    canvas[:size] = scene
    canvas[size:, :tactile_w] = left
    canvas[size:, size - tactile_w:] = right
    cv2.putText(canvas, "LEFT GELSIGHT", (8, size + 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "RIGHT GELSIGHT", (size - tactile_w + 8, size + 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1, cv2.LINE_AA)

    shade = canvas.copy()
    cv2.rectangle(shade, (8, 8), (min(size - 8, 360), 126), (0, 0, 0), -1)
    canvas = cv2.addWeighted(shade, .62, canvas, .38, 0)
    lines = (
        f"EEF force  [{force[0]:6.1f} {force[1]:6.1f} {force[2]:6.1f}] N",
        f"|force|    {np.linalg.norm(force):6.1f} N",
        f"EEF torque [{torque[0]:6.2f} {torque[1]:6.2f} {torque[2]:6.2f}] Nm",
        f"contact L/R {contacts[0]:6.1f} / {contacts[1]:6.1f} N",
    )
    for row, line in enumerate(lines):
        cv2.putText(canvas, line, (18, 31 + row * 25), cv2.FONT_HERSHEY_SIMPLEX, .47, (235,255,235), 1, cv2.LINE_AA)
    return canvas


def render(dataset: Path, output: Path, demo_name: str, view: str, size: int, fps: int, overlay: bool) -> None:
    with h5py.File(dataset, "r") as file:
        data = file["data"]
        meta = json.loads(data.attrs["env_args"])
        demo = data[demo_name]
        states = np.asarray(demo["states"])
        slot_pose = np.r_[demo["obs/slot_pos"][0], demo["obs/slot_quat"][0][[3, 0, 1, 2]]]
        force_key = "ee_force" if "ee_force" in demo["obs"] else "robot0_eef_force"
        torque_key = "ee_torque" if "ee_torque" in demo["obs"] else "robot0_eef_torque"
        forces = demo[f"obs/{force_key}"]
        torques = demo[f"obs/{torque_key}"]
        stored_left = demo["obs/left_gripper_force"] if "left_gripper_force" in demo["obs"] else None
        stored_right = demo["obs/right_gripper_force"] if "right_gripper_force" in demo["obs"] else None
        left = demo["obs/tactile_left_image"]
        right = demo["obs/tactile_right_image"]

        kwargs = dict(meta["env_kwargs"])
        kwargs.update(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, hard_reset=False)
        env = suite.make(meta["env_name"], **kwargs)
        writer = None
        try:
            env.reset()
            env.set_actor_pose("slot", slot_pose)
            camera = CAMERAS[view]
            env.sim.add_render_context(MjRenderContextOffscreen(env.sim, device_id=-1))
            env.sim.render(camera_name=camera, width=size, height=size)
            context = env.sim._render_context_offscreen
            context.vopt.geomgroup[:] = 1
            context.vopt.geomgroup[0] = 0
            output.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(output, fps=fps, codec="libx264", quality=8)
            for index, state in enumerate(states):
                env.sim.set_state_from_flattened(state)
                env.sim.forward()
                frame = env.sim.render(camera_name=camera, width=size, height=size)[::-1].copy()
                if overlay:
                    contacts = (
                        (float(stored_left[index, 0]), float(stored_right[index, 0]))
                        if stored_left is not None else contact_forces(env)
                    )
                    frame = tactile_overlay(frame, left[index], right[index], forces[index], torques[index], contacts)
                writer.append_data(frame)
        finally:
            if writer is not None:
                writer.close()
            env.close()
    print(f"Rendered {len(states)} frames -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/UniVTAC/mujoco/insert_HDMI.hdf5"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/UniVTAC/videos/mujoco"))
    parser.add_argument("--demo", default="demo_0")
    parser.add_argument("--view", choices=(*CAMERAS, "all"), default="all")
    parser.add_argument("--style", choices=("plain", "force_tactile", "all"), default="all")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    views = CAMERAS if args.view == "all" else (args.view,)
    styles = ("plain", "force_tactile") if args.style == "all" else (args.style,)
    for view in views:
        for style in styles:
            suffix = f"_{view}" if view != "agentview" else ""
            if style == "force_tactile":
                suffix += "_force_tactile"
            output = args.output_dir / view / style / f"insert_HDMI_{args.demo}{suffix}.mp4"
            render(args.dataset, output, args.demo, view, args.size, args.fps, style == "force_tactile")


if __name__ == "__main__":
    main()
