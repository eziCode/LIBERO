#!/usr/bin/env python3
"""Render converted ManiSkill MuJoCo demonstrations from recorded states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import h5py
import imageio.v2 as imageio
import cv2
import mujoco
import numpy as np
import robosuite as suite
from robosuite.utils.binding_utils import MjRenderContextOffscreen

import maniskill_mujoco_envs  # noqa: F401 - register task ports


def finger_contact_forces(env) -> tuple[float, float]:
    """Return summed normal contact force for each Panda finger in Newtons."""
    names = env.robots[0].gripper.contact_geoms
    left_ids = {
        env.sim.model.geom_name2id(name) for name in names if "finger1" in name
    }
    right_ids = {
        env.sim.model.geom_name2id(name) for name in names if "finger2" in name
    }
    left = right = 0.0
    wrench = np.zeros(6, dtype=np.float64)
    for index in range(env.sim.data.ncon):
        contact = env.sim.data.contact[index]
        touches_left = contact.geom1 in left_ids or contact.geom2 in left_ids
        touches_right = contact.geom1 in right_ids or contact.geom2 in right_ids
        if not (touches_left or touches_right):
            continue
        mujoco.mj_contactForce(
            env.sim.model._model, env.sim.data._data, index, wrench
        )
        normal = abs(float(wrench[0]))
        if touches_left:
            left += normal
        if touches_right:
            right += normal
    return left, right


def force_overlay(
    frame: np.ndarray,
    force: np.ndarray,
    torque: np.ndarray,
    left_contact: float,
    right_contact: float,
) -> np.ndarray:
    """Draw readable wrench text and tactile bar gauges on an RGB frame."""
    result = frame.copy()
    height, width = result.shape[:2]
    panel = result.copy()
    cv2.rectangle(panel, (8, 8), (min(width - 8, 350), 150), (0, 0, 0), -1)
    result = cv2.addWeighted(panel, 0.62, result, 0.38, 0)
    force_norm = float(np.linalg.norm(force))
    torque_norm = float(np.linalg.norm(torque))
    lines = (
        f"EEF force  [{force[0]:6.1f} {force[1]:6.1f} {force[2]:6.1f}] N",
        f"|force|    {force_norm:6.1f} N",
        f"EEF torque [{torque[0]:6.2f} {torque[1]:6.2f} {torque[2]:6.2f}] Nm",
        f"|torque|   {torque_norm:6.2f} Nm",
        f"tactile L/R {left_contact:6.1f} / {right_contact:6.1f} N",
    )
    for line_index, line in enumerate(lines):
        cv2.putText(
            result,
            line,
            (18, 31 + 24 * line_index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (235, 255, 235),
            1,
            cv2.LINE_AA,
        )

    gauges = (
        ("F", force_norm, 100.0, (80, 220, 255)),
        ("L", left_contact, 40.0, (80, 255, 120)),
        ("R", right_contact, 40.0, (255, 170, 80)),
    )
    bar_left, bar_width = 45, min(230, width - 70)
    for row, (label, value, scale, color) in enumerate(gauges):
        y = height - 68 + row * 20
        cv2.putText(result, label, (18, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.rectangle(result, (bar_left, y), (bar_left + bar_width, y + 11), (35, 35, 35), -1)
        fill = int(bar_width * min(value / scale, 1.0))
        cv2.rectangle(result, (bar_left, y), (bar_left + fill, y + 11), color, -1)
    return result


def render_dataset(
    path: Path,
    output: Path,
    demo_name: str,
    size: int,
    fps: int,
    camera: str,
    overlay_force: bool,
) -> None:
    print(f"Loading {path.stem}/{demo_name}...", flush=True)
    with h5py.File(path, "r") as dataset:
        data = dataset["data"]
        metadata = json.loads(data.attrs["env_args"])
        demo = data[demo_name]
        states = np.asarray(demo["states"])
        forces = np.asarray(demo["obs/robot0_eef_force"])
        torques = np.asarray(demo["obs/robot0_eef_torque"])
        fixed_poses = {}
        for name in ("goal_site", "box_with_hole", "receptacle"):
            position_key, quaternion_key = f"{name}_pos", f"{name}_quat"
            if position_key in demo["obs"] and quaternion_key in demo["obs"]:
                position = np.asarray(demo["obs"][position_key][0])
                quaternion_xyzw = np.asarray(demo["obs"][quaternion_key][0])
                fixed_poses[name] = np.r_[position, quaternion_xyzw[[3, 0, 1, 2]]]

    kwargs = dict(metadata["env_kwargs"])
    kwargs.update(
        has_renderer=False,
        # Defer CGL context creation until sim.render(). Creating it inside
        # suite.make can deadlock against macOS LaunchServices in a CLI process.
        has_offscreen_renderer=False,
        use_camera_obs=False,
        hard_reset=False,
    )
    env = suite.make(metadata["env_name"], **kwargs)
    print(f"Created {metadata['env_name']}; initializing state and renderer...", flush=True)
    writer = None
    try:
        env.reset()
        for name, pose in fixed_poses.items():
            env.set_task_object_pose(name, pose)
        if camera not in env.sim.model.camera_names:
            raise ValueError(
                f"camera {camera!r} is unavailable; choices: {env.sim.model.camera_names}"
            )
        # Initialize the context and enforce visual-only geometry visibility.
        env.sim.add_render_context(MjRenderContextOffscreen(env.sim, device_id=-1))
        env.sim.render(camera_name=camera, width=size, height=size)
        print(f"Renderer ready; encoding {len(states)} frames...", flush=True)
        context = env.sim._render_context_offscreen
        for index in range(len(context.vopt.geomgroup)):
            context.vopt.geomgroup[index] = 1
        context.vopt.geomgroup[0] = 0
        context.vopt.geomgroup[1] = 1

        output.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(output, fps=fps, codec="libx264", quality=8)
        for index, state in enumerate(states):
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            frame = env.sim.render(
                camera_name=camera, width=size, height=size
            )[::-1].copy()
            if overlay_force:
                left, right = finger_contact_forces(env)
                frame = force_overlay(frame, forces[index], torques[index], left, right)
            writer.append_data(frame)
    finally:
        if writer is not None:
            writer.close()
        env.close()
    print(f"{path.stem}/{demo_name}: {len(states)} frames -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("datasets/ManiSkill_Demonstrations/mujoco"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/trajectory_videos")
    )
    parser.add_argument("--demo", default="demo_0")
    parser.add_argument("--task", help="Render only this task stem, e.g. PickCube-v1")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument(
        "--overlay-force",
        action="store_true",
        help="Overlay EEF wrench and MuJoCo left/right finger contact forces",
    )
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("*.hdf5"))
    if args.task:
        paths = [path for path in paths if path.stem == args.task]
    if not paths:
        raise FileNotFoundError(f"no HDF5 datasets found in {args.input_dir}")
    for path in paths:
        render_dataset(
            path,
            args.output_dir
            / (
                f"{path.stem}_{args.demo}"
                f"{'_' + args.camera if args.camera != 'agentview' else ''}"
                f"{'_force_tactile' if args.overlay_force else ''}.mp4"
            ),
            args.demo,
            args.size,
            args.fps,
            args.camera,
            args.overlay_force,
        )


if __name__ == "__main__":
    main()
