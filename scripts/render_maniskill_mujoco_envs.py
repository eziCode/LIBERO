#!/usr/bin/env python3
"""Render labeled multi-view contact sheets for the ManiSkill MuJoCo ports."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
from PIL import Image, ImageDraw
import robosuite as suite
from robosuite.controllers import load_controller_config

import maniskill_mujoco_envs  # noqa: F401 - registers environments


ENVIRONMENTS = (
    ("PickCube-v1", "ManiSkillMujocoPickCube"),
    ("StackCube-v1", "ManiSkillMujocoStackCube"),
    ("StackPyramid-v1", "ManiSkillMujocoStackPyramid"),
    ("PegInsertionSide-v1", "ManiSkillMujocoPegInsertionSide"),
    ("PlugCharger-v1", "ManiSkillMujocoPlugCharger"),
)
PREFERRED_CAMERAS = (
    "agentview",
    "frontview",
    "birdview",
    "sideview",
    "robot0_eye_in_hand",
)


def labeled(image: np.ndarray, title: str) -> Image.Image:
    panel = Image.new("RGB", (image.shape[1], image.shape[0] + 34), "white")
    panel.paste(Image.fromarray(image), (0, 34))
    ImageDraw.Draw(panel).text((10, 9), title, fill="black")
    return panel


def contact_sheet(panels: list[Image.Image], columns: int) -> Image.Image:
    width = max(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    rows = (len(panels) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * height), (235, 235, 235))
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % columns) * width, (index // columns) * height))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/environment_renders"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    overview: list[Image.Image] = []
    for task_id, env_name in ENVIRONMENTS:
        np.random.seed(args.seed)
        env = suite.make(
            env_name,
            robots="Panda",
            controller_configs=load_controller_config(default_controller="JOINT_POSITION"),
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=False,
            hard_reset=False,
        )
        env.reset()
        available = set(env.sim.model.camera_names)
        cameras = [name for name in PREFERRED_CAMERAS if name in available]
        # Manual sim.render() lazily creates the offscreen context after
        # robosuite's usual visibility setup has already run. Prime it, then
        # explicitly show visual geoms (group 1) and hide collision geoms
        # (group 0). Otherwise both meshes overlap and the Panda collision
        # mesh's RGB axis colors bleed through the white visual shell.
        env.sim.render(camera_name=cameras[0], width=args.size, height=args.size)
        visibility = env.sim._render_context_offscreen.vopt.geomgroup
        for group_index in range(len(visibility)):
            env.sim._render_context_offscreen.vopt.geomgroup[group_index] = 1
        env.sim._render_context_offscreen.vopt.geomgroup[0] = 0
        env.sim._render_context_offscreen.vopt.geomgroup[1] = 1
        panels: list[Image.Image] = []
        for camera in cameras:
            # robosuite's legacy renderer returns OpenGL-bottom-up images.
            frame = env.sim.render(
                camera_name=camera, width=args.size, height=args.size
            )[::-1].copy()
            Image.fromarray(frame).save(args.output / f"{task_id}_{camera}.png")
            panels.append(labeled(frame, f"{task_id} — {camera}"))
            if camera == "agentview":
                overview.append(labeled(frame, task_id))
        contact_sheet(panels, columns=3).save(args.output / f"{task_id}_views.png")
        env.close()

    contact_sheet(overview, columns=3).save(args.output / "all_tasks_overview.png")
    print(f"Wrote renders to {args.output.resolve()}")


if __name__ == "__main__":
    main()
