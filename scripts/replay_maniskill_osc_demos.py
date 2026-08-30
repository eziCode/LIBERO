#!/usr/bin/env python3
"""Replay ManiSkill MuJoCo OSC demonstrations and optionally record proof videos.

This executes the stored OSC actions open-loop. It does not play back stored
states or stored RGB frames. Saved states are used only to initialize each
episode and to measure replay drift after every action.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np
from robosuite.utils.binding_utils import MjRenderContextOffscreen

import maniskill_mujoco_envs  # noqa: F401
from scripts.retarget_maniskill_to_mujoco import make_environment
from scripts.validate_maniskill_osc_replay import natural_demo_key, reset_to_demo


DEFAULT_DATASET_DIR = ROOT / "datasets/ManiSkill_Demonstrations/mujoco_osc"
DEFAULT_VIDEO_DIR = ROOT / "artifacts/maniskill_osc_replays"


def demo_names(data: h5py.Group, specification: str) -> list[str]:
    available = sorted(
        (key for key in data if key.startswith("demo_")), key=natural_demo_key
    )
    if specification == "all":
        return available
    requested: list[str] = []
    for item in specification.split(","):
        item = item.strip().removeprefix("demo_")
        if "-" in item:
            start, stop = (int(value) for value in item.split("-", 1))
            requested.extend(f"demo_{index}" for index in range(start, stop + 1))
        else:
            requested.append(f"demo_{int(item)}")
    missing = [name for name in requested if name not in data]
    if missing:
        raise KeyError(f"demonstrations not found: {', '.join(missing)}")
    return requested


def initialize_renderer(env, camera: str, size: int) -> None:
    env.sim.add_render_context(MjRenderContextOffscreen(env.sim, device_id=-1))
    env.sim.render(camera_name=camera, width=size, height=size)
    context = env.sim._render_context_offscreen
    context.vopt.geomgroup[:] = 1
    context.vopt.geomgroup[0] = 0


def live_frame(
    env,
    camera: str,
    size: int,
    step: int,
    total: int,
    success: bool,
    reward: float,
) -> np.ndarray:
    frame = env.sim.render(camera_name=camera, width=size, height=size)[::-1].copy()
    shade = frame.copy()
    cv2.rectangle(shade, (8, 8), (min(size - 8, 390), 92), (0, 0, 0), -1)
    frame = cv2.addWeighted(shade, 0.60, frame, 0.40, 0)
    color = (80, 255, 80) if success else (255, 180, 80)
    lines = (
        f"OPEN-LOOP OSC REPLAY   step {step}/{total}",
        f"task success: {success}",
        f"environment reward: {reward:.4f}",
    )
    for row, line in enumerate(lines):
        cv2.putText(
            frame, line, (18, 31 + row * 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.52, color if row == 1 else (245, 245, 245), 1, cv2.LINE_AA,
        )
    return frame


def replay_demo(
    env,
    demo: h5py.Group,
    dataset_stem: str,
    name: str,
    video_dir: Path | None,
    camera: str,
    size: int,
    fps: float,
) -> dict:
    reset_to_demo(env, demo)
    actions = np.asarray(demo["actions"])
    reference_states = np.asarray(demo["states"])
    success_trace = [bool(env._check_success())]
    rewards = [float(env.reward())]
    state_drifts: list[float] = []
    writer = None
    video_path = None
    if video_dir is not None:
        initialize_renderer(env, camera, size)
        video_path = video_dir / dataset_stem / f"{name}_{camera}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8)
        writer.append_data(
            live_frame(env, camera, size, 0, len(actions), success_trace[-1], rewards[-1])
        )

    try:
        for index, action in enumerate(actions):
            _, reward, _, _ = env.step(action)
            success = bool(env._check_success())
            success_trace.append(success)
            rewards.append(float(reward))
            if index + 1 < len(reference_states):
                actual = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
                expected = reference_states[index + 1]
                state_drifts.append(float(np.linalg.norm(actual - expected)))
            if writer is not None:
                writer.append_data(
                    live_frame(
                        env, camera, size, index + 1, len(actions), success, float(reward)
                    )
                )
    finally:
        if writer is not None:
            writer.close()

    successful_steps = [index for index, value in enumerate(success_trace) if value]
    return {
        "demo": name,
        "actions_executed": len(actions),
        "initial_success": success_trace[0],
        "ever_success": bool(successful_steps),
        "final_success": success_trace[-1],
        "first_success_step": successful_steps[0] if successful_steps else None,
        "last_success_step": successful_steps[-1] if successful_steps else None,
        "successful_step_count": len(successful_steps),
        "max_state_l2_drift": max(state_drifts, default=0.0),
        "mean_state_l2_drift": float(np.mean(state_drifts)) if state_drifts else 0.0,
        "final_reward": rewards[-1],
        "video": str(video_path) if video_path is not None else None,
    }


def replay_file(path: Path, args: argparse.Namespace) -> dict:
    with h5py.File(path, "r") as dataset:
        data = dataset["data"]
        metadata = json.loads(str(data.attrs["env_args"]))
        names = demo_names(data, args.demos)
        env = make_environment(metadata["env_name"], 20, controller="OSC_POSE")
        results = []
        try:
            for ordinal, name in enumerate(names, 1):
                result = replay_demo(
                    env, data[name], path.stem, name,
                    None if args.no_video else args.video_dir,
                    args.camera, args.size, args.fps,
                )
                results.append(result)
                print(
                    f"{path.name} {name} [{ordinal}/{len(names)}]: "
                    f"ever={result['ever_success']} final={result['final_success']} "
                    f"first={result['first_success_step']} "
                    f"max_drift={result['max_state_l2_drift']:.6g}",
                    flush=True,
                )
        finally:
            env.close()
    key = f"{args.success_mode}_success"
    return {
        "dataset": str(path),
        "success_mode": args.success_mode,
        "tested": len(results),
        "successful": sum(bool(result[key]) for result in results),
        "failed": [result["demo"] for result in results if not result[key]],
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", type=Path)
    parser.add_argument(
        "--demos", default="0",
        help="Comma-separated indices/ranges (example: 0,4-7), or 'all'",
    )
    parser.add_argument(
        "--success-mode", choices=("final", "ever"), default="final",
        help="Criterion used for the process exit status (default: final)",
    )
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--camera", choices=("agentview", "robot0_eye_in_hand"), default="agentview")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--json", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    if not args.datasets:
        args.datasets = sorted(DEFAULT_DATASET_DIR.glob("*.hdf5"))
    if not args.datasets:
        parser.error(f"no datasets found in {DEFAULT_DATASET_DIR}")
    if args.size <= 0 or args.fps <= 0:
        parser.error("--size and --fps must be positive")
    return args


def main() -> int:
    args = parse_args()
    reports = [replay_file(path, args) for path in args.datasets]
    for report in reports:
        print(
            f"SUMMARY {Path(report['dataset']).name}: "
            f"{report['successful']}/{report['tested']} pass "
            f"({report['success_mode']}-frame criterion)"
        )
        if report["failed"]:
            print("  failed: " + ", ".join(report["failed"]))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"JSON report: {args.json}")
    return int(any(report["failed"] for report in reports))


if __name__ == "__main__":
    raise SystemExit(main())
