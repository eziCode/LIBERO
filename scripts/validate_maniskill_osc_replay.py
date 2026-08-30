#!/usr/bin/env python3
"""Replay ManiSkill MuJoCo OSC datasets and report true task success."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.retarget_maniskill_to_mujoco import (  # noqa: E402
    make_environment,
    synchronize_gripper_command_state,
)


def natural_demo_key(name: str) -> int:
    return int(name.rsplit("_", 1)[1])


def reset_to_demo(env, demo: h5py.Group) -> None:
    metadata = json.loads(str(demo.attrs["source_episode_metadata"]))
    env.configure_source_episode(metadata)
    env.reset()
    env.reset_from_xml_string(str(demo.attrs["model_file"]))
    env.object_body_ids = {
        name: env.sim.model.body_name2id(obj.root_body)
        for name, obj in env.task_objects.items()
    }
    env.sim.reset()
    env.sim.set_state_from_flattened(np.asarray(demo["states"][0]))
    for name in env.fixed_object_names:
        position_key = f"obs/{name}_pos"
        quaternion_key = f"obs/{name}_quat"
        if position_key not in demo or quaternion_key not in demo:
            continue
        position = np.asarray(demo[position_key][0], dtype=np.float64)
        xyzw = np.asarray(demo[quaternion_key][0], dtype=np.float64)
        env.set_task_object_pose(name, np.r_[position, xyzw[3], xyzw[:3]])
    env.sim.forward()
    synchronize_gripper_command_state(env)

    # Fixed bodies are encoded in model XML, but task-specific Python fields
    # are not. Restore the one such field used by the current task suite.
    if hasattr(env, "goal_position") and "obs/goal_site_pos" in demo:
        env.goal_position = np.asarray(demo["obs/goal_site_pos"][0], dtype=np.float64)

    robot = env.robots[0]
    robot.controller.update_initial_joints(
        np.asarray(robot._joint_positions, dtype=np.float64).copy()
    )
    robot.controller.reset_goal()


def validate_file(path: Path, limit: int | None = None) -> dict:
    with h5py.File(path, "r") as dataset:
        data = dataset["data"]
        metadata = json.loads(str(data.attrs["env_args"]))
        if str(data.attrs.get("controller", "")) != "OSC_POSE":
            raise ValueError(f"{path}: controller is not OSC_POSE")
        names = sorted(data.keys(), key=natural_demo_key)
        if limit is not None:
            names = names[:limit]
        env = make_environment(metadata["env_name"], 20, controller="OSC_POSE")
        results = []
        try:
            for name in names:
                demo = data[name]
                reset_to_demo(env, demo)
                ever_succeeded = bool(env._check_success())
                for action in demo["actions"]:
                    env.step(np.asarray(action, dtype=np.float32))
                    ever_succeeded |= bool(env._check_success())
                results.append(
                    {
                        "demo": name,
                        "steps": len(demo["actions"]),
                        "ever_success": ever_succeeded,
                        "final_success": bool(env._check_success()),
                    }
                )
        finally:
            env.close()
    failures = [item["demo"] for item in results if not item["ever_success"]]
    return {
        "file": str(path),
        "tested": len(results),
        "successful": len(results) - len(failures),
        "failures": failures,
        "results": results,
    }


def drop_failed_demos(path: Path, failures: list[str]) -> None:
    """Atomically rewrite a dataset without demonstrations that failed replay."""
    if not failures:
        return
    rejected = set(failures)
    temporary = path.with_name(f".{path.name}.replay-filter.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with h5py.File(path, "r") as source, h5py.File(temporary, "w") as target:
            source_data = source["data"]
            target_data = target.create_group("data")
            for key, value in source_data.attrs.items():
                target_data.attrs[key] = value
            retained = [
                name
                for name in sorted(source_data.keys(), key=natural_demo_key)
                if name not in rejected
            ]
            total = 0
            for index, name in enumerate(retained):
                source.copy(source_data[name], target_data, name=f"demo_{index}")
                total += int(source_data[name].attrs["num_samples"])
            target_data.attrs["num_demos"] = len(retained)
            target_data.attrs["total"] = total
            target_data.attrs["replay_filtered_demos"] = len(rejected)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--json", type=Path, help="Write the detailed report here")
    parser.add_argument(
        "--drop-failures",
        action="store_true",
        help="Atomically remove demonstrations that fail replay",
    )
    args = parser.parse_args()

    reports = [validate_file(path, args.limit) for path in args.paths]
    for report in reports:
        print(
            f"{Path(report['file']).name}: "
            f"{report['successful']}/{report['tested']} replay successfully"
        )
        if report["failures"]:
            print("  failures: " + ", ".join(report["failures"]))
    if args.json:
        args.json.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    if args.drop_failures:
        for path, report in zip(args.paths, reports):
            drop_failed_demos(path, report["failures"])
            if report["failures"]:
                print(f"{path.name}: removed {len(report['failures'])} failed demonstrations")
    return int(any(report["failures"] for report in reports))


if __name__ == "__main__":
    raise SystemExit(main())
