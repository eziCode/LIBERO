#!/usr/bin/env python3
"""Measure one-step ManiSkill-to-MuJoCo transition fidelity.

Each recorded state is installed independently, its recorded action is executed
once through the evaluation wrapper, and the resulting robot / object state is
compared with the next recorded state. This avoids hiding local dynamics errors
behind accumulated rollout drift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from retarget_maniskill_to_mujoco import (
    TASK_MAPPINGS,
    configure_contact_physics,
    make_environment,
    natural_trajectory_key,
    numeric_observation,
    retarget_action,
    set_mapped_state,
    source_actor_states,
    source_articulation_states,
    source_episode_metadata,
)
from maniskill_mujoco_envs.wrapper import ManiSkillWrapper


def quaternion_angle(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return float(2.0 * np.arccos(np.clip(abs(np.dot(a, b)), 0.0, 1.0)))


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def benchmark(args: argparse.Namespace) -> dict:
    mapping = TASK_MAPPINGS[args.task_id]
    metadata = source_episode_metadata(args.input.with_suffix(".json"))
    env = make_environment(
        mapping.env_name,
        args.control_freq,
        joint_kp=args.joint_kp,
        joint_damping_ratio=args.joint_damping_ratio,
    )
    wrapper = ManiSkillWrapper(env)
    metrics: dict[str, list[float]] = {
        "joint_position_rad": [],
        "joint_velocity_rad_s": [],
        "tcp_position_m": [],
        "tcp_orientation_rad": [],
        "gripper_position_m": [],
        "dynamic_object_position_m": [],
        "dynamic_object_orientation_rad": [],
    }
    samples = 0
    try:
        with h5py.File(args.input, "r") as source_file:
            keys = sorted(
                (key for key in source_file if key.startswith("traj_")),
                key=natural_trajectory_key,
            )[: args.count]
            for key in keys:
                episode_id = int(key.rsplit("_", 1)[1])
                if hasattr(env, "configure_source_episode"):
                    env.configure_source_episode(metadata.get(episode_id, {}))
                env.reset()
                configure_contact_physics(
                    env,
                    args.surface_friction,
                    args.finger_friction,
                    args.contact_timeconstant,
                    args.gripper_force_limit,
                )
                source = source_file[key]
                actions = np.asarray(source["actions"], dtype=np.float64)
                articulation = source_articulation_states(source)
                actors = source_actor_states(source)
                for index in range(0, len(actions), args.stride):
                    current_actors = {name: value[index] for name, value in actors.items()}
                    next_actors = {name: value[index + 1] for name, value in actors.items()}

                    # Obtain the kinematically expected TCP in the target model.
                    set_mapped_state(env, articulation[index + 1], next_actors, mapping.object_map)
                    expected_obs = env._get_observations(force_update=True)
                    expected_tcp_pos = np.asarray(expected_obs["robot0_eef_pos"]).copy()
                    expected_tcp_quat = np.asarray(expected_obs["robot0_eef_quat"]).copy()

                    set_mapped_state(env, articulation[index], current_actors, mapping.object_map)
                    robot = env.robots[0]
                    robot.controller.reset_goal()
                    command = retarget_action(
                        env, actions[index], invert_gripper=not args.no_invert_gripper
                    )
                    next_obs, _, _, _ = wrapper.step(command)
                    actual = numeric_observation(next_obs, env)

                    metrics["joint_position_rad"].append(
                        float(np.max(np.abs(actual["joint_states"] - articulation[index + 1, 13:20])))
                    )
                    metrics["joint_velocity_rad_s"].append(
                        float(
                            np.max(
                                np.abs(
                                    np.asarray(robot._joint_velocities)
                                    - articulation[index + 1, 22:29]
                                )
                            )
                        )
                    )
                    metrics["tcp_position_m"].append(
                        float(np.linalg.norm(actual["ee_pos"] - expected_tcp_pos))
                    )
                    metrics["tcp_orientation_rad"].append(
                        quaternion_angle(actual["robot0_eef_quat"], expected_tcp_quat)
                    )
                    expected_gripper = np.array(
                        [articulation[index + 1, 20], -articulation[index + 1, 21]]
                    )
                    metrics["gripper_position_m"].append(
                        float(np.max(np.abs(actual["gripper_states"] - expected_gripper)))
                    )
                    for source_name, target_name in mapping.object_map:
                        if target_name not in env.dynamic_object_names or source_name not in actors:
                            continue
                        actual_pose = env.get_task_object_pose(target_name)
                        desired_pose = actors[source_name][index + 1, :7].copy()
                        desired_pose[:3] += env.source_world_offset
                        metrics["dynamic_object_position_m"].append(
                            float(np.linalg.norm(actual_pose[:3] - desired_pose[:3]))
                        )
                        metrics["dynamic_object_orientation_rad"].append(
                            quaternion_angle(actual_pose[3:7], desired_pose[3:7])
                        )
                    samples += 1
    finally:
        env.close()

    return {
        "task_id": args.task_id,
        "trajectories": args.count,
        "stride": args.stride,
        "samples": samples,
        "metrics": {name: summarize(values) for name, values in metrics.items() if values},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--task-id", choices=tuple(TASK_MAPPINGS), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--joint-kp", type=float, default=500.0)
    parser.add_argument("--joint-damping-ratio", type=float, default=1.58113883)
    parser.add_argument("--surface-friction", type=float, default=0.5)
    parser.add_argument("--finger-friction", type=float, default=2.0)
    parser.add_argument("--contact-timeconstant", type=float, default=0.002)
    parser.add_argument("--gripper-force-limit", type=float, default=100.0)
    parser.add_argument("--no-invert-gripper", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = benchmark(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
