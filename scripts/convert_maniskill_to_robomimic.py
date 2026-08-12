#!/usr/bin/env python3
"""Package ManiSkill demonstrations in a robomimic / LIBERO-style HDF5.

This is a schema conversion only. It never creates a MuJoCo environment and
does not retarget actions or alter task behavior. ManiSkill simulator states
remain SAPIEN / PhysX states; consumers must not pass them to MuJoCo set_state.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np


def natural_key(name: str) -> tuple[str, int | str]:
    prefix, separator, suffix = name.rpartition("_")
    return (prefix, int(suffix)) if separator and suffix.isdigit() else (name, name)


def datasets(group: h5py.Group, prefix: str = "") -> Iterator[tuple[str, h5py.Dataset]]:
    for name in sorted(group):
        item = group[name]
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(item, h5py.Dataset):
            yield path, item
        else:
            yield from datasets(item, path)


def safe_obs_key(path: str) -> str:
    return path.replace("/", "__")


def time_aligned(array: np.ndarray, length: int, offset: int = 0) -> np.ndarray:
    """Select T samples from arrays recorded with either T or T+1 entries."""
    if array.ndim == 0:
        raise ValueError("observation/state datasets must have a time dimension")
    if array.shape[0] < length + offset:
        raise ValueError(
            f"dataset has {array.shape[0]} steps, need at least {length + offset}"
        )
    return array[offset : offset + length]


def state_components(trajectory: h5py.Group) -> list[tuple[str, np.ndarray]]:
    group = trajectory.get("env_states")
    if group is None:
        return []
    return [(path, np.asarray(value)) for path, value in datasets(group)]


def observation_components(trajectory: h5py.Group) -> list[tuple[str, np.ndarray]]:
    group = trajectory.get("obs")
    if group is None:
        return []
    return [(path, np.asarray(value)) for path, value in datasets(group)]


def flattened_state(components: list[tuple[str, np.ndarray]], length: int) -> np.ndarray:
    flat = []
    for _, value in components:
        aligned = time_aligned(value, length)
        flat.append(aligned.reshape(length, -1))
    if not flat:
        return np.empty((length, 0), dtype=np.float32)
    return np.concatenate(flat, axis=1)


def write_obs(
    demo: h5py.Group,
    name: str,
    components: list[tuple[str, np.ndarray]],
    length: int,
    offset: int,
) -> None:
    output = demo.create_group(name)
    for path, value in components:
        output.create_dataset(
            safe_obs_key(path),
            data=time_aligned(value, length, offset),
            compression="gzip",
        )


def load_metadata(path: Path) -> dict:
    metadata_path = path.with_suffix(".json")
    if not metadata_path.is_file():
        return {}
    with metadata_path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def episode_metadata(metadata: dict, episode_id: int) -> dict:
    episodes = metadata.get("episodes", [])
    for episode in episodes:
        if int(episode.get("episode_id", -1)) == episode_id:
            return episode
    return {}


def convert(input_path: Path, output_path: Path, count: int | None) -> None:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    metadata = load_metadata(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    if temporary.exists():
        temporary.unlink()

    try:
        with h5py.File(input_path, "r") as source, h5py.File(temporary, "w") as target:
            keys = sorted((key for key in source if key.startswith("traj_")), key=natural_key)
            if count is not None:
                keys = keys[:count]
            if not keys:
                raise ValueError(f"no traj_* groups found in {input_path}")

            data = target.create_group("data")
            env_info = metadata.get("env_info", {})
            env_id = env_info.get("env_id", "unknown")
            data.attrs["env_args"] = json.dumps(
                {
                    "env_name": env_id,
                    "type": "maniskill",
                    "env_kwargs": env_info.get("env_kwargs", {}),
                },
                sort_keys=True,
            )
            data.attrs["source_format"] = "maniskill"
            data.attrs["state_backend"] = "sapien_physx"
            data.attrs["states_are_mujoco_compatible"] = False
            data.attrs["source_file"] = str(input_path)

            total = 0
            for output_index, key in enumerate(keys):
                source_demo = source[key]
                actions = np.asarray(source_demo["actions"])
                length = len(actions)
                demo = data.create_group(f"demo_{output_index}")
                demo.create_dataset("actions", data=actions, compression="gzip")

                terminated = np.asarray(source_demo.get("terminated", np.zeros(length, bool)))
                truncated = np.asarray(source_demo.get("truncated", np.zeros(length, bool)))
                dones = np.logical_or(terminated[:length], truncated[:length])
                demo.create_dataset("dones", data=dones.astype(np.uint8), compression="gzip")
                rewards = np.asarray(source_demo.get("rewards", np.zeros(length, np.float32)))
                demo.create_dataset("rewards", data=rewards[:length], compression="gzip")

                states = state_components(source_demo)
                demo.create_dataset("states", data=flattened_state(states, length), compression="gzip")

                # Prefer recorded observations. State components are the lossless
                # fallback for state-only ManiSkill demonstrations.
                observations = observation_components(source_demo) or states
                write_obs(demo, "obs", observations, length, offset=0)
                if observations and all(value.shape[0] >= length + 1 for _, value in observations):
                    write_obs(demo, "next_obs", observations, length, offset=1)

                episode_id = int(key.rsplit("_", 1)[1])
                ep_meta = episode_metadata(metadata, episode_id)
                demo.attrs["num_samples"] = length
                demo.attrs["source_trajectory"] = key
                demo.attrs["source_episode_metadata"] = json.dumps(ep_meta, sort_keys=True)
                demo.attrs["control_mode"] = ep_meta.get("control_mode", "unknown")
                demo.attrs["state_component_order"] = json.dumps([path for path, _ in states])
                total += length
                print(f"{key} -> demo_{output_index}: {length} samples")

            data.attrs["num_demos"] = len(keys)
            data.attrs["total"] = total
            target.flush()
        os.replace(temporary, output_path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    print(f"Wrote {len(keys)} demos / {total} samples to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, help="convert only the first N episodes")
    args = parser.parse_args()
    if args.count is not None and args.count <= 0:
        parser.error("--count must be positive")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    convert(arguments.input, arguments.output, arguments.count)
