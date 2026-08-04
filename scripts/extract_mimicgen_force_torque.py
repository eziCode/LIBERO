#!/usr/bin/env python3
"""Extract simulated end-effector force and torque from MimicGen datasets.

The released MimicGen HDF5 files do not store wrench observations. This script
replays every action in the environment described by the dataset metadata and
writes a compact HDF5 sidecar with matching demo names and timestep indices.

The output datasets are:

    data/<demo>/obs/robot0_eef_force   (T, 3), Newtons
    data/<demo>/obs/robot0_eef_torque  (T, 3), Newton-metres

Each sample is read immediately after applying the corresponding source action.
The wrench is expressed in the MuJoCo force/torque sensor frame attached to the
robot gripper (``gripper0_ft_frame`` for the released Panda datasets).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import robosuite as suite
import robosuite
import mimicgen
from robosuite.models.objects import MujocoObject

# MimicGen's composite objects use this helper from robosuite 1.4.1. The
# PyPI robosuite 1.4.0 package has the same underlying geometry properties but
# predates the convenience method.
if not hasattr(MujocoObject, "get_bounding_box_half_size"):
    def _get_bounding_box_half_size(self):
        return np.array(
            [self.horizontal_radius, self.horizontal_radius, 0.0]
        ) - self.bottom_offset

    MujocoObject.get_bounding_box_half_size = _get_bounding_box_half_size

# Importing this module registers the MimicGen environments with robosuite.
import mimicgen.envs.robosuite  # noqa: F401


FORCE_KEY = "robot0_eef_force"
TORQUE_KEY = "robot0_eef_torque"


def natural_demo_key(name: str) -> tuple[str, int | str]:
    prefix, separator, suffix = name.rpartition("_")
    if separator and suffix.isdigit():
        return prefix, int(suffix)
    return name, name


def source_digest(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_environment(env_meta: dict[str, Any]):
    kwargs = dict(env_meta["env_kwargs"])
    kwargs.update(
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=False,
        reward_shaping=False,
        hard_reset=False,
    )
    return suite.make(env_meta["env_name"], **kwargs)


def localize_robosuite_assets(model_xml: str) -> str:
    """Replace collector-machine asset paths with this robosuite install."""
    root = ET.fromstring(model_xml)
    robosuite_root = Path(robosuite.__file__).resolve().parent
    mimicgen_root = Path(mimicgen.__file__).resolve().parent
    asset = root.find("asset")
    if asset is None:
        return model_xml

    for element in (*asset.findall("mesh"), *asset.findall("texture")):
        old_file = element.get("file")
        if not old_file:
            continue
        parts = Path(old_file).parts
        if "mimicgen_envs" in parts:
            index = parts.index("mimicgen_envs")
            element.set("file", str(mimicgen_root.joinpath(*parts[index + 1 :])))
            continue
        try:
            # Dataset paths end in robosuite/models/assets/... . Preserve the
            # package-relative suffix while replacing the machine-specific root.
            index = max(i for i, part in enumerate(parts) if part == "robosuite")
        except ValueError:
            continue
        element.set("file", str(robosuite_root.joinpath(*parts[index + 1 :])))
    return ET.tostring(root, encoding="unicode")


def reset_to_demo(env, model_xml: str, initial_state: np.ndarray) -> None:
    env.reset()
    model_digest = hashlib.sha1(model_xml.encode("utf-8")).hexdigest()
    if getattr(env, "_force_torque_model_digest", None) != model_digest:
        env.reset_from_xml_string(localize_robosuite_assets(model_xml))
        env._force_torque_model_digest = model_digest
    env.sim.reset()
    env.sim.set_state_from_flattened(initial_state)
    env.sim.forward()


def replay_demo(
    env,
    states: np.ndarray,
    actions: np.ndarray,
    model_xml: str,
    correction_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    if len(states) != len(actions):
        raise ValueError(
            f"states/actions length mismatch: {len(states)} != {len(actions)}"
        )

    reset_to_demo(env, model_xml, states[0])
    forces = np.empty((len(actions), 3), dtype=np.float32)
    torques = np.empty((len(actions), 3), dtype=np.float32)
    errors: list[float] = []
    corrections = 0

    for index, action in enumerate(actions):
        env.step(action)
        forces[index] = np.asarray(env.robots[0].ee_force, dtype=np.float32)
        torques[index] = np.asarray(env.robots[0].ee_torque, dtype=np.float32)

        if index + 1 < len(states):
            replay_state = env.sim.get_state().flatten()
            error = float(np.linalg.norm(states[index + 1] - replay_state))
            errors.append(error)
            if error > correction_threshold:
                # Re-anchor playback to prevent numerical/version drift from
                # accumulating while preserving the controller's goal state.
                env.sim.set_state_from_flattened(states[index + 1])
                env.sim.forward()
                corrections += 1

    stats: dict[str, float | int] = {
        "max_state_error": max(errors, default=0.0),
        "mean_state_error": float(np.mean(errors)) if errors else 0.0,
        "state_corrections": corrections,
    }
    return forces, torques, stats


def initialize_output(
    output: h5py.File,
    source: Path,
    env_meta: dict[str, Any],
    correction_threshold: float,
) -> None:
    data = output.require_group("data")
    data.attrs["env_args"] = json.dumps(env_meta)
    output.attrs["format"] = "mimicgen_force_torque_sidecar_v1"
    output.attrs["source_file"] = source.name
    output.attrs["source_size_bytes"] = source.stat().st_size
    output.attrs["source_sha256"] = source_digest(source)
    output.attrs["sample_alignment"] = "after applying source action[t]"
    output.attrs["force_units"] = "N"
    output.attrs["torque_units"] = "N*m"
    output.attrs["sensor_frame"] = "gripper force/torque site (gripper0_ft_frame for Panda)"
    output.attrs["state_correction_threshold"] = correction_threshold
    output.attrs["created_by"] = Path(__file__).name


def extract_file(
    source_path: Path,
    output_path: Path,
    correction_threshold: float,
    max_demos: int | None,
    overwrite: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and output_path.exists():
        output_path.unlink()

    with h5py.File(source_path, "r") as source:
        env_meta = json.loads(source["data"].attrs["env_args"])
        demos = sorted(source["data"].keys(), key=natural_demo_key)
        if max_demos is not None:
            demos = demos[:max_demos]

        output_exists = output_path.exists()
        with h5py.File(output_path, "a") as output:
            if not output_exists:
                initialize_output(
                    output=output,
                    source=source_path,
                    env_meta=env_meta,
                    correction_threshold=correction_threshold,
                )
            elif output.attrs.get("source_file") != source_path.name:
                raise RuntimeError(
                    f"{output_path} belongs to a different source file; use --overwrite"
                )

            env = make_environment(env_meta)
            try:
                started = time.monotonic()
                for number, demo_name in enumerate(demos, start=1):
                    output_obs_path = f"data/{demo_name}/obs"
                    if (
                        f"{output_obs_path}/{FORCE_KEY}" in output
                        and f"{output_obs_path}/{TORQUE_KEY}" in output
                    ):
                        print(f"[{number}/{len(demos)}] {demo_name}: already complete")
                        continue

                    demo = source[f"data/{demo_name}"]
                    forces, torques, stats = replay_demo(
                        env=env,
                        states=demo["states"][()],
                        actions=demo["actions"][()],
                        model_xml=demo.attrs["model_file"],
                        correction_threshold=correction_threshold,
                    )

                    obs = output.require_group(output_obs_path)
                    for key in (FORCE_KEY, TORQUE_KEY):
                        if key in obs:
                            del obs[key]
                    obs.create_dataset(FORCE_KEY, data=forces, compression="gzip", shuffle=True)
                    obs.create_dataset(TORQUE_KEY, data=torques, compression="gzip", shuffle=True)
                    demo_out = output[f"data/{demo_name}"]
                    demo_out.attrs["num_samples"] = len(forces)
                    for key, value in stats.items():
                        demo_out.attrs[key] = value
                    output.flush()

                    elapsed = time.monotonic() - started
                    print(
                        f"[{number}/{len(demos)}] {demo_name}: {len(forces)} samples, "
                        f"max error={stats['max_state_error']:.6g}, "
                        f"corrections={stats['state_corrections']} ({elapsed:.1f}s elapsed)"
                    )

                completed = sum(
                    f"data/{name}/obs/{FORCE_KEY}" in output
                    and f"data/{name}/obs/{TORQUE_KEY}" in output
                    for name in demos
                )
                output.attrs["completed_demos"] = completed
                output.attrs["requested_demos"] = len(demos)
                output.attrs["processing_complete"] = completed == len(demos)
                output.flush()
            finally:
                env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="MimicGen HDF5 file(s)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for compact force/torque HDF5 sidecars",
    )
    parser.add_argument(
        "--correction-threshold",
        type=float,
        default=1e-2,
        help="Re-anchor to the next recorded state above this L2 replay error (default: 0.01)",
    )
    parser.add_argument("--max-demos", type=int, help="Process only the first N demos")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.correction_threshold < 0:
        raise ValueError("--correction-threshold must be non-negative")
    for source in args.inputs:
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        output = args.output_dir.resolve() / source.name
        print(f"Extracting {source} -> {output}")
        extract_file(
            source_path=source,
            output_path=output,
            correction_threshold=args.correction_threshold,
            max_demos=args.max_demos,
            overwrite=args.overwrite,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; completed demos are safely resumable.", file=sys.stderr)
        raise SystemExit(130)
