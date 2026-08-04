#!/usr/bin/env python3
"""Generate MimicGen videos with highlighted end-effector force and torque.

This follows the visual format of ``generate_force_videos.py`` while joining
the original MimicGen RGB observations to the compact force/torque sidecars.
By default, one ``demo_0`` video is generated for every matching task file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np


FORCE_KEY = "robot0_eef_force"
TORQUE_KEY = "robot0_eef_torque"
IMAGE_KEY = "agentview_image"


def render_task_video(
    source_path: Path,
    force_path: Path,
    output_path: Path,
    demo: str,
    fps: float,
    scale: int,
) -> int:
    with h5py.File(source_path, "r") as source, h5py.File(force_path, "r") as force_file:
        source_obs_path = f"data/{demo}/obs"
        force_obs_path = f"data/{demo}/obs"
        if source_obs_path not in source:
            raise KeyError(f"{demo!r} not found in {source_path}")
        if force_obs_path not in force_file:
            raise KeyError(f"{demo!r} not found in {force_path}")

        source_obs = source[source_obs_path]
        force_obs = force_file[force_obs_path]
        images = source_obs[IMAGE_KEY]
        forces = force_obs[FORCE_KEY]
        torques = force_obs[TORQUE_KEY]

        lengths = (len(images), len(forces), len(torques))
        if len(set(lengths)) != 1:
            raise ValueError(
                f"frame/force/torque length mismatch for {source_path.name} {demo}: {lengths}"
            )
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"unexpected image shape: {images.shape}")
        if forces.shape[1:] != (3,) or torques.shape[1:] != (3,):
            raise ValueError(
                f"unexpected force/torque shapes: {forces.shape}, {torques.shape}"
            )

        height, width = images.shape[1:3]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width * scale, height * scale),
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not open video writer for {output_path}")

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        highlight_color = (0, 255, 0)  # Green in OpenCV BGR order.
        thickness = 1

        try:
            for index in range(len(images)):
                # MimicGen HDF5 images are already stored right-side up. Convert
                # RGB -> BGR, then upscale so the overlay is readable.
                frame = np.asarray(images[index])[..., ::-1].copy()
                frame = cv2.resize(
                    frame,
                    (width * scale, height * scale),
                    interpolation=cv2.INTER_LINEAR,
                )

                ef = np.asarray(forces[index])
                et = np.asarray(torques[index])
                lines = [
                    f"EEF F: [{ef[0]:.1f}, {ef[1]:.1f}, {ef[2]:.1f}] N",
                    f"EEF T: [{et[0]:.1f}, {et[1]:.1f}, {et[2]:.1f}] Nm",
                ]

                y0, dy = 25, 20
                for line_number, line in enumerate(lines):
                    position = (10, y0 + line_number * dy)
                    cv2.putText(
                        frame,
                        line,
                        position,
                        font,
                        font_scale,
                        (0, 0, 0),
                        thickness + 1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        line,
                        position,
                        font,
                        font_scale,
                        highlight_color,
                        thickness,
                        cv2.LINE_AA,
                    )
                writer.write(frame)
        finally:
            writer.release()

    return lengths[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("datasets/mimicgen"),
        help="Directory containing original MimicGen HDF5 files",
    )
    parser.add_argument(
        "--force-dir",
        type=Path,
        default=Path("datasets/mimicgen-force-torque"),
        help="Directory containing force/torque HDF5 sidecars",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/mimicgen-force-torque/videos"),
        help="Output directory for MP4 files",
    )
    parser.add_argument("--demo", default="demo_0", help="Demo key to render")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--scale", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.scale <= 0:
        raise ValueError("--scale must be positive")

    force_files = sorted(args.force_dir.glob("*.hdf5"))
    if not force_files:
        raise FileNotFoundError(f"no HDF5 sidecars found in {args.force_dir}")

    generated = 0
    for force_path in force_files:
        source_path = args.source_dir / force_path.name
        if not source_path.is_file():
            print(f"Skipping {force_path.name}: source file not found at {source_path}")
            continue
        output_path = args.output_dir / f"{force_path.stem}_{args.demo}.mp4"
        print(f"Generating {output_path} ...")
        frame_count = render_task_video(
            source_path=source_path,
            force_path=force_path,
            output_path=output_path,
            demo=args.demo,
            fps=args.fps,
            scale=args.scale,
        )
        print(f"Saved {output_path} ({frame_count} frames at {args.fps:g} FPS)")
        generated += 1

    if generated == 0:
        raise RuntimeError("no videos were generated")
    print(f"Generated {generated} video(s) in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
