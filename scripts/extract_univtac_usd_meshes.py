#!/usr/bin/env python3
"""Extract triangle meshes from UniVTAC's binary USD object assets to OBJ."""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path


def array_body(text: str, declaration: str) -> str:
    match = re.search(rf"{re.escape(declaration)}\s*=\s*\[(.*?)\]\s*\n", text, re.S)
    if match is None:
        raise ValueError(f"could not locate {declaration!r}")
    return match.group(1)


def extract(source: Path, output: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="univtac_usd_") as directory:
        usda = Path(directory) / f"{source.stem}.usda"
        subprocess.run(["usdcat", str(source), "-o", str(usda)], check=True)
        text = usda.read_text(encoding="utf-8")

    points = [
        tuple(float(value) for value in item.split(","))
        for item in re.findall(r"\(([^()]*)\)", array_body(text, "point3f[] points"))
    ]
    counts = [int(value) for value in re.findall(r"-?\d+", array_body(text, "int[] faceVertexCounts"))]
    indices = [int(value) for value in re.findall(r"-?\d+", array_body(text, "int[] faceVertexIndices"))]
    if sum(counts) != len(indices):
        raise ValueError(f"face topology mismatch: sum(counts)={sum(counts)} != {len(indices)}")

    lines = [f"# Extracted from {source.as_posix()}", f"o {source.stem}"]
    lines.extend(f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in points)
    cursor = 0
    for count in counts:
        face = indices[cursor : cursor + count]
        cursor += count
        lines.append("f " + " ".join(str(index + 1) for index in face))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{source.name}: {len(points)} vertices, {len(counts)} faces -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("deps/UniVTAC/assets/objects"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("univtac_mujoco_envs/assets/meshes")
    )
    args = parser.parse_args()
    for name in ("HDMI", "HDMISlot"):
        extract(args.source_dir / f"{name}.usd", args.output_dir / f"{name}.obj")


if __name__ == "__main__":
    main()
