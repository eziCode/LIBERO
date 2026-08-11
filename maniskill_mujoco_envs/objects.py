"""Procedural MuJoCo objects matching selected ManiSkill task geometry."""

from __future__ import annotations

import numpy as np

from robosuite.models.objects import CompositeObject


DEFAULT_FRICTION = (1.0, 5e-3, 1e-4)


def two_color_peg(name: str, half_length: float, radius: float) -> CompositeObject:
    """PegInsertionSide peg: one collision box and two colored visual halves."""
    return CompositeObject(
        name=name,
        total_size=[half_length, radius, radius],
        geom_types=["box", "box", "box"],
        geom_sizes=[
            [half_length, radius, radius],
            [half_length / 2, radius, radius],
            [half_length / 2, radius, radius],
        ],
        geom_locations=[
            [0, 0, 0],
            [half_length / 2, 0, 0],
            [-half_length / 2, 0, 0],
        ],
        geom_names=["collision", "head", "tail"],
        geom_rgbas=[[0.93, 0.45, 0.34, 1], [0.93, 0.45, 0.34, 1], [0.93, 0.96, 0.98, 1]],
        geom_frictions=[DEFAULT_FRICTION] * 3,
        density=1000.0,
        locations_relative_to_center=True,
        joints="default",
        obj_types=["collision", "visual", "visual"],
        duplicate_collision_geoms=False,
    )


def box_with_hole(
    name: str,
    inner_radius: float,
    outer_radius: float,
    depth: float,
    center=(0.0, 0.0),
) -> CompositeObject:
    """Four boxes forming the square hole used by PegInsertionSide."""
    half_center = np.asarray(center, dtype=np.float64) * 0.5
    thickness = (outer_radius - inner_radius) * 0.5
    offset = thickness + inner_radius
    sizes = [
        [depth, thickness - half_center[0], outer_radius],
        [depth, thickness + half_center[0], outer_radius],
        [depth, outer_radius, thickness - half_center[1]],
        [depth, outer_radius, thickness + half_center[1]],
    ]
    locations = [
        [0, offset + half_center[0], 0],
        [0, -offset + half_center[0], 0],
        [0, 0, offset + half_center[1]],
        [0, 0, -offset + half_center[1]],
    ]
    return CompositeObject(
        name=name,
        total_size=[depth, outer_radius, outer_radius],
        geom_types=["box"] * 4,
        geom_sizes=sizes,
        geom_locations=locations,
        geom_names=["top", "bottom", "left", "right"],
        geom_rgbas=[[1.0, 0.82, 0.54, 1.0]] * 4,
        geom_frictions=[DEFAULT_FRICTION] * 4,
        density=1000.0,
        locations_relative_to_center=True,
        joints=None,
        obj_types="all",
    )


def charger(
    name: str,
    peg_size=(8e-3, 0.75e-3, 3.2e-3),
    base_size=(2e-2, 1.5e-2, 1.2e-2),
    gap=7e-3,
) -> CompositeObject:
    """Two-prong charger assembled from the same boxes as ManiSkill."""
    peg_size = np.asarray(peg_size, dtype=np.float64)
    base_size = np.asarray(base_size, dtype=np.float64)
    locations = [
        [peg_size[0], gap, 0],
        [peg_size[0], -gap, 0],
        [-base_size[0], 0, 0],
    ]
    return CompositeObject(
        name=name,
        total_size=[2 * base_size[0], base_size[1], base_size[2]],
        geom_types=["box"] * 3,
        geom_sizes=[peg_size, peg_size, base_size],
        geom_locations=locations,
        geom_names=["positive_prong", "negative_prong", "base"],
        geom_rgbas=[[0.85, 0.85, 0.85, 1], [0.85, 0.85, 0.85, 1], [1, 1, 1, 1]],
        geom_frictions=[DEFAULT_FRICTION] * 3,
        density=1000.0,
        locations_relative_to_center=True,
        joints="default",
        obj_types="all",
    )


def charger_receptacle(
    name: str,
    peg_size=(8e-3, 1.25e-3, 3.7e-3),
    receptacle_size=(1e-2, 5e-2, 5e-2),
    gap=7e-3,
) -> CompositeObject:
    """Kinematic receptacle with the same five collision boxes as ManiSkill."""
    peg_size = np.asarray(peg_size, dtype=np.float64)
    receptacle_size = np.asarray(receptacle_size, dtype=np.float64)
    sy = 0.5 * (receptacle_size[1] - peg_size[1] - gap)
    sz = 0.5 * (receptacle_size[2] - peg_size[2])
    dx = -receptacle_size[0]
    dy = peg_size[1] + gap + sy
    dz = peg_size[2] + sz
    locations = [
        [dx, 0, dz],
        [dx, 0, -dz],
        [dx, dy, 0],
        [dx, -dy, 0],
        [-receptacle_size[0], 0, 0],
    ]
    sizes = [
        [receptacle_size[0], receptacle_size[1], sz],
        [receptacle_size[0], receptacle_size[1], sz],
        [receptacle_size[0], sy, receptacle_size[2]],
        [receptacle_size[0], sy, receptacle_size[2]],
        [receptacle_size[0], gap - peg_size[1], peg_size[2]],
    ]
    return CompositeObject(
        name=name,
        total_size=receptacle_size,
        geom_types=["box"] * 5,
        geom_sizes=sizes,
        geom_locations=locations,
        geom_names=["top", "bottom", "left", "right", "back"],
        geom_rgbas=[[1.0, 0.82, 0.22, 1.0]] * 5,
        geom_frictions=[DEFAULT_FRICTION] * 5,
        density=1000.0,
        locations_relative_to_center=True,
        joints=None,
        obj_types="all",
    )
