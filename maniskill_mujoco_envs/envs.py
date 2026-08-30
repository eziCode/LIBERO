"""Distinct robosuite environments ported from ManiSkill procedural tasks."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject, BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.transform_utils import convert_quat, quat2mat

from .objects import box_with_hole, charger, charger_receptacle, two_color_peg


TABLE_OFFSET = np.array([0.0, 0.0, 0.8])
# The robosuite Panda MJCF and ManiSkill Panda URDF use base-link origins that
# differ by 6.9 mm. This calibrated base height makes their TCP positions agree
# across the same joint trajectory (sub-millimetre residual error).
PANDA_BASE_Z_CORRECTION = -0.0069
TABLE_FULL_SIZE = (0.8, 0.8, 0.05)
TABLE_FRICTION = (1.0, 5e-3, 1e-4)
CUBE_HALF_SIZE = 0.02


def yaw_quat(angle: float) -> np.ndarray:
    return np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])


def pose_matrix(position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = quat2mat(convert_quat(quat_wxyz, to="xyzw"))
    matrix[:3, 3] = position
    return matrix


def sample_separated_xy(count: int, low, high, min_distance: float) -> list[np.ndarray]:
    values: list[np.ndarray] = []
    for _ in range(count):
        for _attempt in range(1000):
            candidate = np.random.uniform(low, high)
            if all(np.linalg.norm(candidate - other) >= min_distance for other in values):
                values.append(candidate)
                break
        else:
            raise RuntimeError("could not sample non-overlapping object positions")
    return values


class ManiSkillMujocoBase(SingleArmEnv):
    """Common table, Panda, observation, and source-pose interface."""

    maniskill_task_id = "UNSET"
    environment_label = "UNSET"
    # ManiSkill's tabletop surface is z=0; robosuite's is z=0.8.
    source_world_offset = TABLE_OFFSET.copy()

    def __init__(
        self,
        robots="Panda",
        env_configuration="default",
        controller_configs=None,
        gripper_types="PandaGripper",
        initialization_noise=None,
        use_camera_obs=False,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera="agentview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=250,
        ignore_done=False,
        hard_reset=False,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        renderer_config=None,
        **kwargs,
    ):
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported arguments for {self.__class__.__name__}: {unknown}")
        self.table_full_size = TABLE_FULL_SIZE
        self.table_friction = TABLE_FRICTION
        self.table_offset = TABLE_OFFSET.copy()
        self.use_object_obs = use_object_obs
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.task_objects: OrderedDict[str, Any] = OrderedDict()
        self.dynamic_object_names: set[str] = set()
        self.fixed_object_names: set[str] = set()
        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            # ManiSkill mounts the Panda directly at [-0.615, 0, 0]. The
            # robosuite default Panda pedestal raises the arm by ~0.913 m and
            # changes every robot-to-object transform, so it must not be used.
            mount_types=None,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def _build_task_objects(self) -> None:
        raise NotImplementedError

    def _load_model(self):
        super()._load_model()
        # The legacy robosuite Panda CAD contains saturated red / green / blue
        # sub-materials. In this mesh export they cover large shell patches,
        # producing a false multicolored robot. A Franka Panda uses neutral
        # white and charcoal shells, so normalize only saturated materials and
        # retain the original neutral whites / grays.
        for material in self.robots[0].robot_model.asset.iter("material"):
            rgba_text = material.get("rgba")
            if not rgba_text:
                continue
            rgba = np.fromstring(rgba_text, sep=" ")
            if rgba.size == 4 and np.ptp(rgba[:3]) > 0.2:
                luminance = float(np.dot(rgba[:3], [0.2126, 0.7152, 0.0722]))
                neutral = 0.92 if luminance >= 0.55 else 0.18
                material.set("rgba", f"{neutral} {neutral} {neutral} {rgba[3]}")
        # MuJoCo 2.3.7's macOS binding exposes mjvOption.geomgroup as a copy,
        # so robosuite cannot hide group-0 collision meshes at render time.
        # Keep collision/contact properties intact while making only the
        # robot's collision geometry visually transparent in the MJCF.
        for geom in self.robots[0].robot_model.worldbody.iter("geom"):
            if geom.get("group", "0") == "0":
                geom.set("rgba", "0 0 0 0")
        # robosuite's Panda grip site is rotated -90 degrees about its local z
        # relative to ManiSkill's panda_hand_tcp. Rotate the site frame back so
        # TCP quaternions and Cartesian controller axes share ManiSkill's
        # convention. This does not move the site position.
        for site in self.robots[0].robot_model.worldbody.iter("site"):
            if site.get("name", "").endswith("grip_site"):
                site.set("quat", "0.7071067811865476 0 0 0.7071067811865475")
        # The whole ManiSkill workspace is translated by TABLE_OFFSET in this
        # MuJoCo port. Apply that translation to the robot as well as objects.
        self.robots[0].robot_model.set_base_xpos(
            (
                -0.615,
                0.0,
                float(self.table_offset[2] + PANDA_BASE_Z_CORRECTION),
            )
        )
        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])
        self.task_objects = OrderedDict()
        self.dynamic_object_names = set()
        self.fixed_object_names = set()
        self._build_task_objects()
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=list(self.task_objects.values()),
        )
        # See the macOS MuJoCo 2.3.7 geomgroup note above. Apply the intended
        # render_collision_mesh=False behavior to the fully assembled task so
        # collision and visual copies do not z-fight and contaminate colors.
        if not self.render_collision_mesh:
            for geom in self.model.worldbody.iter("geom"):
                if geom.get("group", "0") == "0":
                    rgba = np.fromstring(geom.get("rgba", "0.5 0.5 0.5 1"), sep=" ")
                    if rgba.size == 4:
                        rgba[3] = 0.0
                        geom.set("rgba", " ".join(str(value) for value in rgba))

    def _setup_references(self):
        super()._setup_references()
        self.object_body_ids = {
            name: self.sim.model.body_name2id(obj.root_body)
            for name, obj in self.task_objects.items()
        }

    def get_task_object_pose(self, name: str) -> np.ndarray:
        body_id = self.object_body_ids[name]
        return np.concatenate(
            (self.sim.data.body_xpos[body_id].copy(), self.sim.data.body_xquat[body_id].copy())
        )

    def set_task_object_pose(self, name: str, pose: np.ndarray) -> None:
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (7,):
            raise ValueError(f"expected xyz+wxyz pose for {name}, got {pose.shape}")
        obj = self.task_objects[name]
        if name in self.dynamic_object_names:
            self.sim.data.set_joint_qpos(obj.joints[0], pose)
        elif name in self.fixed_object_names:
            body_id = self.object_body_ids[name]
            self.sim.model.body_pos[body_id] = pose[:3]
            self.sim.model.body_quat[body_id] = pose[3:]
        else:
            raise KeyError(f"{name} has no registered pose mode")
        self.sim.forward()

    def _set_source_pose(self, name: str, position, quat=None) -> None:
        position = np.asarray(position, dtype=np.float64)
        quat = yaw_quat(0.0) if quat is None else np.asarray(quat, dtype=np.float64)
        self.set_task_object_pose(name, np.concatenate((position, quat)))

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            self._reset_task()

    def _reset_task(self) -> None:
        raise NotImplementedError

    def configure_source_episode(self, metadata: dict[str, Any]) -> None:
        """Apply source metadata before reset; overridden by variable-geometry tasks."""

    def _object_static(self, name: str, linear=1e-2, angular=0.5) -> bool:
        body_id = self.object_body_ids[name]
        body_name = self.sim.model.body_id2name(body_id)
        velocity = self.sim.data.get_body_xvelp(body_name)
        angular_velocity = self.sim.data.get_body_xvelr(body_name)
        return np.linalg.norm(velocity) <= linear and np.linalg.norm(angular_velocity) <= angular

    def reward(self, action=None):
        reward = float(self._check_success())
        return reward if self.reward_scale is None else reward * self.reward_scale

    def _setup_observables(self):
        observables = super()._setup_observables()
        if not self.use_object_obs:
            return observables

        modality = "maniskill_mujoco_object"
        for object_name in self.task_objects:
            def make_position_sensor(name):
                @sensor(modality=modality)
                def object_position(obs_cache):
                    return self.sim.data.body_xpos[self.object_body_ids[name]].copy()

                return object_position

            def make_quaternion_sensor(name):
                @sensor(modality=modality)
                def object_quaternion(obs_cache):
                    return convert_quat(
                        self.sim.data.body_xquat[self.object_body_ids[name]].copy(),
                        to="xyzw",
                    )

                return object_quaternion

            for suffix, observable_sensor in (
                ("pos", make_position_sensor(object_name)),
                ("quat", make_quaternion_sensor(object_name)),
            ):
                observable_name = f"{object_name}_{suffix}"
                observables[observable_name] = Observable(
                    name=observable_name,
                    sensor=observable_sensor,
                    sampling_rate=self.control_freq,
                )

        @sensor(modality="maniskill_mujoco_wrench")
        def robot0_eef_force(obs_cache):
            return np.asarray(self.robots[0].ee_force).copy()

        @sensor(modality="maniskill_mujoco_wrench")
        def robot0_eef_torque(obs_cache):
            return np.asarray(self.robots[0].ee_torque).copy()

        for observable_sensor in (robot0_eef_force, robot0_eef_torque):
            name = observable_sensor.__name__
            observables[name] = Observable(
                name=name,
                sensor=observable_sensor,
                sampling_rate=self.control_freq,
            )
        return observables


class ManiSkillMujocoPickCube(ManiSkillMujocoBase):
    maniskill_task_id = "PickCube-v1"
    environment_label = "ManiSkill PickCube-v1 MuJoCo Port"

    def __init__(self, goal_threshold=0.025, **kwargs):
        self.goal_threshold = goal_threshold
        self.goal_position = np.zeros(3)
        kwargs.setdefault("horizon", 50)
        super().__init__(**kwargs)

    def _build_task_objects(self):
        self.cube = BoxObject(
            name="cube",
            size=[CUBE_HALF_SIZE] * 3,
            rgba=[1, 0, 0, 1],
            density=1000.0,
        )
        self.goal_site = BallObject(
            name="goal_site",
            size=[self.goal_threshold],
            rgba=[0, 1, 0, 0.35],
            joints=None,
            obj_type="visual",
        )
        self.task_objects.update(cube=self.cube, goal_site=self.goal_site)
        self.dynamic_object_names.add("cube")
        self.fixed_object_names.add("goal_site")

    def _reset_task(self):
        cube_xy = np.random.uniform(-0.1, 0.1, size=2)
        cube_pos = np.r_[cube_xy, self.table_offset[2] + CUBE_HALF_SIZE]
        goal_xy = np.random.uniform(-0.1, 0.1, size=2)
        goal_pos = np.r_[goal_xy, cube_pos[2] + np.random.uniform(0, 0.3)]
        self._set_source_pose("cube", cube_pos, yaw_quat(np.random.uniform(-np.pi, np.pi)))
        self._set_source_pose("goal_site", goal_pos)
        self.goal_position = goal_pos

    def set_task_object_pose(self, name: str, pose: np.ndarray) -> None:
        super().set_task_object_pose(name, pose)
        if name == "goal_site":
            self.goal_position = np.asarray(pose[:3], dtype=np.float64).copy()

    def _check_success(self):
        cube_pos = self.get_task_object_pose("cube")[:3]
        robot_static = np.linalg.norm(self.robots[0]._joint_velocities) < 0.2
        return np.linalg.norm(cube_pos - self.goal_position) <= self.goal_threshold and robot_static


class ManiSkillMujocoStackCube(ManiSkillMujocoBase):
    maniskill_task_id = "StackCube-v1"
    environment_label = "ManiSkill StackCube-v1 MuJoCo Port"

    def __init__(self, **kwargs):
        # ManiSkill registers StackCube-v1 with max_episode_steps=50.
        kwargs.setdefault("horizon", 50)
        super().__init__(**kwargs)

    def _build_task_objects(self):
        self.cubeA = BoxObject(name="cubeA", size=[0.02] * 3, rgba=[1, 0, 0, 1], density=1000.0)
        self.cubeB = BoxObject(name="cubeB", size=[0.02] * 3, rgba=[0, 1, 0, 1], density=1000.0)
        self.task_objects.update(cubeA=self.cubeA, cubeB=self.cubeB)
        self.dynamic_object_names.update(self.task_objects)

    def _reset_task(self):
        positions = sample_separated_xy(2, [-0.1, -0.2], [0.1, 0.2], 0.06)
        for name, xy in zip(("cubeA", "cubeB"), positions):
            self._set_source_pose(
                name,
                np.r_[xy, self.table_offset[2] + CUBE_HALF_SIZE],
                yaw_quat(np.random.uniform(-np.pi, np.pi)),
            )

    def _check_success(self):
        pos_a = self.get_task_object_pose("cubeA")[:3]
        pos_b = self.get_task_object_pose("cubeB")[:3]
        offset = pos_a - pos_b
        on_top = np.linalg.norm(offset[:2]) <= np.sqrt(2) * 0.02 + 0.005
        on_top &= abs(offset[2] - 0.04) <= 0.005
        released = not self._check_grasp(self.robots[0].gripper, self.cubeA)
        return on_top and self._object_static("cubeA") and released


class ManiSkillMujocoStackPyramid(ManiSkillMujocoBase):
    maniskill_task_id = "StackPyramid-v1"
    environment_label = "ManiSkill StackPyramid-v1 MuJoCo Port"

    def __init__(self, **kwargs):
        kwargs.setdefault("horizon", 250)
        super().__init__(**kwargs)

    def _build_task_objects(self):
        self.cubeA = BoxObject(name="cubeA", size=[0.02] * 3, rgba=[1, 0, 0, 1], density=1000.0)
        self.cubeB = BoxObject(name="cubeB", size=[0.02] * 3, rgba=[0, 1, 0, 1], density=1000.0)
        self.cubeC = BoxObject(name="cubeC", size=[0.02] * 3, rgba=[0, 0, 1, 1], density=1000.0)
        self.task_objects.update(cubeA=self.cubeA, cubeB=self.cubeB, cubeC=self.cubeC)
        self.dynamic_object_names.update(self.task_objects)

    def _reset_task(self):
        positions = sample_separated_xy(3, [-0.1, -0.2], [0.1, 0.2], 0.058)
        for name, xy in zip(("cubeA", "cubeB", "cubeC"), positions):
            self._set_source_pose(
                name,
                np.r_[xy, self.table_offset[2] + CUBE_HALF_SIZE],
                yaw_quat(np.random.uniform(-np.pi, np.pi)),
            )

    def _check_success(self):
        a = self.get_task_object_pose("cubeA")[:3]
        b = self.get_task_object_pose("cubeB")[:3]
        c = self.get_task_object_pose("cubeC")[:3]
        xy_threshold = np.linalg.norm([0.04, 0.04]) + 0.005

        def relation(offset, object_name, obj, require_top):
            close_xy = np.linalg.norm(offset[:2]) <= xy_threshold
            correct_height = abs(offset[2]) > 0.02 if require_top else True
            released = not self._check_grasp(self.robots[0].gripper, obj)
            return close_xy and correct_height and self._object_static(object_name) and released

        return (
            relation(a - b, "cubeA", self.cubeA, require_top=False)
            and relation(b - c, "cubeC", self.cubeC, require_top=True)
            and relation(a - c, "cubeC", self.cubeC, require_top=True)
        )


class ManiSkillMujocoPegInsertionSide(ManiSkillMujocoBase):
    maniskill_task_id = "PegInsertionSide-v1"
    environment_label = "ManiSkill PegInsertionSide-v1 MuJoCo Port"
    clearance = 0.003
    insertion_depth_threshold = -0.020

    def __init__(self, geometry_seed=0, peg_half_length=None, peg_radius=None, **kwargs):
        self.geometry_seed = int(geometry_seed)
        self._peg_half_length_override = peg_half_length
        self._peg_radius_override = peg_radius
        self._set_geometry_from_seed(self.geometry_seed)
        # ManiSkill registers PegInsertionSide-v1 with max_episode_steps=100.
        kwargs.setdefault("horizon", 100)
        super().__init__(**kwargs)

    def _set_geometry_from_seed(self, seed: int) -> None:
        rng = np.random.RandomState(seed)
        self.peg_half_length = float(
            rng.uniform(0.085, 0.125)
            if self._peg_half_length_override is None
            else self._peg_half_length_override
        )
        self.peg_radius = float(
            rng.uniform(0.015, 0.025)
            if self._peg_radius_override is None
            else self._peg_radius_override
        )
        self.hole_center = (
            0.5
            * (self.peg_half_length - self.peg_radius)
            * rng.uniform(-1, 1, size=2)
        )
        self.hole_radius = self.peg_radius + self.clearance

    def configure_source_episode(self, metadata: dict[str, Any]) -> None:
        seed = int(metadata.get("episode_seed", self.geometry_seed))
        if seed != self.geometry_seed:
            self.geometry_seed = seed
            self._set_geometry_from_seed(seed)
        # Peg geometry is randomized on every ManiSkill reconfiguration. A
        # hard reset rebuilds the MuJoCo XML with the matching seeded values.
        self.hard_reset = True

    def _build_task_objects(self):
        self.peg = two_color_peg("peg", self.peg_half_length, self.peg_radius)
        self.box_with_hole = box_with_hole(
            "box_with_hole",
            inner_radius=self.hole_radius,
            outer_radius=self.peg_half_length,
            depth=self.peg_half_length,
            center=self.hole_center,
        )
        self.task_objects.update(peg=self.peg, box_with_hole=self.box_with_hole)
        self.dynamic_object_names.add("peg")
        self.fixed_object_names.add("box_with_hole")

    def _reset_task(self):
        peg_xy = np.random.uniform([-0.1, -0.3], [0.1, 0.0])
        box_xy = np.random.uniform([-0.05, 0.2], [0.05, 0.4])
        peg_yaw = np.random.uniform(np.pi / 2 - np.pi / 3, np.pi / 2 + np.pi / 3)
        box_yaw = np.random.uniform(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8)
        self._set_source_pose(
            "peg",
            np.r_[peg_xy, self.table_offset[2] + self.peg_radius],
            yaw_quat(peg_yaw),
        )
        self._set_source_pose(
            "box_with_hole",
            np.r_[box_xy, self.table_offset[2] + self.peg_half_length],
            yaw_quat(box_yaw),
        )

    def peg_head_at_hole(self) -> np.ndarray:
        peg_pose = self.get_task_object_pose("peg")
        box_pose = self.get_task_object_pose("box_with_hole")
        peg_head = pose_matrix(peg_pose[:3], peg_pose[3:])
        peg_head[:3, 3] += peg_head[:3, :3] @ np.array([self.peg_half_length, 0, 0])
        hole = pose_matrix(box_pose[:3], box_pose[3:])
        hole[:3, 3] += hole[:3, :3] @ np.array([0, self.hole_center[0], self.hole_center[1]])
        return (np.linalg.inv(hole) @ peg_head)[:3, 3]

    def _check_success(self):
        relative = self.peg_head_at_hole()
        return (
            relative[0] >= self.insertion_depth_threshold
            and abs(relative[1]) <= self.hole_radius
            and abs(relative[2]) <= self.hole_radius
        )


class ManiSkillMujocoPlugCharger(ManiSkillMujocoBase):
    maniskill_task_id = "PlugCharger-v1"
    environment_label = "ManiSkill PlugCharger-v1 MuJoCo Port"
    base_size = np.array([2e-2, 1.5e-2, 1.2e-2])
    peg_size = np.array([8e-3, 0.75e-3, 3.2e-3])
    peg_gap = 7e-3
    clearance = 5e-4
    receptacle_size = np.array([1e-2, 5e-2, 5e-2])

    def __init__(self, **kwargs):
        # ManiSkill registers PlugCharger-v1 with max_episode_steps=200.
        kwargs.setdefault("horizon", 200)
        super().__init__(**kwargs)

    def _build_task_objects(self):
        self.charger = charger(
            "charger", peg_size=self.peg_size, base_size=self.base_size, gap=self.peg_gap
        )
        receptacle_peg_size = self.peg_size.copy()
        receptacle_peg_size[1:] += self.clearance
        self.receptacle = charger_receptacle(
            "receptacle",
            peg_size=receptacle_peg_size,
            receptacle_size=self.receptacle_size,
            gap=self.peg_gap,
        )
        self.task_objects.update(charger=self.charger, receptacle=self.receptacle)
        self.dynamic_object_names.add("charger")
        self.fixed_object_names.add("receptacle")

    def _reset_task(self):
        charger_xy = np.random.uniform(
            [-0.1, -0.2], [-0.01 - self.peg_size[0] * 2, 0.2]
        )
        receptacle_xy = np.random.uniform([0.01, -0.1], [0.1, 0.1])
        self._set_source_pose(
            "charger",
            np.r_[charger_xy, self.table_offset[2] + self.base_size[2]],
            yaw_quat(np.random.uniform(-np.pi / 3, np.pi / 3)),
        )
        self._set_source_pose(
            "receptacle",
            np.r_[receptacle_xy, self.table_offset[2] + 0.1],
            yaw_quat(np.random.uniform(np.pi - np.pi / 8, np.pi + np.pi / 8)),
        )

    def charger_goal_error(self) -> tuple[float, float]:
        charger_pose = self.get_task_object_pose("charger")
        receptacle_pose = self.get_task_object_pose("receptacle")
        charger_matrix = pose_matrix(charger_pose[:3], charger_pose[3:])
        goal_matrix = pose_matrix(receptacle_pose[:3], receptacle_pose[3:])
        goal_matrix[:3, :3] = goal_matrix[:3, :3] @ pose_matrix(
            np.zeros(3), yaw_quat(np.pi)
        )[:3, :3]
        delta = np.linalg.inv(goal_matrix) @ charger_matrix
        position_error = np.linalg.norm(delta[:3, 3])
        cosine = np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1.0, 1.0)
        return float(position_error), float(np.arccos(cosine))

    def _check_success(self):
        position_error, angle_error = self.charger_goal_error()
        return position_error <= 5e-3 and angle_error <= 0.2
