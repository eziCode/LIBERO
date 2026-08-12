"""Faithful MuJoCo port of UniVTAC's ``insert_HDMI`` task."""

from __future__ import annotations

from collections import OrderedDict
import xml.etree.ElementTree as ET

import numpy as np
import mujoco
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.transform_utils import convert_quat, mat2quat

from .objects import HDMIObject, HDMISlotObject


TABLE_SIZE = (0.8, 0.8, 0.05)
TABLE_OFFSET = np.array([0.4, 0.0, 0.0])
SOURCE_HEAD_CAMERA_POS = (0.74, 0.0, 0.066)
SOURCE_HEAD_CAMERA_QUAT = (0.512, 0.512, 0.487, 0.487)
INSERTION_SPRING_STIFFNESS = 1400.0  # N/m, effective pin + detent resistance


class UniVTACMujocoInsertHDMI(SingleArmEnv):
    """Panda HDMI insertion using UniVTAC world coordinates and source meshes."""

    environment_label = "UniVTAC insert_HDMI MuJoCo Port"
    source_task = "insert_HDMI"

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
        render_camera="univtac_head",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=60,
        horizon=118,
        ignore_done=False,
        hard_reset=False,
        camera_names="univtac_head",
        camera_heights=270,
        camera_widths=480,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        renderer_config=None,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"unsupported arguments: {', '.join(sorted(kwargs))}")
        self.table_full_size = TABLE_SIZE
        self.table_friction = (2.5, 0.01, 0.001)
        self.table_offset = TABLE_OFFSET.copy()
        self.use_object_obs = use_object_obs
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
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

    def _load_model(self):
        super()._load_model()
        # UniVTAC places the Panda at the world origin, with the plate surface
        # at z=0. The 6.9 mm correction reconciles robosuite's Panda MJCF base
        # link with the Franka URDF used by Isaac Lab.
        self.robots[0].robot_model.set_base_xpos((0.0, 0.0, -0.0069))
        arena = EmptyArena()
        arena.set_origin((0, 0, 0))
        # EmptyArena already owns the collision plane. Restyle that plane
        # instead of adding a coplanar second geom, which causes z-fighting.
        floor = arena.worldbody.find("./geom[@name='floor']")
        floor.attrib.pop("material", None)
        floor.set("name", "univtac_plate")
        floor.set("size", "0.8 0.8 0.05")
        floor.set("rgba", "0.62 0.78 0.93 1")
        floor.set("friction", "2.5 0.01 0.001")
        ET.SubElement(
            arena.worldbody,
            "camera",
            name="univtac_head",
            pos=" ".join(map(str, SOURCE_HEAD_CAMERA_POS)),
            quat=" ".join(map(str, SOURCE_HEAD_CAMERA_QUAT)),
            fovy="71.51",
            mode="fixed",
        )
        ET.SubElement(
            arena.worldbody,
            "camera",
            name="univtac_side",
            pos="0.52 -0.48 0.30",
            mode="targetbody",
            target="slot_main",
            fovy="52",
        )
        self.prism = HDMIObject("prism")
        self.slot = HDMISlotObject("slot")
        self.task_objects = OrderedDict(prism=self.prism, slot=self.slot)
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=list(self.task_objects.values()),
        )
        ET.SubElement(
            self.model.equality,
            "weld",
            name="univtac_recorded_grasp",
            body1="gripper0_eef",
            body2=self.prism.root_body,
            active="false",
            solref="0.003 1",
        )
        if not self.render_collision_mesh:
            for geom in self.model.worldbody.iter("geom"):
                if geom.get("group", "0") == "0":
                    rgba = np.fromstring(geom.get("rgba", "0.5 0.5 0.5 1"), sep=" ")
                    if rgba.size == 4:
                        rgba[3] = 0
                        geom.set("rgba", " ".join(map(str, rgba)))

    def _setup_references(self):
        super()._setup_references()
        self.prism_body_id = self.sim.model.body_name2id(self.prism.root_body)
        self.slot_body_id = self.sim.model.body_name2id(self.slot.root_body)
        self.grasp_weld_id = mujoco.mj_name2id(
            self.sim.model._model, mujoco.mjtObj.mjOBJ_EQUALITY, "univtac_recorded_grasp"
        )

    def set_recorded_grasp(self, active: bool = True) -> None:
        """Weld the recorded plug pose to the EEF so contact load reaches the wrist."""
        weld = self.grasp_weld_id
        if active:
            hand = self.sim.model.body_name2id("gripper0_eef")
            plug = self.prism_body_id
            hand_pos = self.sim.data.body_xpos[hand].copy()
            hand_rot = self.sim.data.body_xmat[hand].reshape(3, 3).copy()
            plug_pos = self.sim.data.body_xpos[plug].copy()
            plug_rot = self.sim.data.body_xmat[plug].reshape(3, 3).copy()
            relative_pos = hand_rot.T @ (plug_pos - hand_pos)
            relative_quat_xyzw = convert_quat(mat2quat(hand_rot.T @ plug_rot), to="wxyz")
            # MuJoCo weld data: anchor[0:3], relative position[3:6],
            # relative quaternion[6:10], torque scale[10].
            self.sim.model.eq_data[weld, 3:6] = relative_pos
            self.sim.model.eq_data[weld, 6:10] = relative_quat_xyzw
        self.sim.model.eq_active[weld] = int(active)
        self.sim.forward()

    def insertion_depth(self) -> float:
        """Return positive plug penetration below the socket's upper face."""
        plug_bottom = float(self.actor_pose("prism")[2])
        socket_top = float(self.actor_pose("slot")[2] + 0.0125)
        return max(socket_top - plug_bottom, 0.0)

    @property
    def insertion_spring_force(self) -> float:
        return INSERTION_SPRING_STIFFNESS * self.insertion_depth()

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step=policy_step)
        # The rigid source mesh omits compliant HDMI pins and detent springs.
        # Apply their effective axial reaction to the plug throughout each
        # MuJoCo control interval. The fixed socket receives the opposite load.
        self.sim.data.xfrc_applied[self.prism_body_id] = 0.0
        self.sim.data.xfrc_applied[self.prism_body_id, 2] = self.insertion_spring_force

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            self.set_actor_pose("prism", np.array([0.4, 0, 0.002, 1, 0, 0, 0]))
            offset = np.random.uniform(-0.005, 0.005, 2)
            self.set_actor_pose("slot", np.array([0.55 + offset[0], offset[1], 0.002, 1, 0, 0, 0]))

    def set_actor_pose(self, name: str, pose: np.ndarray) -> None:
        pose = np.asarray(pose, dtype=np.float64)
        obj = self.task_objects[name]
        if obj.joints:
            self.sim.data.set_joint_qpos(obj.joints[0], pose)
        else:
            body_id = self.slot_body_id
            self.sim.model.body_pos[body_id] = pose[:3]
            self.sim.model.body_quat[body_id] = pose[3:]
        self.sim.forward()

    def actor_pose(self, name: str) -> np.ndarray:
        body_id = self.prism_body_id if name == "prism" else self.slot_body_id
        return np.r_[self.sim.data.body_xpos[body_id], self.sim.data.body_xquat[body_id]].copy()

    def _check_success(self):
        prism = self.actor_pose("prism")
        slot = self.actor_pose("slot")
        relative = prism[:3] - (slot[:3] + np.array([0, 0, 0.005]))
        upright = abs(float(prism[3])) > np.cos(np.deg2rad(15) / 2)
        return abs(relative[1]) < 0.005 and relative[2] < 0.005 and upright

    def reward(self, action=None):
        reward = float(self._check_success())
        return reward if self.reward_scale is None else reward * self.reward_scale

    def _setup_observables(self):
        observables = super()._setup_observables()
        if not self.use_object_obs:
            return observables
        for name in ("prism", "slot"):
            @sensor(modality="univtac_object")
            def actor_pos(obs_cache, actor_name=name):
                return self.actor_pose(actor_name)[:3]

            @sensor(modality="univtac_object")
            def actor_quat(obs_cache, actor_name=name):
                return convert_quat(self.actor_pose(actor_name)[3:], to="xyzw")

            for suffix, fn in (("pos", actor_pos), ("quat", actor_quat)):
                key = f"{name}_{suffix}"
                observables[key] = Observable(key, fn, sampling_rate=self.control_freq)
        return observables
