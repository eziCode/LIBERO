"""Standalone MuJoCo ports of selected ManiSkill tabletop tasks.

Importing this package registers five uniquely named robosuite environments.
They intentionally live outside LIBERO's BDDL and custom-task packages.
"""

from .envs import (  # noqa: F401
    ManiSkillMujocoPegInsertionSide,
    ManiSkillMujocoPickCube,
    ManiSkillMujocoPlugCharger,
    ManiSkillMujocoStackCube,
    ManiSkillMujocoStackPyramid,
)

ENVIRONMENT_NAMES = (
    "ManiSkillMujocoPickCube",
    "ManiSkillMujocoStackCube",
    "ManiSkillMujocoStackPyramid",
    "ManiSkillMujocoPegInsertionSide",
    "ManiSkillMujocoPlugCharger",
)

__all__ = (*ENVIRONMENT_NAMES, "ENVIRONMENT_NAMES")
