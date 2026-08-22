"""Evaluation wrapper shared by ManiSkill MuJoCo rollouts and policies."""

from __future__ import annotations

import numpy as np


class ManiSkillWrapper:
    """Small evaluation-style interface around a registered robosuite port."""

    def __init__(self, env):
        self.env = env

    def step(self, action: np.ndarray):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def check_success(self) -> bool:
        return bool(self.env._check_success())

    @property
    def robots(self):
        return self.env.robots

    @property
    def sim(self):
        return self.env.sim

    def close(self) -> None:
        self.env.close()
