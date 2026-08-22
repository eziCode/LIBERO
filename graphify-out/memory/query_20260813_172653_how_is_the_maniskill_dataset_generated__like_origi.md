---
type: "query"
date: "2026-08-13T17:26:53.975655+00:00"
question: "How is the maniskill dataset generated? Like originally, what simulator"
contributor: "graphify"
outcome: "useful"
source_nodes: ["retarget_maniskill_to_mujoco.py", "convert_maniskill_to_robomimic.py", "map_trajectory_states()", "replay_trajectory()"]
---

# Q: How is the maniskill dataset generated? Like originally, what simulator

## Answer

Expanded from original query via graph vocab: [maniskill, dataset, generate, simulator, simulation, trajectory, environment, replay, retarget, mujoco, robomimic]. The original ManiSkill demonstrations are generated in ManiSkill using SAPIEN with the PhysX backend, not MuJoCo. They are stored as native trajectory.h5 plus matching JSON metadata, with traj_N groups containing actions and recorded env_states and observations. In this repository, convert_maniskill_to_robomimic.py performs schema-only packaging and leaves SAPIEN/PhysX states intact. retarget_maniskill_to_mujoco.py is a separate cross-simulator conversion that ports tasks to robosuite/MuJoCo and either maps each recorded PhysX state into MuJoCo or replays retargeted actions. MuJoCo force/torque values are newly simulated, not present in the original dataset.

## Outcome

- Signal: useful

## Source Nodes

- retarget_maniskill_to_mujoco.py
- convert_maniskill_to_robomimic.py
- map_trajectory_states()
- replay_trajectory()