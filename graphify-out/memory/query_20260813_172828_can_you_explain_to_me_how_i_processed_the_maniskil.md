---
type: "query"
date: "2026-08-13T17:28:28.531411+00:00"
question: "can you explain to me how i processed the maniskill dataset"
contributor: "graphify"
outcome: "useful"
source_nodes: ["convert_maniskill_to_robomimic.py", "retarget_maniskill_to_mujoco.py", "map_trajectory_states()", "replay_trajectory()", "libero_actions()"]
---

# Q: can you explain to me how i processed the maniskill dataset

## Answer

Expanded from original query via graph vocab: [maniskill, dataset, convert, trajectory, states, observations, actions, mujoco, robomimic, replay, retarget]. The repository contains two processing routes. convert_maniskill_to_robomimic.py is schema-only: it loads trajectory.h5 and adjacent JSON metadata, iterates traj_N episodes, copies actions/rewards/termination flags, flattens SAPIEN PhysX env_states, preserves observations, aligns T versus T+1 samples, and writes robomimic data/demo_N groups while marking states as sapien_physx and not MuJoCo-compatible. retarget_maniskill_to_mujoco.py performs cross-simulator processing: infer task, instantiate its robosuite MuJoCo port, map object and Panda states per sample by default or replay retargeted joint actions, collect MuJoCo observations and force/torque, convert actions to LIBERO format, then write robomimic/MimicGen HDF5 with provenance and conversion metadata. Thus the native physics is preserved only by the schema converter; the retargeted output is a new MuJoCo realization.

## Outcome

- Signal: useful

## Source Nodes

- convert_maniskill_to_robomimic.py
- retarget_maniskill_to_mujoco.py
- map_trajectory_states()
- replay_trajectory()
- libero_actions()