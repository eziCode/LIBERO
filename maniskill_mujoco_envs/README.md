# ManiSkill MuJoCo environments

This standalone package ports five ManiSkill tabletop tasks to robosuite / MuJoCo:

| ManiSkill source task | Registered robosuite name |
|---|---|
| `PickCube-v1` | `ManiSkillMujocoPickCube` |
| `StackCube-v1` | `ManiSkillMujocoStackCube` |
| `StackPyramid-v1` | `ManiSkillMujocoStackPyramid` |
| `PegInsertionSide-v1` | `ManiSkillMujocoPegInsertionSide` |
| `PlugCharger-v1` | `ManiSkillMujocoPlugCharger` |

The names deliberately include `ManiSkillMujoco`. These environments are not
LIBERO custom tasks, are not registered through BDDL, and do not create or use
files under `libero/libero/bddl_files`.

The object geometry is procedural and follows the source under
`deps/ManiSkill/mani_skill/envs/tasks/tabletop`. The Panda model, contact solver,
and force/torque sensor are MuJoCo / robosuite implementations, so wrench values
are newly simulated MuJoCo measurements rather than converted PhysX values.

## Registration

```python
import robosuite as suite
import maniskill_mujoco_envs  # registers the five distinct names

env = suite.make(
    "ManiSkillMujocoStackCube",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)
```

Every observation includes:

- `robot0_eef_force`, shape `(3,)`, Newtons
- `robot0_eef_torque`, shape `(3,)`, Newton-metres

## Dataset conversion

```bash
MUJOCO_GL=disable .venv-maniskill-mujoco/bin/python \
  scripts/retarget_maniskill_to_mujoco.py \
  --input datasets/ManiSkill_Demonstrations/native_demos/StackCube-v1/motionplanning/trajectory.h5 \
  --output datasets/ManiSkill_Demonstrations/mujoco/StackCube-v1.hdf5
```

For this repository's bundled robosuite 1.4.1, use MuJoCo 2.3.7. The
`MUJOCO_GL=disable` setting is only needed for headless operation on systems
where an OpenGL context is unavailable.
