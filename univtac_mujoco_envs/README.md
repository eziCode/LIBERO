# UniVTAC MuJoCo environments

This package is intentionally separate from LIBERO BDDL environments and from
the ManiSkill ports. It currently registers `UniVTACMujocoInsertHDMI`.

Source-of-truth files are the official UniVTAC `envs/insert_HDMI.py`,
`assets/objects/HDMI.usd`, and `assets/objects/HDMISlot.usd` under
`deps/UniVTAC`.

The HDMI and slot visual meshes are exact USD-to-OBJ extractions. MuJoCo treats
a mesh collision geom as a convex hull, so the concave slot collision is a
five-piece analytic decomposition that leaves the source-sized opening usable.
GelSight RGB, depth, and marker observations are preserved from the source
dataset; MuJoCo does not attempt to regenerate deformable-gel pixels.

Convert one episode and validate it:

```bash
MUJOCO_GL=disable .venv-maniskill-mujoco/bin/python \
  scripts/convert_univtac_to_mujoco.py --count 1 --output /tmp/univtac_test.hdf5
.venv-maniskill-mujoco/bin/python \
  scripts/validate_univtac_mujoco.py /tmp/univtac_test.hdf5
```

Omit `--count` to convert all downloaded `insert_HDMI` episodes. The default
output is `datasets/UniVTAC/mujoco/insert_HDMI.hdf5`.
