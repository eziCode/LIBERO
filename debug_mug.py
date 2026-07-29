import numpy as np
import robosuite
from libero.libero.envs.problems.libero_tabletop_manipulation import LiberoTabletopManipulation
from libero.libero.envs.objects.custom_objects import FlippedMug
import robosuite.utils.transform_utils as T

env = LiberoTabletopManipulation(
    bddl_file_name="libero/libero/bddl_files/custom/upright_flipped_cup.bddl",
    robots="Panda",
)
obs = env.reset()

# Find the mug
obj_name = "flipped_mug_1"
obj = env.objects_dict[obj_name]
print(f"Object Class: {obj.__class__.__name__}")
print(f"Object rotation: {obj.rotation}")
print(f"Object rotation_axis: {obj.rotation_axis}")

# Get physical state
quat = env.sim.data.get_joint_qpos(obj.joints[0])[-4:]
# Wait, FlippedMug has joints. It's a free joint, so last 7 values are pos(3) and quat(4).
# Actually, robosuite free joints are [x,y,z, x,y,z,w] or [x,y,z, w,x,y,z]?
# Standard MuJoCo free joint is [x,y,z, q_w, q_x, q_y, q_z]
print(f"Physical Quat (MuJoCo): {quat}")
mat = T.quat2mat(T.convert_quat(quat, to="xyzw")) # robosuite utilities usually like xyzw
print(f"Rotation Matrix:\n{mat}")
print(f"IsUpright value (mat[2,2]): {mat[2, 2]}")

env.close()
