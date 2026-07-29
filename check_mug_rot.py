import numpy as np
import robosuite.utils.transform_utils as T
from libero.libero.envs.objects.turbosquid_objects import WhiteYellowMug

def get_z_alignment(rotation, axis):
    # This is a simplification of how the sampler works
    if axis == 'x':
        q = np.array([np.sin(rotation / 2), 0, 0, np.cos(rotation / 2)])
    elif axis == 'y':
        q = np.array([0, np.sin(rotation / 2), 0, np.cos(rotation / 2)])
    else:
        q = np.array([0, 0, np.sin(rotation / 2), np.cos(rotation / 2)])
    
    mat = T.quat2mat(q)
    # This is the world-frame Z axis of an object that was originally at identity
    # But WhiteYellowMug might have an internal offset if we don't know the XML.
    return mat[2, 2]

print("WhiteYellowMug default rot (around X): -1.57")
# If WhiteYellowMug is upright at -1.57, then identity (0) is...
# Let's see the relative rotation between 0 and -1.57
q_upright = np.array([np.sin(-1.57/2), 0, 0, np.cos(-1.57/2)])
mat_upright = T.quat2mat(q_upright)
print("Mat at -1.57:\n", mat_upright)

q_side = np.array([0, 0, 0, 1])
mat_side = T.quat2mat(q_side)
print("Mat at 0:\n", mat_side)

