import h5py
import numpy as np
import os
import json
import robosuite.utils.transform_utils as T

def repackage_to_mimicgen(input_path, output_path):
    print(f"Repackaging to MimicGen Format: {input_path} -> {output_path}...")
    if os.path.exists(output_path):
        os.remove(output_path)
        
    with h5py.File(input_path, "r") as f_in, h5py.File(output_path, "w") as f_out:
        # 1. Setup Root 'data' Group and Attributes
        data_in = f_in["data"]
        data_out = f_out.create_group("data")
        
        # MimicGen/Robomimic sometimes use 'env_meta', but standard eval scripts expect 'env_args'
        if "env_args" in data_in.attrs:
            old_args = json.loads(data_in.attrs["env_args"])
            env_meta = {
                "env_name": old_args.get("env_name", data_in.attrs.get("env_name", "Libero_Tabletop_Manipulation")),
                "type": 1,
                "env_kwargs": old_args.get("env_kwargs", {})
            }
            data_out.attrs["env_meta"] = json.dumps(env_meta)
            data_out.attrs["env_args"] = json.dumps(env_meta)
        
        # Copy other root attributes
        for attr in data_in.attrs:
            if attr not in ["env_args", "env_meta"]:
                data_out.attrs[attr] = data_in.attrs[attr]

        for demo in sorted(data_in.keys()):
            demo_in = data_in[demo]
            demo_out = data_out.create_group(demo)
            
            # Copy demo attributes
            for attr in demo_in.attrs:
                demo_out.attrs[attr] = demo_in.attrs[attr]
                
            # 2. Process Actions: Retain original magnitudes, convert to float32
            actions = demo_in["actions"][()]
            actions_f32 = actions.astype(np.float32)
            demo_out.create_dataset("actions", data=actions_f32)
            
            # 3. Process Observations: MimicGen/Robomimic strict naming
            obs_in = demo_in["obs"]
            obs_out = demo_out.create_group("obs")
            
            # Direct Mappings
            obs_out.create_dataset("agentview_image", data=obs_in["agentview_rgb"][()])
            obs_out.create_dataset("robot0_eye_in_hand_image", data=obs_in["eye_in_hand_rgb"][()])
            obs_out.create_dataset("robot0_joint_pos", data=obs_in["joint_states"][()].astype(np.float32))
            obs_out.create_dataset("robot0_gripper_qpos", data=obs_in["gripper_states"][()].astype(np.float32))
            obs_out.create_dataset("robot0_eef_pos", data=obs_in["ee_pos"][()].astype(np.float32))
            
            # 4. Critical Conversion: Axis-Angle -> Quaternion
            # ee_ori is Axis-Angle (3D), MimicGen expects Quaternion (4D)
            ee_ori = obs_in["ee_ori"][()]
            quats = []
            for aa in ee_ori:
                quats.append(T.axisangle2quat(aa))
            obs_out.create_dataset("robot0_eef_quat", data=np.array(quats).astype(np.float32))
            
            # 5. Robot State Vector (Proprioception concatenation)
            if "robot_states" in demo_in:
                demo_out.create_dataset("robot0_robot_state", data=demo_in["robot_states"][()].astype(np.float32))
            
            # 6. Copy standard datasets
            for k in ["rewards", "dones", "states"]:
                if k in demo_in:
                    demo_out.create_dataset(k, data=demo_in[k][()])
                    
    print(f"Success! Converted dataset saved to: {output_path}")

if __name__ == "__main__":
    tasks = ["shake_cup_demo.hdf5", "upright_flipped_cup_demo.hdf5"]
    base_dir = "datasets/custom"
    
    for task in tasks:
        in_p = os.path.join(base_dir, task)
        out_p = os.path.join(base_dir, "mimicgen_" + task)
        if os.path.exists(in_p):
            repackage_to_mimicgen(in_p, out_p)
        else:
            print(f"Skipping {task}, file not found.")
