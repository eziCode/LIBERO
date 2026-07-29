import os
import h5py
import json
import numpy as np
import imageio
import tqdm
import cv2
from libero.libero.envs import OffScreenRenderEnv

def create_comparison_videos():
    teleop_dir = "demonstration_data"
    original_data_dir = "datasets/libero_90"
    output_dir = "comparison_videos"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all teleop datasets
    teleop_folders = [d for d in os.listdir(teleop_dir) if os.path.isdir(os.path.join(teleop_dir, d)) and "robosuite_ln_libero_" in d]
    
    for folder in teleop_folders:
        teleop_h5_file = os.path.join(teleop_dir, folder, "demo.hdf5")
        if not os.path.exists(teleop_h5_file):
            continue
            
        with h5py.File(teleop_h5_file, "r") as f:
            # Extract bddl name to find the original dataset
            bddl_file = ""
            if "env_args" in f["data"].attrs:
                env_args = json.loads(f["data"].attrs["env_args"])
                bddl_file = env_args["bddl_file_name"]
            elif "bddl_file_name" in f["data"].attrs:
                bddl_file = f["data"].attrs["bddl_file_name"]
            
            if not bddl_file:
                print(f"Skipping {folder}: Could not find BDDL file in metadata")
                continue
                
            task_basename = os.path.basename(bddl_file).replace(".bddl", "")
            original_h5_file = os.path.join(original_data_dir, f"{task_basename}_demo.hdf5")
            
            if not os.path.exists(original_h5_file):
                print(f"Skipping {task_basename}: Original dataset not found at {original_h5_file}")
                continue
            
            # Ensure bddl_file uses correct path relative to cwd if needed
            if not os.path.exists(bddl_file):
                potential_path = os.path.join(os.getcwd(), bddl_file)
                if os.path.exists(potential_path):
                    bddl_file = potential_path
                    
            try:
                env = OffScreenRenderEnv(
                    bddl_file_name=bddl_file,
                    camera_heights=512,
                    camera_widths=512,
                )
            except Exception as e:
                print(f"Failed to create env for {task_basename}: {e}")
                continue
                
            # Get states for original
            try:
                with h5py.File(original_h5_file, "r") as orig_f:
                    orig_demo_name = list(orig_f["data"].keys())[0]
                    orig_states = orig_f[f"data/{orig_demo_name}/states"][()]
            except Exception as e:
                print(f"Failed to read original dataset states for {task_basename}: {e}")
                env.close()
                continue
                
            # Get states for teleop
            teleop_demo_name = list(f["data"].keys())[0]
            teleop_states = f[f"data/{teleop_demo_name}/states"][()]
            
        print(f"\n--- Rendering comparison for {task_basename} ---")
        
        orig_frames = []
        for i in tqdm.tqdm(range(len(orig_states)), desc="Rendering Original"):
            obs = env.regenerate_obs_from_state(orig_states[i])
            img = obs["agentview_image"]
            orig_frames.append(img[::-1]) # The image is typically upside down in some views
            
        teleop_frames = []
        for i in tqdm.tqdm(range(len(teleop_states)), desc="Rendering Teleop"):
            obs = env.regenerate_obs_from_state(teleop_states[i])
            img = obs["agentview_image"]
            teleop_frames.append(img[::-1])
            
        env.close()
        
        if len(orig_frames) == 0 or len(teleop_frames) == 0:
            print(f"Failed to render frames for {task_basename}. Skipping.")
            continue
            
        # Pad to same length
        max_len = max(len(orig_frames), len(teleop_frames))
        
        while len(orig_frames) < max_len:
            orig_frames.append(orig_frames[-1])
        while len(teleop_frames) < max_len:
            teleop_frames.append(teleop_frames[-1])
            
        # Write video side-by-side
        video_path = os.path.join(output_dir, f"{task_basename}_comparison.mp4")
        writer = imageio.get_writer(video_path, fps=30)
        
        for i in tqdm.tqdm(range(max_len), desc="Writing Video"):
            img1 = orig_frames[i].copy()
            img2 = teleop_frames[i].copy()
            
            # Add text labels on the top left of each view
            cv2.putText(img1, "Original (Clean)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img2, "Teleop (Human)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            combined = np.concatenate([img1, img2], axis=1) # Horizontally concatenate
            writer.append_data(combined)
            
        writer.close()
        print(f"Saved {video_path}")

if __name__ == "__main__":
    create_comparison_videos()
