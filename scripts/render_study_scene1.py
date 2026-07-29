import h5py
import json
import os
import numpy as np
import tqdm
import imageio
from libero.libero.envs import OffScreenRenderEnv

def render_single_demo():
    dataset_path = "/Users/ezraakresh/Documents/LIBERO/datasets/libero_90/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy_demo.hdf5"
    bddl_file = "libero/libero/bddl_files/libero_90/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy.bddl"
    output_dir = "demonstration_videos/study_scene1_single"
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found.")
        return

    # Create environment
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=512,
        camera_widths=512,
    )

    with h5py.File(dataset_path, "r") as f:
        demo_name = "demo_0" # HF datasets start with demo_0
        if demo_name not in f["data"]:
            demo_name = list(f["data"].keys())[0]
            
        print(f"Rendering {demo_name}...")
        demo_grp = f[f"data/{demo_name}"]
        states = demo_grp["states"][()]
        
        video_path = os.path.join(output_dir, f"{demo_name}.mp4")
        writer = imageio.get_writer(video_path, fps=30)
        
        for i in tqdm.tqdm(range(len(states))):
            obs = env.regenerate_obs_from_state(states[i])
            img = obs["agentview_image"]
            writer.append_data(img[::-1])
        
        writer.close()
        print(f"Finished rendering {demo_name} to {video_path}")

    env.close()

if __name__ == "__main__":
    render_single_demo()
