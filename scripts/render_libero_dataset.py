import argparse
import h5py
import json
import init_path
import os
import numpy as np
import tqdm
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils.video_utils import VideoWriter

def render_dataset(dataset_path, output_dir, fps=30, camera_name="agentview"):
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(dataset_path, "r") as f:
        # Load metadata
        if "env_args" in f["data"].attrs:
            env_args = json.loads(f["data"].attrs["env_args"])
            bddl_file = env_args["bddl_file_name"]
        elif "bddl_file_name" in f["data"].attrs:
            bddl_file = f["data"].attrs["bddl_file_name"]
        else:
            # Try to find it in problem_info
            problem_info = json.loads(f["data"].attrs["problem_info"])
            # We might need to reconstruct the path
            bddl_file = f"libero/libero/bddl_files/{problem_info['problem_folder']}/{problem_info['bddl_file']}"

        # Adjust BDDL path if it's relative to libero root
        if not os.path.exists(bddl_file):
            potential_path = os.path.join(os.getcwd(), bddl_file)
            if os.path.exists(potential_path):
                bddl_file = potential_path

        print(f"Using BDDL file: {bddl_file}")

        # Create environment
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_heights=512,
            camera_widths=512,
        )

        demos = sorted(list(f["data"].keys()), key=lambda x: int(x.split("_")[1]) if "_" in x else 0)
        
        for demo_name in tqdm.tqdm(demos, desc="Rendering demos"):
            demo_grp = f[f"data/{demo_name}"]
            states = demo_grp["states"][()]
            
            video_path = os.path.join(output_dir, f"{demo_name}.mp4")
            # Using imageio directly for simple per-demo control
            import imageio
            writer = imageio.get_writer(video_path, fps=fps)
            
            for i in range(len(states)):
                obs = env.regenerate_obs_from_state(states[i])
                # agentview_image is (H, W, 3), but often needs flip
                img = obs[f"{camera_name}_image"]
                writer.append_data(img[::-1])
            
            writer.close()
            print(f"Finished rendering {demo_name} to {video_path}")

        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to HDF5 dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save videos")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera", type=str, default="agentview")
    args = parser.parse_args()

    render_dataset(args.dataset, args.output_dir, args.fps, args.camera)
