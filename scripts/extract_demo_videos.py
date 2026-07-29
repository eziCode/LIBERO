import os
import h5py
import imageio
import numpy as np

def extract_videos(hdf5_path, output_base_dir, num_demos=10):
    dataset_name = os.path.basename(hdf5_path).replace(".hdf5", "")
    output_dir = os.path.join(output_base_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Extracting videos from {hdf5_path}...")
    
    with h5py.File(hdf5_path, "r") as f:
        demos = list(f["data"].keys())
        # Sort demos to ensure demo_0 to demo_9 are picked
        demos = sorted(demos, key=lambda x: int(x.split("_")[1]))
        
        for ep in demos[:num_demos]:
            video_path = os.path.join(output_dir, f"{ep}.mp4")
            
            if "obs/agentview_rgb" not in f[f"data/{ep}"]:
                print(f"  [Warning] agentview_rgb missing for {ep}")
                continue
                
            images = f[f"data/{ep}/obs/agentview_rgb"][()]
            
            # Check if images are 0-255 uint8
            if images.dtype != np.uint8:
                images = (images * 255).astype(np.uint8)
                
            imageio.mimsave(video_path, images, fps=20)
            print(f"  Saved {video_path}")

if __name__ == "__main__":
    custom_datasets_dir = "/Users/ezraakresh/Documents/LIBERO/datasets/custom"
    output_dir = "/Users/ezraakresh/Documents/LIBERO/datasets/custom/demo_videos"
    
    hdf5_files = [
        os.path.join(custom_datasets_dir, "shake_cup_demo.hdf5"),
        os.path.join(custom_datasets_dir, "upright_flipped_cup_demo.hdf5")
    ]
    
    for hdf5_file in hdf5_files:
        if os.path.exists(hdf5_file):
            extract_videos(hdf5_file, output_dir)
        else:
            print(f"[Error] {hdf5_file} does not exist.")
