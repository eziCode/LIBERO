import os
import glob
import h5py
import cv2
import numpy as np

def generate_videos():
    input_dir = "/Users/ezraakresh/Documents/LIBERO/custom_tasks"
    output_dir = "/Users/ezraakresh/Documents/LIBERO/datasets/custom/videos"
    
    os.makedirs(output_dir, exist_ok=True)
    
    hdf5_files = glob.glob(os.path.join(input_dir, "*.hdf5"))
    if not hdf5_files:
        print(f"No HDF5 files found in {input_dir}")
        return
        
    for file_path in hdf5_files:
        task_name = os.path.basename(file_path).replace(".hdf5", "")
        print(f"\nProcessing task: {task_name}")
        
        try:
            with h5py.File(file_path, 'r') as f:
                demo_keys = list(f['data'].keys())
                # Take the first 3 demos
                demos_to_process = demo_keys[:3]
                
                for demo_key in demos_to_process:
                    print(f"  Generating video for {demo_key}...")
                    obs_group = f[f'data/{demo_key}/obs']
                    
                    # Extract arrays
                    images = obs_group['agentview_rgb'][:]
                    # Reverse RGB to BGR for OpenCV
                    images = images[..., ::-1]
                    
                    ee_forces = obs_group['ee_force'][:]
                    ee_torques = obs_group['ee_torque'][:]
                    
                    # Handle new separate forces or fallback to old if not re-run yet
                    has_split_forces = 'left_gripper_force' in obs_group
                    if has_split_forces:
                        left_forces = obs_group['left_gripper_force'][:]
                        right_forces = obs_group['right_gripper_force'][:]
                    else:
                        print("  WARNING: Split left/right forces not found. Did you re-run create_dataset.py?")
                        continue
                    
                    num_frames = len(images)
                    height, width, _ = images[0].shape
                    scale = 4 # Upscale factor to make text readable on 128x128 native images
                    
                    # Initialize video writer (using mp4v codec)
                    out_path = os.path.join(output_dir, f"{task_name}_{demo_key}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    # robosuite usually runs at 20 control freq (20 FPS)
                    video_writer = cv2.VideoWriter(out_path, fourcc, 20.0, (width * scale, height * scale))
                    
                    for i in range(num_frames):
                        frame = images[i].copy()
                        
                        # 1. Flip vertically to correct OpenGL rendering
                        frame = cv2.flip(frame, 0)
                        
                        # 2. Resize to higher resolution for crisp text
                        frame = cv2.resize(frame, (width * scale, height * scale), interpolation=cv2.INTER_LINEAR)
                        
                        # Data for this frame
                        ef = ee_forces[i]
                        et = ee_torques[i]
                        lf = left_forces[i][0]
                        rf = right_forces[i][0]
                        
                        # Text parameters
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        color = (0, 255, 0) # Green text
                        thickness = 1
                        
                        # Overlay Text
                        lines = [
                            f"EEF F: [{ef[0]:.1f}, {ef[1]:.1f}, {ef[2]:.1f}]",
                            f"EEF T: [{et[0]:.1f}, {et[1]:.1f}, {et[2]:.1f}]",
                            f"L Prong F: {lf:.1f} N",
                            f"R Prong F: {rf:.1f} N"
                        ]
                        
                        y0, dy = 25, 20
                        for j, line in enumerate(lines):
                            y = y0 + j * dy
                            # Add black outline for better contrast
                            cv2.putText(frame, line, (10, y), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
                            cv2.putText(frame, line, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)
                            
                        video_writer.write(frame)
                        
                    video_writer.release()
                    print(f"  Saved: {out_path}")
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    generate_videos()
