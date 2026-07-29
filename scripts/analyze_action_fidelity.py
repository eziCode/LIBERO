import os
import json
import h5py
import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import robosuite
from robosuite.utils.transform_utils import *
from libero.libero.envs.problems.libero_tabletop_manipulation import Libero_Tabletop_Manipulation
import tqdm

def spectral_filter(action_chunk, keep_ratio=0.5):
    """
    Applies DCT, zeros out high frequency coefficients, and applies IDCT.
    action_chunk: (L, D) where L is sequence length, D is action dimension.
    """
    if keep_ratio >= 1.0:
        return action_chunk
    
    coeffs = dct(action_chunk, axis=0, norm='ortho')
    
    L = len(coeffs)
    cutoff = max(1, int(L * keep_ratio))
    coeffs[cutoff:] = 0
    
    # Reconstruct
    return idct(coeffs, axis=0, norm='ortho')

def process_full_sequence(actions, chunk_size, keep_ratio):
    """
    Processes the entire action sequence in chunks.
    chunk_size: int, 'Full', or 'gripper'
    """
    if chunk_size == 'Full':
        return spectral_filter(actions, keep_ratio)
    
    num_steps = actions.shape[0]
    processed_actions = np.zeros_like(actions)
    
    if chunk_size == 'gripper':
        # Split chunks based on gripper signal (index 6)
        gripper = actions[:, 6]
        # Find indices where it changes
        diffs = np.where(np.diff(gripper) != 0)[0]
        # Add start and end
        boundaries = [0] + (diffs + 1).tolist() + [num_steps]
        print(f"  Gripper phases: {len(boundaries)-1} | Boundaries: {boundaries}")
        
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            if start < end:
                processed_actions[start:end] = spectral_filter(actions[start:end], keep_ratio)
        return processed_actions

    # Fixed chunking
    for i in range(0, num_steps, chunk_size):
        end = min(i + chunk_size, num_steps)
        chunk = actions[i:end]
        processed_actions[i:end] = spectral_filter(chunk, keep_ratio)
        
    return processed_actions

def run_experiment(demo_file, task_name):
    print(f"\n>>> Running Spectral Analysis for Task: {task_name}")
    os.makedirs("results/videos", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    f = h5py.File(demo_file, "r")
    env_name = f["data"].attrs["env"]
    env_info = json.loads(f["data"].attrs["env_info"])
    bddl_file = f["data"].attrs["bddl_file_name"]
    
    # Create the environment with a retry loop for RandomizationErrors
    max_retries = 5
    for attempt in range(max_retries):
        try:
            env = Libero_Tabletop_Manipulation(
                bddl_file_name=bddl_file,
                **env_info,
                has_renderer=False,
                has_offscreen_renderer=True,
                render_camera="agentview",
                ignore_done=True,
                control_freq=20,
            )
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"  Placement failed (attempt {attempt+1}/{max_retries}), retrying...")

    ep = "demo_1"
    model_xml = f[f"data/{ep}"].attrs["model_file"]
    states = f[f"data/{ep}/states"][()]
    original_actions = f[f"data/{ep}/actions"][()]
    
    chunk_sizes = ["gripper"]
    keep_ratios = [1.0, 0.5, 0.1]
    
    results = []
    
    for L in chunk_sizes:
        for P in keep_ratios:
            print(f"Testing ChunkSize={L}, KeepRatio={P}...")
            
            reconstructed_actions = process_full_sequence(original_actions, L, P)
            mse = np.mean((original_actions - reconstructed_actions)**2)
            
            try:
                xml = env.edit_model_xml(model_xml)
            except Exception:
                xml = model_xml
            env.reset_from_xml_string(xml)
            env.sim.reset()
            env.sim.set_state_from_flattened(states[0])
            env.sim.forward()
            
            video_frames = []
            success = False
            
            for action in reconstructed_actions:
                obs, reward, done, info = env.step(action)
                if len(video_frames) * 2 <= len(reconstructed_actions):
                    if len(video_frames) == 0 or len(video_frames) * 2 < len(reconstructed_actions):
                        if int(env.timestep) % 2 == 0:
                            video_frames.append(env.sim.render(width=512, height=512, camera_name="agentview")[::-1])
            
            # Stricter success: Must be successful at the final frame
            success = bool(env._check_success())
            
            status = "SUCCESS" if success else "FAILURE"
            print(f"  Result: {status} | MSE: {mse:.6f} | Frames: {len(video_frames)}")
            results.append({
                "chunk_size": L,
                "keep_ratio": P,
                "mse": mse,
                "success": success
            })
            
            vid_name = f"{task_name}_L{L}_P{P}.mp4"
            vid_path = os.path.join("results/videos", vid_name)
            if len(video_frames) > 0:
                imageio.mimsave(vid_path, video_frames, fps=10)

    f.close()
    return results

if __name__ == "__main__":
    demo_dirs = [
        "demonstration_data_shaking_and_accel/robosuite_ln_libero_tabletop_manipulation_1776085564_317537_pick_up_a_cup_with_a_lid_on_it_and_you_shake_it_to_mix_the_contents",
        "demonstration_data_shaking_and_accel/robosuite_ln_libero_tabletop_manipulation_1776086280_428284_hammer_a_nail_into_a_board",
        "demonstration_data_shaking_and_accel/robosuite_ln_libero_tabletop_manipulation_1776093696_115486_pick_up_a_cup_that_has_been_flipped_over_and_you_put_it_rightside_up",
    ]
    
    all_experiments = {}
    for d in demo_dirs:
        task_id = "_".join(d.split("/")[-1].split("_")[-4:])
        results = run_experiment(os.path.join(d, "demo.hdf5"), task_id)
        all_experiments[task_id] = results
        
    print("\nSummary of Experiments:")
    for task, res in all_experiments.items():
        print(f"\nTask: {task}")
        print("Chunk | Ratio | MSE | Success")
        for r in res:
            print(f"{r['chunk_size']} | {r['keep_ratio']} | {r['mse']:.5f} | {r['success']}")
