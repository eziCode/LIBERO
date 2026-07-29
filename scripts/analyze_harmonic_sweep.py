import os
import h5py
import json
import numpy as np
import cv2
import imageio
from scipy.fftpack import dct, idct
from libero.libero.envs import Libero_Tabletop_Manipulation
from libero.libero.utils.video_utils import VideoWriter

def spectral_filter_bands(action_chunk, num_bands=16):
    """Zeroes out all harmonics except the lowest K bands."""
    coeffs = dct(action_chunk, axis=0, norm='ortho')
    # Limit num_bands to chunk length
    actual_bands = min(num_bands, len(coeffs))
    coeffs[actual_bands:] = 0
    return idct(coeffs, axis=0, norm='ortho')

def process_sequence_harmonic(actions, chunk_size, num_bands):
    processed = np.zeros_like(actions)
    for i in range(0, actions.shape[0], chunk_size):
        end = min(i + chunk_size, actions.shape[0])
        processed[i:end] = spectral_filter_bands(actions[i:end], num_bands)
    return processed

def run_experiment(demo_file, demo_name):
    print(f"\n>>> Running Harmonic Sweep for Task: {demo_name}")
    f = h5py.File(demo_file, "r")
    env_info = json.loads(f["data"].attrs["env_info"])
    bddl_file = f["data"].attrs["bddl_file_name"]
    ep = "demo_1"
    model_xml = f[f"data/{ep}"].attrs["model_file"]
    states = f[f"data/{ep}/states"][()]
    original_actions = f[f"data/{ep}/actions"][()]
    f.close()
    
    # Configuration
    chunk_size = 32
    bands_list = [16, 13, 10, 7, 4, 1]
    
    results = []
    
    for B in bands_list:
        print(f"  Testing Bands={B}...")
        recon_actions = process_sequence_harmonic(original_actions, chunk_size, B)
        
        # Retry loop for simulator stability
        success = False
        video_frames = []
        for attempt in range(3):
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
                
                # Robust initialization
                try:
                    xml = env.edit_model_xml(model_xml)
                except Exception:
                    xml = model_xml
                env.reset_from_xml_string(xml)
                env.sim.reset()
                env.sim.set_state_from_flattened(states[0])
                env.sim.forward()
                
                # Rollout
                video_frames = []
                for i in range(len(recon_actions)):
                    obs, reward, done, info = env.step(recon_actions[i])
                    # Capture every 2nd frame for size/performance
                    if i % 2 == 0:
                        video_frames.append(obs["agentview_image"][::-1])
                
                # Success is determined at the FINAL frame (Strict Metric)
                success = env._check_success()
                env.close()
                break
            except Exception as e:
                print(f"    Sim error (attempt {attempt+1}): {e}")
                continue

        # Save video
        vid_dir = "results/action_filtering/sweep/videos"
        os.makedirs(vid_dir, exist_ok=True)
        vid_name = f"{demo_name}_B{B}.mp4"
        vid_path = os.path.join(vid_dir, vid_name)
        if video_frames:
            imageio.mimsave(vid_path, video_frames, fps=10)
        
        # Log result
        mse = np.mean((original_actions - recon_actions)**2)
        results.append({
            "bands": B,
            "success": success,
            "mse": mse
        })
        print(f"    Result: {'SUCCESS' if success else 'FAILURE'} | MSE: {mse:.6f}")

    return results

if __name__ == "__main__":
    demo_base = "demonstration_data_shaking_and_accel"
    tasks = {
        "to_mix_the_contents": "robosuite_ln_libero_tabletop_manipulation_1776085564_317537_pick_up_a_cup_with_a_lid_on_it_and_you_shake_it_to_mix_the_contents",
        "nail_into_a_board": "robosuite_ln_libero_tabletop_manipulation_1776086280_428284_hammer_a_nail_into_a_board",
        "put_it_rightside_up": "robosuite_ln_libero_tabletop_manipulation_1776093696_115486_pick_up_a_cup_that_has_been_flipped_over_and_you_put_it_rightside_up"
    }

    master_results = {}
    for t_name, t_folder in tasks.items():
        h5_path = os.path.join(demo_base, t_folder, "demo.hdf5")
        if os.path.exists(h5_path):
            master_results[t_name] = run_experiment(h5_path, t_name)
        else:
            print(f"Error: {h5_path} not found.")

    # Output simple summary
    print("\n" + "="*40)
    print("HARMONIC SWEEP SUMMARY (Chunk=32)")
    print("="*40)
    for task, res in master_results.items():
        print(f"\nTask: {task}")
        print("Bands | MSE | Success")
        for r in res:
            print(f"{r['bands']:5} | {r['mse']:.5f} | {r['success']}")
