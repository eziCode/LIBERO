import os
import json
import h5py
import numpy as np
import imageio
from scipy.fftpack import dct, idct
from libero.libero.envs.problems.libero_tabletop_manipulation import Libero_Tabletop_Manipulation

def spectral_filter_bands(action_chunk, num_bands=16):
    """Applies absolute band filtering to an action chunk."""
    coeffs = dct(action_chunk, axis=0, norm='ortho')
    actual_bands = min(num_bands, len(coeffs))
    coeffs[actual_bands:] = 0
    return idct(coeffs, axis=0, norm='ortho')

def process_hybrid_sequence(actions, chunk_size, num_bands, grasp_idx):
    """
    Keeps everything UNFILTERED until the grasp_idx.
    After the grasp, applies the chunked spectral filter.
    """
    processed = np.copy(actions)
    for i in range(grasp_idx, actions.shape[0], chunk_size):
        end = min(i + chunk_size, actions.shape[0])
        processed[i:end] = spectral_filter_bands(actions[i:end], num_bands)
    return processed

def check_checkpoint(env):
    if "checkpoint_state" not in env.parsed_problem or not env.parsed_problem["checkpoint_state"]:
        return False
    checkpoint_state = env.parsed_problem["checkpoint_state"]
    result = True
    for state in checkpoint_state:
        result = env._eval_predicate(state) and result
    return result

def run_isolated_experiment(demo_file):
    print("\n>>> Running Isolated Harmonic Sweep (Physical Checkpoint): Mixing Task")
    out_dir = "/Users/ezraakresh/Documents/LIBERO/results/isolated_shake"
    os.makedirs(f"{out_dir}/videos", exist_ok=True)
    
    f = h5py.File(demo_file, "r")
    env_info = json.loads(f["data"].attrs["env_info"])
    ep = "demo_1"
    model_xml = f[f"data/{ep}"].attrs["model_file"]
    states = f[f"data/{ep}/states"][()]
    original_actions = f[f"data/{ep}/actions"][()]
    f.close()
    
    # We force the environment to use our edited custom BDDL file
    bddl_file = "/Users/ezraakresh/Documents/LIBERO/libero/libero/bddl_files/custom/shake_cup.bddl"
    
    print("  Initializing Dry Run to find physical grasp checkpoint...")
    env = Libero_Tabletop_Manipulation(
        bddl_file_name=bddl_file,
        **env_info,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        control_freq=20,
    )
    
    try:
        xml = env.edit_model_xml(model_xml)
    except Exception:
        xml = model_xml
    env.reset_from_xml_string(xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()
    
    grasp_idx = 0
    for step, action in enumerate(original_actions):
        env.step(action)
        if check_checkpoint(env):
            grasp_idx = step
            print(f"  -> Physical Grasp Detected at Step {grasp_idx}!")
            break
            
    if grasp_idx == 0:
        print("  -> Warning: Checkpoint never reached. Using entire sequence.")
    
    # Configuration
    chunk_size = 32
    bands_list = [16, 13, 10, 7, 4, 1]
    
    results = []
    
    for B in bands_list:
        print(f"\nTesting Bands={B} (Post-Checkpoint)...")
        recon_actions = process_hybrid_sequence(original_actions, chunk_size, B, grasp_idx)
        mse = np.mean((original_actions - recon_actions)**2)
        
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
                
                env.reset_from_xml_string(xml)
                env.sim.reset()
                env.sim.set_state_from_flattened(states[0])
                env.sim.forward()
                
                video_frames = []
                for step, action in enumerate(recon_actions):
                    obs, reward, done, info = env.step(action)
                    if step % 2 == 0:
                        video_frames.append(env.sim.render(width=512, height=512, camera_name="agentview")[::-1])
                
                success = bool(env._check_success())
                break 
            except Exception as e:
                print(f"  Simulation failed on attempt {attempt+1}: {e}")
                
        status = "SUCCESS" if success else "FAILURE"
        print(f"  Result: {status} | MSE: {mse:.6f} | Frames: {len(video_frames)}")
        
        results.append({
            "bands": B,
            "mse": mse,
            "success": success
        })
        
        if len(video_frames) > 0:
            vid_path = f"{out_dir}/videos/Mixing_Isolated_B{B}.mp4"
            imageio.mimsave(vid_path, video_frames, fps=10)
            
    print("\n========================================")
    print("ISOLATED HARMONIC SWEEP SUMMARY (Chunk=32)")
    print("========================================")
    print("Bands | MSE | Success")
    for r in results:
        print(f"{r['bands']:>5} | {r['mse']:.5f} | {r['success']}")

if __name__ == "__main__":
    demo_path = "demonstration_data_shaking_and_accel/robosuite_ln_libero_tabletop_manipulation_1776085564_317537_pick_up_a_cup_with_a_lid_on_it_and_you_shake_it_to_mix_the_contents/demo.hdf5"
    if os.path.exists(demo_path):
        run_isolated_experiment(demo_path)
    else:
        print("Demo file not found.")
