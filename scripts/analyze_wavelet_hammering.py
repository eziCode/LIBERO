import os
import json
import h5py
import numpy as np
import imageio
import pywt
from libero.libero.envs.problems.libero_tabletop_manipulation import Libero_Tabletop_Manipulation

def wavelet_filter_hybrid(actions, grasp_idx, wavelet_type='db4', levels_to_remove=1):
    """
    Keeps everything UNFILTERED until the grasp_idx.
    After the grasp, applies Wavelet filtering to the remaining sequence.
    """
    processed = np.copy(actions)
    if levels_to_remove == 0:
        return processed
        
    post_grasp_actions = actions[grasp_idx:]
    
    # Decompose
    coeffs = pywt.wavedec(post_grasp_actions, wavelet_type, axis=0)
    
    for i in range(1, min(levels_to_remove + 1, len(coeffs))):
        coeffs[-i] = np.zeros_like(coeffs[-i])
        
    # Reconstruct
    filtered = pywt.waverec(coeffs, wavelet_type, axis=0)
    
    processed[grasp_idx:] = filtered[:post_grasp_actions.shape[0]]
    return processed

def check_checkpoint(env):
    if "checkpoint_state" not in env.parsed_problem or not env.parsed_problem["checkpoint_state"]:
        return False
    checkpoint_state = env.parsed_problem["checkpoint_state"]
    result = True
    for state in checkpoint_state:
        result = env._eval_predicate(state) and result
    return result

def run_wavelet_experiment(demo_file):
    print("\n>>> Running Wavelet Depth Sweep (Post-Grasp Only): Hammering Task")
    out_dir = "/Users/ezraakresh/Documents/LIBERO/results/wavelet_analysis_post_grasp"
    os.makedirs(f"{out_dir}/videos", exist_ok=True)
    
    f = h5py.File(demo_file, "r")
    env_info = json.loads(f["data"].attrs["env_info"])
    ep = "demo_1"
    model_xml = f[f"data/{ep}"].attrs["model_file"]
    states = f[f"data/{ep}/states"][()]
    original_actions = f[f"data/{ep}/actions"][()]
    f.close()
    
    bddl_file = "/Users/ezraakresh/Documents/LIBERO/libero/libero/bddl_files/custom/hammer_nail_into_board.bddl"
    
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
    levels_list = [0, 1, 2, 3, 4]
    
    results = []
    
    for L in levels_list:
        print(f"\nTesting Levels Removed={L} (Post-Checkpoint)...")
        recon_actions = wavelet_filter_hybrid(original_actions, grasp_idx, 'db4', L)
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
            "levels_removed": L,
            "mse": mse,
            "success": success
        })
        
        if len(video_frames) > 0:
            vid_path = f"{out_dir}/videos/Hammering_Wavelet_Hybrid_L{L}.mp4"
            imageio.mimsave(vid_path, video_frames, fps=10)
            
    print("\n========================================")
    print("HYBRID WAVELET SWEEP SUMMARY (db4)")
    print("========================================")
    print("Levels Removed | MSE | Success")
    for r in results:
        print(f"{r['levels_removed']:>14} | {r['mse']:.5f} | {r['success']}")

if __name__ == "__main__":
    demo_path = "demonstration_data_shaking_and_accel/robosuite_ln_libero_tabletop_manipulation_1776086280_428284_hammer_a_nail_into_a_board/demo.hdf5"
    if os.path.exists(demo_path):
        run_wavelet_experiment(demo_path)
    else:
        print("Demo file not found.")
