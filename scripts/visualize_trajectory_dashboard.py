import os
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.fftpack import dct, idct

def spectral_filter(action_chunk, keep_ratio=0.5):
    if keep_ratio >= 1.0: return action_chunk
    coeffs = dct(action_chunk, axis=0, norm='ortho')
    cutoff = max(1, int(len(coeffs) * keep_ratio))
    coeffs[cutoff:] = 0
    return idct(coeffs, axis=0, norm='ortho')

def process_full_sequence(actions, chunk_size, keep_ratio):
    if chunk_size == 'Full': return spectral_filter(actions, keep_ratio)
    processed = np.zeros_like(actions)
    
    if chunk_size == 'gripper':
        gripper = actions[:, 6]
        diffs = np.where(np.diff(gripper) != 0)[0]
        boundaries = [0] + (diffs + 1).tolist() + [actions.shape[0]]
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            if start < end:
                processed[start:end] = spectral_filter(actions[start:end], keep_ratio)
        return processed
        
    for i in range(0, actions.shape[0], chunk_size):
        end = min(i + chunk_size, actions.shape[0])
        processed[i:end] = spectral_filter(actions[i:end], keep_ratio)
    return processed

def get_trajectory(actions):
    # Scale from [-1, 1] to [-0.05, 0.05] as per OSC_POSE config
    # and integrate deltas to get relative path
    return np.cumsum(actions[:, :3] * 0.05, axis=0)

def generate_dashboard(demo_file, task_name, chunk_size, out_dir):
    print(f"Processing Dashboard for: {task_name}")
    f = h5py.File(demo_file, "r")
    original_actions = f["data/demo_1/actions"][()]
    f.close()
    
    keep_ratios = [1.0, 0.5, 0.1]
    colors = ['#444444', '#1f77b4', '#d62728']
    alphas = [0.3, 0.8, 0.8]
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"Trajectory Fidelity Dashboard: {task_name} (Chunk={chunk_size})", fontsize=18, fontweight='bold')
    
    # 1. 3D Trajectory Plot
    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    # 2. 2D Planar Plot (XY)
    ax2d = fig.add_subplot(2, 2, 2)
    # 3. Euclidean Error Plot
    ax_err = fig.add_subplot(2, 1, 2)
    
    expert_traj = get_trajectory(original_actions)
    
    for i, P in enumerate(keep_ratios):
        recon_actions = process_full_sequence(original_actions, chunk_size, P)
        recon_traj = get_trajectory(recon_actions)
        
        # Plot 3D
        label = "Expert" if P == 1.0 else f"Ratio {P}"
        ax3d.plot(recon_traj[:, 0], recon_traj[:, 1], recon_traj[:, 2], 
                  color=colors[i], alpha=alphas[i], linewidth=2, label=label)
        
        # Plot 2D
        ax2d.plot(recon_traj[:, 0], recon_traj[:, 1], color=colors[i], alpha=alphas[i], linewidth=2)
        
        # Calculate Euclidean Error over time
        # error = distance between original and reconstructed at each step
        error = np.linalg.norm(expert_traj - recon_traj, axis=1)
        ax_err.plot(error, color=colors[i], alpha=alphas[i], linewidth=2, label=label)

    # Styling 3D
    ax3d.set_title("3D Commanded Path", fontsize=14)
    ax3d.set_xlabel("X (Forward)")
    ax3d.set_ylabel("Y (Lateral)")
    ax3d.set_zlabel("Z (Height)")
    ax3d.legend(loc='upper left', fontsize=8)
    
    # Styling 2D
    ax2d.set_title("Top-Down Footprint (XY Projection)", fontsize=14)
    ax2d.set_xlabel("X (Forward)")
    ax2d.set_ylabel("Y (Lateral)")
    ax2d.grid(True, linestyle='--', alpha=0.5)
    
    # Styling Error
    ax_err.set_title("Cumulative Translational Error (Meters)", fontsize=14)
    ax_err.set_xlabel("Time (Steps)")
    ax_err.set_ylabel("Euclidean Distance from Expert")
    ax_err.grid(True, linestyle='--', alpha=0.5)
    ax_err.legend(loc='upper left', ncol=5)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    os.makedirs(out_dir, exist_ok=True)
    save_path = f"{out_dir}/DASHBOARD_{task_name}_Chunk{chunk_size}.png"
    plt.savefig(save_path, dpi=150)
    print(f"  Dashboard saved: {save_path}")
    plt.close()

if __name__ == "__main__":
    demo_base = "demonstration_data_shaking_and_accel"
    tasks = [
        "robosuite_ln_libero_tabletop_manipulation_1776085564_317537_pick_up_a_cup_with_a_lid_on_it_and_you_shake_it_to_mix_the_contents",
        "robosuite_ln_libero_tabletop_manipulation_1776086280_428284_hammer_a_nail_into_a_board",
        "robosuite_ln_libero_tabletop_manipulation_1776093696_115486_pick_up_a_cup_that_has_been_flipped_over_and_you_put_it_rightside_up"
    ]
    
    out_path = "results/action_filtering/gripper_chunks/graphs"
    
    for t_folder in tasks:
        task_id = "_".join(t_folder.split("_")[-4:])
        h5_path = os.path.join(demo_base, t_folder, "demo.hdf5")
        if os.path.exists(h5_path):
            generate_dashboard(h5_path, task_id, 'gripper', out_path)
        else:
            print(f"Error: {h5_path} not found.")
