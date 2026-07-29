import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

def spectral_filter_bands(action_chunk, num_bands=16):
    coeffs = dct(action_chunk, axis=0, norm='ortho')
    actual_bands = min(num_bands, len(coeffs))
    coeffs[actual_bands:] = 0
    return idct(coeffs, axis=0, norm='ortho')

def process_hybrid_sequence(actions, chunk_size, num_bands):
    processed = np.copy(actions)
    gripper = actions[:, 6]
    grasp_indices = np.where(gripper > 0)[0]
    grasp_idx = grasp_indices[0] if len(grasp_indices) > 0 else 0
    
    for i in range(grasp_idx, actions.shape[0], chunk_size):
        end = min(i + chunk_size, actions.shape[0])
        processed[i:end] = spectral_filter_bands(actions[i:end], num_bands)
    return processed, grasp_idx

def generate_isolated_dashboard(demo_file, out_dir):
    print("Generating Isolated Dashboard for Mixing Task...")
    f = h5py.File(demo_file, "r")
    expert_actions = f["data/demo_1/actions"][()]
    f.close()
    
    bands_list = [16, 13, 10, 7, 4, 1]
    chunk_size = 32
    colors = ['#444444', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Isolated Interaction Dashboard: Mixing Task (Post-Grasp Filter Only)", fontsize=18, fontweight='bold')
    
    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    ax2d = fig.add_subplot(2, 2, 2)
    ax_err = fig.add_subplot(2, 1, 2)
    
    expert_traj = np.cumsum(expert_actions[:, :3] * 0.05, axis=0)
    
    _, grasp_idx = process_hybrid_sequence(expert_actions, chunk_size, 16)
    
    for i, B in enumerate(bands_list):
        recon_actions, _ = process_hybrid_sequence(expert_actions, chunk_size, B)
        recon_traj = np.cumsum(recon_actions[:, :3] * 0.05, axis=0)
        
        label = f"Bands={B}"
        alpha = 0.4 if B <= 4 else 0.8
        
        ax3d.plot(recon_traj[:, 0], recon_traj[:, 1], recon_traj[:, 2], 
                  color=colors[i], alpha=alpha, linewidth=2, label=label)
        ax2d.plot(recon_traj[:, 0], recon_traj[:, 1], color=colors[i], alpha=alpha, linewidth=2)
        
        error = np.linalg.norm(expert_traj - recon_traj, axis=1) * 1000 # mm
        ax_err.plot(error, color=colors[i], alpha=alpha, linewidth=2, label=label)

    # Highlight the Grasp Point
    grasp_pos = expert_traj[grasp_idx]
    ax3d.scatter(*grasp_pos, color='magenta', s=150, marker='*', label="Grasp Point (Filter Starts)")
    ax2d.scatter(grasp_pos[0], grasp_pos[1], color='magenta', s=150, marker='*')
    
    # Error plot styling
    ax_err.axvline(x=grasp_idx, color='magenta', linestyle='--', linewidth=2, label="Grasp Point")
    ax_err.set_title("Cumulative Translational Error (mm) - Note the zero error before grasp", fontsize=14)
    ax_err.set_xlabel("Time (Steps)")
    ax_err.set_ylabel("Error from Expert (mm)")
    ax_err.grid(True, linestyle='--', alpha=0.5)
    ax_err.legend(loc='upper left', ncol=4)
    
    ax3d.set_title("3D Trajectory Path (Overlapping Reach)", fontsize=14)
    ax3d.legend()
    ax2d.set_title("XY Projection", fontsize=14)
    
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/Isolated_Dashboard_Mixing.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    demo_path = "demonstration_data_shaking_and_accel/robosuite_ln_libero_tabletop_manipulation_1776085564_317537_pick_up_a_cup_with_a_lid_on_it_and_you_shake_it_to_mix_the_contents/demo.hdf5"
    out = "results/action_filtering/isolated_shake/graphs"
    if os.path.exists(demo_path):
        generate_isolated_dashboard(demo_path, out)
