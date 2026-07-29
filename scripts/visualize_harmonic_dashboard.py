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

def process_sequence_harmonic(actions, chunk_size, num_bands):
    processed = np.zeros_like(actions)
    for i in range(0, actions.shape[0], chunk_size):
        end = min(i + chunk_size, actions.shape[0])
        processed[i:end] = spectral_filter_bands(actions[i:end], num_bands)
    return processed

def generate_dashboard(demo_file, task_name, out_dir):
    print(f"Processing Dashboard for: {task_name}")
    f = h5py.File(demo_file, "r")
    original_actions = f["data/demo_1/actions"][()]
    f.close()
    
    bands_list = [16, 13, 10, 7, 4, 1]
    chunk_size = 32
    colors = ['#444444', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"Harmonic Fidelity Dashboard: {task_name} (Chunk=32)", fontsize=18, fontweight='bold')
    
    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    ax2d = fig.add_subplot(2, 2, 2)
    ax_err = fig.add_subplot(2, 1, 2)
    
    expert_traj = np.cumsum(original_actions[:, :3] * 0.05, axis=0)
    
    for i, B in enumerate(bands_list):
        recon_actions = process_sequence_harmonic(original_actions, chunk_size, B)
        recon_traj = np.cumsum(recon_actions[:, :3] * 0.05, axis=0)
        
        label = f"Bands={B}"
        alpha = 0.3 if B == 1 else 0.8
        
        ax3d.plot(recon_traj[:, 0], recon_traj[:, 1], recon_traj[:, 2], 
                  color=colors[i], alpha=alpha, linewidth=2, label=label)
        ax2d.plot(recon_traj[:, 0], recon_traj[:, 1], color=colors[i], alpha=alpha, linewidth=2)
        
        error = np.linalg.norm(expert_traj - recon_traj, axis=1) * 1000 # to mm
        ax_err.plot(error, color=colors[i], alpha=alpha, linewidth=2, label=label)

    # Styling
    ax3d.set_title("3D Trajectory Path", fontsize=14)
    ax2d.set_title("XY Projection", fontsize=14)
    ax_err.set_title("Cumlative Translational Error (mm)", fontsize=14)
    ax_err.set_xlabel("Time (Steps)")
    ax_err.set_ylabel("Error from Expert (mm)")
    ax_err.grid(True, linestyle='--', alpha=0.5)
    ax_err.legend(loc='upper left', ncol=3)
    
    os.makedirs(out_dir, exist_ok=True)
    save_path = f"{out_dir}/HARMONIC_DASHBOARD_{task_name}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    demo_base = "demonstration_data_shaking_and_accel"
    tasks = {
        "to_mix_the_contents": "robosuite_ln_libero_tabletop_manipulation_1776085564_317537_pick_up_a_cup_with_a_lid_on_it_and_you_shake_it_to_mix_the_contents",
        "nail_into_a_board": "robosuite_ln_libero_tabletop_manipulation_1776086280_428284_hammer_a_nail_into_a_board",
        "put_it_rightside_up": "robosuite_ln_libero_tabletop_manipulation_1776093696_115486_pick_up_a_cup_that_has_been_flipped_over_and_you_put_it_rightside_up"
    }
    out_path = "/Users/ezraakresh/Documents/LIBERO/results/chunk_32_no_keep_ratio"
    
    for name, folder in tasks.items():
        h5_path = os.path.join(demo_base, folder, "demo.hdf5")
        generate_dashboard(h5_path, name, out_path)
