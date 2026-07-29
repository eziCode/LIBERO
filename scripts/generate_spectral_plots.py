import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

def spectral_filter(action_chunk, keep_ratio=0.5):
    if keep_ratio >= 1.0:
        return action_chunk
    coeffs = dct(action_chunk, axis=0, norm='ortho')
    L = len(coeffs)
    cutoff = max(1, int(L * keep_ratio))
    coeffs[cutoff:] = 0
    return idct(coeffs, axis=0, norm='ortho')

def process_full_sequence(actions, chunk_size, keep_ratio):
    if chunk_size == 'Full':
        return spectral_filter(actions, keep_ratio)
    
    num_steps = actions.shape[0]
    processed_actions = np.zeros_like(actions)
    for i in range(0, num_steps, chunk_size):
        end = min(i + chunk_size, num_steps)
        chunk = actions[i:end]
        processed_actions[i:end] = spectral_filter(chunk, keep_ratio)
    return processed_actions

def generate_task_plots(demo_file, task_name):
    print(f"Generating plots for: {task_name}")
    os.makedirs("results/plots", exist_ok=True)
    
    f = h5py.File(demo_file, "r")
    ep = "demo_1"
    original_actions = f[f"data/{ep}/actions"][()]
    f.close()
    
    # Target Configurations
    chunk_sizes = [16, 32, 64, 128, "Full"]
    keep_ratios = [1.0, 0.5, 0.25, 0.1, 0.05]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for L in chunk_sizes:
        plt.figure(figsize=(12, 6))
        
        # Plot Original in background
        plt.plot(original_actions[:, 0], color='black', alpha=0.2, linewidth=3, label="Original Expert")
        
        # Plot each Keep Ratio
        for i, P in enumerate(keep_ratios):
            reconstructed = process_full_sequence(original_actions, L, P)
            plt.plot(reconstructed[:, 0], color=colors[i], linewidth=1.5, label=f"Ratio {P}")
        
        plt.title(f"Spectral Fidelity Comparison (Task: {task_name}, Chunk Size: {L})")
        plt.xlabel("Time (Steps)")
        plt.ylabel("Action Value (X-Dim)")
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = f"results/plots/{task_name}_Chunk{L}_Comparison.png"
        plt.savefig(plot_path)
        print(f"  Saved: {plot_path}")
        plt.close()

if __name__ == "__main__":
    demo_dirs = [
        "demonstration_data_shaking_and_accel/robosuite_ln_libero_tabletop_manipulation_1776085564_317537_pick_up_a_cup_with_a_lid_on_it_and_you_shake_it_to_mix_the_contents",
        "demonstration_data_shaking_and_accel/robosuite_ln_libero_tabletop_manipulation_1776086280_428284_hammer_a_nail_into_a_board",
        "demonstration_data_shaking_and_accel/robosuite_ln_libero_tabletop_manipulation_1776093696_115486_pick_up_a_cup_that_has_been_flipped_over_and_you_put_it_rightside_up"
    ]
    
    for d in demo_dirs:
        task_id = "_".join(d.split("/")[-1].split("_")[-4:])
        generate_task_plots(os.path.join(d, "demo.hdf5"), task_id)
