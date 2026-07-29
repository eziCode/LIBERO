import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pywt

def wavelet_filter_sequence(actions, wavelet_type='db4', levels_to_remove=1):
    if levels_to_remove == 0:
        return np.copy(actions)
    coeffs = pywt.wavedec(actions, wavelet_type, axis=0)
    for i in range(1, min(levels_to_remove + 1, len(coeffs))):
        coeffs[-i] = np.zeros_like(coeffs[-i])
    filtered = pywt.waverec(coeffs, wavelet_type, axis=0)
    return filtered[:actions.shape[0]]

def generate_wavelet_dashboard(demo_file, out_dir):
    print("Generating Wavelet Dashboard for Hammering Task...")
    f = h5py.File(demo_file, "r")
    expert_actions = f["data/demo_1/actions"][()]
    f.close()
    
    # We sweep over levels removed
    levels_list = [1, 2, 3]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Calculate Velocity (diff of positions) and Acceleration (diff of velocity)
    # The hammering motion is mainly Z-axis (index 2)
    expert_vel = np.diff(expert_actions[:, 2]) * 20 # scaled to roughly represent velocity
    expert_accel = np.diff(expert_vel) * 20
    
    fig, (ax_vel, ax_accel) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("Wavelet Fidelity: Hammering Impact (No Gibbs Ringing)", fontsize=18, fontweight='bold')
    
    # We zoom in on the actual strike (around frame 100-250 for hammering)
    strike_start = 100
    strike_end = 250
    
    # Plot Expert
    ax_vel.plot(expert_vel[strike_start:strike_end], color='grey', linewidth=3, label="Expert (Raw)", alpha=0.8)
    ax_accel.plot(expert_accel[strike_start:strike_end], color='grey', linewidth=3, label="Expert (Raw)", alpha=0.8)
    
    for i, L in enumerate(levels_list):
        recon_actions = wavelet_filter_sequence(expert_actions, 'db4', L)
        recon_vel = np.diff(recon_actions[:, 2]) * 20
        recon_accel = np.diff(recon_vel) * 20
        
        label = f"Wavelet (L={L} Removed)"
        
        ax_vel.plot(recon_vel[strike_start:strike_end], color=colors[i], linewidth=2, label=label, alpha=0.9, linestyle='--')
        ax_accel.plot(recon_accel[strike_start:strike_end], color=colors[i], linewidth=2, label=label, alpha=0.9, linestyle='--')

    ax_vel.set_title("Vertical Velocity Profile (Preserving the Impact Spike)", fontsize=14)
    ax_vel.set_ylabel("Z Velocity")
    ax_vel.grid(True, linestyle='--', alpha=0.5)
    ax_vel.legend()
    
    ax_accel.set_title("Vertical Acceleration Profile (No Pre-Strike Wobble)", fontsize=14)
    ax_accel.set_ylabel("Z Acceleration")
    ax_accel.grid(True, linestyle='--', alpha=0.5)
    ax_accel.legend()
    
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/Wavelet_Dashboard_Hammering.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    demo_path = "demonstration_data_shaking_and_accel/robosuite_ln_libero_tabletop_manipulation_1776086280_428284_hammer_a_nail_into_a_board/demo.hdf5"
    out = "results/wavelet_action/hammering/graphs"
    if os.path.exists(demo_path):
        generate_wavelet_dashboard(demo_path, out)
