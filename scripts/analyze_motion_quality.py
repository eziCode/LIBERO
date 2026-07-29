import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

def spectral_filter(actions, keep_ratio=0.05):
    """Applies a heavy spectral filter to the entire continuous sequence."""
    coeffs = dct(actions, axis=0, norm='ortho')
    cutoff = max(1, int(len(coeffs) * keep_ratio))
    coeffs[cutoff:] = 0
    return idct(coeffs, axis=0, norm='ortho')

def analyze_mixing_diagonal(hdf5_path, out_dir):
    f = h5py.File(hdf5_path, "r")
    expert_actions = f["data/demo_1/actions"][()]
    f.close()
    
    filtered_actions = spectral_filter(expert_actions, keep_ratio=0.03) # 3% - Heavy filter
    
    # Reconstruct positions (integration)
    expert_pos = np.cumsum(expert_actions[:, :3] * 0.05, axis=0)
    filtered_pos = np.cumsum(filtered_actions[:, :3] * 0.05, axis=0)
    
    # Plotting Y (Lateral Shake) vs Z (Vertical Lift)
    plt.figure(figsize=(10, 8))
    plt.title("Mixing Task: The 'Diagonal Wiggle' Effect (Cross-Axis Coupling)", fontsize=14)
    
    # Extract the interaction phase roughly based on Z height or gripper
    # We will plot the whole trajectory to see the reach -> lift -> shake
    plt.plot(expert_pos[:, 1], expert_pos[:, 2], label="Expert (Raw)", color='black', alpha=0.6, linewidth=1.5)
    plt.plot(filtered_pos[:, 1], filtered_pos[:, 2], label="Filtered (3% Fidelity)", color='red', linewidth=3)
    
    # Mark start and end
    plt.scatter(expert_pos[0, 1], expert_pos[0, 2], color='green', marker='o', s=100, label='Start')
    plt.scatter(expert_pos[-1, 1], expert_pos[-1, 2], color='blue', marker='x', s=100, label='End')
    
    plt.xlabel("Y Position (Lateral Shake Axis)")
    plt.ylabel("Z Position (Vertical Lift Axis)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "ANALYSIS_Mixing_Diagonal.png"), dpi=200)
    plt.close()

def analyze_rotation_continuity(hdf5_path, out_dir):
    f = h5py.File(hdf5_path, "r")
    expert_actions = f["data/demo_1/actions"][()]
    f.close()
    
    filtered_actions = spectral_filter(expert_actions, keep_ratio=0.05) # 5%
    
    # Actions are velocities. Let's look at rotation (index 3, 4, 5)
    # The flip is usually a strong rotation around X or Y. Let's find the max variance axis.
    rot_variance = np.var(expert_actions[:, 3:6], axis=0)
    dom_axis = np.argmax(rot_variance) + 3
    axis_names = ['RX', 'RY', 'RZ']
    
    expert_vel = expert_actions[:, dom_axis]
    filtered_vel = filtered_actions[:, dom_axis]
    
    # Acceleration (Delta Velocity)
    expert_acc = np.diff(expert_vel)
    filtered_acc = np.diff(filtered_vel)
    
    time = np.arange(len(expert_vel)) / 20.0
    
    plt.figure(figsize=(12, 10))
    plt.suptitle(f"Flip Cup Task: Rotational Continuity Analysis ({axis_names[dom_axis-3]} Axis)", fontsize=16)
    
    plt.subplot(2, 1, 1)
    plt.plot(time, expert_vel, label="Expert Velocity", color='black', alpha=0.4)
    plt.plot(time, filtered_vel, label="Filtered Velocity (Smoothed)", color='red', linewidth=2)
    plt.title("Commanded Angular Velocity (The 'Flip' Momentum)")
    plt.ylabel("Velocity (rad/step)")
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    plt.subplot(2, 1, 2)
    plt.plot(time[:-1], expert_acc, label="Expert Acceleration", color='black', alpha=0.3)
    plt.plot(time[:-1], filtered_acc, label="Filtered Acceleration (Artifacts)", color='blue', linewidth=2)
    plt.title("Angular Acceleration (The 'Snap' & Artifact Detection)")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration")
    
    # Highlight potential Gibbs ringing artifacts where filtered acceleration oscillates heavily
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(out_dir, "ANALYSIS_FlipCup_Continuity.png"), dpi=200)
    plt.close()

def analyze_hammering_impact(hdf5_path, out_dir):
    f = h5py.File(hdf5_path, "r")
    expert_actions = f["data/demo_1/actions"][()]
    f.close()
    
    filtered_actions = spectral_filter(expert_actions, keep_ratio=0.05) # 5%
    
    # Hammering is dominated by downward Z-velocity (index 2)
    expert_vel_z = expert_actions[:, 2]
    filtered_vel_z = filtered_actions[:, 2]
    
    # Acceleration
    expert_acc_z = np.diff(expert_vel_z)
    filtered_acc_z = np.diff(filtered_vel_z)
    
    time = np.arange(len(expert_vel_z)) / 20.0
    
    plt.figure(figsize=(12, 10))
    plt.suptitle("Hammering Task: Vertical Impact & Continuity Analysis", fontsize=16)
    
    plt.subplot(2, 1, 1)
    plt.plot(time, expert_vel_z, label="Expert Z-Velocity", color='black', alpha=0.4)
    plt.plot(time, filtered_vel_z, label="Filtered Z-Velocity", color='red', linewidth=2)
    plt.title("Commanded Vertical Velocity (The 'Strike' Momentum)")
    plt.ylabel("Velocity (m/step)")
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    plt.subplot(2, 1, 2)
    plt.plot(time[:-1], expert_acc_z, label="Expert Z-Acceleration", color='black', alpha=0.3)
    plt.plot(time[:-1], filtered_acc_z, label="Filtered Z-Acceleration (Ringing)", color='blue', linewidth=2)
    plt.title("Vertical Acceleration (Jerk Profile & Artifact Detection)")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration")
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(out_dir, "ANALYSIS_Hammering_Impact.png"), dpi=200)
    plt.close()

if __name__ == "__main__":
    out_path = "results/action_filtering/motion_fidelity"
    
    mix_path = "demonstration_data_shaking_and_accel/robosuite_ln_libero_tabletop_manipulation_1776085564_317537_pick_up_a_cup_with_a_lid_on_it_and_you_shake_it_to_mix_the_contents/demo.hdf5"
    flip_path = "demonstration_data_shaking_and_accel/robosuite_ln_libero_tabletop_manipulation_1776093696_115486_pick_up_a_cup_that_has_been_flipped_over_and_you_put_it_rightside_up/demo.hdf5"
    hammer_path = "demonstration_data_shaking_and_accel/robosuite_ln_libero_tabletop_manipulation_1776086280_428284_hammer_a_nail_into_a_board/demo.hdf5"
    
    if os.path.exists(mix_path):
        analyze_mixing_diagonal(mix_path, out_path)
        print("Generated Mixing Phase Distortion Plot.")
    
    if os.path.exists(flip_path):
        analyze_rotation_continuity(flip_path, out_path)
        print("Generated Flip Cup Continuity Plot.")
        
    if os.path.exists(hammer_path):
        analyze_hammering_impact(hammer_path, out_path)
        print("Generated Hammering Impact Plot.")
