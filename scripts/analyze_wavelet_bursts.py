import os
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.ndimage import label
from libero.libero.envs.problems.libero_tabletop_manipulation import Libero_Tabletop_Manipulation

# Constants
WAVELET_TYPE = 'db4'
NUM_LEVELS = 5
THRESHOLD_SIGMA = 2.0  # Threshold for energy peaks (multiplier of mean energy)
CLUSTER_GAP = 10       # Max steps between peaks to be in the same cluster
RESULTS_DIR = "results/wavelet_action/burst_analysis"

TASKS = [
    {
        "name": "Hammering",
        "demo_path": "demonstration_data_shaking_and_accel/robosuite_ln_libero_tabletop_manipulation_1776086280_428284_hammer_a_nail_into_a_board/demo.hdf5",
        "bddl": "/Users/ezraakresh/Documents/LIBERO/libero/libero/bddl_files/custom/hammer_nail_into_board.bddl"
    },
    {
        "name": "Shaking",
        "demo_path": "demonstration_data_shaking_and_accel/robosuite_ln_libero_tabletop_manipulation_1776085564_317537_pick_up_a_cup_with_a_lid_on_it_and_you_shake_it_to_mix_the_contents/demo.hdf5",
        "bddl": "/Users/ezraakresh/Documents/LIBERO/libero/libero/bddl_files/custom/shake_cup.bddl"
    },
    {
        "name": "Upright Flipped Cup",
        "demo_path": "demonstration_data_shaking_and_accel/robosuite_ln_libero_tabletop_manipulation_1776093696_115486_pick_up_a_cup_that_has_been_flipped_over_and_you_put_it_rightside_up/demo.hdf5",
        "bddl": "/Users/ezraakresh/Documents/LIBERO/libero/libero/bddl_files/custom/upright_flipped_cup.bddl"
    }
]

def upsample_signal(signal, target_len):
    if len(signal) == target_len:
        return signal
    x = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, target_len)
    f = interp1d(x, signal, axis=0, kind='linear', fill_value="extrapolate")
    return f(x_new)

def calculate_wavelet_energies(actions):
    """
    Decomposes actions into low, mid, and high frequency components.
    """
    T = actions.shape[0]
    # Use norm of action vector at each timestep for overall energy analysis
    action_norm = np.linalg.norm(actions, axis=1)
    
    coeffs = pywt.wavedec(action_norm, WAVELET_TYPE, level=NUM_LEVELS)
    
    low_energy = upsample_signal(coeffs[0]**2, T)
    
    # Mid: Middle levels
    mid_coeffs = coeffs[1:-2]
    mid_energy = np.zeros(T)
    for c in mid_coeffs:
        mid_energy += upsample_signal(c**2, T)
        
    # High: Last 2 detail levels (highest frequencies)
    high_coeffs = coeffs[-2:]
    high_energy = np.zeros(T)
    for c in high_coeffs:
        high_energy += upsample_signal(c**2, T)
        
    return low_energy, mid_energy, high_energy

def extract_physical_events(demo_file, bddl_file):
    print(f"  Extracting physical events from {os.path.basename(demo_file)}...")
    f = h5py.File(demo_file, "r")
    ep = "demo_1"
    states = f[f"data/{ep}/states"][()]
    actions = f[f"data/{ep}/actions"][()]
    env_info = json.loads(f["data"].attrs["env_info"])
    model_xml = f[f"data/{ep}"].attrs["model_file"]
    f.close()
    
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
    
    contacts = []
    eef_vels = []
    ori_errors = []
    
    # Pre-identify robot, table, and floor geoms
    robot_geom_ids = set()
    for i in range(env.sim.model.ngeom):
        geom_name = env.sim.model.geom_id2name(i)
        if "robot0" in geom_name or "gripper0" in geom_name:
            robot_geom_ids.add(i)
            
    table_geom_id = env.sim.model.geom_name2id("table_collision")
    floor_geom_id = env.sim.model.geom_name2id("floor_collision") if "floor_collision" in env.sim.model._geom_name2id else -1

    for step, action in enumerate(actions):
        env.step(action)
        
        # 1. Contact check: 
        # We want to detect:
        # - Robot contact with objects
        # - Object contact with other objects (e.g., hammer hitting nail)
        # We exclude contacts involving the table or floor as the primary pair
        in_contact = False
        for contact in env.sim.data.contact[:env.sim.data.ncon]:
            g1, g2 = contact.geom1, contact.geom2
            
            # Exclude table/floor contacts
            if g1 == table_geom_id or g1 == floor_geom_id or g2 == table_geom_id or g2 == floor_geom_id:
                continue
                
            # If we are here, it's a contact between two things that are NOT table/floor
            # This could be Robot-Object or Object-Object
            # We also exclude Robot-Robot contacts (self-collision)
            if g1 in robot_geom_ids and g2 in robot_geom_ids:
                continue
                
            in_contact = True
            break
        contacts.append(float(in_contact))
        
        # 2. Velocity: magnitude of EEF linear velocity
        vel = np.linalg.norm(env.sim.data.get_site_xvelp("gripper0_grip_site"))
        eef_vels.append(vel)
        
        # 3. Orientation Error: EEF angular velocity as a proxy for stabilization effort
        ang_vel = np.linalg.norm(env.sim.data.get_site_xvelr("gripper0_grip_site"))
        ori_errors.append(ang_vel)
        
    print(f"    -> Max Vel: {np.max(eef_vels):.4f}, Max Ang Vel: {np.max(ori_errors):.4f}, Contact Steps: {sum(contacts)}")
    return np.array(contacts), np.array(eef_vels), np.array(ori_errors), actions

def analyze_bursts(high_energy):
    mean_val = np.mean(high_energy)
    std_val = np.std(high_energy)
    threshold = mean_val + THRESHOLD_SIGMA * std_val
    
    active_mask = high_energy > threshold
    activity_ratio = np.mean(active_mask)
    
    # Use labeling to find contiguous bursts
    labeled_array, num_clusters = label(active_mask)
    
    if num_clusters == 0:
        return activity_ratio, 0, 0, threshold
    
    # Calculate durations of each cluster
    cluster_durations = []
    for i in range(1, num_clusters + 1):
        duration = np.sum(labeled_array == i)
        cluster_durations.append(duration)
        
    avg_duration = np.mean(cluster_durations)
    
    return activity_ratio, num_clusters, avg_duration, threshold

def run_experiment():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_stats = []
    
    for task in TASKS:
        name = task["name"]
        print(f"\n>>> Analyzing Task: {name}")
        
        if not os.path.exists(task["demo_path"]):
            print(f"  Warning: Path {task['demo_path']} does not exist. Skipping.")
            continue
            
        contacts, vels, ori_errors, actions = extract_physical_events(task["demo_path"], task["bddl"])
        low, mid, high = calculate_wavelet_energies(actions)
        
        ratio, num_clusters, avg_dur, thresh = analyze_bursts(high)
        
        summary_stats.append({
            "Task": name,
            "Activity Ratio": f"{ratio:.4f}",
            "Clusters": num_clusters,
            "Avg Cluster Duration": f"{avg_dur:.2f}"
        })
        
        # Visualization
        plt.figure(figsize=(15, 12))
        
        # Panel 1: Energy Distribution (Log Scale for visibility)
        ax1 = plt.subplot(3, 1, 1)
        T = np.arange(len(low))
        plt.plot(T, low, label='Low Freq (Approx)', color='#1f77b4', alpha=0.8)
        plt.plot(T, mid, label='Mid Freq (Details)', color='#ff7f0e', alpha=0.8)
        plt.plot(T, high, label='High Freq (Details)', color='#2ca02c', alpha=0.9)
        plt.yscale('log')
        plt.title(f"Wavelet Energy Spectrum (Log Scale): {name}", fontsize=14)
        plt.ylabel("Energy (Log Scale)")
        plt.legend(loc='upper right')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        # Panel 2: High-Frequency Energy + Threshold (The "Burst" Panel)
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        plt.plot(T, high, color='#2ca02c', linewidth=1.5, label='High Frequency Energy')
        plt.axhline(thresh, color='red', linestyle='--', alpha=0.6, label=f'Threshold ({THRESHOLD_SIGMA}σ)')
        plt.fill_between(T, 0, high, where=(high > thresh), color='red', alpha=0.2, label='Active Bursts')
        plt.title(f"High-Frequency Bursts | Activity Ratio: {ratio:.4f} | Clusters: {num_clusters} | Avg Dur: {avg_dur:.2f}", fontsize=14)
        plt.ylabel("HF Energy")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        # Panel 3: Physical Events Overlay
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        
        # Normalize event signals for visualization
        def norm(s): 
            if np.max(s) == np.min(s): return np.zeros_like(s)
            return (s - np.min(s)) / (np.max(s) - np.min(s) + 1e-6)
        
        plt.plot(T, norm(vels), color='blue', linewidth=1.5, label='EEF Velocity (norm)', alpha=0.7)
        plt.plot(T, norm(ori_errors), color='orange', linewidth=1.5, label='EEF Ang Vel (norm)', alpha=0.7)
        
        # Contact events as background highlights
        plt.fill_between(T, 0, 1.0, where=(contacts > 0.5), color='grey', alpha=0.2, label='Contact Period')
        
        plt.title("Physical Events Correlation", fontsize=14)
        plt.xlabel("Timestep")
        plt.ylabel("Normalized Signal")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = f"{RESULTS_DIR}/{name.replace(' ', '_')}_Wavelet_Burst_Analysis.png"
        plt.savefig(save_path, dpi=150)
        print(f"  Saved plot to {save_path}")
        plt.close()

        # --- VIDEO GENERATION ---
        print(f"  Generating synced analysis video for {name}...")
        import imageio
        from PIL import Image
        
        video_path = f"{RESULTS_DIR}/{name.replace(' ', '_')}_Synced_Analysis.mp4"
        writer = imageio.get_writer(video_path, fps=10)
        
        # Prepare the base plot (static background)
        fig_vid, ax_vid = plt.subplots(figsize=(8, 4))
        ax_vid.plot(T, high, color='#2ca02c', linewidth=1.5)
        ax_vid.axhline(thresh, color='red', linestyle='--', alpha=0.6)
        ax_vid.set_title(f"High-Frequency Analysis: {name}", fontsize=12)
        ax_vid.set_ylabel("HF Energy")
        ax_vid.set_xlabel("Timestep")
        ax_vid.grid(True, alpha=0.3)
        
        # We'll add the ticker line in the loop
        ticker = ax_vid.axvline(0, color='red', linewidth=2)
        
        # Re-extracting frames (to save memory, we do it step-by-step or use the ones from before if we had them)
        # For simplicity and to ensure sync, I'll re-run a quick render loop or use captured frames
        # Let's assume we capture them in extract_physical_events now.
        
    print("\n========================================")
    print("WAVELET BURST EXPERIMENT SUMMARY")
    print("========================================")
    for s in summary_stats:
        print(f"Task: {s['Task']:<20} | Ratio: {s['Activity Ratio']} | Clusters: {s['Clusters']:>3} | Dur: {s['Avg Cluster Duration']}")

def create_synced_video(name, T, high, thresh, frames, out_path):
    import imageio
    from PIL import Image
    import io

    print(f"  Creating synced video at {out_path}...")
    writer = imageio.get_writer(out_path, fps=10)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    for i, frame in enumerate(frames):
        # frame index i corresponds to simulation step i * 2
        sim_step = i * 2
        if sim_step >= len(T): break
        
        ax.clear()
        # Plot full resolution background
        ax.plot(T, high, color='#2ca02c', linewidth=1.5, label='HF Energy', alpha=0.8)
        ax.axhline(thresh, color='red', linestyle='--', alpha=0.6)
        
        # Ticker at the correct sim step
        ax.axvline(sim_step, color='red', linewidth=2, label='Current Time')
        
        ax.set_title(f"High-Frequency Energy Profile: {name}", fontsize=12)
        ax.set_ylabel("HF Energy")
        ax.set_xlim(0, len(T))
        ax.set_ylim(0, np.max(high) * 1.1)
        ax.grid(True, alpha=0.3)
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80)
        buf.seek(0)
        plot_img = np.array(Image.open(buf))
        
        # Resize plot image to match frame width if needed
        h_f, w_f, _ = frame.shape
        h_p, w_p, _ = plot_img.shape
        
        # Resize plot to match frame width
        new_h_p = int(h_p * (w_f / w_p))
        plot_img_resized = np.array(Image.fromarray(plot_img).resize((w_f, new_h_p)))
        
        # Stack
        combined = np.vstack((frame, plot_img_resized[:, :, :3]))
        writer.append_data(combined)
        
    writer.close()
    plt.close(fig)

def extract_physical_events(demo_file, bddl_file):
    print(f"  Extracting physical events and frames from {os.path.basename(demo_file)}...")
    f = h5py.File(demo_file, "r")
    ep = "demo_1"
    states = f[f"data/{ep}/states"][()]
    actions = f[f"data/{ep}/actions"][()]
    env_info = json.loads(f["data"].attrs["env_info"])
    model_xml = f[f"data/{ep}"].attrs["model_file"]
    f.close()
    
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
    
    contacts = []
    eef_vels = []
    ori_errors = []
    video_frames = []
    
    # Pre-identify robot, table, and floor geoms
    robot_geom_ids = set()
    for i in range(env.sim.model.ngeom):
        geom_name = env.sim.model.geom_id2name(i)
        if "robot0" in geom_name or "gripper0" in geom_name:
            robot_geom_ids.add(i)
            
    table_geom_id = env.sim.model.geom_name2id("table_collision")
    floor_geom_id = env.sim.model.geom_name2id("floor_collision") if "floor_collision" in env.sim.model._geom_name2id else -1

    for step, action in enumerate(actions):
        env.step(action)
        
        # Render every 2nd frame for video to save time/space
        if step % 2 == 0:
            frame = env.sim.render(width=512, height=512, camera_name="agentview")[::-1]
            video_frames.append(frame)
        
        # 1. Contact check
        in_contact = False
        for contact in env.sim.data.contact[:env.sim.data.ncon]:
            g1, g2 = contact.geom1, contact.geom2
            if g1 == table_geom_id or g1 == floor_geom_id or g2 == table_geom_id or g2 == floor_geom_id:
                continue
            if g1 in robot_geom_ids and g2 in robot_geom_ids:
                continue
            in_contact = True
            break
        contacts.append(float(in_contact))
        
        vel = np.linalg.norm(env.sim.data.get_site_xvelp("gripper0_grip_site"))
        eef_vels.append(vel)
        
        ang_vel = np.linalg.norm(env.sim.data.get_site_xvelr("gripper0_grip_site"))
        ori_errors.append(ang_vel)
        
    return np.array(contacts), np.array(eef_vels), np.array(ori_errors), actions, video_frames

def run_experiment():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_stats = []
    
    for task in TASKS:
        name = task["name"]
        print(f"\n>>> Analyzing Task: {name}")
        
        if not os.path.exists(task["demo_path"]):
            print(f"  Warning: Path {task['demo_path']} does not exist. Skipping.")
            continue
            
        contacts, vels, ori_errors, actions, frames = extract_physical_events(task["demo_path"], task["bddl"])
        low, mid, high = calculate_wavelet_energies(actions)
        
        ratio, num_clusters, avg_dur, thresh = analyze_bursts(high)
        
        summary_stats.append({
            "Task": name,
            "Activity Ratio": f"{ratio:.4f}",
            "Clusters": num_clusters,
            "Avg Cluster Duration": f"{avg_dur:.2f}"
        })
        
        # Panel plotting...
        T = np.arange(len(low))
        plt.figure(figsize=(15, 12))
        # (Same plotting code as before)
        # Panel 1: Energy Distribution (Log Scale)
        ax1 = plt.subplot(3, 1, 1)
        plt.plot(T, low, label='Low Freq (Approx)', color='#1f77b4', alpha=0.8)
        plt.plot(T, mid, label='Mid Freq (Details)', color='#ff7f0e', alpha=0.8)
        plt.plot(T, high, label='High Freq (Details)', color='#2ca02c', alpha=0.9)
        plt.yscale('log')
        plt.title(f"Wavelet Energy Spectrum (Log Scale): {name}", fontsize=14)
        plt.ylabel("Energy (Log Scale)")
        plt.legend(loc='upper right')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        # Panel 2: High-Frequency Energy + Threshold
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        plt.plot(T, high, color='#2ca02c', linewidth=1.5, label='High Frequency Energy')
        plt.axhline(thresh, color='red', linestyle='--', alpha=0.6, label=f'Threshold ({THRESHOLD_SIGMA}σ)')
        plt.fill_between(T, 0, high, where=(high > thresh), color='red', alpha=0.2, label='Active Bursts')
        plt.title(f"High-Frequency Bursts | Activity Ratio: {ratio:.4f} | Clusters: {num_clusters} | Avg Dur: {avg_dur:.2f}", fontsize=14)
        plt.ylabel("HF Energy")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        # Panel 3: Physical Events Overlay
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        def norm(s): 
            if np.max(s) == np.min(s): return np.zeros_like(s)
            return (s - np.min(s)) / (np.max(s) - np.min(s) + 1e-6)
        plt.plot(T, norm(vels), color='blue', linewidth=1.5, label='EEF Velocity (norm)', alpha=0.7)
        plt.plot(T, norm(ori_errors), color='orange', linewidth=1.5, label='EEF Ang Vel (norm)', alpha=0.7)
        plt.fill_between(T, 0, 1.0, where=(contacts > 0.5), color='grey', alpha=0.2, label='Contact Period')
        plt.title("Physical Events Correlation", fontsize=14)
        plt.xlabel("Timestep")
        plt.ylabel("Normalized Signal")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = f"{RESULTS_DIR}/{name.replace(' ', '_')}_Wavelet_Burst_Analysis.png"
        plt.savefig(save_path, dpi=150)
        plt.close()

        # Create synced video
        video_out = f"{RESULTS_DIR}/{name.replace(' ', '_')}_Synced_Analysis.mp4"
        # Pass FULL resolution data for the graph background
        create_synced_video(name, T, high, thresh, frames, video_out)

    # Save summary as text/json
    with open(f"{RESULTS_DIR}/summary_stats.json", "w") as f:
        json.dump(summary_stats, f, indent=4)
        
    print("\n========================================")
    print("WAVELET BURST EXPERIMENT SUMMARY")
    print("========================================")
    for s in summary_stats:
        print(f"Task: {s['Task']:<20} | Ratio: {s['Activity Ratio']} | Clusters: {s['Clusters']:>3} | Dur: {s['Avg Cluster Duration']}")

if __name__ == "__main__":
    run_experiment()
