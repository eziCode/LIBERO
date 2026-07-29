import os
import h5py
import json
import numpy as np
from scipy.stats import entropy

def extract_chunks(actions_list, chunk_size, stride=None):
    if stride is None:
        stride = chunk_size // 2
    chunks = []
    for seq in actions_list:
        if len(seq) < chunk_size:
            continue
        for start in range(0, len(seq) - chunk_size + 1, stride):
            chunks.append(seq[start:start+chunk_size])
    if not chunks:
        return np.empty((0, chunk_size, actions_list[0].shape[1] if actions_list else 7))
    return np.array(chunks)

def compute_action_kl(orig_chunks, teleop_chunks, bins=50):
    if len(orig_chunks) == 0 or len(teleop_chunks) == 0:
        return np.nan
        
    D = orig_chunks.shape[-1]
    kl_divs = []
    
    for d in range(D):
        orig_vals = orig_chunks[..., d].flatten()
        teleop_vals = teleop_chunks[..., d].flatten()
        
        min_val = min(np.min(orig_vals), np.min(teleop_vals)) - 1e-4
        max_val = max(np.max(orig_vals), np.max(teleop_vals)) + 1e-4
        
        P, _ = np.histogram(orig_vals, bins=bins, range=(min_val, max_val), density=True)
        Q, _ = np.histogram(teleop_vals, bins=bins, range=(min_val, max_val), density=True)
        
        P = P + 1e-8
        Q = Q + 1e-8
        
        P /= P.sum()
        Q /= Q.sum()
        
        kl_divs.append(entropy(P, Q))
        
    return np.mean(kl_divs)

def compute_freq_kl(orig_chunks, teleop_chunks):
    if len(orig_chunks) == 0 or len(teleop_chunks) == 0:
        return np.nan
        
    D = orig_chunks.shape[-1]
    kl_divs = []
    
    for d in range(D):
        orig_seqs = orig_chunks[..., d] 
        teleop_seqs = teleop_chunks[..., d] 
        
        # Absolute of rfft magnitude
        orig_fft = np.abs(np.fft.rfft(orig_seqs, axis=1)) 
        teleop_fft = np.abs(np.fft.rfft(teleop_seqs, axis=1)) 
        
        # Average spectrum across all chunks
        P = np.mean(orig_fft, axis=0) + 1e-8
        Q = np.mean(teleop_fft, axis=0) + 1e-8
        
        P /= P.sum()
        Q /= Q.sum()
        
        kl_divs.append(entropy(P, Q))
        
    return np.mean(kl_divs)

def analyze():
    teleop_dir = "demonstration_data"
    original_data_dir = "datasets/libero_90"
    
    teleop_folders = [d for d in os.listdir(teleop_dir) if os.path.isdir(os.path.join(teleop_dir, d)) and "robosuite_ln_libero_" in d]
    
    results = {}
    
    for folder in teleop_folders:
        teleop_h5_file = os.path.join(teleop_dir, folder, "demo.hdf5")
        if not os.path.exists(teleop_h5_file):
            continue
            
        with h5py.File(teleop_h5_file, "r") as f:
            bddl_file = ""
            if "env_args" in f["data"].attrs:
                env_args = json.loads(f["data"].attrs["env_args"])
                bddl_file = env_args["bddl_file_name"]
            elif "bddl_file_name" in f["data"].attrs:
                bddl_file = f["data"].attrs["bddl_file_name"]
            
            if not bddl_file:
                continue
                
            task_basename = os.path.basename(bddl_file).replace(".bddl", "")
            original_h5_file = os.path.join(original_data_dir, f"{task_basename}_demo.hdf5")
            
            if not os.path.exists(original_h5_file):
                continue
                
            teleop_actions = []
            for demo in f["data"]:
                teleop_actions.append(f[f"data/{demo}/actions"][()])
                
            with h5py.File(original_h5_file, "r") as orig_f:
                orig_actions = []
                for demo in orig_f["data"]:
                    orig_actions.append(orig_f[f"data/{demo}/actions"][()])
                    
            results[task_basename] = {"orig_demos": len(orig_actions), "teleop_demos": len(teleop_actions)}
            
            chunk_sizes = [16, 32, 64]
            for W in chunk_sizes:
                orig_chunks = extract_chunks(orig_actions, W)
                teleop_chunks = extract_chunks(teleop_actions, W)
                
                action_kl = compute_action_kl(orig_chunks, teleop_chunks)
                freq_kl = compute_freq_kl(orig_chunks, teleop_chunks)
                
                results[task_basename][f"W{W}_action_kl"] = float(action_kl)
                results[task_basename][f"W{W}_freq_kl"] = float(freq_kl)
                
    with open("kl_analysis_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Metrics written to kl_analysis_results.json")

if __name__ == "__main__":
    analyze()
