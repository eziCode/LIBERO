import cv2
import numpy as np
import os

def create_triple_side_by_side(v1, v2, v3, out_path, labels):
    cap1 = cv2.VideoCapture(v1)
    cap2 = cv2.VideoCapture(v2)
    cap3 = cv2.VideoCapture(v3)
    
    w = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w * 3, h))
    
    print(f"Generating Triple View: {out_path}")
    
    count = 0
    while True:
        ret1, f1 = cap1.read()
        ret2, f2 = cap2.read()
        ret3, f3 = cap3.read()
        
        if not ret1 and not ret2 and not ret3:
            break
            
        def pad(ret, f):
            if not ret: return np.zeros((h, w, 3), dtype=np.uint8)
            return cv2.resize(f, (w, h)) if (f.shape[1], f.shape[0]) != (w, h) else f

        f1, f2, f3 = pad(ret1, f1), pad(ret2, f2), pad(ret3, f3)
        combined = np.hstack((f1, f2, f3))
        
        # Labels
        cv2.rectangle(combined, (0, 0), (w*3, 40), (0,0,0), -1)
        colors = [(255, 255, 255), (0, 255, 0), (0, 165, 255)]
        for i, label in enumerate(labels):
            cv2.putText(combined, label, (i*w + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
        
        out.write(combined)
        count += 1
        
    cap1.release()
    cap2.release()
    cap3.release()
    out.release()
    print(f"Saved {count} frames.")

if __name__ == "__main__":
    tasks = ["to_mix_the_contents", "nail_into_a_board", "put_it_rightside_up"]
    vid_dir = "results/action_filtering/sweep/videos"
    out_dir = "results/action_filtering/comparisons/overlays"
    os.makedirs(out_dir, exist_ok=True)
    
    labels = ["Expert", "Gripper-Phase (Ratio 0.5)", "Gripper-Phase (Ratio 0.1)"]
    
    for tid in tasks:
        v1 = f"{vid_dir}/{tid}_Lgripper_P1.0.mp4"
        v2 = f"{vid_dir}/{tid}_Lgripper_P0.5.mp4"
        v3 = f"{vid_dir}/{tid}_Lgripper_P0.1.mp4"
        out = f"{out_dir}/{tid}_LGripper_TripleView.mp4"
        
        if os.path.exists(v1) and os.path.exists(v2) and os.path.exists(v3):
            create_triple_side_by_side(v1, v2, v3, out, labels)
        else:
            print(f"Skipping {tid}: Files not found. (V1: {os.path.exists(v1)}, V2: {os.path.exists(v2)}, V3: {os.path.exists(v3)})")
