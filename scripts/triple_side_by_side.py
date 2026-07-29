import cv2
import numpy as np
import sys
import os

def create_triple_side_by_side(v1, v2, v3, out_path, labels):
    cap1 = cv2.VideoCapture(v1)
    cap2 = cv2.VideoCapture(v2)
    cap3 = cv2.VideoCapture(v3)
    
    # Metadata
    w = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    
    # Output is triple width
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w * 3, h))
    
    print(f"Generating Triple Side-by-Side: {out_path}")
    
    count = 0
    while True:
        ret1, f1 = cap1.read()
        ret2, f2 = cap2.read()
        ret3, f3 = cap3.read()
        
        if not ret1 and not ret2 and not ret3:
            break
            
        # Padding for shorter videos
        def pad_frame(ret, f):
            if not ret: return np.zeros((h, w, 3), dtype=np.uint8)
            if (f.shape[1], f.shape[0]) != (w, h): return cv2.resize(f, (w, h))
            return f

        f1 = pad_frame(ret1, f1)
        f2 = pad_frame(ret2, f2)
        f3 = pad_frame(ret3, f3)
            
        # Combine
        combined = np.hstack((f1, f2, f3))
        
        # Add Header Labels
        cv2.rectangle(combined, (0, 0), (w*3, 40), (0,0,0), -1)
        cv2.putText(combined, labels[0], (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, labels[1], (w + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, labels[2], (w*2 + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        out.write(combined)
        count += 1
        
    cap1.release()
    cap2.release()
    cap3.release()
    out.release()
    print(f"Done! {count} frames saved.")

if __name__ == "__main__":
    task_ids = ["to_mix_the_contents", "nail_into_a_board", "put_it_rightside_up"]
    base_dir = "results/videos"
    out_dir = "results/overlayed_videos"
    os.makedirs(out_dir, exist_ok=True)
    
    config_labels = ["Expert", "keep_ratio=0.5", "keep_ratio=0.1"]
    
    for tid in task_ids:
        v1 = f"{base_dir}/{tid}_L16_P1.0.mp4"
        v2 = f"{base_dir}/{tid}_L16_P0.5.mp4"
        v3 = f"{base_dir}/{tid}_L16_P0.1.mp4"
        out = f"{out_dir}/{tid}_L16_TripleView.mp4"
        
        if os.path.exists(v1) and os.path.exists(v2) and os.path.exists(v3):
            create_triple_side_by_side(v1, v2, v3, out, config_labels)
        else:
            print(f"Skipping {tid}: missing source files.")
