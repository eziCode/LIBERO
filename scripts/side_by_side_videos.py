import cv2
import numpy as np
import sys
import os

def create_side_by_side(v1_path, v2_path, out_path, label1="Expert", label2="Compressed"):
    cap1 = cv2.VideoCapture(v1_path)
    cap2 = cv2.VideoCapture(v2_path)
    
    # Metadata
    w = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    
    # Output is double width
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w * 2, h))
    
    print(f"Generating Side-by-Side: {out_path}")
    
    count = 0
    while True:
        ret1, f1 = cap1.read()
        ret2, f2 = cap2.read()
        
        if not ret1 and not ret2:
            break
            
        # Padding
        if not ret1: f1 = np.zeros((h, w, 3), dtype=np.uint8)
        if not ret2: f2 = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Resize f2 if needed
        if (f2.shape[1], f2.shape[0]) != (w, h):
            f2 = cv2.resize(f2, (w, h))
            
        # Combine
        combined = np.hstack((f1, f2))
        
        # Add Header Labels
        cv2.rectangle(combined, (0, 0), (w*2, 40), (0,0,0), -1)
        cv2.putText(combined, label1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(combined, label2, (w + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        out.write(combined)
        count += 1
        
    cap1.release()
    cap2.release()
    out.release()
    print(f"Done! {count} frames saved.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        # Default for the Flip Cup task as requested
        v1 = "results/videos/put_it_rightside_up_L16_P1.0.mp4"
        v2 = "results/videos/put_it_rightside_up_L16_P0.05.mp4"
        out = "results/overlayed_videos/put_it_rightside_up_L16_SideBySide.mp4"
        os.makedirs("results/overlayed_videos", exist_ok=True)
        create_side_by_side(v1, v2, out)
    else:
        v1, v2, out = sys.argv[1:4]
        create_side_by_side(v1, v2, out)
