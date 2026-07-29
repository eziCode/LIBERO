import cv2
import numpy as np
import sys
import os

def overlay_videos(video_path_1, video_path_2, output_path, alpha=0.5):
    """
    Overlays two videos with transparency.
    video_path_1: The bottom video (e.g. Expert)
    video_path_2: The top video (e.g. Compressed)
    """
    cap1 = cv2.VideoCapture(video_path_1)
    cap2 = cv2.VideoCapture(video_path_2)
    
    # Get metadata
    w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    
    w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use dimensions of the first video
    width, height = w1, h1
    
    # Define codec and output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps1, (width, height))
    
    print(f"Overlaying {video_path_1} and {video_path_2}...")
    
    frame_count = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 and not ret2:
            break
            
        # If one video is shorter, use a black frame for the missing part
        if not ret1:
            frame1 = np.zeros((height, width, 3), dtype=np.uint8)
        if not ret2:
            frame2 = np.zeros((height, width, 3), dtype=np.uint8)
            
        # Resize frame2 to match frame1 if necessary
        if (w2, h2) != (width, height):
            frame2 = cv2.resize(frame2, (width, height))
            
        # Blend the images
        # beta is transparency of frame2
        blended = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
        
        # Add labels for clarity
        cv2.putText(blended, "Video 1 (Expert)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(blended, "Video 2 (Compressed)", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        out.write(blended)
        frame_count += 1
        
    cap1.release()
    cap2.release()
    out.release()
    print(f"Successfully saved overlay with {frame_count} frames to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        # Default to the shaking task comparison for internal verification
        v1 = "results/action_filtering/sweep/videos/to_mix_the_contents_LFull_P1.0.mp4"
        v2 = "results/action_filtering/sweep/videos/to_mix_the_contents_LFull_P0.25.mp4"
        out = "results/shaking_fidelity_overlay.mp4"
        
        if os.path.exists(v1) and os.path.exists(v2):
            overlay_videos(v1, v2, out)
        else:
            print("Usage: python overlay_videos.py video1.mp4 video2.mp4 [output.mp4]")
    else:
        v1 = sys.argv[1]
        v2 = sys.argv[2]
        out = sys.argv[3] if len(sys.argv) > 3 else "overlay_result.mp4"
        overlay_videos(v1, v2, out)
