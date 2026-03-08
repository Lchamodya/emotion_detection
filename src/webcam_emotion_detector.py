"""
=============================================================
  Smart Energy Systems – Webcam Application
  Real-time facial analysis using DeepFace
  Emotions detected: angry, disgust, fear, happy, sad, surprise, neutral
=============================================================

SETUP INSTRUCTIONS
------------------
1. Install dependencies:
       pip install deepface opencv-python tf-keras

2. Run this script:
       python webcam_emotion_detector.py

3. Press 'q' to quit the webcam feed

NOTE: Make sure your webcam is connected and accessible
      First run downloads models (~100 MB)
=============================================================
"""

import sys
import cv2
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ── 1. Check dependencies ────────────────────────────────────────────────────
try:
    from deepface import DeepFace
except ImportError:
    print("[ERROR] 'deepface' not found. Run:  pip install deepface opencv-python tf-keras")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("[ERROR] 'numpy' not found. Run:  pip install numpy")
    sys.exit(1)


# ── 2. Initialize ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  Smart Energy Systems – Webcam Application")
print(f"{'='*60}")
print("  Press 'q' to quit")
print(f"{'='*60}\n")

print("[INFO] Initializing DeepFace...")
print("[INFO] First run may download models (~100 MB)...\n")


# ── 3. Emoji and color mapping ──────────────────────────────────────────────
EMOJI = {
    "angry":    "😠",
    "disgust":  "🤢",
    "fear":     "😨",
    "happy":    "😊",
    "neutral":  "😐",
    "sad":      "😢",
    "surprise": "😲",
}

# Color mapping for bounding boxes (BGR format for OpenCV)
EMOTION_COLORS = {
    "angry":    (0, 0, 255),      # Red
    "disgust":  (0, 165, 255),    # Orange
    "fear":     (255, 0, 255),    # Magenta
    "happy":    (0, 255, 0),      # Green
    "neutral":  (128, 128, 128),  # Gray
    "sad":      (255, 0, 0),      # Blue
    "surprise": (0, 255, 255),    # Yellow
}


# ── 4. Start webcam capture ──────────────────────────────────────────────────
print("[INFO] Starting webcam... (Press 'q' to quit)")
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("[ERROR] Cannot access webcam. Make sure it's connected and not in use.")
    sys.exit(1)

# Set resolution
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# ── 5. Main processing loop ──────────────────────────────────────────────────
frame_count = 0
analyze_frequency = 15  # Analyze every 15 frames (better performance)
last_result = None

try:
    while True:
        # Capture frame
        ret, frame = video_capture.read()
        if not ret:
            print("[ERROR] Failed to capture frame.")
            break

        frame_count += 1
        
        # Analyze emotions periodically
        if frame_count % analyze_frequency == 0:
            try:
                result = DeepFace.analyze(
                    frame, 
                    actions=['emotion'], 
                    enforce_detection=False, 
                    silent=True
                )
                # Handle both single and multiple faces
                last_result = result if isinstance(result, list) else [result]
            except:
                pass  # Keep using last result if analysis fails
        
        # Draw results
        if last_result:
            for face_data in last_result:
                try:
                    # Get face region
                    region = face_data.get("region", {})
                    x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 100), region.get("h", 100)
                    
                    # Get emotions
                    emotions = face_data.get("emotion", {})
                    if not emotions:
                        continue
                    
                    # Normalize to lowercase
                    emotions_norm = {k.lower(): v / 100.0 for k, v in emotions.items()}
                    
                    top_emotion = max(emotions_norm, key=emotions_norm.get)
                    top_score = emotions_norm[top_emotion]
                    
                    if top_score > 0.05:
                        # Draw bounding box
                        color = EMOTION_COLORS.get(top_emotion, (255, 255, 255))
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        
                        # Draw label
                        emoji = EMOJI.get(top_emotion, "")
                        label = f"{emoji} {top_emotion.upper()} {top_score:.0%}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (x, y - 30), (x + label_size[0], y), color, -1)
                        cv2.putText(frame, label, (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Draw emotion scores sidebar
                        bar_x, bar_y = 10, 30
                        cv2.putText(frame, "Emotion Scores:", (bar_x, bar_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        sorted_emotions = sorted(emotions_norm.items(), key=lambda x: x[1], reverse=True)
                        
                        for i, (emotion, score) in enumerate(sorted_emotions[:7]):
                            y_pos = bar_y + 30 + (i * 25)
                            emoji_icon = EMOJI.get(emotion, "")
                            
                            # Emotion name
                            cv2.putText(frame, f"{emoji_icon} {emotion:<8}", (bar_x, y_pos),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                            
                            # Score bar
                            bar_width = int(score * 150)
                            bar_color = EMOTION_COLORS.get(emotion, (128, 128, 128))
                            cv2.rectangle(frame, (bar_x + 90, y_pos - 12),
                                        (bar_x + 90 + bar_width, y_pos - 2), bar_color, -1)
                            
                            # Percentage
                            cv2.putText(frame, f"{score:.0%}", (bar_x + 250, y_pos),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                except:
                    continue
        else:
            cv2.putText(frame, "Detecting face...", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit", (frame.shape[1] - 180, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display
        cv2.imshow('Smart Energy Systems', frame)
        
        # Quit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] Quitting...")
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted. Quitting...")

finally:
    video_capture.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam released. Goodbye!")
