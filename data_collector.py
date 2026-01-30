"""
==============================================================================
DATA COLLECTOR - Collect Hand Gesture Training Data
==============================================================================
This script captures hand landmarks and saves them to create a training dataset.

HOW TO USE:
1. Run this script
2. Press a key (0-9, a-z) to assign a gesture label
3. Show your hand gesture to the camera
4. Press 's' to save samples
5. Press 'q' to quit

The data is saved to 'gesture_data.pickle' for training.
==============================================================================
"""

import cv2
import mediapipe as mp
import pickle
import os
import numpy as np

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Data storage
data = []
labels = []

# Gesture labels mapping
GESTURE_LABELS = {
    '0': 'hello',       # Open palm
    '1': 'yes',         # Fist  
    '2': 'peace',       # Peace sign (V)
    '3': 'thumbs_up',   # Thumbs up
    '4': 'pointing',    # Index finger pointing
    '5': 'stop',        # Stop/halt gesture
    '6': 'ok',          # OK gesture (circle)
    '7': 'rock',        # Rock sign
    '8': 'call',        # Call me gesture
    '9': 'love',        # I love you (ASL)
}

current_label = None

print("=" * 50)
print("  HAND GESTURE DATA COLLECTOR")
print("=" * 50)
print("\nControls:")
print("  0-9: Select gesture label")
print("  s: Save current samples")
print("  c: Clear all data")
print("  q: Quit and save")
print("\nGesture Labels:")
for key, name in GESTURE_LABELS.items():
    print(f"  {key}: {name}")
print("=" * 50)

# Open camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

samples_count = {label: 0 for label in GESTURE_LABELS.values()}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip for mirror view
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect hands
    results = hands.process(frame_rgb)
    
    # Draw landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
            
            # Extract landmark data (21 landmarks x 3 coordinates = 63 features)
            landmarks_data = []
            for landmark in hand_landmarks.landmark:
                landmarks_data.extend([landmark.x, landmark.y, landmark.z])
            
            # Show that hand is detected
            cv2.putText(frame, "Hand Detected!", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # If we have a label, add to dataset
            if current_label is not None:
                data.append(landmarks_data)
                labels.append(current_label)
                samples_count[current_label] = samples_count.get(current_label, 0) + 1
    
    # Display info
    cv2.putText(frame, "GESTURE DATA COLLECTOR", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    if current_label:
        cv2.putText(frame, f"Recording: {current_label}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "Press 0-9 to select gesture", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
    
    # Show sample counts
    y_pos = 120
    for label, count in samples_count.items():
        if count > 0:
            cv2.putText(frame, f"{label}: {count}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_pos += 20
    
    cv2.putText(frame, f"Total samples: {len(data)}", (10, 460), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("Data Collector", frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('c'):
        data = []
        labels = []
        samples_count = {label: 0 for label in GESTURE_LABELS.values()}
        current_label = None
        print("Data cleared!")
    elif key == ord('s'):
        current_label = None
        print(f"Stopped recording. Total samples: {len(data)}")
    elif chr(key) in GESTURE_LABELS:
        current_label = GESTURE_LABELS[chr(key)]
        print(f"Recording: {current_label}")

# Save data
cap.release()
cv2.destroyAllWindows()
hands.close()

if len(data) > 0:
    data_file = os.path.join(os.path.dirname(__file__), 'gesture_data.pickle')
    with open(data_file, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print(f"\nSaved {len(data)} samples to gesture_data.pickle")
    print("Sample distribution:")
    for label in set(labels):
        count = labels.count(label)
        print(f"  {label}: {count}")
else:
    print("\nNo data collected.")
