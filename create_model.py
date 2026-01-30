"""
==============================================================================
CREATE SAMPLE MODEL - Generate Pre-trained Model for Demo
==============================================================================
This script creates a pre-trained model using synthetic landmarks.
For production, use data_collector.py and train_model.py instead.
==============================================================================
"""

import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier

print("Creating pre-trained gesture model...")

# Define gesture patterns based on typical hand landmark positions
# 21 landmarks x 3 coordinates (x, y, z) = 63 features
# Values are normalized (0-1)

def generate_samples(gesture_type, num_samples=100):
    """Generate synthetic landmark data for a gesture with noise."""
    samples = []
    
    # Base patterns for different gestures
    # These approximate real hand positions
    
    if gesture_type == "hello":  # Open palm - all fingers extended
        base = np.array([
            0.5, 0.9, 0,    # Wrist
            0.35, 0.75, 0,  # Thumb CMC
            0.28, 0.6, 0,   # Thumb MCP
            0.22, 0.45, 0,  # Thumb IP
            0.18, 0.35, 0,  # Thumb TIP
            0.38, 0.5, 0,   # Index MCP
            0.38, 0.32, 0,  # Index PIP
            0.38, 0.2, 0,   # Index DIP
            0.38, 0.1, 0,   # Index TIP
            0.5, 0.48, 0,   # Middle MCP
            0.5, 0.28, 0,   # Middle PIP
            0.5, 0.15, 0,   # Middle DIP
            0.5, 0.05, 0,   # Middle TIP
            0.62, 0.5, 0,   # Ring MCP
            0.62, 0.32, 0,  # Ring PIP
            0.62, 0.2, 0,   # Ring DIP
            0.62, 0.1, 0,   # Ring TIP
            0.74, 0.55, 0,  # Pinky MCP
            0.74, 0.42, 0,  # Pinky PIP
            0.74, 0.32, 0,  # Pinky DIP
            0.74, 0.22, 0,  # Pinky TIP
        ])
    
    elif gesture_type == "yes":  # Fist - all fingers closed
        base = np.array([
            0.5, 0.9, 0,    # Wrist
            0.4, 0.75, 0,   # Thumb CMC
            0.35, 0.65, 0,  # Thumb MCP
            0.38, 0.6, 0,   # Thumb IP
            0.42, 0.58, 0,  # Thumb TIP (tucked)
            0.42, 0.55, 0,  # Index MCP
            0.42, 0.6, 0,   # Index PIP (bent)
            0.42, 0.65, 0,  # Index DIP
            0.42, 0.68, 0,  # Index TIP (curled)
            0.5, 0.52, 0,   # Middle MCP
            0.5, 0.58, 0,   # Middle PIP
            0.5, 0.63, 0,   # Middle DIP
            0.5, 0.66, 0,   # Middle TIP
            0.58, 0.55, 0,  # Ring MCP
            0.58, 0.6, 0,   # Ring PIP
            0.58, 0.65, 0,  # Ring DIP
            0.58, 0.68, 0,  # Ring TIP
            0.66, 0.58, 0,  # Pinky MCP
            0.66, 0.63, 0,  # Pinky PIP
            0.66, 0.67, 0,  # Pinky DIP
            0.66, 0.7, 0,   # Pinky TIP
        ])
    
    elif gesture_type == "peace":  # Peace sign - index and middle up
        base = np.array([
            0.5, 0.9, 0,    # Wrist
            0.38, 0.75, 0,  # Thumb CMC
            0.32, 0.68, 0,  # Thumb MCP
            0.35, 0.63, 0,  # Thumb IP
            0.4, 0.6, 0,    # Thumb TIP (folded)
            0.4, 0.52, 0,   # Index MCP
            0.38, 0.35, 0,  # Index PIP (extended)
            0.36, 0.22, 0,  # Index DIP
            0.35, 0.1, 0,   # Index TIP
            0.52, 0.5, 0,   # Middle MCP
            0.54, 0.33, 0,  # Middle PIP (extended)
            0.55, 0.2, 0,   # Middle DIP
            0.56, 0.08, 0,  # Middle TIP
            0.62, 0.55, 0,  # Ring MCP
            0.62, 0.6, 0,   # Ring PIP (folded)
            0.62, 0.65, 0,  # Ring DIP
            0.62, 0.68, 0,  # Ring TIP
            0.7, 0.6, 0,    # Pinky MCP
            0.7, 0.65, 0,   # Pinky PIP (folded)
            0.7, 0.68, 0,   # Pinky DIP
            0.7, 0.7, 0,    # Pinky TIP
        ])
    
    elif gesture_type == "thumbs_up":  # Thumb up, others closed
        base = np.array([
            0.5, 0.85, 0,   # Wrist
            0.38, 0.7, 0,   # Thumb CMC
            0.32, 0.55, 0,  # Thumb MCP
            0.28, 0.4, 0,   # Thumb IP (up)
            0.26, 0.25, 0,  # Thumb TIP (up)
            0.45, 0.6, 0,   # Index MCP
            0.45, 0.65, 0,  # Index PIP (folded)
            0.46, 0.7, 0,   # Index DIP
            0.47, 0.72, 0,  # Index TIP
            0.52, 0.58, 0,  # Middle MCP
            0.52, 0.64, 0,  # Middle PIP
            0.53, 0.68, 0,  # Middle DIP
            0.53, 0.71, 0,  # Middle TIP
            0.59, 0.6, 0,   # Ring MCP
            0.59, 0.65, 0,  # Ring PIP
            0.6, 0.69, 0,   # Ring DIP
            0.6, 0.72, 0,   # Ring TIP
            0.66, 0.63, 0,  # Pinky MCP
            0.66, 0.67, 0,  # Pinky PIP
            0.66, 0.7, 0,   # Pinky DIP
            0.67, 0.73, 0,  # Pinky TIP
        ])
    
    elif gesture_type == "pointing":  # Index finger pointing
        base = np.array([
            0.5, 0.88, 0,   # Wrist
            0.4, 0.75, 0,   # Thumb CMC
            0.35, 0.68, 0,  # Thumb MCP
            0.38, 0.62, 0,  # Thumb IP
            0.42, 0.58, 0,  # Thumb TIP (folded on fist)
            0.42, 0.52, 0,  # Index MCP
            0.4, 0.35, 0,   # Index PIP (extended)
            0.38, 0.2, 0,   # Index DIP
            0.37, 0.08, 0,  # Index TIP (pointing)
            0.52, 0.55, 0,  # Middle MCP
            0.52, 0.62, 0,  # Middle PIP (folded)
            0.52, 0.67, 0,  # Middle DIP
            0.52, 0.7, 0,   # Middle TIP
            0.6, 0.58, 0,   # Ring MCP
            0.6, 0.64, 0,   # Ring PIP
            0.6, 0.68, 0,   # Ring DIP
            0.6, 0.71, 0,   # Ring TIP
            0.67, 0.62, 0,  # Pinky MCP
            0.67, 0.67, 0,  # Pinky PIP
            0.67, 0.7, 0,   # Pinky DIP
            0.67, 0.73, 0,  # Pinky TIP
        ])
    
    elif gesture_type == "stop":  # Stop/Halt - palm facing out, fingers spread
        base = np.array([
            0.5, 0.92, 0,   # Wrist
            0.3, 0.78, 0,   # Thumb CMC
            0.22, 0.65, 0,  # Thumb MCP
            0.16, 0.52, 0,  # Thumb IP
            0.12, 0.42, 0,  # Thumb TIP (extended out)
            0.35, 0.48, 0,  # Index MCP
            0.32, 0.3, 0,   # Index PIP
            0.3, 0.18, 0,   # Index DIP
            0.28, 0.08, 0,  # Index TIP
            0.5, 0.45, 0,   # Middle MCP
            0.5, 0.25, 0,   # Middle PIP
            0.5, 0.12, 0,   # Middle DIP
            0.5, 0.02, 0,   # Middle TIP
            0.65, 0.48, 0,  # Ring MCP
            0.68, 0.3, 0,   # Ring PIP
            0.7, 0.18, 0,   # Ring DIP
            0.72, 0.08, 0,  # Ring TIP
            0.78, 0.55, 0,  # Pinky MCP
            0.82, 0.42, 0,  # Pinky PIP
            0.85, 0.32, 0,  # Pinky DIP
            0.88, 0.24, 0,  # Pinky TIP
        ])
    
    else:
        base = np.random.rand(63) * 0.8 + 0.1
    
    # Generate variations with noise
    for _ in range(num_samples):
        noise = np.random.normal(0, 0.03, len(base))
        sample = base + noise
        sample = np.clip(sample, 0, 1)
        samples.append(sample.tolist())
    
    return samples

# Generate dataset
all_data = []
all_labels = []

gestures = ["hello", "yes", "peace", "thumbs_up", "pointing", "stop"]

for gesture in gestures:
    samples = generate_samples(gesture, num_samples=150)
    all_data.extend(samples)
    all_labels.extend([gesture] * len(samples))
    print(f"  Generated {len(samples)} samples for '{gesture}'")

# Convert to numpy
X = np.array(all_data)
y = np.array(all_labels)

print(f"\nTotal samples: {len(X)}")

# Train model
print("Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42
)
model.fit(X, y)

# Save model
model_file = os.path.join(os.path.dirname(__file__), 'gesture_model.pickle')
with open(model_file, 'wb') as f:
    pickle.dump({
        'model': model,
        'labels': gestures
    }, f)

print(f"✓ Model saved to: {model_file}")
print("\nGesture labels in model:")
for g in gestures:
    print(f"  - {g}")
