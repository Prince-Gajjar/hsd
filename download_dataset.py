"""
==============================================================================
DOWNLOAD DATASET - Download Pre-collected Hand Gesture Dataset
==============================================================================
Downloads the hand gesture landmark dataset from GitHub and trains a model.

SOURCE: kinivi/hand-gesture-recognition-mediapipe (GitHub)
Dataset contains MediaPipe hand landmarks for gesture recognition.
==============================================================================
"""

import os
import urllib.request
import csv
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("=" * 60)
print("  HAND GESTURE DATASET DOWNLOADER & MODEL TRAINER")
print("=" * 60)

# URLs for the dataset
KEYPOINT_CSV_URL = "https://raw.githubusercontent.com/kinivi/hand-gesture-recognition-mediapipe/main/model/keypoint_classifier/keypoint.csv"
LABELS_CSV_URL = "https://raw.githubusercontent.com/kinivi/hand-gesture-recognition-mediapipe/main/model/keypoint_classifier/keypoint_classifier_label.csv"

# Local paths
data_dir = os.path.dirname(__file__)
keypoint_file = os.path.join(data_dir, "keypoint.csv")
labels_file = os.path.join(data_dir, "keypoint_labels.csv")

# Download keypoint data
print("\n📥 Downloading keypoint dataset...")
try:
    urllib.request.urlretrieve(KEYPOINT_CSV_URL, keypoint_file)
    print(f"   ✓ Downloaded: keypoint.csv")
except Exception as e:
    print(f"   ✗ Error downloading keypoint.csv: {e}")
    exit(1)

# Download labels
print("📥 Downloading labels...")
try:
    urllib.request.urlretrieve(LABELS_CSV_URL, labels_file)
    print(f"   ✓ Downloaded: keypoint_labels.csv")
except Exception as e:
    print(f"   ⚠ Labels file not found, using default labels")
    labels_file = None

# Load labels
gesture_names = {}
if labels_file and os.path.exists(labels_file):
    with open(labels_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            gesture_names[idx] = line.strip()
    print(f"\n📋 Loaded {len(gesture_names)} gesture labels:")
    for idx, name in gesture_names.items():
        print(f"   {idx}: {name}")
else:
    # Default labels based on common gesture recognition datasets
    gesture_names = {
        0: "Open",      # Open palm
        1: "Close",     # Closed fist
        2: "Pointer",   # Pointing
        3: "OK",        # OK sign
        4: "Peace",     # Peace/Victory
        5: "ThumbsUp",  # Thumbs up
    }
    print("\n📋 Using default gesture labels")

# Load keypoint data
print("\n📊 Loading keypoint data...")
data = []
labels = []

with open(keypoint_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) > 1:
            label = int(row[0])
            features = [float(x) for x in row[1:]]
            labels.append(label)
            data.append(features)

print(f"   ✓ Loaded {len(data)} samples")
print(f"   ✓ Features per sample: {len(data[0])}")

# Show distribution
print("\n📈 Sample distribution:")
label_counts = {}
for label in labels:
    label_counts[label] = label_counts.get(label, 0) + 1

for label_id, count in sorted(label_counts.items()):
    name = gesture_names.get(label_id, f"Gesture_{label_id}")
    print(f"   {name}: {count} samples")

# Convert to numpy
X = np.array(data)
y = np.array(labels)

# Create label name mapping
label_to_name = {}
for label_id in set(labels):
    label_to_name[label_id] = gesture_names.get(label_id, f"Gesture_{label_id}")

# Split data
print("\n🔬 Splitting data for training...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# Train model
print("\n🤖 Training Random Forest classifier...")
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'=' * 60}")
print(f"   🎯 MODEL ACCURACY: {accuracy * 100:.2f}%")
print(f"{'=' * 60}")

# Per-class accuracy
print("\n📊 Per-gesture accuracy:")
for label_id in sorted(set(y_test)):
    mask = y_test == label_id
    if sum(mask) > 0:
        class_acc = accuracy_score(y_test[mask], y_pred[mask])
        name = label_to_name.get(label_id, f"Gesture_{label_id}")
        print(f"   {name}: {class_acc * 100:.1f}%")

# Save model
model_file = os.path.join(data_dir, "gesture_model.pickle")
with open(model_file, 'wb') as f:
    pickle.dump({
        'model': model,
        'labels': list(label_to_name.values()),
        'label_to_name': label_to_name,
        'feature_count': len(data[0])
    }, f)

print(f"\n✅ Model saved to: gesture_model.pickle")
print(f"   Gestures: {list(label_to_name.values())}")
print("\n🚀 You can now run: python main.py")
