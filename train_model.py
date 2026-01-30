"""
==============================================================================
TRAIN MODEL - Train Gesture Classification Model
==============================================================================
This script trains a Random Forest classifier on the collected gesture data.

HOW IT WORKS:
1. Loads gesture_data.pickle (created by data_collector.py)
2. Trains a Random Forest classifier
3. Evaluates accuracy
4. Saves the model to gesture_model.pickle

REQUIREMENTS:
- scikit-learn
- Run data_collector.py first to create training data
==============================================================================
"""

import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("=" * 50)
print("  GESTURE MODEL TRAINER")
print("=" * 50)

# Load data
data_file = os.path.join(os.path.dirname(__file__), 'gesture_data.pickle')

if not os.path.exists(data_file):
    print("\nERROR: gesture_data.pickle not found!")
    print("Run data_collector.py first to collect training data.")
    exit(1)

with open(data_file, 'rb') as f:
    dataset = pickle.load(f)

data = np.array(dataset['data'])
labels = np.array(dataset['labels'])

print(f"\nLoaded {len(data)} samples")
print(f"Features per sample: {data.shape[1]}")
print(f"Unique gestures: {len(set(labels))}")

# Show distribution
print("\nGesture distribution:")
for label in set(labels):
    count = list(labels).count(label)
    print(f"  {label}: {count} samples")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Train model
print("\nTraining Random Forest classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'=' * 50}")
print(f"  MODEL ACCURACY: {accuracy * 100:.2f}%")
print(f"{'=' * 50}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
model_file = os.path.join(os.path.dirname(__file__), 'gesture_model.pickle')
with open(model_file, 'wb') as f:
    pickle.dump({
        'model': model,
        'labels': list(set(labels))
    }, f)

print(f"\nModel saved to: gesture_model.pickle")
print("You can now run main.py to use the trained model!")
