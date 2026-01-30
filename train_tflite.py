"""
==============================================================================
TENSORFLOW LITE MODEL TRAINER
==============================================================================
Trains a neural network for gesture recognition and converts to TFLite format.
The resulting .tflite model works on Android!

GESTURES:
- Open (Hello)
- Close (Yes/Fist)
- Pointer (Pointing)
- OK (OK sign)
==============================================================================
"""

import os
import csv
import numpy as np

# TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

print("=" * 60)
print("  TENSORFLOW LITE MODEL TRAINER")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")

# ============================================================================
# 1. LOAD DATA
# ============================================================================

data_file = os.path.join(os.path.dirname(__file__), 'keypoint.csv')

if not os.path.exists(data_file):
    print("\n❌ keypoint.csv not found!")
    print("Run: python download_dataset.py first")
    exit(1)

print("\n📊 Loading training data...")

data = []
labels = []

with open(data_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) > 1:
            label = int(row[0])
            features = [float(x) for x in row[1:]]
            labels.append(label)
            data.append(features)

X = np.array(data, dtype=np.float32)
y = np.array(labels, dtype=np.int32)

# Get unique classes
num_classes = len(set(labels))
num_features = X.shape[1]

print(f"   ✓ Loaded {len(X)} samples")
print(f"   ✓ Features: {num_features}")
print(f"   ✓ Classes: {num_classes}")

# Label names
LABEL_NAMES = {0: 'Open', 1: 'Close', 2: 'Pointer', 3: 'OK'}
for i in range(num_classes):
    count = sum(1 for l in labels if l == i)
    name = LABEL_NAMES.get(i, f'Class_{i}')
    print(f"     {name}: {count} samples")

# ============================================================================
# 2. PREPARE DATA
# ============================================================================

print("\n🔬 Preparing data...")

# One-hot encode labels
y_onehot = keras.utils.to_categorical(y, num_classes=num_classes)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42, stratify=y
)

print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# ============================================================================
# 3. BUILD MODEL
# ============================================================================

print("\n🧠 Building neural network...")

model = keras.Sequential([
    # Input layer
    keras.layers.InputLayer(input_shape=(num_features,)),
    
    # Hidden layers
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation='relu'),
    
    # Output layer
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================================
# 4. TRAIN MODEL
# ============================================================================

print("\n🏋️ Training model...")

# Early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# ============================================================================
# 5. EVALUATE MODEL
# ============================================================================

print("\n📊 Evaluating model...")

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"\n{'=' * 60}")
print(f"   🎯 TEST ACCURACY: {accuracy * 100:.2f}%")
print(f"{'=' * 60}")

# Per-class accuracy
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\n📈 Per-class accuracy:")
for i in range(num_classes):
    mask = y_true_classes == i
    if sum(mask) > 0:
        class_acc = np.mean(y_pred_classes[mask] == y_true_classes[mask])
        name = LABEL_NAMES.get(i, f'Class_{i}')
        print(f"   {name}: {class_acc * 100:.1f}%")

# ============================================================================
# 6. CONVERT TO TFLITE
# ============================================================================

print("\n📱 Converting to TensorFlow Lite...")

# Standard conversion
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite model
tflite_path = os.path.join(os.path.dirname(__file__), 'gesture_model.tflite')
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"   ✓ Saved: gesture_model.tflite ({len(tflite_model) / 1024:.1f} KB)")

# ============================================================================
# 7. QUANTIZED VERSION (Even smaller!)
# ============================================================================

print("\n📱 Creating quantized version (smaller)...")

converter_quant = tf.lite.TFLiteConverter.from_keras_model(model)
converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative dataset for quantization
def representative_dataset():
    for i in range(min(100, len(X_train))):
        yield [X_train[i:i+1]]

converter_quant.representative_dataset = representative_dataset
converter_quant.target_spec.supported_types = [tf.float16]

tflite_quant = converter_quant.convert()

# Save quantized model
tflite_quant_path = os.path.join(os.path.dirname(__file__), 'gesture_model_quant.tflite')
with open(tflite_quant_path, 'wb') as f:
    f.write(tflite_quant)

print(f"   ✓ Saved: gesture_model_quant.tflite ({len(tflite_quant) / 1024:.1f} KB)")

# ============================================================================
# 8. SAVE LABELS
# ============================================================================

labels_path = os.path.join(os.path.dirname(__file__), 'gesture_labels.txt')
with open(labels_path, 'w') as f:
    for i in range(num_classes):
        f.write(f"{LABEL_NAMES.get(i, f'Class_{i}')}\n")

print(f"   ✓ Saved: gesture_labels.txt")

# ============================================================================
# 9. TEST TFLITE MODEL
# ============================================================================

print("\n🧪 Testing TFLite model...")

# Load and test
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"   Input shape: {input_details[0]['shape']}")
print(f"   Output shape: {output_details[0]['shape']}")

# Test prediction
test_sample = X_test[0:1]
interpreter.set_tensor(input_details[0]['index'], test_sample)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
predicted = np.argmax(output)
actual = np.argmax(y_test[0])

print(f"   Test prediction: {LABEL_NAMES.get(predicted)} (actual: {LABEL_NAMES.get(actual)})")

# ============================================================================
# DONE
# ============================================================================

print(f"\n{'=' * 60}")
print("   ✅ TENSORFLOW LITE MODEL READY!")
print(f"{'=' * 60}")
print("\nFiles created:")
print(f"   📁 gesture_model.tflite     (Full precision)")
print(f"   📁 gesture_model_quant.tflite (Quantized, smaller)")
print(f"   📁 gesture_labels.txt       (Class labels)")
print("\n🚀 You can now build the Android app!")
