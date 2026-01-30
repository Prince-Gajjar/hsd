# utils/constants.py

"""
This file contains constants used throughout the application.
These constants define gesture mappings and app configuration.
"""

# ==============================================================================
# GESTURE MAPPINGS
# ==============================================================================

# TFLite model gesture labels (must match gesture_labels.txt)
GESTURE_LABELS = ['Open', 'Close', 'Pointer', 'OK']

# Gesture to Text Mapping for TTS and display
GESTURE_MAP = {
    "Open": "Hello",
    "Close": "Yes", 
    "Pointer": "Look",
    "OK": "OK",
    # Legacy mappings for backward compatibility
    "OPEN_PALM": "Hello",
    "FIST": "Yes",
    "TWO_FINGERS": "No",
    "THUMBS_UP": "Thank You",
    "PINCH": "I am sorry"
}

# Gesture to Emoji Mapping
GESTURE_EMOJI = {
    "Open": "👋",
    "Close": "✊",
    "Pointer": "👆",
    "OK": "👌",
}

# ==============================================================================
# UI CONSTANTS
# ==============================================================================

APP_TITLE = "Hand Sign Detection System"
APP_VERSION = "1.0.0"

# ==============================================================================
# CAMERA SETTINGS
# ==============================================================================

CAMERA_RESOLUTION = (640, 480)
CAMERA_INDEX = 0
DETECTION_FPS = 15

# ==============================================================================
# MODEL SETTINGS
# ==============================================================================

TFLITE_MODEL = "gesture_model_quant.tflite"  # Smaller, quantized model
TFLITE_MODEL_FULL = "gesture_model.tflite"   # Full precision model
LABELS_FILE = "gesture_labels.txt"

# Classification confidence threshold
CONFIDENCE_THRESHOLD = 0.6

# Gesture debounce time (seconds) - prevents rapid repeated detections
GESTURE_DEBOUNCE_TIME = 1.5
