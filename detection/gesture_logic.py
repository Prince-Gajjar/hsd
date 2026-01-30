# detection/gesture_logic.py

"""
This module implements the logic to classify hand gestures based on landmark positions.

For this MVP, we use a simple rule-based approach. We check the positions of
specific landmarks (like fingertips) to determine the gesture. This is not a 
full-fledged ML model, but it is effective, fast, and easy to explain.
"""

from utils.constants import GESTURE_MAP

def classify_gesture(landmarks):
    """
    Classifies a gesture based on a list of hand landmarks.

    Args:
        landmarks (list): A list of (x, y) coordinates for the 21 hand landmarks.

    Returns:
        str: The name of the detected gesture (e.g., "OPEN_PALM", "FIST"), or None.
    """
    if not landmarks:
        return None

    # The landmarks are indexed as follows:
    # 0: WRIST
    # 4: THUMB_TIP
    # 8: INDEX_FINGER_TIP
    # 12: MIDDLE_FINGER_TIP
    # 16: RING_FINGER_TIP
    # 20: PINKY_TIP

    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # --- Rule-based Gesture Logic ---

    # Rule for FIST: All fingertips are below their respective mid-points.
    # We'll simplify by checking if fingertips are below the middle finger's base.
    if (
        index_tip[1] > landmarks[5][1] and
        middle_tip[1] > landmarks[9][1] and
        ring_tip[1] > landmarks[13][1] and
        pinky_tip[1] > landmarks[17][1]
    ):
        return "FIST"

    # Rule for TWO_FINGERS (Peace Sign): Index and middle fingers are up, others are down.
    if (
        index_tip[1] < landmarks[6][1] and
        middle_tip[1] < landmarks[10][1] and
        ring_tip[1] > landmarks[14][1] and
        pinky_tip[1] > landmarks[18][1]
    ):
        return "TWO_FINGERS"

    # Rule for OPEN_PALM: All five fingertips are above their base joints.
    # This is a good sign for "Hello".
    if (
        thumb_tip[0] < landmarks[2][0] and # Thumb is to the left of its base
        index_tip[1] < landmarks[5][1] and
        middle_tip[1] < landmarks[9][1] and
        ring_tip[1] < landmarks[13][1] and
        pinky_tip[1] < landmarks[17][1]
    ):
        return "OPEN_PALM"
    
    # Rule for THUMBS_UP
    if (
        thumb_tip[1] < landmarks[2][1] and
        index_tip[1] > landmarks[5][1] and
        middle_tip[1] > landmarks[9][1] and
        ring_tip[1] > landmarks[13][1] and
        pinky_tip[1] > landmarks[17][1]
    ):
        return "THUMBS_UP"

    # Rule for PINCH
    if (
        abs(thumb_tip[0] - index_tip[0]) < 20 and 
        abs(thumb_tip[1] - index_tip[1]) < 20
    ):
        return "PINCH"

    return None

def get_gesture_text(gesture):
    """
    Maps a gesture name to its corresponding text from the constants file.

    Args:
        gesture (str): The name of the gesture.

    Returns:
        str: The text associated with the gesture, or an empty string.
    """
    return GESTURE_MAP.get(gesture, "")
