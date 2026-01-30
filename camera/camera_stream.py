# camera/camera_stream.py

"""
This module contains the CameraStream class, which handles all camera-related operations.

It uses OpenCV to capture video from the webcam, processes each frame for hand detection,
and updates a Kivy Texture to display the feed in the UI. This approach keeps the
camera logic separate from the main application UI.
"""

import cv2
from kivy.graphics.texture import Texture
from detection.hand_detector import HandDetector
from detection.gesture_logic import classify_gesture, get_gesture_text

class CameraStream:
    """
    Manages the camera feed, hand detection, and gesture classification.
    """
    def __init__(self):
        self.capture = None
        self.hand_detector = HandDetector()
        self.detected_text = ""

    def start(self):
        """
        Starts the camera capture if it's not already started.
        """
        if self.capture is None:
            # 0 is usually the default webcam
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                print("Error: Could not open camera.")
                self.capture = None

    def stop(self):
        """
        Stops and releases the camera.
        """
        if self.capture:
            self.capture.release()
            self.capture = None

    def get_texture(self):
        """
        Captures a frame, processes it, and returns it as a Kivy Texture.

        Returns:
            Texture: A Kivy texture to be displayed in the UI, or None if camera is off.
        """
        if not self.capture:
            return None

        ret, frame = self.capture.read()
        if not ret:
            return None

        # 1. Find hands and draw landmarks
        processed_frame = self.hand_detector.find_hands(frame, draw=True)

        # 2. Get landmark positions
        landmarks = self.hand_detector.get_landmark_positions(processed_frame.shape)

        # 3. Classify the gesture
        gesture = classify_gesture(landmarks)
        
        # 4. Get the text for the gesture
        self.detected_text = get_gesture_text(gesture)

        # Kivy textures are in RGB format, so we need to convert the BGR frame.
        # We also need to flip the image vertically for correct display.
        processed_frame = cv2.flip(processed_frame, 0)
        buf = processed_frame.tobytes()
        texture = Texture.create(
            size=(processed_frame.shape[1], processed_frame.shape[0]), colorfmt='bgr'
        )
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        
        return texture

    def get_detected_text(self):
        """
        Returns the text of the last detected gesture.
        """
        return self.detected_text
