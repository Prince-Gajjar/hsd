# detection/hand_detector.py

"""
This module contains the HandDetector class, which uses MediaPipe to detect hand landmarks.

It processes image frames from the camera, finds hand landmarks, and provides methods
to get landmark positions and draw them on the image. This isolates the complexity
of MediaPipe from the main application logic.
"""

import cv2
import mediapipe as mp

class HandDetector:
    """
    A class to detect hands and their landmarks in an image frame.
    """
    def __init__(self, max_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        """
        Initializes the HandDetector.

        Args:
            max_hands (int): Maximum number of hands to detect.
            min_detection_confidence (float): Minimum confidence value for hand detection.
            min_tracking_confidence (float): Minimum confidence value for hand tracking.
        """
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=self.max_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None

    def find_hands(self, img, draw=True):
        """
        Finds hands in a BGR image.

        Args:
            img (numpy.ndarray): The image frame from OpenCV.
            draw (bool): If True, draws the landmarks and connections on the image.

        Returns:
            numpy.ndarray: The image with or without drawn landmarks.
        """
        # MediaPipe works with RGB images, but OpenCV provides BGR.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to find hand landmarks
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                # Draw the landmarks and connections on the original BGR image
                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
        return img

    def get_landmark_positions(self, img_shape):
        """
        Extracts the landmark positions for the detected hand.

        Args:
            img_shape (tuple): The shape of the image (height, width) to scale landmarks.

        Returns:
            list: A list of landmark coordinates [(x1, y1), (x2, y2), ...].
                  Returns an empty list if no hand is detected.
        """
        landmark_list = []
        if self.results and self.results.multi_hand_landmarks:
            # We are working with only one hand for this project
            hand_landmarks = self.results.multi_hand_landmarks[0]
            height, width, _ = img_shape

            for landmark in hand_landmarks.landmark:
                # The landmark coordinates are normalized; we need to convert them to pixel values.
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                landmark_list.append((cx, cy))
        
        return landmark_list
