# main.py

"""
This is the main entry point for the Hand Sign Detection application.

It sets up the Kivy application, builds the UI from the 'main.kv' file, and handles
the application lifecycle. It connects the UI events (like button presses) to the
backend logic (camera control, TTS).
"""

import kivy
kivy.require('2.1.0') # Or a version compatible with your environment

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.uix.image import Image

# Local imports for the app's components
from camera.camera_stream import CameraStream
from services import tts
from utils.constants import APP_TITLE

class MainLayout(BoxLayout):
    """
    The root widget of the application that contains all other UI elements.
    It manages the camera stream and updates the UI accordingly.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera_stream = CameraStream()
        self.update_event = None

    def start_camera(self):
        """
        Starts the camera feed and schedules the UI to be updated.
        """
        self.camera_stream.start()
        if not self.update_event:
            # Schedule the update method to be called 30 times a second
            self.update_event = Clock.schedule_interval(self.update, 1.0 / 30.0)

    def stop_camera(self):
        """
        Stops the camera feed and the UI updates.
        """
        if self.update_event:
            self.update_event.cancel()
            self.update_event = None
        self.camera_stream.stop()
        # Clear the camera view
        self.ids.camera_view.texture = None

    def update(self, dt):
        """
        This method is called repeatedly to update the camera view and detected text.
        """
        texture = self.camera_stream.get_texture()
        if texture:
            self.ids.camera_view.texture = texture
        
        detected_text = self.camera_stream.get_detected_text()
        if detected_text:
            self.ids.detected_text_label.text = f"Detected Text: {detected_text}"
        else:
            self.ids.detected_text_label.text = "Detected Text: "

    def speak_text(self):
        """
        Uses the TTS service to speak the currently detected text.
        """
        text_to_speak = self.camera_stream.get_detected_text()
        if text_to_speak:
            tts.speak(text_to_speak)

    def clear_text(self):
        """
        Clears the detected text label.
        """
        self.ids.detected_text_label.text = "Detected Text: "

class HandSignApp(App):
    """
    The main Kivy application class.
    """
    def build(self):
        """
        Builds the application's UI by returning the root widget.
        """
        self.title = APP_TITLE
        return MainLayout()

    def on_stop(self):
        """
        Ensures the camera is released when the app is closed.
        """
        # Safely stop the camera when the app closes
        self.root.stop_camera()

if __name__ == '__main__':
    HandSignApp().run()
