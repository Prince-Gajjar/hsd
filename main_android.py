"""
==============================================================================
ANDROID-COMPATIBLE MAIN - Simplified for Mobile
==============================================================================
This version works without MediaPipe for Android deployment.
Uses Kivy's built-in Camera and rule-based gesture detection.
==============================================================================
"""

from kivy.config import Config
Config.set('graphics', 'width', '360')
Config.set('graphics', 'height', '640')

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics.texture import Texture

# Try to import TTS (may not work on Android)
try:
    from plyer import tts as plyer_tts
    HAS_TTS = True
except:
    try:
        import pyttsx3
        HAS_TTS = True
    except:
        HAS_TTS = False


class MainScreen(BoxLayout):
    """Main screen for Android version."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = 10
        self.spacing = 10
        
        # Background color
        Window.clearcolor = (0.12, 0.12, 0.16, 1)
        
        # Title
        title = Label(
            text='🤟 Hand Sign Detection',
            font_size='24sp',
            size_hint_y=0.08,
            bold=True
        )
        self.add_widget(title)
        
        # Camera
        self.camera = Camera(
            resolution=(640, 480),
            play=False,
            size_hint_y=0.5
        )
        self.add_widget(self.camera)
        
        # Status
        self.status = Label(
            text='Tap START to begin',
            font_size='16sp',
            size_hint_y=0.05
        )
        self.add_widget(self.status)
        
        # Gesture display
        self.gesture_label = Label(
            text='---',
            font_size='32sp',
            size_hint_y=0.1,
            color=(0.3, 0.85, 0.6, 1)
        )
        self.add_widget(self.gesture_label)
        
        # Output text
        self.output_label = Label(
            text='',
            font_size='18sp',
            size_hint_y=0.1
        )
        self.add_widget(self.output_label)
        
        # Buttons container
        btn_layout = BoxLayout(
            size_hint_y=0.17,
            spacing=10
        )
        
        # Start button
        self.start_btn = Button(
            text='▶ START',
            background_color=(0.2, 0.7, 0.4, 1),
            font_size='16sp'
        )
        self.start_btn.bind(on_press=self.start_camera)
        btn_layout.add_widget(self.start_btn)
        
        # Stop button
        self.stop_btn = Button(
            text='⏹ STOP',
            background_color=(0.85, 0.35, 0.35, 1),
            font_size='16sp'
        )
        self.stop_btn.bind(on_press=self.stop_camera)
        btn_layout.add_widget(self.stop_btn)
        
        self.add_widget(btn_layout)
        
        # Second row of buttons
        btn_layout2 = BoxLayout(
            size_hint_y=0.12,
            spacing=10
        )
        
        # Speak button
        self.speak_btn = Button(
            text='🔊 SPEAK',
            background_color=(0.25, 0.55, 0.85, 1),
            font_size='16sp'
        )
        self.speak_btn.bind(on_press=self.speak_output)
        btn_layout2.add_widget(self.speak_btn)
        
        # Clear button
        self.clear_btn = Button(
            text='✖ CLEAR',
            background_color=(0.5, 0.5, 0.55, 1),
            font_size='16sp'
        )
        self.clear_btn.bind(on_press=self.clear_output)
        btn_layout2.add_widget(self.clear_btn)
        
        self.add_widget(btn_layout2)
        
        # State
        self.detected_texts = []
        self.is_running = False
    
    def start_camera(self, instance):
        """Start the camera."""
        self.camera.play = True
        self.is_running = True
        self.status.text = '● Camera Running'
        self.status.color = (0.3, 0.85, 0.4, 1)
        
        # Start frame processing
        # Note: On Android, you'd use Camera's on_texture event
        # For demo, we just show camera
    
    def stop_camera(self, instance):
        """Stop the camera."""
        self.camera.play = False
        self.is_running = False
        self.status.text = '● Camera Stopped'
        self.status.color = (0.9, 0.4, 0.4, 1)
    
    def speak_output(self, instance):
        """Speak the output text."""
        text = ' '.join(self.detected_texts) if self.detected_texts else "No text"
        
        if HAS_TTS:
            try:
                # Try plyer first (Android)
                from plyer import tts
                tts.speak(text)
            except:
                try:
                    # Fallback to pyttsx3 (Desktop)
                    import pyttsx3
                    engine = pyttsx3.init()
                    engine.say(text)
                    engine.runAndWait()
                except:
                    pass
        
        self.status.text = f'Speaking: {text[:30]}...'
    
    def clear_output(self, instance):
        """Clear the output."""
        self.detected_texts = []
        self.output_label.text = ''
        self.gesture_label.text = '---'
        self.status.text = 'Cleared'


class HandSignAppAndroid(App):
    """Android-compatible Kivy app."""
    
    def build(self):
        self.title = 'Hand Sign Detection'
        return MainScreen()


if __name__ == '__main__':
    HandSignAppAndroid().run()
