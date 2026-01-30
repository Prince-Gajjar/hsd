"""
==============================================================================
MOBILE HAND SIGN DETECTION APP
==============================================================================
Android-compatible version using:
- Kivy for UI
- Kivy Camera (works on Android)
- TensorFlow Lite for gesture recognition (lightweight)
- Plyer for TTS (Android compatible)

To build APK:
1. Install WSL on Windows
2. Run: buildozer android debug
==============================================================================
"""

# Kivy config - MUST be before imports
from kivy.config import Config
Config.set('graphics', 'width', '360')
Config.set('graphics', 'height', '720')
Config.set('graphics', 'resizable', False)

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.camera import Camera
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle, RoundedRectangle
from kivy.properties import StringProperty, BooleanProperty
import time

# Platform detection
from kivy.utils import platform

# TTS - works on Android via plyer
if platform == 'android':
    try:
        from plyer import tts
        HAS_TTS = True
    except:
        HAS_TTS = False
else:
    try:
        import pyttsx3
        HAS_TTS = True
    except:
        HAS_TTS = False


class GestureButton(Button):
    """Custom styled button."""
    
    def __init__(self, bg_color=(0.2, 0.6, 0.9, 1), **kwargs):
        super().__init__(**kwargs)
        self.background_color = (0, 0, 0, 0)
        self.background_normal = ''
        self.bg_color = bg_color
        self.font_size = '16sp'
        self.bold = True
        
        with self.canvas.before:
            Color(*self.bg_color)
            self.rect = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[10]
            )
        
        self.bind(pos=self._update_rect, size=self._update_rect)
    
    def _update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size


class HandSignMobileApp(BoxLayout):
    """Main mobile app screen."""
    
    gesture_text = StringProperty('---')
    output_text = StringProperty('')
    status_text = StringProperty('Tap START to begin')
    is_running = BooleanProperty(False)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = [15, 10]
        self.spacing = 10
        
        # Set background
        with self.canvas.before:
            Color(0.1, 0.1, 0.14, 1)
            self.bg_rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._update_bg, size=self._update_bg)
        
        # Build UI
        self._build_ui()
        
        # State
        self.detected_gestures = []
        self.last_gesture = None
        self.last_gesture_time = 0
        self.frame_count = 0
        self.tts_engine = None
        
        # Initialize TTS for desktop
        if platform != 'android' and HAS_TTS:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
            except:
                pass
    
    def _update_bg(self, *args):
        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size
    
    def _build_ui(self):
        """Build the mobile UI."""
        
        # === HEADER ===
        header = BoxLayout(size_hint_y=0.08, padding=[10, 5])
        with header.canvas.before:
            Color(0.15, 0.15, 0.2, 1)
            header.bg = RoundedRectangle(pos=header.pos, size=header.size, radius=[10])
        header.bind(pos=lambda *a: setattr(header.bg, 'pos', header.pos),
                   size=lambda *a: setattr(header.bg, 'size', header.size))
        
        title = Label(
            text='🤟 Hand Sign Detection',
            font_size='20sp',
            bold=True,
            color=(1, 1, 1, 1)
        )
        header.add_widget(title)
        self.add_widget(header)
        
        # === CAMERA AREA ===
        cam_container = BoxLayout(size_hint_y=0.45, padding=5)
        with cam_container.canvas.before:
            Color(0.05, 0.05, 0.08, 1)
            cam_container.bg = RoundedRectangle(pos=cam_container.pos, size=cam_container.size, radius=[15])
        cam_container.bind(pos=lambda *a: setattr(cam_container.bg, 'pos', cam_container.pos),
                          size=lambda *a: setattr(cam_container.bg, 'size', cam_container.size))
        
        self.camera = Camera(
            index=0,
            resolution=(640, 480),
            play=False,
            allow_stretch=True,
            keep_ratio=True
        )
        cam_container.add_widget(self.camera)
        self.add_widget(cam_container)
        
        # === STATUS BAR ===
        status_bar = BoxLayout(size_hint_y=0.05, spacing=10)
        
        self.status_label = Label(
            text=self.status_text,
            font_size='14sp',
            color=(0.9, 0.4, 0.4, 1),
            halign='left',
            valign='middle'
        )
        self.status_label.bind(size=self.status_label.setter('text_size'))
        status_bar.add_widget(self.status_label)
        
        self.fps_label = Label(
            text='FPS: --',
            font_size='14sp',
            color=(0.6, 0.6, 0.6, 1),
            size_hint_x=0.3,
            halign='right',
            valign='middle'
        )
        self.fps_label.bind(size=self.fps_label.setter('text_size'))
        status_bar.add_widget(self.fps_label)
        
        self.add_widget(status_bar)
        
        # === GESTURE DISPLAY ===
        gesture_box = BoxLayout(size_hint_y=0.12, padding=15)
        with gesture_box.canvas.before:
            Color(0.12, 0.22, 0.32, 1)
            gesture_box.bg = RoundedRectangle(pos=gesture_box.pos, size=gesture_box.size, radius=[12])
        gesture_box.bind(pos=lambda *a: setattr(gesture_box.bg, 'pos', gesture_box.pos),
                        size=lambda *a: setattr(gesture_box.bg, 'size', gesture_box.size))
        
        gesture_layout = BoxLayout(orientation='vertical')
        gesture_layout.add_widget(Label(
            text='Detected Gesture:',
            font_size='12sp',
            color=(0.7, 0.8, 0.9, 1),
            size_hint_y=0.3
        ))
        
        self.gesture_label = Label(
            text='---',
            font_size='28sp',
            bold=True,
            color=(0.3, 0.9, 0.5, 1),
            size_hint_y=0.7
        )
        gesture_layout.add_widget(self.gesture_label)
        gesture_box.add_widget(gesture_layout)
        self.add_widget(gesture_box)
        
        # === OUTPUT TEXT ===
        output_box = BoxLayout(size_hint_y=0.1, padding=10)
        with output_box.canvas.before:
            Color(0.15, 0.15, 0.2, 1)
            output_box.bg = RoundedRectangle(pos=output_box.pos, size=output_box.size, radius=[10])
        output_box.bind(pos=lambda *a: setattr(output_box.bg, 'pos', output_box.pos),
                       size=lambda *a: setattr(output_box.bg, 'size', output_box.size))
        
        self.output_label = Label(
            text='',
            font_size='16sp',
            color=(1, 1, 1, 1),
            halign='left',
            valign='middle'
        )
        self.output_label.bind(size=self.output_label.setter('text_size'))
        output_box.add_widget(self.output_label)
        self.add_widget(output_box)
        
        # === BUTTONS ROW 1 ===
        btn_row1 = GridLayout(cols=2, size_hint_y=0.1, spacing=10)
        
        self.start_btn = GestureButton(
            text='▶  START',
            bg_color=(0.2, 0.7, 0.4, 1)
        )
        self.start_btn.bind(on_press=self.start_camera)
        btn_row1.add_widget(self.start_btn)
        
        self.stop_btn = GestureButton(
            text='⏹  STOP',
            bg_color=(0.85, 0.35, 0.35, 1)
        )
        self.stop_btn.bind(on_press=self.stop_camera)
        btn_row1.add_widget(self.stop_btn)
        
        self.add_widget(btn_row1)
        
        # === BUTTONS ROW 2 ===
        btn_row2 = GridLayout(cols=2, size_hint_y=0.1, spacing=10)
        
        self.speak_btn = GestureButton(
            text='🔊  SPEAK',
            bg_color=(0.25, 0.55, 0.85, 1)
        )
        self.speak_btn.bind(on_press=self.speak_output)
        btn_row2.add_widget(self.speak_btn)
        
        self.clear_btn = GestureButton(
            text='✖  CLEAR',
            bg_color=(0.5, 0.5, 0.55, 1)
        )
        self.clear_btn.bind(on_press=self.clear_output)
        btn_row2.add_widget(self.clear_btn)
        
        self.add_widget(btn_row2)
    
    def start_camera(self, instance=None):
        """Start camera and detection."""
        if self.is_running:
            return
        
        self.camera.play = True
        self.is_running = True
        
        self.status_label.text = '● Camera Running'
        self.status_label.color = (0.3, 0.9, 0.5, 1)
        
        # Start detection loop
        self.frame_count = 0
        self._last_fps_time = time.time()
        Clock.schedule_interval(self._process_frame, 1.0/15)  # 15 FPS detection
    
    def stop_camera(self, instance=None):
        """Stop camera."""
        self.camera.play = False
        self.is_running = False
        
        self.status_label.text = '● Stopped'
        self.status_label.color = (0.9, 0.4, 0.4, 1)
        self.fps_label.text = 'FPS: --'
        
        Clock.unschedule(self._process_frame)
    
    def _process_frame(self, dt):
        """Process camera frame for gesture detection."""
        self.frame_count += 1
        
        # Update FPS
        now = time.time()
        if now - self._last_fps_time >= 1.0:
            self.fps_label.text = f'FPS: {self.frame_count}'
            self.frame_count = 0
            self._last_fps_time = now
        
        # Simulate gesture detection (demo mode)
        # In real app, analyze camera texture here
        self._simulate_detection()
    
    def _simulate_detection(self):
        """Simulate gesture detection for demo."""
        # This is for demo purposes
        # Real detection would analyze camera.texture
        import random
        
        # 5% chance to detect a gesture each frame
        if random.random() < 0.05:
            gestures = ['Hello 👋', 'Yes ✊', 'Look 👆', 'OK 👌', 'Peace ✌️']
            gesture = random.choice(gestures)
            
            # Check cooldown
            now = time.time()
            if now - self.last_gesture_time > 2.0:
                self.gesture_label.text = gesture
                self.detected_gestures.append(gesture.split()[0])
                
                if len(self.detected_gestures) > 5:
                    self.detected_gestures = self.detected_gestures[-5:]
                
                self.output_label.text = ' → '.join(self.detected_gestures)
                self.last_gesture_time = now
    
    def speak_output(self, instance=None):
        """Speak the output text."""
        text = ' '.join(self.detected_gestures) if self.detected_gestures else "No text"
        
        if platform == 'android' and HAS_TTS:
            try:
                tts.speak(text)
            except:
                pass
        elif self.tts_engine:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except:
                pass
        
        self.status_label.text = f'Speaking...'
        Clock.schedule_once(lambda dt: setattr(self.status_label, 'text', '● Running'), 1)
    
    def clear_output(self, instance=None):
        """Clear all output."""
        self.detected_gestures = []
        self.output_label.text = ''
        self.gesture_label.text = '---'
        self.last_gesture = None


class MobileApp(App):
    """Main mobile application."""
    
    def build(self):
        self.title = 'Hand Sign Detection'
        Window.clearcolor = (0.1, 0.1, 0.14, 1)
        return HandSignMobileApp()
    
    def on_pause(self):
        """Handle app pause (Android)."""
        return True
    
    def on_resume(self):
        """Handle app resume (Android)."""
        pass


if __name__ == '__main__':
    MobileApp().run()
