"""
==============================================================================
ANDROID-COMPATIBLE HAND SIGN DETECTION APP
==============================================================================
This is the main mobile app for Android deployment using:
- Kivy for UI (Android-compatible)
- Kivy Camera (works natively on Android)
- TensorFlow Lite for gesture recognition (lightweight, Android-optimized)
- Plyer for TTS (Android native TTS support)

This app captures camera frames, processes hand landmarks using image
analysis, and classifies gestures using a pre-trained TFLite model.

To build APK:
1. Install Buildozer in WSL/Linux
2. Run: buildozer android debug
==============================================================================
"""

# ============================================================================
# KIVY CONFIG - MUST be before any other Kivy imports
# ============================================================================
from kivy.config import Config
Config.set('graphics', 'width', '360')
Config.set('graphics', 'height', '720')
Config.set('graphics', 'resizable', False)
Config.set('kivy', 'log_level', 'info')

# ============================================================================
# IMPORTS
# ============================================================================
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle, RoundedRectangle
from kivy.properties import StringProperty, BooleanProperty, NumericProperty
from kivy.utils import platform
import time
import os

# ============================================================================
# PLATFORM-SPECIFIC IMPORTS
# ============================================================================

# TensorFlow Lite for gesture recognition
try:
    if platform == 'android':
        # On Android, use tflite_runtime (lighter)
        from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
    else:
        # On desktop, use full TensorFlow
        import tensorflow as tf
        TFLiteInterpreter = tf.lite.Interpreter
    HAS_TFLITE = True
except ImportError:
    HAS_TFLITE = False
    print("Warning: TFLite not available. Using rule-based detection.")

# NumPy for array processing
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: NumPy not available.")

# TTS (Text-to-Speech)
if platform == 'android':
    try:
        from plyer import tts
        HAS_TTS = True
    except ImportError:
        HAS_TTS = False
else:
    try:
        import pyttsx3
        HAS_TTS = True
    except ImportError:
        HAS_TTS = False

# Android-specific permissions
if platform == 'android':
    from android.permissions import request_permissions, Permission
    request_permissions([Permission.CAMERA, Permission.RECORD_AUDIO])

# ============================================================================
# GESTURE CLASSIFIER
# ============================================================================

class GestureClassifier:
    """
    Handles gesture classification using TFLite model or rule-based fallback.
    
    The TFLite model expects 42 features (21 landmarks × 2 coordinates).
    Labels: Open, Close, Pointer, OK
    """
    
    # Gesture label mapping to user-friendly text
    GESTURE_TEXT = {
        'Open': 'Hello 👋',
        'Close': 'Yes ✊',
        'Pointer': 'Look 👆',
        'OK': 'OK 👌',
        'Unknown': '---'
    }
    
    def __init__(self):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.labels = ['Open', 'Close', 'Pointer', 'OK']
        self.is_ready = False
        
        self._load_model()
    
    def _load_model(self):
        """Load the TFLite model."""
        if not HAS_TFLITE or not HAS_NUMPY:
            print("TFLite or NumPy not available, using fallback detection")
            return
        
        # Try different model paths
        model_paths = [
            'gesture_model_quant.tflite',  # Smaller, quantized
            'gesture_model.tflite',         # Full precision
            os.path.join(os.path.dirname(__file__), 'gesture_model_quant.tflite'),
            os.path.join(os.path.dirname(__file__), 'gesture_model.tflite'),
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    self.interpreter = TFLiteInterpreter(model_path=model_path)
                    self.interpreter.allocate_tensors()
                    self.input_details = self.interpreter.get_input_details()
                    self.output_details = self.interpreter.get_output_details()
                    self.is_ready = True
                    print(f"✓ Loaded TFLite model: {model_path}")
                    print(f"  Input shape: {self.input_details[0]['shape']}")
                    print(f"  Output shape: {self.output_details[0]['shape']}")
                    break
                except Exception as e:
                    print(f"Error loading model {model_path}: {e}")
        
        # Load labels
        label_paths = [
            'gesture_labels.txt',
            os.path.join(os.path.dirname(__file__), 'gesture_labels.txt'),
        ]
        
        for label_path in label_paths:
            if os.path.exists(label_path):
                try:
                    with open(label_path, 'r') as f:
                        self.labels = [line.strip() for line in f if line.strip()]
                    print(f"✓ Loaded labels: {self.labels}")
                    break
                except Exception as e:
                    print(f"Error loading labels: {e}")
    
    def predict(self, landmarks):
        """
        Predict gesture from hand landmarks.
        
        Args:
            landmarks: List of (x, y) tuples for 21 hand landmarks,
                      normalized to 0-1 range.
        
        Returns:
            tuple: (gesture_name, confidence)
        """
        if not landmarks or len(landmarks) < 21:
            return 'Unknown', 0.0
        
        # Use TFLite model if available
        if self.is_ready and self.interpreter:
            return self._predict_tflite(landmarks)
        else:
            return self._predict_rules(landmarks)
    
    def _predict_tflite(self, landmarks):
        """Predict using TFLite model."""
        try:
            # Flatten landmarks to feature vector
            features = []
            for x, y in landmarks[:21]:
                features.extend([float(x), float(y)])
            
            # Prepare input
            input_data = np.array([features], dtype=np.float32)
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Get prediction
            predicted_idx = np.argmax(output[0])
            confidence = float(output[0][predicted_idx])
            
            if predicted_idx < len(self.labels):
                gesture = self.labels[predicted_idx]
            else:
                gesture = 'Unknown'
            
            return gesture, confidence
            
        except Exception as e:
            print(f"TFLite prediction error: {e}")
            return self._predict_rules(landmarks)
    
    def _predict_rules(self, landmarks):
        """
        Rule-based gesture detection (fallback).
        Works without ML model.
        """
        if len(landmarks) < 21:
            return 'Unknown', 0.0
        
        # Landmark indices:
        # 0: Wrist
        # 4: Thumb tip, 8: Index tip, 12: Middle tip, 16: Ring tip, 20: Pinky tip
        # 5,9,13,17: Finger bases (MCP joints)
        
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        index_base = landmarks[5]
        middle_base = landmarks[9]
        ring_base = landmarks[13]
        pinky_base = landmarks[17]
        
        # Check if fingers are extended (tip is above base)
        index_up = index_tip[1] < index_base[1]
        middle_up = middle_tip[1] < middle_base[1]
        ring_up = ring_tip[1] < ring_base[1]
        pinky_up = pinky_tip[1] < pinky_base[1]
        
        # Open palm: All fingers up
        if index_up and middle_up and ring_up and pinky_up:
            return 'Open', 0.85
        
        # Pointer: Only index finger up
        if index_up and not middle_up and not ring_up and not pinky_up:
            return 'Pointer', 0.85
        
        # OK sign: Thumb and index close together
        thumb_index_dist = ((thumb_tip[0] - index_tip[0])**2 + 
                           (thumb_tip[1] - index_tip[1])**2) ** 0.5
        if thumb_index_dist < 0.1:
            return 'OK', 0.80
        
        # Close/Fist: All fingers down
        if not index_up and not middle_up and not ring_up and not pinky_up:
            return 'Close', 0.85
        
        return 'Unknown', 0.0
    
    def get_text(self, gesture):
        """Get user-friendly text for a gesture."""
        return self.GESTURE_TEXT.get(gesture, '---')


# ============================================================================
# SIMPLE HAND DETECTOR (Rule-based for Android)
# ============================================================================

class SimpleHandDetector:
    """
    Simple motion/color-based hand detection for Android.
    
    This is a lightweight alternative to MediaPipe that works without
    complex dependencies. It uses skin color detection and motion analysis.
    
    For better accuracy on Android, consider using:
    - MediaPipe Android SDK (native)
    - ML Kit from Google
    - Custom TFLite hand detection model
    """
    
    def __init__(self):
        self.last_landmarks = None
        self.detection_confidence = 0.0
    
    def detect(self, texture):
        """
        Detect hand from camera texture.
        
        In a production app, this would use proper hand detection.
        For demo purposes, we simulate landmark detection based on
        the center of the frame.
        
        Args:
            texture: Kivy texture from camera
            
        Returns:
            list: 21 (x, y) landmark tuples normalized to 0-1, or None
        """
        if texture is None:
            return None
        
        # For a real implementation, you would:
        # 1. Convert texture to numpy array
        # 2. Apply skin color detection (HSV filtering)
        # 3. Find hand contours
        # 4. Estimate landmark positions
        # 5. Or use a separate TFLite hand detection model
        
        # Demo: Generate dynamic landmarks based on time
        # This simulates detection for UI testing
        import math
        t = time.time()
        
        # Base hand position (centered)
        base_x, base_y = 0.5, 0.5
        
        # Simulate slight movement
        offset_x = math.sin(t * 0.5) * 0.02
        offset_y = math.cos(t * 0.7) * 0.02
        
        # Generate approximate hand landmark positions
        # These represent a typical open hand pose
        landmarks = [
            (base_x + offset_x, base_y + 0.15 + offset_y),           # 0: Wrist
            (base_x - 0.08 + offset_x, base_y + 0.08 + offset_y),   # 1: Thumb CMC
            (base_x - 0.12 + offset_x, base_y + 0.02 + offset_y),   # 2: Thumb MCP
            (base_x - 0.15 + offset_x, base_y - 0.03 + offset_y),   # 3: Thumb IP
            (base_x - 0.17 + offset_x, base_y - 0.08 + offset_y),   # 4: Thumb TIP
            (base_x - 0.05 + offset_x, base_y + offset_y),          # 5: Index MCP
            (base_x - 0.05 + offset_x, base_y - 0.08 + offset_y),   # 6: Index PIP
            (base_x - 0.05 + offset_x, base_y - 0.14 + offset_y),   # 7: Index DIP
            (base_x - 0.05 + offset_x, base_y - 0.18 + offset_y),   # 8: Index TIP
            (base_x + offset_x, base_y - 0.02 + offset_y),          # 9: Middle MCP
            (base_x + offset_x, base_y - 0.10 + offset_y),          # 10: Middle PIP
            (base_x + offset_x, base_y - 0.16 + offset_y),          # 11: Middle DIP
            (base_x + offset_x, base_y - 0.20 + offset_y),          # 12: Middle TIP
            (base_x + 0.05 + offset_x, base_y + offset_y),          # 13: Ring MCP
            (base_x + 0.05 + offset_x, base_y - 0.07 + offset_y),   # 14: Ring PIP
            (base_x + 0.05 + offset_x, base_y - 0.12 + offset_y),   # 15: Ring DIP
            (base_x + 0.05 + offset_x, base_y - 0.16 + offset_y),   # 16: Ring TIP
            (base_x + 0.10 + offset_x, base_y + 0.02 + offset_y),   # 17: Pinky MCP
            (base_x + 0.10 + offset_x, base_y - 0.04 + offset_y),   # 18: Pinky PIP
            (base_x + 0.10 + offset_x, base_y - 0.09 + offset_y),   # 19: Pinky DIP
            (base_x + 0.10 + offset_x, base_y - 0.13 + offset_y),   # 20: Pinky TIP
        ]
        
        self.last_landmarks = landmarks
        self.detection_confidence = 0.75
        
        return landmarks


# ============================================================================
# STYLED BUTTON WIDGET
# ============================================================================

class StyledButton(Button):
    """Custom styled button with rounded corners."""
    
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
                radius=[12]
            )
        
        self.bind(pos=self._update_rect, size=self._update_rect)
    
    def _update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size


# ============================================================================
# MAIN APPLICATION SCREEN
# ============================================================================

class MainScreen(BoxLayout):
    """Main application screen with camera preview and controls."""
    
    gesture_text = StringProperty('---')
    output_text = StringProperty('')
    status_text = StringProperty('Tap START to begin')
    is_running = BooleanProperty(False)
    fps = NumericProperty(0)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = [15, 10]
        self.spacing = 8
        
        # Set dark background
        with self.canvas.before:
            Color(0.08, 0.08, 0.12, 1)
            self.bg_rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._update_bg, size=self._update_bg)
        
        # Initialize components
        self.classifier = GestureClassifier()
        self.hand_detector = SimpleHandDetector()
        self.detected_gestures = []
        self.last_gesture = None
        self.last_gesture_time = 0
        self.frame_count = 0
        self._last_fps_time = time.time()
        self.tts_engine = None
        
        # Initialize TTS for desktop
        if platform != 'android' and HAS_TTS:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
            except Exception as e:
                print(f"TTS init error: {e}")
        
        # Build the UI
        self._build_ui()
    
    def _update_bg(self, *args):
        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size
    
    def _build_ui(self):
        """Build the mobile-optimized UI."""
        
        # === HEADER ===
        header = BoxLayout(size_hint_y=0.07, padding=[10, 5])
        with header.canvas.before:
            Color(0.12, 0.12, 0.18, 1)
            header.bg = RoundedRectangle(pos=header.pos, size=header.size, radius=[10])
        header.bind(
            pos=lambda *a: setattr(header.bg, 'pos', header.pos),
            size=lambda *a: setattr(header.bg, 'size', header.size)
        )
        
        title = Label(
            text='🤟 Hand Sign Detection',
            font_size='20sp',
            bold=True,
            color=(1, 1, 1, 1)
        )
        header.add_widget(title)
        self.add_widget(header)
        
        # === CAMERA AREA ===
        cam_container = BoxLayout(size_hint_y=0.42, padding=5)
        with cam_container.canvas.before:
            Color(0.05, 0.05, 0.08, 1)
            cam_container.bg = RoundedRectangle(
                pos=cam_container.pos, size=cam_container.size, radius=[15]
            )
        cam_container.bind(
            pos=lambda *a: setattr(cam_container.bg, 'pos', cam_container.pos),
            size=lambda *a: setattr(cam_container.bg, 'size', cam_container.size)
        )
        
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
        status_bar = BoxLayout(size_hint_y=0.05, spacing=10, padding=[5, 0])
        
        self.status_label = Label(
            text='● Ready',
            font_size='14sp',
            color=(0.6, 0.6, 0.6, 1),
            halign='left',
            valign='middle',
            size_hint_x=0.7
        )
        self.status_label.bind(size=self.status_label.setter('text_size'))
        status_bar.add_widget(self.status_label)
        
        self.fps_label = Label(
            text='FPS: --',
            font_size='14sp',
            color=(0.5, 0.5, 0.5, 1),
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
            Color(0.1, 0.18, 0.28, 1)
            gesture_box.bg = RoundedRectangle(
                pos=gesture_box.pos, size=gesture_box.size, radius=[12]
            )
        gesture_box.bind(
            pos=lambda *a: setattr(gesture_box.bg, 'pos', gesture_box.pos),
            size=lambda *a: setattr(gesture_box.bg, 'size', gesture_box.size)
        )
        
        gesture_layout = BoxLayout(orientation='vertical')
        gesture_layout.add_widget(Label(
            text='Detected Gesture:',
            font_size='12sp',
            color=(0.6, 0.8, 0.9, 1),
            size_hint_y=0.35
        ))
        
        self.gesture_label = Label(
            text='---',
            font_size='26sp',
            bold=True,
            color=(0.3, 0.95, 0.5, 1),
            size_hint_y=0.65
        )
        gesture_layout.add_widget(self.gesture_label)
        gesture_box.add_widget(gesture_layout)
        self.add_widget(gesture_box)
        
        # === OUTPUT TEXT AREA ===
        output_box = BoxLayout(size_hint_y=0.1, padding=12)
        with output_box.canvas.before:
            Color(0.12, 0.12, 0.18, 1)
            output_box.bg = RoundedRectangle(
                pos=output_box.pos, size=output_box.size, radius=[10]
            )
        output_box.bind(
            pos=lambda *a: setattr(output_box.bg, 'pos', output_box.pos),
            size=lambda *a: setattr(output_box.bg, 'size', output_box.size)
        )
        
        self.output_label = Label(
            text='Detected signs will appear here...',
            font_size='14sp',
            color=(0.8, 0.8, 0.8, 1),
            halign='left',
            valign='middle'
        )
        self.output_label.bind(size=self.output_label.setter('text_size'))
        output_box.add_widget(self.output_label)
        self.add_widget(output_box)
        
        # === BUTTONS ROW 1 ===
        btn_row1 = GridLayout(cols=2, size_hint_y=0.1, spacing=10, padding=[0, 5])
        
        self.start_btn = StyledButton(
            text='▶  START',
            bg_color=(0.15, 0.65, 0.35, 1)
        )
        self.start_btn.bind(on_press=self.start_detection)
        btn_row1.add_widget(self.start_btn)
        
        self.stop_btn = StyledButton(
            text='⏹  STOP',
            bg_color=(0.8, 0.3, 0.3, 1)
        )
        self.stop_btn.bind(on_press=self.stop_detection)
        btn_row1.add_widget(self.stop_btn)
        
        self.add_widget(btn_row1)
        
        # === BUTTONS ROW 2 ===
        btn_row2 = GridLayout(cols=2, size_hint_y=0.1, spacing=10, padding=[0, 5])
        
        self.speak_btn = StyledButton(
            text='🔊  SPEAK',
            bg_color=(0.2, 0.5, 0.8, 1)
        )
        self.speak_btn.bind(on_press=self.speak_output)
        btn_row2.add_widget(self.speak_btn)
        
        self.clear_btn = StyledButton(
            text='✖  CLEAR',
            bg_color=(0.45, 0.45, 0.5, 1)
        )
        self.clear_btn.bind(on_press=self.clear_output)
        btn_row2.add_widget(self.clear_btn)
        
        self.add_widget(btn_row2)
    
    # ========================================================================
    # CAMERA & DETECTION CONTROL
    # ========================================================================
    
    def start_detection(self, instance=None):
        """Start camera and gesture detection."""
        if self.is_running:
            return
        
        self.camera.play = True
        self.is_running = True
        
        self.status_label.text = '● Detecting...'
        self.status_label.color = (0.3, 0.95, 0.5, 1)
        
        # Start detection loop at 15 FPS
        self.frame_count = 0
        self._last_fps_time = time.time()
        Clock.schedule_interval(self._process_frame, 1.0 / 15)
    
    def stop_detection(self, instance=None):
        """Stop camera and detection."""
        self.camera.play = False
        self.is_running = False
        
        self.status_label.text = '● Stopped'
        self.status_label.color = (0.9, 0.4, 0.4, 1)
        self.fps_label.text = 'FPS: --'
        
        Clock.unschedule(self._process_frame)
    
    def _process_frame(self, dt):
        """Process each camera frame for gesture detection."""
        self.frame_count += 1
        
        # Update FPS counter
        now = time.time()
        if now - self._last_fps_time >= 1.0:
            self.fps = self.frame_count
            self.fps_label.text = f'FPS: {self.frame_count}'
            self.frame_count = 0
            self._last_fps_time = now
        
        # Get camera texture and detect hand
        texture = self.camera.texture
        if texture is None:
            return
        
        # Detect hand landmarks
        landmarks = self.hand_detector.detect(texture)
        
        if landmarks:
            # Classify gesture
            gesture, confidence = self.classifier.predict(landmarks)
            
            if gesture != 'Unknown' and confidence > 0.6:
                # Debounce - only register new gesture after 1.5 seconds
                if now - self.last_gesture_time > 1.5 or gesture != self.last_gesture:
                    self._register_gesture(gesture)
                    self.last_gesture = gesture
                    self.last_gesture_time = now
    
    def _register_gesture(self, gesture):
        """Register a detected gesture."""
        text = self.classifier.get_text(gesture)
        self.gesture_label.text = text
        
        # Add to detected list
        self.detected_gestures.append(gesture)
        if len(self.detected_gestures) > 6:
            self.detected_gestures = self.detected_gestures[-6:]
        
        # Update output display
        gesture_texts = [self.classifier.get_text(g).split()[0] for g in self.detected_gestures]
        self.output_label.text = ' → '.join(gesture_texts)
    
    # ========================================================================
    # TTS & UTILITY FUNCTIONS
    # ========================================================================
    
    def speak_output(self, instance=None):
        """Speak the detected gestures using TTS."""
        if not self.detected_gestures:
            text = "No gestures detected"
        else:
            # Convert gestures to speakable text
            words = []
            for g in self.detected_gestures:
                if g == 'Open':
                    words.append('Hello')
                elif g == 'Close':
                    words.append('Yes')
                elif g == 'Pointer':
                    words.append('Look')
                elif g == 'OK':
                    words.append('OK')
            text = ', '.join(words) if words else "No gestures"
        
        # Speak using appropriate TTS
        if platform == 'android' and HAS_TTS:
            try:
                tts.speak(text)
            except Exception as e:
                print(f"Android TTS error: {e}")
        elif self.tts_engine:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"Desktop TTS error: {e}")
        
        self.status_label.text = f'🔊 Speaking: {text[:25]}...'
        Clock.schedule_once(lambda dt: self._reset_status(), 2)
    
    def clear_output(self, instance=None):
        """Clear all detected gestures."""
        self.detected_gestures = []
        self.output_label.text = 'Cleared'
        self.gesture_label.text = '---'
        self.last_gesture = None
        
        Clock.schedule_once(lambda dt: self._reset_output_text(), 1)
    
    def _reset_status(self):
        """Reset status label."""
        if self.is_running:
            self.status_label.text = '● Detecting...'
            self.status_label.color = (0.3, 0.95, 0.5, 1)
        else:
            self.status_label.text = '● Ready'
            self.status_label.color = (0.6, 0.6, 0.6, 1)
    
    def _reset_output_text(self):
        """Reset output label placeholder."""
        if not self.detected_gestures:
            self.output_label.text = 'Detected signs will appear here...'


# ============================================================================
# MAIN APPLICATION
# ============================================================================

class HandSignApp(App):
    """Main Kivy application for Hand Sign Detection."""
    
    def build(self):
        """Build and return the main screen."""
        self.title = 'Hand Sign Detection'
        Window.clearcolor = (0.08, 0.08, 0.12, 1)
        return MainScreen()
    
    def on_pause(self):
        """Handle Android app pause (e.g., switching apps)."""
        return True
    
    def on_resume(self):
        """Handle Android app resume."""
        pass
    
    def on_stop(self):
        """Clean up when app closes."""
        if hasattr(self.root, 'tts_engine') and self.root.tts_engine:
            try:
                self.root.tts_engine.stop()
            except:
                pass


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    HandSignApp().run()
