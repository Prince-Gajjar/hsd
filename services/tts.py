# services/tts.py

"""
This module handles the Text-to-Speech (TTS) functionality.

It uses 'plyer' for cross-platform compatibility, which works on Android.
For desktop systems where 'plyer' might not have a TTS implementation,
it falls back to 'pyttsx3'.
"""

from kivy.utils import platform

# Use plyer for Android's native TTS
if platform == 'android':
    from plyer import tts

    def speak(text):
        """
        Uses plyer's TTS to speak the given text on Android.
        """
        try:
            tts.speak(message=text)
        except Exception as e:
            print(f"Error using plyer TTS: {e}")

# Use pyttsx3 as a fallback for desktop (Windows, Linux, macOS)
else:
    import pyttsx3
    try:
        engine = pyttsx3.init()
    except Exception as e:
        engine = None
        print(f"Could not initialize pyttsx3 engine: {e}")

    def speak(text):
        """
        Uses pyttsx3 to speak the given text on desktop systems.
        """
        if engine:
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"Error using pyttsx3: {e}")
        else:
            print("TTS engine not initialized. Cannot speak.")
