"""
==============================================================================
ANDROID BUILD INSTRUCTIONS
==============================================================================

⚠️  IMPORTANT LIMITATIONS FOR ANDROID:
=====================================
1. MediaPipe's Python library does NOT work on Android
2. pyttsx3 does NOT work on Android (need plyer or android TTS)
3. Building requires LINUX (use WSL on Windows)

📋 STEPS TO BUILD APK:
======================

OPTION A: Using WSL on Windows 10/11
------------------------------------
1. Install WSL (Windows Subsystem for Linux):
   wsl --install

2. Open WSL terminal and install dependencies:
   sudo apt update
   sudo apt install python3 python3-pip python3-venv
   sudo apt install git zip unzip openjdk-17-jdk
   sudo apt install autoconf libtool pkg-config
   sudo apt install zlib1g-dev libncurses5-dev libffi-dev
   sudo apt install libssl-dev libsqlite3-dev

3. Install Buildozer:
   pip3 install buildozer cython

4. Navigate to project (Windows path in WSL):
   cd /mnt/g/collegeProject/HSL

5. Initialize and build:
   buildozer init  # Only if buildozer.spec doesn't exist
   buildozer android debug

6. APK will be in: bin/ folder


OPTION B: Using Google Colab (Easier!)
--------------------------------------
1. Upload your project to Google Drive

2. Open Google Colab and run:

   # Mount Drive
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Install Buildozer
   !pip install buildozer cython
   !sudo apt update
   !sudo apt install -y git zip unzip openjdk-17-jdk
   
   # Navigate to project
   %cd /content/drive/MyDrive/HSL
   
   # Build APK
   !buildozer android debug

3. Download APK from bin/ folder


⚠️  MODIFIED FILES FOR ANDROID:
===============================
For Android compatibility, you need to:

1. Replace mediapipe with rule-based detection (already in gesture_logic.py)
2. Replace pyttsx3 with plyer TTS or remove TTS feature
3. Use opencv4android or camera4kivy for camera access


📱 SIMPLIFIED ANDROID VERSION:
==============================
For a quick Android demo, consider using only:
- Kivy Camera widget (built-in)
- Rule-based gesture detection
- Simple text display (no TTS)

This will work without MediaPipe!

==============================================================================
"""

print(__doc__)
