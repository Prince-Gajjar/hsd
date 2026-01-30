# ==============================================================================
# BUILDOZER SPEC - Hand Sign Detection Android App
# ==============================================================================
# This file configures the Android APK build process.
# 
# To build:
# 1. Clean old build: buildozer android clean
# 2. Build debug: buildozer android debug
# ==============================================================================

[app]

# Application Metadata
# --------------------
title = Hand Sign Detection
package.name = handsigndetection
package.domain = org.college.hsl

# Version
version = 1.0.0

# Source Configuration
# --------------------
source.dir = .
source.main = android_app.py

# Include files with these extensions
source.include_exts = py,tflite,txt,png,jpg,kv,atlas

# Explicitly include model files
source.include_patterns = android_app.py,gesture_model.tflite,gesture_model_quant.tflite,gesture_labels.txt,assets/*

# Exclude unnecessary files from APK
source.exclude_dirs = tests,bin,.git,__pycache__,.idea,build,.buildozer,venv
source.exclude_patterns = *.pyc,*.pyo,*.pickle,*.csv,*test*.py,train*.py,create*.py,download*.py,data*.py,BUILD*.py

# Requirements
# ------------
# SIMPLIFIED requirements for better compatibility
# Removed tflite-runtime to avoid pyjnius build issues
requirements = python3,kivy==2.2.1,pillow,plyer

# Android Configuration
# ---------------------
android.permissions = CAMERA,INTERNET

# API levels
android.minapi = 21
android.api = 33
android.ndk = 25b

# Target only arm64 for faster build (add armeabi-v7a for older phones)
android.archs = arm64-v8a

# Accept SDK license automatically
android.accept_sdk_license = True

# Orientation (portrait only for this app)
orientation = portrait

# Fullscreen mode (0 = no, 1 = yes)
fullscreen = 0

# Application Icon & Splash
# -------------------------
# Uncomment and set paths when you have these assets:
# icon.filename = assets/icon.png
# presplash.filename = assets/splash.png

# Android Specific Settings
# -------------------------
# Bootstrap - use p4a.bootstrap (not deprecated android.bootstrap)
p4a.bootstrap = sdl2

# Enable AndroidX
android.enable_androidx = True

# Build Settings
# --------------
# Set to False for debug builds (faster), True for release
android.release = False

# Debug mode
android.debug = True

# Skip NDK updates (faster builds)
android.skip_update = True

# Presplash color (hex)
android.presplash_color = #0D0D14

# Logcat filters
android.logcat_filters = *:S python:D

# ==============================================================================
# BUILDOZER SETTINGS
# ==============================================================================

[buildozer]

# Log level (0 = error, 1 = info, 2 = debug)
log_level = 2

# Display warning for root user
warn_on_root = 1

# Build directory
build_dir = ./.buildozer

# Bin directory for APK output
bin_dir = ./bin
