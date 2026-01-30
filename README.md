<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Kivy-2.2+-5E5E5E?style=for-the-badge&logo=kivy&logoColor=white" alt="Kivy"/>
  <img src="https://img.shields.io/badge/TensorFlow_Lite-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow Lite"/>
  <img src="https://img.shields.io/badge/MediaPipe-0097A7?style=for-the-badge&logo=google&logoColor=white" alt="MediaPipe"/>
</p>

<h1 align="center">🤟 Hand Sign Detection System</h1>

<p align="center">
  <strong>Real-time hand gesture recognition with Text-to-Speech conversion</strong>
</p>

<p align="center">
  A cross-platform application that uses computer vision and machine learning to detect hand signs in real-time, convert them to text, and speak them aloud using text-to-speech technology.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Platform-Windows%20|%20Linux%20|%20macOS%20|%20Android-blue?style=flat-square" alt="Platforms"/>
  <img src="https://img.shields.io/badge/License-Educational-green?style=flat-square" alt="License"/>
  <img src="https://img.shields.io/badge/Status-Active-success?style=flat-square" alt="Status"/>
</p>

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🎯 Supported Gestures](#-supported-gestures)
- [🛠️ Technology Stack](#️-technology-stack)
- [📁 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [📱 Android APK Build](#-android-apk-build)
- [🧠 Model Training](#-model-training)
- [⚙️ Configuration](#️-configuration)
- [🔧 Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🎥 Real-time Detection

- Live camera feed processing at 15-30 FPS
- MediaPipe-powered hand landmark detection
- Visual landmark overlay on detected hands

</td>
<td width="50%">

### 🧠 Smart Recognition

- TensorFlow Lite model for gesture classification
- Rule-based fallback detection
- 60%+ confidence threshold for accuracy

</td>
</tr>
<tr>
<td width="50%">

### 🔊 Text-to-Speech

- Convert gestures to spoken words
- Cross-platform TTS support
- Native Android TTS integration

</td>
<td width="50%">

### 📱 Cross-Platform

- Desktop: Windows, Linux, macOS
- Mobile: Android APK support
- Consistent UI across platforms

</td>
</tr>
</table>

---

## 🎯 Supported Gestures

The system currently recognizes **4 hand gestures**:

|     Gesture     | Visual | Recognized As | Speech Output |
| :-------------: | :----: | :-----------: | :-----------: |
|  **Open Palm**  |   👋   |    `Open`     |    "Hello"    |
| **Closed Fist** |   ✊   |    `Close`    |     "Yes"     |
|  **Pointing**   |   👆   |   `Pointer`   |    "Look"     |
|   **OK Sign**   |   👌   |     `OK`      |     "OK"      |

---

## 🛠️ Technology Stack

| Component            | Technology      | Purpose                            |
| -------------------- | --------------- | ---------------------------------- |
| **UI Framework**     | Kivy 2.2+       | Cross-platform user interface      |
| **Hand Detection**   | MediaPipe       | Hand landmark detection (Desktop)  |
| **ML Model**         | TensorFlow Lite | Gesture classification             |
| **Desktop TTS**      | pyttsx3         | Text-to-speech (Windows/Linux/Mac) |
| **Mobile TTS**       | Plyer           | Text-to-speech (Android)           |
| **Image Processing** | OpenCV, NumPy   | Camera and frame processing        |
| **Build Tool**       | Buildozer       | Android APK generation             |

---

## 📁 Project Structure

```
hsd/
├── 📄 main.py                  # Desktop application (MediaPipe)
├── 📄 android_app.py           # Android-compatible app (TFLite)
├── 📄 main.kv                  # Kivy UI layout file
├── 📄 buildozer.spec           # Android build configuration
├── 📄 requirements.txt         # Python dependencies
│
├── 🧠 Models & Data
│   ├── gesture_model.tflite        # Full precision TFLite model
│   ├── gesture_model_quant.tflite  # Quantized model (smaller)
│   ├── gesture_model.pickle        # Trained model (pickle format)
│   ├── gesture_labels.txt          # Label mapping file
│   ├── keypoint.csv                # Training data
│   └── keypoint_labels.csv         # Label definitions
│
├── 📁 camera/                  # Camera handling module
│   ├── __init__.py
│   └── camera_stream.py        # Camera capture and processing
│
├── 📁 detection/               # Detection logic
│   ├── __init__.py
│   ├── hand_detector.py        # MediaPipe hand detection
│   └── gesture_logic.py        # Gesture classification rules
│
├── 📁 services/                # Application services
│   ├── __init__.py
│   ├── tts.py                  # Text-to-speech service
│   └── translator.py           # Translation service (future)
│
├── 📁 utils/                   # Utility modules
│   ├── __init__.py
│   └── constants.py            # App constants and config
│
├── 🛠️ Training Scripts
│   ├── data_collector.py       # Collect training data
│   ├── download_dataset.py     # Download external datasets
│   ├── create_model.py         # Create and export model
│   ├── train_model.py          # Train classifier
│   └── train_tflite.py         # Convert to TFLite format
│
└── 📁 assets/                  # App assets (icons, images)
```

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Version    | Notes                            |
| ----------- | ---------- | -------------------------------- |
| **Python**  | 3.8 - 3.11 | Python 3.12+ not fully supported |
| **pip**     | Latest     | Package manager                  |
| **Webcam**  | Any        | For hand detection               |
| **Git**     | Any        | For cloning repository           |

### Installation

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Prince-Gajjar/hsd.git
cd hsd
```

#### 2️⃣ Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

#### 3️⃣ Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install additional desktop dependencies
pip install opencv-python mediapipe tensorflow
```

### Running the Application

#### 🖥️ Desktop Version (Full Features)

The desktop version uses **MediaPipe** for accurate hand detection:

```bash
python main.py
```

#### 📱 Android-Compatible Version

For testing the Android-compatible version on desktop:

```bash
python android_app.py
```

---

## 📱 Android APK Build

### Requirements

- **Linux** or **Windows with WSL** (Ubuntu 20.04+ recommended)
- **Buildozer** and **Cython**
- **Android SDK/NDK** (auto-installed by Buildozer)

### Step-by-Step Build Guide

#### 1️⃣ Set Up WSL (Windows Users)

```powershell
# Run in PowerShell as Administrator
wsl --install -d Ubuntu
```

#### 2️⃣ Install Build Dependencies (WSL/Linux)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y \
    python3-pip python3-venv git zip unzip \
    openjdk-17-jdk autoconf libtool pkg-config \
    zlib1g-dev libncurses5-dev libncursesw5-dev libtinfo5 \
    cmake libffi-dev libssl-dev

# Install Buildozer and Cython
pip3 install --upgrade buildozer cython virtualenv
```

#### 3️⃣ Navigate to Project Directory

```bash
# Adjust the path based on your Windows drive
cd /mnt/g/collegeProject/hsd
```

#### 4️⃣ Build the APK

```bash
# Debug build (for testing)
buildozer android debug

# The APK will be generated at:
# bin/handsigndetection-1.0.0-arm64-v8a-debug.apk
```

#### 5️⃣ Install on Android Device

```bash
# Connect your device via USB (enable USB debugging)
buildozer android deploy run logcat
```

> 💡 **Tip:** First build may take 20-40 minutes as it downloads Android SDK/NDK.

---

## 🧠 Model Training

Train your own gesture recognition model:

### Step 1: Collect Training Data

```bash
python data_collector.py
```

- Shows camera preview
- Press keys (0-3) to record gestures
- Data saved to `keypoint.csv`

### Step 2: Download Additional Dataset (Optional)

```bash
python download_dataset.py
```

### Step 3: Train the Model

```bash
# Train and create pickle model
python train_model.py

# Convert to TFLite format
python train_tflite.py
```

### Output Files

| File                         | Description        | Size    |
| ---------------------------- | ------------------ | ------- |
| `gesture_model.pickle`       | Full trained model | ~2.7 MB |
| `gesture_model.tflite`       | TFLite model       | ~24 KB  |
| `gesture_model_quant.tflite` | Quantized TFLite   | ~14 KB  |
| `gesture_labels.txt`         | Label mapping      | <1 KB   |

---

## ⚙️ Configuration

### Gesture Mappings

Edit `utils/constants.py` to customize gestures:

```python
# Gesture to Text Mapping
GESTURE_MAP = {
    "Open": "Hello",      # Open palm → "Hello"
    "Close": "Yes",       # Closed fist → "Yes"
    "Pointer": "Look",    # Pointing → "Look"
    "OK": "OK",           # OK sign → "OK"
}

# Add custom gestures here
```

### Camera Settings

```python
# In utils/constants.py
CAMERA_RESOLUTION = (640, 480)  # Camera resolution
CAMERA_INDEX = 0                 # Camera device index
DETECTION_FPS = 15               # Detection rate
```

### Model Settings

```python
# In utils/constants.py
CONFIDENCE_THRESHOLD = 0.6       # Min confidence (0-1)
GESTURE_DEBOUNCE_TIME = 1.5      # Seconds between detections
```

---

## 🔧 Troubleshooting

### Common Issues & Solutions

<details>
<summary><b>❌ Camera not opening</b></summary>

**Solution:**

1. Check if another app is using the camera
2. Try different camera index: Change `CAMERA_INDEX = 1` in `constants.py`
3. On Linux: Run `sudo chmod 666 /dev/video0`

</details>

<details>
<summary><b>❌ MediaPipe import error</b></summary>

**Solution:**

```bash
pip uninstall mediapipe
pip install mediapipe==0.10.0
```

</details>

<details>
<summary><b>❌ TTS not speaking (Windows)</b></summary>

**Solution:**

1. Install/reinstall pyttsx3: `pip install --upgrade pyttsx3`
2. Check Windows speech settings
3. Try: `pip install pywin32`

</details>

<details>
<summary><b>❌ Buildozer build fails</b></summary>

**Solution:**

1. Clean build: `buildozer android clean`
2. Update Buildozer: `pip install --upgrade buildozer`
3. Check Java version: `java -version` (need JDK 17)
4. Accept SDK licenses: Add `android.accept_sdk_license = True` to `buildozer.spec`

</details>

<details>
<summary><b>❌ Model not loading</b></summary>

**Solution:**

1. Ensure `.tflite` files are in project root
2. Check file permissions
3. Verify TensorFlow is installed: `pip install tensorflow`

</details>

### Build Errors Quick Reference

| Error                      | Solution                                                    |
| -------------------------- | ----------------------------------------------------------- |
| `SDK license not accepted` | Add `android.accept_sdk_license = True` to `buildozer.spec` |
| `NDK not found`            | Let Buildozer auto-install or set path manually             |
| `Recipe failed`            | Run `buildozer android clean` and rebuild                   |
| `AAPT2 error`              | Delete `.buildozer` folder and rebuild                      |

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Ideas for Contribution

- [ ] Add more gesture types
- [ ] Improve hand detection accuracy
- [ ] Add multi-language TTS support
- [ ] Create iOS version
- [ ] Add gesture recording/playback

---

## 📄 License

This project is developed for **educational purposes** as a college project.

---

<p align="center">
  <strong>Made with ❤️ for accessibility</strong>
</p>

<p align="center">
  <a href="#-hand-sign-detection-system">⬆️ Back to Top</a>
</p>
