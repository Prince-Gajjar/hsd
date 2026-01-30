# 🤟 Hand Sign Detection System

A Kivy-based mobile/desktop application that performs **real-time hand sign detection** and converts detected signs into text and speech.

![Platform](https://img.shields.io/badge/Platform-Android%20%7C%20Windows%20%7C%20Linux-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Kivy](https://img.shields.io/badge/Kivy-2.2%2B-orange)

---

## ✨ Features

- 📹 **Live Camera Feed** - Real-time camera processing
- 🖐️ **Gesture Detection** - Recognizes 4 hand signs (Open, Close, Pointer, OK)
- 📝 **Text Conversion** - Converts gestures to readable text
- 🔊 **Text-to-Speech** - Speaks detected gestures aloud
- 📱 **Android Compatible** - Build APK for Android devices
- 🖥️ **Desktop Support** - Works on Windows, Linux, macOS

---

## 🎯 Supported Gestures

| Gesture     | Sign | Text Output |
| ----------- | ---- | ----------- |
| Open Palm   | 👋   | Hello       |
| Closed Fist | ✊   | Yes         |
| Pointing    | 👆   | Look        |
| OK Sign     | 👌   | OK          |

---

## 📁 Project Structure

```
HSL/
├── android_app.py          # 📱 Android-compatible main app
├── main.py                 # 🖥️ Desktop app (with MediaPipe)
├── main.kv                 # UI layout (Kivy language)
├── buildozer.spec          # Android build configuration
├── requirements.txt        # Python dependencies
│
├── gesture_model.tflite    # TensorFlow Lite model (full)
├── gesture_model_quant.tflite # TensorFlow Lite model (quantized)
├── gesture_labels.txt      # Model label mapping
│
├── camera/                 # Camera handling module
│   └── camera_stream.py
├── detection/              # Detection logic
│   ├── hand_detector.py    # MediaPipe hand detection
│   └── gesture_logic.py    # Gesture classification
├── services/               # Services
│   ├── tts.py             # Text-to-speech
│   └── translator.py      # Translation (future)
└── utils/                  # Utilities
    └── constants.py        # App constants
```

---

## 🚀 Quick Start

### Desktop (Windows/Linux/macOS)

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd HSL
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the desktop app**

   ```bash
   # With MediaPipe (full accuracy)
   python main.py

   # Android-compatible version (TFLite)
   python android_app.py
   ```

---

## 📱 Building Android APK

### Prerequisites

- **Linux** or **Windows with WSL** (Ubuntu recommended)
- Python 3.8+
- Android SDK & NDK (auto-installed by Buildozer)

### Step-by-Step Guide

#### 1. Install WSL (Windows only)

```powershell
# Run in PowerShell as Administrator
wsl --install -d Ubuntu
```

#### 2. Set up build environment (in WSL/Linux)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3-pip python3-venv git zip unzip \
    openjdk-17-jdk autoconf libtool pkg-config \
    zlib1g-dev libncurses5-dev libncursesw5-dev libtinfo5 \
    cmake libffi-dev libssl-dev

# Install Buildozer and Cython
pip3 install --upgrade buildozer cython virtualenv
```

#### 3. Navigate to project directory

```bash
cd /mnt/g/collegeProject/HSL  # Adjust path as needed
```

#### 4. Build the APK

```bash
# Debug build (for testing)
buildozer android debug

# The APK will be in: bin/handsigndetection-1.0.0-arm64-v8a-debug.apk
```

#### 5. Install on Android device

```bash
# Connect your Android device and enable USB debugging
buildozer android deploy run logcat
```

---

## 🔧 Configuration

### Buildozer Settings (`buildozer.spec`)

| Setting         | Description                                    |
| --------------- | ---------------------------------------------- |
| `title`         | App name shown on device                       |
| `package.name`  | Unique identifier                              |
| `android.archs` | CPU architectures (`arm64-v8a`, `armeabi-v7a`) |
| `android.api`   | Target Android API level                       |
| `requirements`  | Python packages to include                     |

### Modifying Gestures

Edit `utils/constants.py` to change gesture mappings:

```python
GESTURE_MAP = {
    "Open": "Hello",    # Change the text output
    "Close": "Yes",
    "Pointer": "Look",
    "OK": "OK",
}
```

---

## 🧠 Model Training

To train a new gesture model:

1. **Collect data**

   ```bash
   python data_collector.py
   ```

2. **Download additional dataset**

   ```bash
   python download_dataset.py
   ```

3. **Train TFLite model**
   ```bash
   python train_tflite.py
   ```

This generates:

- `gesture_model.tflite` - Full precision model
- `gesture_model_quant.tflite` - Quantized (smaller) model
- `gesture_labels.txt` - Label mapping

---

## 🛠️ Troubleshooting

### Build Errors

| Error                      | Solution                                            |
| -------------------------- | --------------------------------------------------- |
| `SDK license not accepted` | Run with `android.accept_sdk_license = True`        |
| `NDK not found`            | Let Buildozer auto-install or set `ANDROIDSDK` path |
| `Recipe failed`            | Clean build: `buildozer android clean`              |

### Runtime Errors

| Error              | Solution                                      |
| ------------------ | --------------------------------------------- |
| Camera not working | Check `CAMERA` permission in Android settings |
| TTS not speaking   | Enable TTS engine in Android settings         |
| Model not loading  | Ensure `.tflite` files are in project root    |

---

## 📄 License

This project is for educational purposes (College Project).

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a pull request

---

## 📞 Support

For issues or questions, please open a GitHub issue.
