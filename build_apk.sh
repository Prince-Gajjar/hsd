#!/bin/bash
# ==============================================================================
# BUILD ANDROID APK - Run this in WSL or Linux
# ==============================================================================
# 
# PREREQUISITES:
# 1. On Windows, open WSL terminal first
# 2. Navigate to project: cd /mnt/g/collegeProject/HSL
# 3. Run: bash build_apk.sh
#
# ==============================================================================

echo "=========================================="
echo "  HAND SIGN DETECTION - APK BUILDER"
echo "=========================================="
echo ""

# Check if running in WSL/Linux
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "ERROR: This script must run in WSL or Linux!"
    echo "Open WSL terminal and navigate to:"
    echo "  cd /mnt/g/collegeProject/HSL"
    echo "Then run: bash build_apk.sh"
    exit 1
fi

# Update system
echo "[1/5] Updating system packages..."
sudo apt update -qq

# Install dependencies
echo "[2/5] Installing build dependencies..."
sudo apt install -y -qq \
    python3 python3-pip python3-venv \
    git zip unzip openjdk-17-jdk \
    autoconf libtool pkg-config \
    zlib1g-dev libncurses5-dev \
    libffi-dev libssl-dev

# Install Buildozer
echo "[3/5] Installing Buildozer and Cython..."
pip3 install --upgrade --quiet buildozer cython

# Clean previous builds
echo "[4/5] Cleaning previous builds..."
rm -rf .buildozer/android/platform/build-*
rm -rf bin/

# Build APK
echo "[5/5] Building APK (this takes 10-30 minutes first time)..."
echo ""
buildozer android debug

# Check result
if [ -f bin/*.apk ]; then
    echo ""
    echo "=========================================="
    echo "  SUCCESS! APK created in bin/ folder"
    echo "=========================================="
    ls -la bin/*.apk
    echo ""
    echo "To install on phone:"
    echo "  1. Copy APK to phone"
    echo "  2. Enable 'Unknown sources' in settings"
    echo "  3. Tap APK to install"
else
    echo ""
    echo "=========================================="
    echo "  BUILD FAILED - Check errors above"
    echo "=========================================="
fi
