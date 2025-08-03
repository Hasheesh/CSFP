#!/bin/bash

# Install dependencies for CSFP (Complete Speech Processing Framework)
echo "Installing dependencies for CSFP..."

# Activate virtual environment
source csfp-venv/bin/activate

# Install required packages for audio processing
pip install soundfile
pip install librosa
pip install numpy
pip install torch
pip install transformers
pip install scipy
pip install sounddevice

# Install packages for image processing
pip install paddlepaddle
pip install opencv-python
pip install paddleocr

# Install packages for TTS
pip install piper-tts

echo "Dependencies installed successfully!"
echo "You can now run: python main.py" 