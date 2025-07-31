#!/bin/bash

# Install dependencies for ImageReader
echo "Installing dependencies for ImageReader..."

# Activate virtual environment
source csfp-venv/bin/activate

# Install required packages
pip install paddlepaddle
pip install opencv-python
pip install numpy
pip install paddleocr

echo "Dependencies installed successfully!"
echo "You can now run: python img_read_engine.py" 