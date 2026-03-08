#!/bin/bash
# Setup script for Emotion Detection project

echo "Setting up Emotion Detection environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed. Please install Miniconda or Anaconda first."
    exit 1
fi

# Create conda environment from config
echo "Creating conda environment from config/environment.yml..."
conda env create -f config/environment.yml

echo ""
echo "Setup complete! To activate the environment, run:"
echo "    conda activate emotion_webcam"
echo ""
echo "To run the webcam emotion detector:"
echo "    python src/webcam_emotion_detector.py"
