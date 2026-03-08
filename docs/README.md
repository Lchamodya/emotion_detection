# Webcam Emotion Detection – Smart Energy Systems Project
### Real-time Facial Expression Analysis | University of Vaasa – TECH1001

---

## What This Project Does

This project provides **real-time webcam-based facial emotion detection**:

### Webcam Facial Emotion Detection
Real-time facial emotion recognition using DeepFace and computer vision:

| Emotion   | Description |
|-----------|-------------|
| Happy     | 😊 Positive joy/happiness |
| Sad       | 😢 Low mood/sadness |
| Angry     | 😠 Frustration/anger |
| Fear      | 😨 Anxiety/fear |
| Surprise  | 😲 Unexpected reaction |
| Disgust   | 🤢 Repulsion/disgust |
| Neutral   | 😐 No strong emotion |

**Features:**
- Live webcam feed processing
- Face detection with bounding boxes
- Real-time emotion classification
- Visual score bars showing confidence for all 7 emotions
- Uses **DeepFace** with TensorFlow backend

---

## Connection to Smart Energy Systems

This project is relevant to the course in several ways:

- **Machine Learning Applications (Lecture 5):** This is a real-world
  application of a pre-trained transformer model — the same pipeline
  used for energy demand forecasting and anomaly detection.

- **Data-driven AI approach:** Just like load forecasting (collect data →
  define model → train → predict), emotion detection follows the same
  paradigm taught by Prof. Elmusrati.

- **Human-in-the-loop (Lecture 10):** Emotion detection can be used in
  smart grid customer service systems to route frustrated consumers to
  human operators automatically.
---

## Connection to Smart Energy Systems

This project is relevant to the course in several ways:

- **Machine Learning Applications (Lecture 5):** Real-world application of 
  deep learning for computer vision — similar techniques are used for 
  equipment monitoring and fault detection in smart grids.

- **Data-driven AI approach:** Face detection and emotion recognition follow 
  the same pipeline: collect data → train model → deploy → predict.

- **Human-in-the-loop (Lecture 10):** Emotion detection can be integrated into 
  smart grid customer service systems to:
  - Route frustrated customers to human operators automatically
  - Improve customer satisfaction during outages
  - Provide real-time feedback on service quality

- **Energy security & public trust:** Understanding public sentiment about grid 
  outages, renewable transitions, or pricing policies is increasingly important 
  for utilities. Non-verbal cues can be valuable data.

---

## Project Structure

```
emotion_detection/
├── src/
│   ├── __init__.py
│   └── webcam_emotion_detector.py    ← main application
├── docs/
│   └── README.md                     ← this file
├── models/
│   └── .gitkeep                      ← model weights stored here
├── tests/                            ← unit tests (future)
├── config/
│   └── environment.yml               ← conda environment specification
├── requirements.txt                  ← pip dependencies
├── setup.sh                          ← automated setup script
└── .gitignore
```

---

## Setup Instructions

### Quick Setup (Recommended)

If you have conda installed:
```bash
./setup.sh
conda activate emotion_webcam
python src/webcam_emotion_detector.py
```

### Manual Setup

#### Step 1 – Install Conda
This project requires a conda environment due to TensorFlow dependencies.
```bash
# Download Miniconda (if not installed)
# Visit: https://docs.conda.io/en/latest/miniconda.html
```

#### Step 2 – Create environment from config
```bash
conda env create -f config/environment.yml
conda activate emotion_webcam
```

#### Step 3 – Run the application
```bash
python src/webcam_emotion_detector.py
```

On **first run**, DeepFace will download the emotion recognition model 
(~6 MB) to `~/.deepface/weights/`. Subsequent runs are instant.

**Controls:**
- Press **'q'** to quit
- The window shows your webcam feed with:
  - Green bounding box around detected faces
  - Dominant emotion label
  - Score bars for all 7 emotions with percentages

---

## How It Works (Technical Overview)

```
Webcam feed (OpenCV)
    │
    ▼
Face detection
    │
    ▼
DeepFace emotion analysis
(TensorFlow backend)
(Pre-trained CNN model)
    │
    ▼
7-class emotion prediction
    │
    ▼
Visual overlay with scores
```

**DeepFace** is a hybrid face recognition framework that includes emotion 
detection capabilities. It uses a convolutional neural network trained on 
facial expression datasets to classify expressions into 7 categories.

The model analyzes facial features such as:
- Eye shape and openness
- Mouth position and curvature
- Eyebrow position
- Overall facial muscle tension

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `conda env create -f config/environment.yml` |
| Webcam not opening | Check camera permissions in System Preferences (macOS) |
| TensorFlow errors | Make sure you activated the conda environment |
| Model download fails | Check internet connection; model downloads on first run |
| Low FPS / laggy video | Normal on older hardware; model runs every 15 frames |
| Face not detected | Ensure good lighting and face clearly visible to camera |

---

## Requirements

- **Python:** 3.10 (recommended for TensorFlow compatibility)
- **Camera:** Built-in or external webcam
- **OS:** macOS, Linux, or Windows
- **RAM:** 4GB minimum, 8GB recommended
- **Storage:** ~500MB for packages and models

## Key Dependencies

- `deepface==0.0.99` - Facial analysis framework
- `opencv-python==4.13.0.74` - Computer vision and webcam capture
- `tensorflow==2.21.0` - Deep learning backend
- `tf-keras` - Keras API for TensorFlow

---

## Next Steps / Extensions

1. **Record emotion logs** — Save timestamped emotion data to CSV for analysis
2. **Multiple face detection** — Handle multiple people in frame simultaneously
3. **Emotion trends dashboard** — Build a real-time visualization with matplotlib/streamlit
4. **Integration with IoT** — Connect to smart home systems to adjust lighting/music based on mood
5. **Custom training** — Fine-tune the model on domain-specific expressions
6. **Mobile deployment** — Port to mobile app using TensorFlow Lite

---

## License

This project is for educational purposes as part of TECH1001 - Smart Energy Systems at University of Vaasa.

## Author

Isuru Pathirathna  
University of Vaasa  
March 2024

---

*Project for TECH1001 – Smart Energy Systems, University of Vaasa, 2026*
