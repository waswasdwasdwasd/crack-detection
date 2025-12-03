#Crack Detection System
## Overview
Automated crack detection using deep learning. Achieved **99.81% accuracy** on 40,000 images.

## Results
- ✅ Accuracy: 99.81%
- ✅ Precision: 99.65%
- ✅ Recall: 99.98%
- ✅ F1-Score: 99.81%
- ✅ Training Time: 18 minutes (5 epochs)

## How to Use

### Installation
pip install torch torchvision opencv-python scikit-learn grad-cam matplotlib seaborn pillow tqdm

### Run the Code
python crack_detection.py

## What's Included
- `crack_detection.py` - Main program
- `best_crack_model.pth` - Trained model
- Research report (PDF)

## Technical Details
- **Model:** ResNet18 (transfer learning)
- **Dataset:** 40,000 images (20k crack, 20k no-crack)
- **Framework:** PyTorch
- **GPU:** NVIDIA CUDA

## Project Files
crack-detection-project/
├── crack_detection.py
├── requirements.txt # Package list
├── best_crack_model.pth # Trained model
├── README.md # This file
└── docs/
└── research_report.pdf # Full report

## Download Trained Model

The trained model is available in the releases:

📥 **[Download best_crack_model.pth (43 MB)](https://github.com/waswasdwasdwasd/crack-detection/releases/download/v1.0/best_crack_model.pth)**

Place the downloaded file in the same folder as `crack_detection.py` before running.

