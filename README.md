# Face Emotion Detector

Real-time facial emotion recognition using a CNN trained from scratch on FER2013, with live webcam inference via MediaPipe and OpenCV.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green) ![MediaPipe](https://img.shields.io/badge/MediaPipe-0.9.3-orange)

---

## Overview

This project builds a real-time emotion classifier entirely from scratch — no pretrained backbone, no API. A 4-block CNN is trained on the FER2013 dataset and deployed on live webcam input, where MediaPipe handles face detection and the CNN classifies the cropped face region into one of 7 emotion categories.

**Recognizes:** Angry · Disgusted · Fearful · Happy · Neutral · Sad · Surprised

---

## Architecture

```
Webcam frame
    │
    ▼
MediaPipe Face Detector  ──→  bounding box coordinates
    │
    ▼
Crop + Resize (48×48 grayscale)
    │
    ▼
Normalize (mean=0.5, std=0.5)
    │
    ▼
4-Block CNN
    ├── Block 1: Conv→BN→ReLU→Conv→BN→ReLU→MaxPool→Dropout2d
    ├── Block 2: Conv→BN→ReLU→Conv→BN→ReLU→MaxPool→Dropout2d
    ├── Block 3: Conv→BN→ReLU→Conv→BN→ReLU→MaxPool→Dropout2d
    └── Classifier: Flatten→Linear→ReLU→Dropout→Linear→ReLU→Dropout→Linear
    │
    ▼
Softmax probabilities (7 classes)
    │
    ▼
10-frame prediction smoothing + confidence threshold
    │
    ▼
Emotion label overlaid on frame
```

---

## Results

| Metric | Value |
|---|---|
| Dataset | FER2013 (28,709 train / 7,178 test) |
| Validation accuracy | ~65% |
| Human-level accuracy on FER2013 | ~65% |
| Training epochs | 60 |
| Parameters | ~2.1M |

65% is on par with human agreement on FER2013 — the dataset itself is noisy and ambiguously labeled, making it a hard ceiling for from-scratch training.

---

## Key Implementation Details

**Class imbalance handling** — FER2013 is heavily skewed (Happy: ~8000 samples, Disgusted: ~500). Class-weighted CrossEntropyLoss ensures minority emotions aren't ignored during training.

**Cosine annealing scheduler** — smoothly decays learning rate over training rather than keeping it flat, typically adding 2–3% accuracy over a fixed LR.

**Prediction smoothing** — a 10-frame rolling buffer averages softmax probabilities across recent frames, eliminating the per-frame flicker common in live inference systems.

**Confidence thresholding** — predictions below 50% confidence are displayed as "Uncertain" rather than forcing a low-confidence label, reducing visible false positives.

**Dropout2d in conv blocks** — drops entire feature map channels (not individual neurons) during training, which is the correct regularization form for spatial data.

---

## Project Structure

```
face-emotion-detector/
├── train.py               # training loop, dataset loading, class weights
├── model.py               # EmotionCNN architecture
├── inference.py           # real-time webcam inference
├── transforms.py          # train and val transform pipelines
├── best_emotion_model.pth # saved model weights (download separately)
├── requirements.txt
└── README.md
```


> **Note:** MediaPipe must be pinned to `0.9.3`. Version 0.10+ removed the `solutions` API used in this project.

---

## Known Limitations

- **FER2013 quality** — the dataset contains mislabeled and low-resolution images scraped from the internet. Real-world webcam performance can vary from validation accuracy.
- **Single-face buffer** — the prediction smoothing buffer is shared across detections. In multi-face scenes, predictions from different faces mix into one buffer.
- **Subtle emotions** — Disgusted and Fearful have far fewer training samples and are harder to classify reliably.

---

## What's Next

- Fine-tune a pretrained ResNet18 on FER2013 for improved real-world accuracy
- Per-face prediction buffers for multi-face support
- Gradio web interface for browser-based demo

---

## License

MIT
