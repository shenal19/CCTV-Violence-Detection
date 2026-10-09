# CCTV Violence Detection with Temporal Confidence Gating and Automated Alerts

## Overview

This project explores automated violence detection in CCTV-style video streams using deep learning. It classifies video clips as violent or non-violent and incorporates temporal confidence filtering to reduce false alarms. When a sustained violence event is detected, the system can generate an email alert containing a timestamp and snapshot.

The detection pipeline uses an R(2+1)D-18 spatiotemporal convolutional neural network, pretrained on Kinetics-400 and fine-tuned on the RWF-2000 surveillance dataset.

## Key Features

- **Video-based violence detection:** Classifies CCTV video clips and webcam streams.
- **Spatiotemporal deep learning:** Uses R(2+1)D-18 to learn spatial and temporal patterns.
- **Temporal confidence filtering:** Smooths prediction scores and requires sustained confidence before confirming an event.
- **Input-quality screening:** Filters unsuitable inputs, such as blank, blocked, or static-camera footage.
- **Automated alerting:** Supports email notifications with event timestamps and snapshots.
- **Modular configuration:** Keeps model settings, detection thresholds, and alert options configurable.

## System Architecture

The processing pipeline follows these stages:

1. Read video frames from a clip or webcam.
2. Prepare frames using resizing, color conversion, and normalization.
3. Check input quality before inference.
4. Generate violence probability scores using the R(2+1)D-18 model.
5. Apply temporal confidence smoothing and event confirmation.
6. Display detection results and trigger configured alerts when appropriate.

## Technology Stack

- Python
- PyTorch and Torchvision
- OpenCV
- NumPy
- Deep learning and video processing
- Email integration for event notifications

## Dataset

The primary dataset is **RWF-2000**, a surveillance-video dataset containing violent and non-violent clips.

Hockey Fights and Real Life Violence Situations (RLVS) are also referenced for cross-dataset evaluation. Dataset access, licensing, and usage conditions should be checked before downloading or redistributing data.

## Reported Results

The original project README reports the following results on its held-out RWF-2000 test set:

| Metric | Reported value |
|---|---:|
| Accuracy | 83.8% |
| Precision (violence) | 83.3% |
| Recall (violence) | 84.5% |
| F1-score (violence) | 83.9% |
| AUC | 92.7% |
| False-positive rate | 17.0% |

These figures should be reproduced and verified against the actual evaluation code and model checkpoints before being presented as results of another implementation.

## Repository Structure

```text
CCTV_Violence_Detection/
├── configs/
├── notebooks/
├── scripts/
├── src/
├── README.md
├── requirements.txt
└── .gitignore
```

The source repository describes separate modules for model construction, dataset loading, environment checks, temporal gating, training, evaluation, inference, and alert management. Confirm the actual files in your copy before documenting them as implemented.

## Setup

1. Install a compatible Python version and create a virtual environment.
2. Install a PyTorch build compatible with your hardware.
3. Install the remaining dependencies listed in `requirements.txt`.
4. Download the permitted datasets separately.
5. Configure environment variables for email alerts, if required.
6. Run the environment checks before training or inference.

Refer to the actual scripts and configuration files in your repository for the precise commands.

## Responsible Use

This system is a research proof of concept. Violence predictions can be incorrect, and automated alerts should not be treated as definitive evidence of an incident. A human reviewer should verify detected events before any consequential action is taken.

## Attribution

This README is adapted from the project structure and description of [TAB-0705/CCTV_Violence_Detection](https://github.com/TAB-0705/CCTV_Violence_Detection). Acknowledge the original implementation and clearly identify any modifications, experiments, or additional work performed in this repository.
