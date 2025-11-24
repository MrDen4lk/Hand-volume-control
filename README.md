# Hand Volume Control

![C++](https://img.shields.io/badge/C++-20-blue.svg?style=flat&logo=c%2B%2B)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=flat&logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg?style=flat&logo=opencv)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-Inference-orange.svg?style=flat&logo=onnx)
![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux-lightgrey)

> **Real-time system volume control powered by Computer Vision.**
> Built with C++ for high performance and low latency.

## 📖 Overview

This project implements a contactless volume controller using hand gestures. It captures video from a webcam, detects hand keypoints using a deep learning model, and maps the distance between the **Thumb** and **Index** finger to the system volume level.

The core logic is implemented in **C++** to ensure maximum performance and efficient resource usage, making it suitable for running in the background.

## 🚀 Key Features

* **High Performance:** ~40 FPS on CPU M3 pro using ONNX Runtime.
* **State-of-the-Art Model:** Uses **YOLOv8-Small Pose** for robust keypoint detection.
* **Architecture Pattern:** Implements **Strategy Pattern** to easily switch between different deep learing models.
* **Smooth Control:** Exponential Moving Average (EMA) filtering to prevent volume jitter.

## 🔬 Research & Approach

Before the final C++ implementation, I conducted experiments with different architectures to find the best balance between speed and accuracy.

| Model Architecture | Framework | Input Size | FPS (CPU) | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Custom U-Net (ConvNext-base)** | ONNX Runtime | 224x224 | ~7 | Good for educational purposes (Heatmaps), but slow for real-time. |
| **YOLOv8s-Pose** | ONNX Runtime | 224x224 | **~40** | **Selected.** Best trade-off between speed and accuracy. |

*The custom U-Net you can try changing model in main.cpp

## 🛠️ Tech Stack

* **Language:** C++20
* **Deep Learning:** PyTorch
* **Computer Vision:** OpenCV
* **ML Inference:** ONNX Runtime (C++ API)
* **Build System:** CMake
* **System Integration:** AppleScript (`osascript`) for macOS volume control.
