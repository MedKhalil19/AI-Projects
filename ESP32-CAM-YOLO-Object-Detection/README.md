# 📷 ESP32-CAM Object Detection Using YOLO

## 📌 Project Overview

This project demonstrates a wireless AI vision system using an **ESP32-CAM module** as a remote camera and a computer running deep learning-based object detection algorithms.

The ESP32-CAM captures images and transfers them through a Wi-Fi HTTP connection to a computer. The received images are then processed using YOLO-based object detection models running with Python.

> ⚠️ The ESP32-CAM is used only as an image acquisition device. The AI inference is performed on the computer.

The system provides two different object detection implementations:

1. **YOLOv3 with OpenCV DNN**
   - Direct integration with the official YOLOv3 Darknet model.
   - Requires manual download of YOLO configuration and weight files.

2. **cvlib Object Detection**
   - Simplified implementation using the cvlib library.
   - Does not require manual YOLO model file management because cvlib automatically handles the required files.

---

# 🌐 ESP32-CAM Network Configuration

## 📌 Important Setup Step

Before running the Python detection programs, the ESP32-CAM must be connected to the same Wi-Fi network as the computer.

The communication architecture is:
