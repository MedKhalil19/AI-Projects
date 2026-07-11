# 📷 ESP32-CAM Object Detection Using YOLO

## 📌 Project Overview

This project demonstrates real-time object detection using an **ESP32-CAM module** as a wireless camera and a computer vision model running on a PC.

The ESP32-CAM captures images and sends them through an HTTP connection over Wi-Fi. The received images are processed on the computer using deep learning-based object detection algorithms.

Two different implementation approaches are provided:

1. **YOLOv3 with OpenCV DNN**
   - Direct integration with YOLOv3 Darknet model files.
   - Requires manual download of YOLO configuration and weights files.

2. **cvlib Object Detection**
   - Simplified implementation using the cvlib library.
   - Does not require manual YOLOv3 model file management because the required model files are automatically handled by the library.

---

# 🚀 Implementations

## 1️⃣ YOLOv3 Object Detection using OpenCV DNN

### 📌 Description

This implementation uses the official **YOLOv3 Darknet model** with OpenCV's Deep Neural Network (DNN) module.

The ESP32-CAM sends JPEG images through an HTTP request. Each image is decoded and passed to the YOLOv3 neural network for object detection.

The model detects objects from the COCO dataset and displays:

- Object name
- Confidence percentage
- Bounding boxes

---

## 📂 Required YOLOv3 Files

Unlike the cvlib implementation, this method requires manually downloading the YOLOv3 model files:

- `yolov3.cfg` → Network architecture configuration
- `yolov3.weights` → Pre-trained neural network weights
- `coco.names` → Object class names

These files must be placed in the project directory.

---

## ⬇️ Download YOLOv3 Files

The official YOLOv3 Darknet files can be downloaded from the original YOLO website:

🔗 https://pjreddie.com/darknet/yolo/
yolov3.cfg
yolov3.weights
coco.names


---

## ⚙️ How It Works

1. ESP32-CAM captures an image.
2. The image is transferred through Wi-Fi using HTTP.
3. OpenCV decodes the received image.
4. The image is converted into a YOLO input blob.
5. YOLOv3 performs object detection.
6. Non-Maximum Suppression (NMS) removes duplicated detections.
7. Bounding boxes and labels are displayed.

---

## 📦 Main Libraries


OpenCV
NumPy
urllib
YOLOv3 Darknet

---

# 2️⃣ Object Detection using cvlib

## 📌 Description

This implementation provides a simpler way to perform object detection using the **cvlib** library.

Unlike the OpenCV DNN implementation, the YOLOv3 model files do not need to be downloaded manually.

The reason is that cvlib internally manages the required YOLOv3 model files and automatically downloads them when the detection function is executed for the first time.

---

## ⚙️ How It Works
1.ESP32-CAM captures images.
2.Images are transmitted through Wi-Fi.
3.Python receives and decodes the images.
4.cvlib processes the image using its integrated object detection model.
5.Detected objects are displayed with:
6.Bounding boxes
7.Object labels
8.Confidence scores

---

## 🔧 Hardware Requirements
Hardware Components
ESP32-CAM module
USB-to-Serial programmer (for flashing)
Wi-Fi network
Computer for AI processing

---

## 💻 Software Requirements
Python 3.x
OpenCV
NumPy
urllib
cvlib (for cvlib version)

---

## 🎯 Possible Improvements
Replace YOLOv3 with YOLOv8/YOLO11 for better accuracy and speed.
Use ESP32-CAM MJPEG streaming instead of single image requests.
Add object tracking.
Deploy lightweight models for edge AI.
Train custom YOLO models for specific objects.

---

## 👨‍💻 Author
Med Khalil Khlifi

Required files:

