# 🚗 Automatic Tunisian License Plate Recognition System (ALPR)

## 📌 Overview

This project implements an **Automatic License Plate Recognition (ALPR)** system based on Deep Learning and Computer Vision techniques.

The system is designed specifically for **Tunisian vehicle license plates**, combining object detection, image processing, and Optical Character Recognition (OCR) to automatically detect and recognize vehicle registration numbers.

The project can process license plates in two different ways:

- 🎥 **Real-Time Recognition** using a webcam.
- 🖼️ **Image Recognition** by processing a saved vehicle image.

The complete pipeline consists of three main stages:

1. License plate detection.
2. Character segmentation.
3. Character recognition.

---

# ✨ Features

✅ Real-time Tunisian license plate detection  
✅ License plate extraction from vehicle images  
✅ Automatic character segmentation  
✅ Deep Learning based OCR recognition  
✅ Supports webcam and image processing modes  
✅ CPU inference support (no GPU required)  
✅ Detection confidence filtering using Non-Maximum Suppression (NMS)

---

# 🧠 System Architecture

The recognition pipeline is divided into three main modules:

```
Input Image / Webcam
          |
          ↓
┌──────────────────────┐
│ YOLOv3 License Plate │
│      Detection       │
└──────────────────────┘
          |
          ↓
License Plate Extraction
          |
          ↓
┌──────────────────────┐
│ Character Segmentation│
│ Histogram Projection │
└──────────────────────┘
          |
          ↓
┌──────────────────────┐
│ CNN OCR Recognition  │
│    ocrmodel.h5       │
└──────────────────────┘
          |
          ↓
Recognized Plate Number
```

---

# 🔍 License Plate Detection

The first stage uses **YOLOv3 Darknet** to locate the vehicle license plate.

The model uses:

- `darknet-yolov3.cfg` → YOLO network architecture
- `lapi.weights` → trained weights
- `classes.names` → detection classes

The detector is executed using OpenCV DNN module:

```python
cv2.dnn.readNetFromDarknet()
```

The model runs on CPU using:

```python
cv2.dnn.DNN_TARGET_CPU
```

No GPU acceleration is required.

---

# 🇹🇳 Tunisian License Plate Recognition

The system is adapted for the Tunisian license plate format.

A Tunisian plate generally follows this structure:

```
XXX تونس XXXX
```

or:

```
XXX Tunisia XXXX
```

Where:

- First part → 3 numerical digits
- Middle part → "Tunisie / Tunisia"
- Last part → 4 numerical digits

Example:

```
123 تونس 4567
```

The OCR model is trained to recognize:

```
0 1 2 3 4 5 6 7 8 9 T
```

where:

```
T = Tunisie
```

---

# 🔤 Character Segmentation

After detecting the plate, the system extracts individual characters using:

**Histogram of Pixel Projection**

The algorithm:

1. Converts the plate image to grayscale.
2. Applies binary thresholding.
3. Calculates horizontal and vertical projections.
4. Detects character boundaries.
5. Extracts individual character images.

Each extracted character is resized to:

```
28 × 28 pixels
```

before being sent to the OCR model.

---

# 🤖 OCR Recognition Model

The character recognition stage uses a trained CNN model:

```
ocrmodel.h5
```

The model receives segmented characters and predicts the corresponding class.

Supported classes:

```
0-9 + Tunisia
```

The output is reconstructed to generate the final license plate number.

---

# 🛠️ Technologies Used

| Technology | Purpose |
|-|-|
| Python | Main programming language |
| OpenCV | Image processing and camera handling |
| YOLOv3 | License plate detection |
| Darknet | YOLO model format |
| TensorFlow / Keras | OCR deep learning model |
| NumPy | Numerical operations |

---

# 📂 Project Structure

```
Vehicle-Recognition-System/
│
├── Licence_plate_detection/
│   ├── darknet-yolov3.cfg
│   ├── lapi.weights
│   └── classes.names
│
├── Licence_Plate_Recognition/
│   └── ocrmodel.h5
│
├── realtime_recognition.py
├── image_recognition.py
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Vehicle-Recognition-System.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Required libraries:

```
opencv-python
numpy
tensorflow
keras
```

---

# ▶️ Usage

## 🎥 Real-Time Recognition

Run:

```bash
python realtime_recognition.py
```

The webcam will open and detected license plates will be recognized automatically.

Press:

```
q
```

to exit.

---

## 🖼️ Image Recognition

Run:

```bash
python image_recognition.py
```

Select an image containing a vehicle.

The system will:

1. Detect the license plate.
2. Extract characters.
3. Recognize the plate number.
4. Display the result.

---

# 📌 Model Files

The YOLOv3 model files are required:

```
darknet-yolov3.cfg
lapi.weights
classes.names
```

The OCR model:

```
ocrmodel.h5
```

must also be included.

Because GitHub has a file size limitation, large model files may need to be downloaded separately or stored using Git LFS.

---

# 🚀 Possible Improvements

Future improvements:

- Replace YOLOv3 with YOLOv8 for better accuracy.
- Improve OCR accuracy with a larger Tunisian plate dataset.
- Add automatic plate format validation.
- Add vehicle tracking for traffic monitoring.
- Deploy on embedded platforms (Raspberry Pi / Jetson Nano).

---

# 👨‍💻 Author

**Khlifi Med Khalil**

Mechatronics Engineering Student

---

# ⭐ Acknowledgment

This project demonstrates the integration of:

- Deep Learning
- Computer Vision
- Object Detection
- OCR
- Image Processing

for real-world vehicle identification applications.
