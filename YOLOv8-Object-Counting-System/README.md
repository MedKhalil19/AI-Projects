# 🤖 YOLOv8 Object Counting System

## 📌 Project Overview

This project implements an **intelligent object counting system** using **YOLOv8 deep learning technology**.

The main objective is to detect and count objects automatically in different input sources while maintaining high accuracy and efficient performance.

The system supports three different processing modes:

- 🎥 **Real-Time Object Counting** using a webcam.
- 🖼️ **Image Object Counting** from static images.
- 🎬 **Video Object Counting** by analyzing videos frame-by-frame.

This project demonstrates how modern computer vision techniques can be applied to real-world applications such as:

- Traffic monitoring
- Vehicle analysis
- Surveillance systems
- Industrial automation
- Smart vision applications

---

# ✨ Features

✅ YOLOv8-based object detection  
✅ Automatic object counting  
✅ Real-time webcam processing  
✅ Image-based analysis  
✅ Video processing and result generation  
✅ User-friendly graphical interface  
✅ Support for custom YOLOv8 models  
✅ CPU-based inference support  
✅ Optional GPU acceleration with CUDA

---

# 🧠 System Architecture

The project workflow:

```
              Input Source
        ┌────────┼────────┐
        ↓        ↓        ↓

     Webcam    Image    Video

        └────────┼────────┘
                 ↓
          ┌─────────────┐
          │   YOLOv8    │
          │ Detection   │
          └─────────────┘
                 ↓
          Object Tracking
                 ↓
          Object Counting
                 ↓
        Display / Save Results
```

---

# 📂 Project Components

The project is divided into four main modules:

---

## 1️⃣ Real-Time Object Counting

### `yolov8_real_time.py`

This module performs live object detection and counting using a webcam.

The YOLOv8 model analyzes each frame in real-time, detects objects, and updates the object count continuously.

Example applications:

- Live traffic monitoring
- People counting
- Smart surveillance

---

## 2️⃣ Image Object Counting

### `yolov8_img.py`

This module processes individual images.

The system detects all objects present in the image and displays the total count for each detected class.

Example:

```
Person: 5
Car: 3
Bicycle: 2
```

Useful for:

- Image analysis
- Dataset inspection
- Automated image processing

---

## 3️⃣ Video Object Counting

### `yolov8_vid.py`

This module processes video files frame-by-frame.

It detects and counts objects throughout the entire video while generating a processed output video.

Applications:

- Traffic analysis
- Security monitoring
- Video analytics

---

## 4️⃣ Integrated User Interface

### `main.py`

A graphical interface that combines all counting modes into one application.

The user can easily select:

- Real-Time Counting
- Image Counting
- Video Counting

without running individual scripts manually.

---

# 🤖 Why YOLOv8?

The project uses **YOLOv8 (You Only Look Once version 8)** because it provides an excellent balance between:

- Detection accuracy
- Processing speed
- Ease of deployment

Unlike traditional computer vision methods, YOLOv8 can detect multiple objects simultaneously and classify them in a single forward pass.

This makes it suitable for real-time object counting applications.

---

# 🛠️ Technologies Used

| Technology | Purpose |
|-|-|
| Python | Main programming language |
| YOLOv8 | Object detection model |
| Ultralytics | YOLO framework |
| OpenCV | Image and video processing |
| NumPy | Numerical operations |
| PyTorch | Deep learning backend |
| Tkinter | User interface |

---

# ⚙️ Installation and Setup

## 1. Install Python

Recommended version:

```
Python 3.7.9
```

(Other compatible Python versions may also work depending on installed libraries.)

---

## 2. Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

---

## 3. Optional GPU Acceleration

The project can run on CPU.

For users with an NVIDIA GPU, PyTorch can be installed with CUDA support for faster inference:

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```

Modify the CUDA version according to your GPU configuration.

---

# ▶️ Usage

## Run the Graphical Interface

```bash
python main.py
```

Then select:

```
1. Real-Time Counting
2. Image Counting
3. Video Counting
```

---

## Run Individual Modules

### Webcam:

```bash
python yolov8_real_time.py
```

### Image:

```bash
python yolov8_img.py
```

### Video:

```bash
python yolov8_vid.py
```

---

# 📊 Results

The system provides:

- Detected object labels
- Bounding boxes
- Object quantities
- Processed images/videos

Example output:

```
Person: 12
Car: 8
Truck: 2
```

---

# 🎬 Demo

A complete demonstration video showing the project execution is available on LinkedIn:

🔗 **Project Demo:**  
[objects_counting's LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7261845336736808963/)

---

# 🚀 Possible Improvements

Future improvements:

- Add object tracking with ByteTrack.
- Improve counting accuracy in crowded scenes.
- Deploy on embedded AI platforms.
- Add automatic data logging.
- Integrate custom-trained YOLO models.
- Add cloud-based monitoring.

---

# 👨‍💻 Author

**Khlifi Med Khalil**

Mechatronics Engineering Student

## Contact Information For more details,
feel free to contact me:
- **Email:** khlifimedkhalil@gmail.com -
- **LinkedIn:** [Khlifi Med Khalil](https://www.linkedin.com/in/khlifi-medkhalil/)
  
---

# ⭐ Project Highlights

This project combines:

- Artificial Intelligence
- Deep Learning
- Computer Vision
- Object Detection
- Automated Counting

to create a practical intelligent vision system.
