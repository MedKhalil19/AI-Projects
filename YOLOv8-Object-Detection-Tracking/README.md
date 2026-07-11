# 🎯 YOLOv8 Interactive Object Tracking with Spotlight Visualization

## 📌 Overview

This project implements an **interactive object tracking system** using **YOLOv8 and ByteTrack**.

The system combines object detection, multi-object tracking, and user interaction to create a smart vision application where the user can select any detected object and automatically follow it throughout a video.

After selecting an object by clicking on it, the system creates a **real-time spotlight window** by displaying a cropped view of the selected target while keeping the complete scene visible.

This approach can be useful for:

- Smart surveillance systems
- Robotics vision applications
- Traffic monitoring
- Human-object interaction
- Intelligent camera systems

---

# ✨ Features

✅ YOLOv8 object detection  
✅ ByteTrack multi-object tracking  
✅ Click-to-select object tracking  
✅ Persistent object IDs  
✅ Live spotlight crop visualization  
✅ Video processing and output generation  
✅ Supports custom YOLOv8 models  
✅ Supports video files and webcam input  
✅ CPU compatible

---

# 🧠 How It Works

The system follows this pipeline:

```
              Input Source
           (Video / Webcam)
                    |
                    ↓
          ┌─────────────────┐
          │     YOLOv8      │
          │ Object Detection│
          └─────────────────┘
                    |
                    ↓
          ┌─────────────────┐
          │    ByteTrack    │
          │ Object Tracking │
          └─────────────────┘
                    |
                    ↓
             Assign IDs
                    |
                    ↓
          User selects object
              by clicking
                    |
                    ↓
          Crop selected object
                    |
                    ↓
          Spotlight Overlay
                    |
                    ↓
              Output Video
```

---

# 🔍 Object Detection

The detection stage uses the Ultralytics YOLOv8 model.

Default model:

```
yolov8n.pt
```

YOLOv8 provides:

- Object class
- Bounding box coordinates
- Confidence score

The model can detect multiple objects simultaneously.

Example:

```
Person ID: 1
Car ID: 2
Dog ID: 3
```

---

# 🚀 Object Tracking with ByteTrack

After detection, objects are tracked using:

```
bytetrack.yaml
```

ByteTrack allows the system to maintain the same identity of an object between frames.

Example:

Frame 1:

```
Person ID: 5
```

Frame 100:

```
Person ID: 5
```

The object keeps the same tracking ID even while moving.

---

# 🎯 Interactive Spotlight Feature

The main feature of this project is interactive object selection.

Workflow:

1. Run the application.
2. Multiple objects are detected and tracked.
3. Click on the desired object.
4. The system saves its tracking ID.
5. The object is continuously followed.
6. A zoomed crop appears in the corner of the video.

Example:

```
+--------------------------------+
|  Spotlight                     |
|  +-------------+               |
|  | Selected    |               |
|  | Object Crop |               |
|  +-------------+               |
|                                |
|      Full Video Scene          |
|                                |
+--------------------------------+
```

---

# 🎥 Input Modes

## 1️⃣ Video File Mode

By default, the system processes a video file:

```python
source="video.mp4"
```

Example:

```python
tracker = YOLOSpotlight(
    model="yolov8n.pt",
    source="video.mp4",
    output="YOLO_spotlight.mp4"
)
```

The processed result is saved as:

```
YOLO_spotlight.mp4
```

---

## 2️⃣ Real-Time Webcam Mode

The same system can work with a live camera.

Replace:

```python
source="video.mp4"
```

with:

```python
source=0
```

Example:

```python
tracker = YOLOSpotlight(
    model="yolov8n.pt",
    source=0,
    output="webcam_result.mp4"
)
```

Explanation:

```
0 → Default webcam
1 → Second camera
2 → Third camera
```

This allows real-time object detection and tracking directly from a camera.

---

# 🎬 Results

Example output:

```
Input:
video.mp4

Processing:
YOLOv8 Detection
+
ByteTrack Tracking
+
Interactive Spotlight

Output:
YOLO_spotlight.mp4
```

## Demo Result

(Add your result video here)

Example:

```
![YOLOv8 Spotlight Demo](results/demo.gif)
```

or upload your video:

```
results/YOLO_spotlight.mp4
```

---

# 🛠️ Technologies Used

| Technology | Usage |
|-|-|
| Python | Main programming language |
| OpenCV | Video processing and visualization |
| YOLOv8 | Object detection |
| Ultralytics | Deep learning framework |
| ByteTrack | Multi-object tracking |
| NumPy | Data processing |

---

# 📂 Project Structure

```
YOLOv8-Spotlight-Tracker/

│
├── main.py
├── yolov8n.pt
├── bytetrack.yaml
│
├── videos/
│   └── video.mp4
│
├── results/
│   └── YOLO_spotlight.mp4
│
└── README.md
```

---

# ⚙️ Installation

Clone repository:

```bash
git clone https://github.com/yourusername/YOLOv8-Spotlight-Tracker.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Required packages:

```
opencv-python
numpy
ultralytics
```

---

# ▶️ Run The Project

Execute:

```bash
python main.py
```

Controls:

| Action | Function |
|-|-|
| Left Mouse Click | Select object |
| C | Clear selection |
| Q | Exit |

---

# 🔧 Customization

You can change the YOLO model:

```python
model="yolov8n.pt"
```

Examples:

```
yolov8n.pt  → Faster, lightweight
yolov8s.pt  → Better accuracy
yolov8m.pt  → Balanced
yolov8l.pt  → High accuracy
yolov8x.pt  → Maximum accuracy
```

---

# 🚀 Future Improvements

Possible improvements:

- Real-time webcam optimization
- Multiple object spotlight selection
- Object counting integration
- Automatic camera following
- Object segmentation using YOLOv8-seg
- Deployment on embedded AI platforms

---

# 👨‍💻 Author

**Khlifi Med Khalil**

Mechatronics Engineering Student

---

# ⭐ Project Highlights

This project demonstrates the integration of:

- Artificial Intelligence
- Computer Vision
- Object Detection
- Object Tracking
- Human-Computer Interaction

to build an interactive intelligent vision system.
