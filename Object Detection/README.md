# 🎯 YOLOv8 Interactive Object Tracking with Spotlight View

## 📌 Overview

This project implements an **interactive object tracking system** using **YOLOv8 and ByteTrack**.

Unlike traditional object detection systems that only identify objects in each frame, this project allows the user to **select any detected object by clicking on it** and continuously track it throughout the video.

After selecting an object, the system creates a real-time **spotlight view** by displaying a cropped close-up of the tracked object while maintaining the complete scene visibility.

---

# ✨ Features

✅ Real-time object detection using YOLOv8  
✅ Multi-object tracking with ByteTrack  
✅ Click-to-select object interaction  
✅ Persistent object IDs across frames  
✅ Live cropped spotlight overlay  
✅ Supports video input  
✅ Automatic output video saving  
✅ CPU compatible  
✅ Works with different YOLOv8 models

---

# 🧠 System Architecture

The processing pipeline:

```
             Input Video
                  |
                  ↓
        ┌──────────────────┐
        │     YOLOv8       │
        │ Object Detection │
        └──────────────────┘
                  |
                  ↓
        ┌──────────────────┐
        │    ByteTrack     │
        │ Object Tracking  │
        └──────────────────┘
                  |
                  ↓
          Assign Object IDs
                  |
                  ↓
        User Click Selection
                  |
                  ↓
        Crop Selected Object
                  |
                  ↓
        Spotlight Overlay
                  |
                  ↓
          Output Video
```

---

# 🔍 How It Works

## 1. Object Detection

The system uses the Ultralytics YOLOv8 model:

```
yolov8n.pt
```

YOLO detects objects in every frame and provides:

- Bounding boxes
- Object classes
- Confidence scores

Example detected objects:

```
Person
Car
Bus
Dog
Bicycle
...
```

The model can be replaced with other YOLOv8 versions:

```
yolov8n.pt
yolov8s.pt
yolov8m.pt
yolov8l.pt
yolov8x.pt
```

depending on the required speed and accuracy.

---

# 🎯 Interactive Object Selection

The user can select an object directly from the video window:

1. Run the program.
2. Click on any detected object.
3. The system stores the object's tracking ID.
4. The selected object remains highlighted.

Example:

```
Person ID: 3
Car ID: 7
Dog ID: 12
```

The selected object is tracked even when it moves through the scene.

---

# 🚀 Object Tracking

Tracking is performed using:

```
ByteTrack
```

ByteTrack provides:

- Stable object identities.
- Reduced ID switching.
- Efficient real-time tracking.

The tracker configuration:

```
bytetrack.yaml
```

is included with Ultralytics YOLO.

---

# 🔎 Spotlight Feature

After selecting an object, the system:

1. Extracts the object's bounding box.
2. Creates a cropped image.
3. Resizes it while maintaining aspect ratio.
4. Displays it in the top corner of the video.

The original scene remains visible while providing a zoomed view of the selected target.

---

# 🎥 Controls

| Key / Action | Function |
|-|-|
| Mouse Left Click | Select an object |
| `c` | Clear current selection |
| `q` | Exit program |

---

# 🛠️ Technologies Used

| Technology | Purpose |
|-|-|
| Python | Main programming language |
| OpenCV | Video processing and visualization |
| YOLOv8 | Object detection |
| Ultralytics | Deep learning framework |
| ByteTrack | Object tracking |
| NumPy | Numerical operations |

---

# 📂 Project Structure

```
YOLOv8-Spotlight-Tracker/
│
├── main.py
├── yolov8n.pt
├── bytetrack.yaml
├── video.mp4
├── YOLO_spotlight.mp4
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/YOLOv8-Spotlight-Tracker.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Required libraries:

```
opencv-python
numpy
ultralytics
```

---

# ▶️ Usage

Run:

```bash
python main.py
```

The program starts processing:

```
video.mp4
```

and generates:

```
YOLO_spotlight.mp4
```

with:

- Detection boxes
- Tracking IDs
- Selected object spotlight

---

# 🔧 Configuration

The tracker can be customized:

```python
tracker = YOLOSpotlight(
    model="yolov8n.pt",
    source="video.mp4",
    output="YOLO_spotlight.mp4"
)
```

Parameters:

| Parameter | Description |
|-|-|
| model | YOLOv8 model file |
| source | Input video |
| output | Saved output video |
| conf | Detection confidence threshold |
| tracker | Tracking algorithm |

---

# 📈 Possible Improvements

Future improvements:

- Add webcam real-time tracking.
- Add multiple selected objects.
- Add automatic object following with camera movement.
- Add object counting and analytics.
- Integrate segmentation models (YOLOv8-seg).
- Deploy on embedded AI platforms.

---

# 👨‍💻 Author

**Khlifi Med Khalil**

Mechatronics Engineering Student

---

# ⭐ Project Highlights

This project demonstrates the combination of:

- Artificial Intelligence
- Computer Vision
- Object Detection
- Object Tracking
- Human-Computer Interaction

to create an interactive real-world vision system.
