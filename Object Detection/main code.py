import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

class YOLOSpotlight:
    """Click a tracked object to show a live cropped spotlight overlay."""

    def __init__(self, model="yolov8n.pt", source="video.mp4",
                 output="YOLO_spotlight.mp4", crop_size=(400, 400),
                 conf=0.4, tracker="bytetrack.yaml"):
        self.model = YOLO(model)
        self.names = self.model.names
        self.conf = conf
        self.tracker = tracker

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30

        self.writer = cv2.VideoWriter(
            output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h)
        )

        self.crop_size = crop_size
        self.crop_pad = 8
        self.crop_margin = 10

        self.window = "YOLOv8 Spotlight"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.mouse_callback)

        self.current_boxes = None
        self.current_ids = None
        self.selected_id = None

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.current_boxes is None or self.current_ids is None:
            return

        self.selected_id = None
        for box, obj_id in zip(self.current_boxes, self.current_ids):
            x1, y1, x2, y2 = box.astype(int)
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.selected_id = int(obj_id)
                print(f"Selected ID: {self.selected_id}")
                break

    def make_crop(self, frame, box):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, x1 - self.crop_pad)
        y1 = max(0, y1 - self.crop_pad)
        x2 = min(w, x2 + self.crop_pad)
        y2 = min(h, y2 + self.crop_pad)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        ch, cw = crop.shape[:2]
        scale = min(self.crop_size[0] / cw, self.crop_size[1] / ch)
        return cv2.resize(crop, (int(cw * scale), int(ch * scale)))

    def overlay_crop(self, frame, crop):
        if crop is None:
            return
        h, w = frame.shape[:2]
        ch, cw = crop.shape[:2]
        x = w - cw - self.crop_margin
        y = self.crop_margin
        cv2.rectangle(frame, (x-4, y-4), (x+cw+4, y+ch+4), (40,40,40), -1)
        frame[y:y+ch, x:x+cw] = crop
        cv2.rectangle(frame, (x-4, y-4), (x+cw+4, y+ch+4), (0,255,0), 2)
        cv2.putText(frame, "Spotlight", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    def run(self):
        print("Click an object to spotlight it.")
        print("'c' = clear selection | 'q' = quit")

        while True:
            ok, frame = self.cap.read()
            if not ok:
                break

            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker,
                conf=self.conf,
                verbose=False
            )

            annotator = Annotator(frame, line_width=2)

            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                r = results[0]
                boxes = r.boxes.xyxy.cpu().numpy()
                ids = r.boxes.id.cpu().numpy().astype(int)
                classes = r.boxes.cls.cpu().numpy().astype(int)

                self.current_boxes = boxes
                self.current_ids = ids

                if self.selected_id is not None:
                    idx = np.where(ids == self.selected_id)[0]
                    if len(idx):
                        crop = self.make_crop(frame, boxes[idx[0]])
                        self.overlay_crop(frame, crop)
                    else:
                        self.selected_id = None

                for box, obj_id, cls in zip(boxes, ids, classes):
                    color = (0,0,255) if obj_id == self.selected_id else colors(int(cls), True)
                    annotator.box_label(box, f"{self.names[int(cls)]} ID:{obj_id}", color=color)

            self.writer.write(frame)
            cv2.imshow(self.window, frame)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("c"):
                self.selected_id = None

        self.cap.release()
        self.writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = YOLOSpotlight(
        model="yolov8n.pt",
        source="video.mp4",
        output="YOLO_spotlight.mp4"
    )
    tracker.run()
