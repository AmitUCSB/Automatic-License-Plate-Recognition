# src/detector.py
from ultralytics import YOLO

class PlateDetector:
    def __init__(self, weights, device="cpu", imgsz=960, conf=0.25):
        self.model = YOLO(weights)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf

    def predict(self, frame):
        """Run YOLO detection on a frame (image or video frame)."""
        results = self.model.predict(
            source=frame, imgsz=self.imgsz,
            conf=self.conf, device=self.device,
            verbose=False
        )
        return results[0]  # first result
