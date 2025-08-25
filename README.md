# Automatic License Plate Recognition — YOLOv8 + EasyOCR

Real-time license plate detection, tracking, and OCR using **Ultralytics YOLOv8** for object detection and **EasyOCR** for reading plate text. Works with a webcam, video files, or image folders. Outputs cropped plates and a CSV of recognized text with confidences.

> Repo layout includes `src/`, `data/`, `runs/detect/`, `requirements.txt`, and a YOLO weight file `yolov8n.pt`. See GitHub file list for details.

## Demo

<video src="assets/demo.mp4" controls loop muted playsinline width="640"></video>

## Features

- **Detect** license plates with YOLOv8 (default: `yolov8n.pt`)
- **Track** detections across frames (ID stitching-friendly utils)
- **OCR** plate crops with EasyOCR, with configurable preprocessing
- **Save** crops and an `ocr_results.csv` (timestamp, text, confidence, bbox)
- **Run** on webcam (`--source 0`), videos, or image folders

---

## Quickstart

### 1) Clone & set up
```bash
git clone https://github.com/AmitUCSB/license-plate-detection-and-tracking.git
cd license-plate-detection-and-tracking

# (Recommended) Python 3.10–3.12 virtual env
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
