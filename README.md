
# YOLO License Plate Detector — Starter Repo

A minimal, production-leaning starter kit to train and run a **YOLOv8** detector for **license plates** (detection only).

## Quickstart

```bash
# 1) Create & activate a virtual environment (optional but recommended)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3) Verify ultralytics install
yolo checks

# 4) Put your dataset in YOLO format under ./data/ (see data/data.yaml)
#    data/
#      images/{train,val,test}/*.jpg
#      labels/{train,val,test}/*.txt  # class x_center y_center width height (normalized)

# 5) Train a baseline model
python src/train.py --data data/data.yaml --model yolov8n.pt --imgsz 1280 --epochs 80 --batch 16

# 6) Run the live demo (webcam 0) using the best weights
python src/infer_webcam.py --weights runs/detect/lp_yolov8n/weights/best.pt --imgsz 1280 --conf 0.25 --source 0
```

## Dataset Format

- Single class: `license_plate`.
- Labels are in YOLO format:
  - One `.txt` per image with lines: `0 x_center y_center width height` (all **normalized 0..1**).
- Update `data/data.yaml` if your dataset lives elsewhere.

## Useful Scripts

- `src/utils/video_to_frames.py` — Extract frames from a video every N frames to speed up labeling.

## Tips

- Plates are **small objects** → prefer `--imgsz 1280` or higher; consider `yolov8s.pt` if recall is low.
- To improve night/glare performance, add those conditions to your dataset (or use heavy augmentations).
- Prioritize **Recall** for detection tasks: catching more plates is usually better than a tiny precision gain.

## Legal & Privacy

- Check local laws on recording license plates and storing video. Use encryption and limit retention where required.
