
import argparse
from pathlib import Path
import sys
import yaml

try:
    from ultralytics import YOLO
except Exception as e:
    print("Ultralytics not installed or failed to import. Did you run 'pip install -r requirements.txt'?")
    raise

def main():
    p = argparse.ArgumentParser(description="Train a YOLOv8 license-plate detector")
    p.add_argument("--data", type=str, default="data/data.yaml", help="Path to YOLO data.yaml")
    p.add_argument("--model", type=str, default="yolov8n.pt", help="Base model (e.g., yolov8n.pt, yolov8s.pt)")
    p.add_argument("--imgsz", type=int, default=1280, help="Training image size (pixels)")
    p.add_argument("--epochs", type=int, default=80, help="Training epochs")
    p.add_argument("--batch", type=int, default=16, help="Batch size")
    p.add_argument("--name", type=str, default="lp_yolov8n", help="Run name")
    p.add_argument("--project", type=str, default="runs/detect", help="Project directory")
    p.add_argument("--device", type=str, default="auto", help="cuda, cpu, or auto")
    p.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    p.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[!] data.yaml not found at: {data_path.resolve()}")
        print("    Create your dataset in YOLO format and update data/data.yaml accordingly.")
        sys.exit(1)

    # Basic sanity check on YAML
    try:
        with open(data_path, "r") as f:
            y = yaml.safe_load(f)
        for k in ["train", "val", "names"]:
            assert k in y, f"Missing key '{k}' in {data_path}"
    except Exception as e:
        print(f"[!] Failed to parse {data_path}: {e}")
        sys.exit(1)

    print("[i] Loading model:", args.model)
    model = YOLO(args.model)

    print("[i] Starting training...")
    results = model.train(
        data=str(data_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project=args.project,
        workers=args.workers,
        patience=args.patience,
        # You can tune augmentations via overrides:
        # hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, mosaic=1, translate=0.1, scale=0.5,
    )
    print("[✓] Training finished. Best weights saved under 'runs/detect/<run-name>/weights/best.pt'")

if __name__ == "__main__":
    main()
