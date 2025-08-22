# src/infer_webcam.py
import argparse
import csv
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ---- EasyOCR (optional) ----
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False


# ----------------- Helpers -----------------
def draw_fps(img, fps):
    txt = f"FPS: {fps:.1f}"
    cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)


def is_image_path(p: str) -> bool:
    ext = Path(p).suffix.lower()
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def init_easyocr(device_arg: str):
    """
    Initialize EasyOCR reader.
    - On Macs (M1/M2/M3), GPU=False (no CUDA).
    """
    if not EASYOCR_AVAILABLE:
        raise RuntimeError("EasyOCR not installed. Run: pip install easyocr")
    gpu = False
    # If you're on a CUDA machine and want GPU, set gpu=True manually.
    # Verbose=True so you can see first-time model downloads.
    reader = easyocr.Reader(["en"], gpu=gpu, verbose=True)
    return reader


# Plate text post-processing
PLATE_RE = re.compile(r"[A-Z0-9]{5,8}")  # adjust for your locale as needed


def clean_plate_text(s: str) -> str:
    s = s.upper()
    s = re.sub(r"[^A-Z0-9]", "", s)  # keep only alphanumerics
    # Common visual confusions
    if "O" in s and "0" not in s:
        s = s.replace("O", "0")
    if "I" in s and "1" not in s:
        s = s.replace("I", "1")
    return s


def preprocess_for_ocr(bgr: np.ndarray) -> np.ndarray:
    """Light preprocessing → clearer characters for OCR."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=75, sigmaSpace=75)
    th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return th


def ocr_plate(reader, crop_bgr: np.ndarray):
    """Run OCR with preprocessing and pick the best plate-like candidate."""
    roi = preprocess_for_ocr(crop_bgr)
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
    results = reader.readtext(roi_rgb)  # list of (bbox, text, conf)

    if not results:
        return "", 0.0

    candidates = []
    for _, text, conf in results:
        t = clean_plate_text(text)
        if not t:
            continue
        bonus = 1.3 if PLATE_RE.fullmatch(t) else 1.0
        score = conf * bonus * max(1, len(t))
        candidates.append((t, score, conf))

    if not candidates:
        return "", 0.0

    # best by our composite score
    t, _, conf = max(candidates, key=lambda x: x[1])
    # final sanity: prefer plate-like tokens
    if not PLATE_RE.fullmatch(t):
        alnums = [c[0] for c in candidates if len(c[0]) >= 4]
        t = max(alnums, key=len, default=t)
    return t, float(conf)


def annotate_text(img, x1, y1, text):
    """Draw a label box above detection."""
    label = str(text)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    cv2.rectangle(img, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), (0, 0, 0), -1)
    cv2.putText(img, label, (x1 + 5, y1 - 5), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def tightened_crop(orig_img: np.ndarray, x1, y1, x2, y2, shrink_x=0.10, shrink_y=0.10):
    """
    Shrink/expand the YOLO box by fractions per side; negatives expand.
    shrink_x/shrink_y are per-side fractions (0.10 = 10% on each side).
    """
    H, W = orig_img.shape[:2]
    w = x2 - x1
    h = y2 - y1
    nx1 = max(0, x1 + int(w * shrink_x))
    ny1 = max(0, y1 + int(h * shrink_y))
    nx2 = min(W, x2 - int(w * shrink_x))
    ny2 = min(H, y2 - int(h * shrink_y))
    if nx2 <= nx1 or ny2 <= ny1:
        # fall back to original if over-shrunk
        nx1, ny1, nx2, ny2 = x1, y1, x2, y2
    return orig_img[ny1:ny2, nx1:nx2], (nx1, ny1, nx2, ny2)


def plausible_plate(t: str) -> bool:
    t = t.upper()
    return 5 <= len(t) <= 8 and any(c.isdigit() for c in t) and re.fullmatch(r"[A-Z0-9]+", t) is not None


def try_ocr_with_retries(reader, orig_img, x1, y1, x2, y2, base_sx, base_sy, save_dir=None, tag=""):
    """
    Try a few shrink/expand combos to avoid cutting digits or picking frame text.
    Returns (text, conf, (cx1,cy1,cx2,cy2)).
    """
    trials = [
        (base_sx, base_sy),     # as requested
        (0.05, 0.05),           # gentle shrink
        (0.00, 0.00),           # no shrink
        (-0.03, -0.03),         # slight expansion
        (0.02, 0.02),           # tiny shrink
    ]
    best_text, best_conf, best_box = "", 0.0, (x1, y1, x2, y2)
    for idx, (sx, sy) in enumerate(trials):
        crop, box = tightened_crop(orig_img, x1, y1, x2, y2, sx, sy)
        text, conf = ocr_plate(reader, crop)
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(str(Path(save_dir) / f"retry_{tag}_{idx}_{sx}_{sy}.jpg"), crop)
        if plausible_plate(text):
            return text, conf, box
        if conf > best_conf:
            best_text, best_conf, best_box = text, conf, box
    return best_text, best_conf, best_box


# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser(description="YOLO license-plate detector with OCR (image/video/webcam)")
    ap.add_argument("--weights", type=str, required=True, help="Path to model weights (e.g., runs/detect/.../best.pt)")
    ap.add_argument("--source", type=str, default="0", help="0 for webcam, path/URL for video, or an image path")
    ap.add_argument("--imgsz", type=int, default=960, help="Inference image size (pixels)")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--device", type=str, default="auto", help="cuda, cpu, mps, or auto")

    # Saving / viewing
    ap.add_argument("--save-crops", action="store_true", help="Save cropped plate images")
    ap.add_argument("--crop-dir", type=str, default="data/crops", help="Dir to save crops")
    ap.add_argument("--save-annotated", action="store_true", help="Save annotated outputs (image or per-frame jpgs)")
    ap.add_argument("--out-dir", type=str, default="data/outs", help="Dir for annotated outputs")
    ap.add_argument("--show", action="store_true", help="Show a window (disable for headless)")

    # OCR controls
    ap.add_argument("--ocr", action="store_true", help="Enable EasyOCR on detected plates")
    ap.add_argument("--log-csv", type=str, default="", help="Optional CSV path to append OCR results")

    # Crop tightening (asymmetric; negatives expand)
    ap.add_argument("--shrink", type=float, default=0.10, help="Default shrink per side (used if -x/-y not set)")
    ap.add_argument("--shrink-x", type=float, default=None, help="Horizontal shrink per side (neg to expand)")
    ap.add_argument("--shrink-y", type=float, default=None, help="Vertical shrink per side (neg to expand)")

    # Debug crops from retries
    ap.add_argument("--debug-crops", action="store_true", help="Save all retry crops for debugging")

    args = ap.parse_args()

    # Create model
    model = YOLO(args.weights)

    # Make dirs
    if args.save_crops:
        os.makedirs(args.crop_dir, exist_ok=True)
    if args.save_annotated:
        os.makedirs(args.out_dir, exist_ok=True)

    # OCR init
    reader = None
    if args.ocr:
        reader = init_easyocr(args.device)

    # CSV header
    if args.log_csv:
        first_time = not Path(args.log_csv).exists()
        with open(args.log_csv, "a", newline="") as f:
            w = csv.writer(f)
            if first_time:
                w.writerow(["timestamp", "source", "frame_idx", "text", "confidence",
                            "x1", "y1", "x2", "y2", "crop_path"])

    source_arg = args.source
    is_image = is_image_path(source_arg)
    if not is_image and source_arg.isdigit():
        source_arg = int(source_arg)  # webcam index

    # Effective shrink values
    def eff_shrinks():
        sx = args.shrink_x if args.shrink_x is not None else args.shrink
        sy = args.shrink_y if args.shrink_y is not None else args.shrink
        return sx, sy

    # ---------- Single image ----------
    if is_image:
        img = cv2.imread(args.source)
        if img is None:
            raise RuntimeError(f"Could not read image: {args.source}")

        results = model.predict(source=img, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
        r = results[0]
        annotated = r.plot()

        saved_crops = 0
        frame_idx = 0
        ts_now = time.time_ns()
        sx, sy = eff_shrinks()

        if r.boxes is not None and len(r.boxes) > 0:
            for i, box in enumerate(r.boxes.xyxy.cpu().numpy().astype(int)):
                x1, y1, x2, y2 = box.tolist()

                # OCR with retries to avoid cutoff
                text, conf, (cx1, cy1, cx2, cy2) = ("", 0.0, (x1, y1, x2, y2))
                if reader is not None:
                    text, conf, (cx1, cy1, cx2, cy2) = try_ocr_with_retries(
                        reader, r.orig_img, x1, y1, x2, y2, sx, sy,
                        save_dir=(args.out_dir if args.debug_crops else None),
                        tag=f"img_{i}"
                    )

                # Save the final (best) crop if requested
                final_crop, _ = tightened_crop(r.orig_img, cx1, cy1, cx2, cy2, 0.0, 0.0)
                crop_path = ""
                if args.save_crops:
                    crop_path = str(Path(args.crop_dir) / f"plate_{ts_now}_{i}.jpg")
                    cv2.imwrite(crop_path, final_crop)
                    saved_crops += 1

                # Annotate & print
                if text:
                    annotate_text(annotated, cx1, cy1, f"{text} ({conf:.2f})")
                    print(f"Plate: {text}  conf={conf:.2f}  box=({cx1},{cy1},{cx2},{cy2})")

                # Log CSV
                if args.log_csv:
                    with open(args.log_csv, "a", newline="") as f:
                        csv.writer(f).writerow(
                            [int(ts_now / 1e9), args.source, frame_idx, text, f"{conf:.4f}",
                             cx1, cy1, cx2, cy2, crop_path]
                        )

        if args.save_annotated:
            base = Path(args.source).stem
            out_ann = Path(args.out_dir) / f"{base}_annotated.jpg"
            cv2.imwrite(str(out_ann), annotated)

        if args.show:
            cv2.imshow("ALPR (YOLO + OCR)", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print(f"[✓] Image done. Detections: {len(r.boxes) if r.boxes is not None else 0}. "
              f"Crops saved: {saved_crops if args.save_crops else 0}.")
        return

    # ---------- Video / webcam ----------
    cap = cv2.VideoCapture(source_arg)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    prev = time.time()
    fps = 0.0
    frame_idx = 0
    sx, sy = eff_shrinks()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
        r = results[0]
        annotated = r.plot()

        if r.boxes is not None and len(r.boxes) > 0:
            ts_now = time.time_ns()
            for i, box in enumerate(r.boxes.xyxy.cpu().numpy().astype(int)):
                x1, y1, x2, y2 = box.tolist()

                # OCR with retries
                text, conf, (cx1, cy1, cx2, cy2) = ("", 0.0, (x1, y1, x2, y2))
                if reader is not None:
                    text, conf, (cx1, cy1, cx2, cy2) = try_ocr_with_retries(
                        reader, r.orig_img, x1, y1, x2, y2, sx, sy,
                        save_dir=(args.out_dir if args.debug_crops else None),
                        tag=f"{frame_idx}_{i}"
                    )

                # Save final crop if requested
                final_crop, _ = tightened_crop(r.orig_img, cx1, cy1, cx2, cy2, 0.0, 0.0)
                crop_path = ""
                if args.save_crops:
                    crop_path = str(Path(args.crop_dir) / f"plate_f{frame_idx}_{ts_now}_{i}.jpg")
                    cv2.imwrite(crop_path, final_crop)

                if text:
                    annotate_text(annotated, cx1, cy1, f"{text} ({conf:.2f})")
                    print(f"Plate: {text}  conf={conf:.2f}  frame={frame_idx}  box=({cx1},{cy1},{cx2},{cy2})")

                if args.log_csv:
                    with open(args.log_csv, "a", newline="") as f:
                        csv.writer(f).writerow(
                            [int(ts_now / 1e9), str(args.source), frame_idx, text, f"{conf:.4f}",
                             cx1, cy1, cx2, cy2, crop_path]
                        )

        # Optional: save annotated frames (careful: many files)
        if args.save_annotated:
            out = Path(args.out_dir) / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out), annotated)

        # Show + FPS
        now = time.time()
        dt = now - prev
        prev = now
        if dt > 0:
            fps = 1.0 / dt
        if args.show:
            draw_fps(annotated, fps)
            cv2.imshow("ALPR (YOLO + OCR)", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

        frame_idx += 1

    cap.release()
    if args.show:
        cv2.destroyAllWindows()
    print("[✓] Finished processing stream.")


if __name__ == "__main__":
    main()
