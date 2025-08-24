# src/infer_webcam.py
import argparse, cv2, time, os
from pathlib import Path

from detector import PlateDetector
from ocr_utils import init_easyocr, read_plate_text  # using the focused OCR
from utils import draw_fps, annotate_text, save_csv_row, timestamp


def is_image_path(p: str) -> bool:
    return Path(p).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _clamp(v, a, b):
    return max(a, min(b, v))


def _shrink_box(x1, y1, x2, y2, sx, sy, W, H):
    """Shrink the box by fractions sx, sy on each side, clamp to image bounds."""
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    dx = int(round(w * sx))
    dy = int(round(h * sy))
    nx1 = _clamp(x1 + dx, 0, W - 1)
    ny1 = _clamp(y1 + dy, 0, H - 1)
    nx2 = _clamp(x2 - dx, 1, W)
    ny2 = _clamp(y2 - dy, 1, H)
    if nx2 <= nx1:
        nx1, nx2 = x1, x2
    if ny2 <= ny1:
        ny1, ny2 = y1, y2
    return nx1, ny1, nx2, ny2


def _crop(img, x1, y1, x2, y2):
    H, W = img.shape[:2]
    x1 = _clamp(int(x1), 0, W - 1)
    x2 = _clamp(int(x2), 1, W)
    y1 = _clamp(int(y1), 0, H - 1)
    y2 = _clamp(int(y2), 1, H)
    return img[y1:y2, x1:x2].copy()


def _band_rect_global(shrunk_box, band_y, full_shape):
    """
    Convert a (y0,y1) band within the shrunk ROI back to global image coords.
    shrunk_box = (sx1, sy1, sx2, sy2); band_y = (y0, y1) relative to shrunk ROI.
    Returns (gx1, gy0, gx2, gy1).
    """
    sx1, sy1, sx2, sy2 = shrunk_box
    H, W = full_shape[:2]
    y0, y1 = band_y
    gx1, gx2 = _clamp(sx1, 0, W - 1), _clamp(sx2, 1, W)
    gy0, gy1 = _clamp(sy1 + int(y0), 0, H - 1), _clamp(sy1 + int(y1), 1, H)
    return gx1, gy0, gx2, gy1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", default="0")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cpu")

    # OCR & logging
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--log-csv", default="")

    # Saving / display
    ap.add_argument("--save-crops", action="store_true")
    ap.add_argument("--crop-dir", default="data/crops")
    ap.add_argument("--save-annotated", action="store_true")
    ap.add_argument("--out-dir", default="data/outs")
    ap.add_argument("--show", action="store_true")

    # Crop controls (shrink detector box before OCR to avoid borders/bolts)
    ap.add_argument("--shrink", type=float, default=None, help="default shrink per side if -x/-y not set")
    ap.add_argument("--shrink-x", type=float, default=0.06)
    ap.add_argument("--shrink-y", type=float, default=0.15)
    ap.add_argument("--debug-crops", action="store_true")

    args = ap.parse_args()

    detector = PlateDetector(args.weights, args.device, args.imgsz, args.conf)
    reader = init_easyocr() if args.ocr else None

    if args.save_crops:
        os.makedirs(args.crop_dir, exist_ok=True)
    if args.save_annotated or args.debug_crops:
        os.makedirs(args.out_dir, exist_ok=True)

    def eff_shrinks():
        sx = args.shrink_x if args.shrink_x is not None else args.shrink
        sy = args.shrink_y if args.shrink_y is not None else args.shrink
        return sx, sy

    # ---------- Single image ----------
    if is_image_path(args.source):
        img = cv2.imread(args.source)
        if img is None:
            raise RuntimeError(f"Could not read image: {args.source}")

        r = detector.predict(img)
        annotated = r.plot()
        sx, sy = eff_shrinks()
        ts_now = time.time_ns()

        if r.boxes is not None and len(r.boxes) > 0:
            for i, box in enumerate(r.boxes.xyxy.cpu().numpy().astype(int)):
                x1, y1, x2, y2 = box.tolist()

                # Shrink the detected plate box before OCR
                sx1, sy1, sx2, sy2 = _shrink_box(x1, y1, x2, y2, sx, sy, r.orig_img.shape[1], r.orig_img.shape[0])
                roi = _crop(r.orig_img, sx1, sy1, sx2, sy2)

                text, conf, band = "", 0.0, (0, roi.shape[0])
                if reader is not None:
                    save_dir = os.path.join(args.out_dir, f"ocr_img_{i}") if args.debug_crops else None
                    text, conf, band, dbg = read_plate_text(reader, roi, save_debug_dir=save_dir)

                # Compute a global band rectangle for visualization
                bx1, by0, bx2, by1 = _band_rect_global((sx1, sy1, sx2, sy2), band, r.orig_img.shape)

                # Save final (shrunk) crop
                if args.save_crops:
                    cv2.imwrite(str(Path(args.crop_dir) / f"plate_{ts_now}_{i}.jpg"), roi)

                # Annotate with best text
                if text and conf >= 0.7:
                    annotate_text(annotated, bx1, max(by0 - 10, 0), f"{text} ({conf:.2f})")
                    # Draw the character band box for clarity
                    cv2.rectangle(annotated, (bx1, by0), (bx2, by1), (0, 255, 0), 2)
                    print(f"Plate: {text}  conf={conf:.2f}  box=({bx1},{by0},{bx2},{by1})")

                if args.log_csv:
                    save_csv_row(args.log_csv, [timestamp(), args.source, 0, text, conf, bx1, by0, bx2, by1, ""])

        if args.save_annotated:
            out = Path(args.out_dir) / f"{Path(args.source).stem}_annotated.jpg"
            cv2.imwrite(str(out), annotated)

        if args.show:
            cv2.imshow("ALPR (image)", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # ---------- Video / webcam ----------
    cap = cv2.VideoCapture(0 if args.source == "0" else args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    prev = time.time()
    frame_idx = 0
    sx, sy = eff_shrinks()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        r = detector.predict(frame)
        annotated = r.plot()
        ts_now = time.time_ns()

        if r.boxes is not None and len(r.boxes) > 0:
            for i, box in enumerate(r.boxes.xyxy.cpu().numpy().astype(int)):
                x1, y1, x2, y2 = box.tolist()

                sx1, sy1, sx2, sy2 = _shrink_box(x1, y1, x2, y2, sx, sy, r.orig_img.shape[1], r.orig_img.shape[0])
                roi = _crop(r.orig_img, sx1, sy1, sx2, sy2)

                text, conf, band = "", 0.0, (0, roi.shape[0])
                if reader is not None:
                    save_dir = os.path.join(args.out_dir, f"ocr_f{frame_idx}_{i}") if args.debug_crops else None
                    text, conf, band, dbg = read_plate_text(reader, roi, save_debug_dir=save_dir)

                bx1, by0, bx2, by1 = _band_rect_global((sx1, sy1, sx2, sy2), band, r.orig_img.shape)

                if args.save_crops:
                    cv2.imwrite(str(Path(args.crop_dir) / f"plate_f{frame_idx}_{ts_now}_{i}.jpg"), roi)

                if text and conf >= 0.7:
                    annotate_text(annotated, bx1, max(by0 - 10, 0), f"{text} ({conf:.2f})")
                    cv2.rectangle(annotated, (bx1, by0), (bx2, by1), (0, 255, 0), 2)
                    print(f"Frame {frame_idx}: {text} (conf {conf:.2f})  box=({bx1},{by0},{bx2},{by1})")

                if args.log_csv:
                    save_csv_row(
                        args.log_csv,
                        [timestamp(), args.source, frame_idx, text, conf, bx1, by0, bx2, by1, ""],
                    )

        # save annotated frames if requested
        if args.save_annotated:
            cv2.imwrite(str(Path(args.out_dir) / f"frame_{frame_idx:06d}.jpg"), annotated)

        # draw & show
        dt = time.time() - prev
        prev = time.time()
        fps = 1.0 / dt if dt > 0 else 0.0
        if args.show:
            draw_fps(annotated, fps)
            cv2.imshow("ALPR (video/webcam)", annotated)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

        frame_idx += 1

    cap.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
