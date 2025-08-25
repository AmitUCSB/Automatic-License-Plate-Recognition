# src/infer_webcam.py
import argparse, cv2, time, os
from pathlib import Path

from detector import PlateDetector
from gemini_utils import init_gemini, read_plate_text  # Gemini-based plate reader
from utils import draw_fps, save_csv_row, timestamp  # note: no annotate_text import

# ---- Fixed styling (no CLI knobs) ----
TEXT_SCALE = 1.8       # bump this if you want even bigger text
TEXT_THICKNESS = 4     # thicker text stroke
TEXT_MARGIN = 10       # padding around the text background


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


def _annotate_text_big(img, x1, y1, text):
    """Draw a bigger text box using fixed style constants."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, TEXT_SCALE, TEXT_THICKNESS)
    top = max(0, y1 - th - TEXT_MARGIN)
    cv2.rectangle(img, (x1, top), (x1 + tw + TEXT_MARGIN, y1), (0, 0, 0), -1)
    cv2.putText(img, text, (x1 + 5, y1 - 5), font, TEXT_SCALE, (255, 255, 255), TEXT_THICKNESS, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", default="0")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cpu")

    # Gemini (LLM) reading
    ap.add_argument("--use-gemini", action="store_true", help="Enable Gemini plate reading")
    ap.add_argument("--gemini-model", default=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--min-conf", type=float, default=0.7, help="Min confidence to overlay/print")

    # Logging / saving / display
    ap.add_argument("--log-csv", default="")
    ap.add_argument("--save-crops", action="store_true")
    ap.add_argument("--crop-dir", default="data/crops")
    ap.add_argument("--save-annotated", action="store_true", help="Dump per-frame JPEGs in --out-dir")
    ap.add_argument("--out-dir", default="data/outs")
    ap.add_argument("--show", action="store_true")

    # Crop controls
    ap.add_argument("--shrink", type=float, default=None, help="Default shrink per side if -x/-y not set")
    ap.add_argument("--shrink-x", type=float, default=0)
    ap.add_argument("--shrink-y", type=float, default=0)
    ap.add_argument("--debug-crops", action="store_true")

    # Annotated video writer
    ap.add_argument("--out-video", default="", help="Write a single annotated video (e.g., data/outs/annotated.mp4)")
    ap.add_argument("--fourcc", default="mp4v", help="Codec fourcc: mp4v, avc1, XVID, MJPG, etc.")
    ap.add_argument("--fps", type=float, default=0.0, help="Override output FPS; 0 uses source FPS")

    args = ap.parse_args()

    detector = PlateDetector(args.weights, args.device, args.imgsz, args.conf)
    gemini = init_gemini(args.gemini_model) if args.use_gemini else None

    if args.save_crops:
        os.makedirs(args.crop_dir, exist_ok=True)
    if args.save_annotated or args.debug_crops:
        os.makedirs(args.out_dir, exist_ok=True)
    if args.out_video:
        out_parent = Path(args.out_video).parent
        if str(out_parent):
            os.makedirs(out_parent, exist_ok=True)

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
        annotated = r.orig_img.copy()  # <- no YOLO overlay labels
        sx, sy = eff_shrinks()
        ts_now = time.time_ns()

        if r.boxes is not None and len(r.boxes) > 0:
            for i, box in enumerate(r.boxes.xyxy.cpu().numpy().astype(int)):
                x1, y1, x2, y2 = box.tolist()

                # Shrink the detected plate box before Gemini reading
                sx1, sy1, sx2, sy2 = _shrink_box(
                    x1, y1, x2, y2, sx, sy, r.orig_img.shape[1], r.orig_img.shape[0]
                )
                roi = _crop(r.orig_img, sx1, sy1, sx2, sy2)

                text, conf, band = "", 0.0, (0, roi.shape[0])
                if gemini is not None:
                    save_dir = os.path.join(args.out_dir, f"ocr_img_{i}") if args.debug_crops else None
                    text, conf, band, dbg = read_plate_text(
                        gemini, roi, save_debug_dir=save_dir, temperature=args.temperature
                    )

                # Compute a global band rectangle for visualization
                bx1, by0, bx2, by1 = _band_rect_global((sx1, sy1, sx2, sy2), band, r.orig_img.shape)

                # Save final (shrunk) crop
                if args.save_crops:
                    cv2.imwrite(str(Path(args.crop_dir) / f"plate_{ts_now}_{i}.jpg"), roi)

                # Annotate with bigger text
                if text and conf >= args.min_conf:
                    _annotate_text_big(annotated, bx1, max(by0 - 10, 0), f"{text} ({conf:.2f})")
                    # comment out next line if you don't want the green band rectangle:
                    cv2.rectangle(annotated, (bx1, by0), (bx2, by1), (0, 255, 0), 2)
                    print(f"Plate: {text}  conf={conf:.2f}  box=({bx1},{by0},{bx2},{by1})")

                if args.log_csv:
                    save_csv_row(args.log_csv, [timestamp(), args.source, 0, text, conf, bx1, by0, bx2, by1, ""])

        # Write annotated image if requested
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

    # Prepare writer lazily to match the first annotated frame's size
    writer = None
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 1.0:
        src_fps = 30.0
    out_fps = args.fps if args.fps and args.fps > 0 else src_fps
    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        r = detector.predict(frame)
        annotated = r.orig_img.copy()  # <- no YOLO overlay labels
        ts_now = time.time_ns()

        if r.boxes is not None and len(r.boxes) > 0:
            for i, box in enumerate(r.boxes.xyxy.cpu().numpy().astype(int)):
                x1, y1, x2, y2 = box.tolist()

                sx1, sy1, sx2, sy2 = _shrink_box(
                    x1, y1, x2, y2, sx, sy, r.orig_img.shape[1], r.orig_img.shape[0]
                )
                roi = _crop(r.orig_img, sx1, sy1, sx2, sy2)

                text, conf, band = "", 0.0, (0, roi.shape[0])
                if gemini is not None:
                    save_dir = os.path.join(args.out_dir, f"ocr_f{frame_idx}_{i}") if args.debug_crops else None
                    text, conf, band, dbg = read_plate_text(
                        gemini, roi, save_debug_dir=save_dir, temperature=args.temperature
                    )

                bx1, by0, bx2, by1 = _band_rect_global((sx1, sy1, sx2, sy2), band, r.orig_img.shape)

                if args.save_crops:
                    cv2.imwrite(str(Path(args.crop_dir) / f"plate_f{frame_idx}_{ts_now}_{i}.jpg"), roi)

                if text and conf >= args.min_conf:
                    _annotate_text_big(annotated, bx1, max(by0 - 10, 0), f"{text} ({conf:.2f})")
                    # comment out next line if you don't want the green band rectangle:
                    cv2.rectangle(annotated, (bx1, by0), (bx2, by1), (0, 255, 0), 2)
                    print(f"Frame {frame_idx}: {text} (conf {conf:.2f})  box=({bx1},{by0},{bx2},{by1})")

                if args.log_csv:
                    save_csv_row(
                        args.log_csv,
                        [timestamp(), args.source, frame_idx, text, conf, bx1, by0, bx2, by1, ""],
                    )

        # Dump per-frame JPEGs if requested
        if args.save_annotated:
            cv2.imwrite(str(Path(args.out_dir) / f"frame_{frame_idx:06d}.jpg"), annotated)

        # Lazy-init and write annotated video if requested
        if args.out_video:
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(args.out_video, fourcc, out_fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer at {args.out_video}")
            writer.write(annotated)

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
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
