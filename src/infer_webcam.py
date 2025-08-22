# src/infer_webcam.py
import argparse, cv2, time, os
from pathlib import Path

from detector import PlateDetector
from ocr_utils import init_easyocr, try_ocr_with_retries, tightened_crop
from utils import draw_fps, annotate_text, save_csv_row, timestamp

def is_image_path(p: str) -> bool:
    return Path(p).suffix.lower() in {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", default="0")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="auto")

    # OCR & logging
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--log-csv", default="")

    # Saving / display
    ap.add_argument("--save-crops", action="store_true")
    ap.add_argument("--crop-dir", default="data/crops")
    ap.add_argument("--save-annotated", action="store_true")
    ap.add_argument("--out-dir", default="data/outs")
    ap.add_argument("--show", action="store_true")

    # Crop controls
    ap.add_argument("--shrink", type=float, default=0.10, help="default shrink per side if -x/-y not set")
    ap.add_argument("--shrink-x", type=float, default=None)
    ap.add_argument("--shrink-y", type=float, default=None)
    ap.add_argument("--debug-crops", action="store_true")

    args = ap.parse_args()

    detector = PlateDetector(args.weights, args.device, args.imgsz, args.conf)
    reader = init_easyocr() if args.ocr else None

    if args.save_crops: os.makedirs(args.crop_dir, exist_ok=True)
    if args.save_annotated: os.makedirs(args.out_dir, exist_ok=True)

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

                text, conf, (cx1, cy1, cx2, cy2) = ("", 0.0, (x1,y1,x2,y2))
                if reader:
                    text, conf, (cx1, cy1, cx2, cy2) = try_ocr_with_retries(
                        reader, r.orig_img, x1, y1, x2, y2, sx, sy,
                        save_dir=(args.out_dir if args.debug_crops else None),
                        tag=f"img_{i}"
                    )

                # final crop to save (no extra shrink)
                final_crop, _ = tightened_crop(r.orig_img, cx1, cy1, cx2, cy2, 0.0, 0.0)
                if args.save_crops:
                    cv2.imwrite(str(Path(args.crop_dir)/f"plate_{ts_now}_{i}.jpg"), final_crop)

                if text:
                    annotate_text(annotated, cx1, cy1, f"{text} ({conf:.2f})")
                    print(f"Plate: {text}  conf={conf:.2f}  box=({cx1},{cy1},{cx2},{cy2})")

                if args.log_csv:
                    save_csv_row(args.log_csv, [timestamp(), args.source, 0, text, conf, cx1, cy1, cx2, cy2, ""])

        if args.save_annotated:
            out = Path(args.out_dir)/f"{Path(args.source).stem}_annotated.jpg"
            cv2.imwrite(str(out), annotated)

        if args.show:
            cv2.imshow("ALPR (image)", annotated)
            cv2.waitKey(0)     # <— keeps the window open until you press a key
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
        if not ok: break

        r = detector.predict(frame)
        annotated = r.plot()
        ts_now = time.time_ns()

        if r.boxes is not None and len(r.boxes) > 0:
            for i, box in enumerate(r.boxes.xyxy.cpu().numpy().astype(int)):
                x1, y1, x2, y2 = box.tolist()
                text, conf, (cx1, cy1, cx2, cy2) = ("", 0.0, (x1,y1,x2,y2))
                if reader:
                    text, conf, (cx1, cy1, cx2, cy2) = try_ocr_with_retries(
                        reader, r.orig_img, x1, y1, x2, y2, sx, sy,
                        save_dir=(args.out_dir if args.debug_crops else None),
                        tag=f"{frame_idx}_{i}"
                    )

                final_crop, _ = tightened_crop(r.orig_img, cx1, cy1, cx2, cy2, 0.0, 0.0)
                if args.save_crops:
                    cv2.imwrite(str(Path(args.crop_dir)/f"plate_f{frame_idx}_{ts_now}_{i}.jpg"), final_crop)

                if text:
                    annotate_text(annotated, cx1, cy1, f"{text} ({conf:.2f})")
                    print(f"Frame {frame_idx}: {text} (conf {conf:.2f})  box=({cx1},{cy1},{cx2},{cy2})")

                if args.log_csv:
                    save_csv_row(args.log_csv, [timestamp(), args.source, frame_idx, text, conf, cx1, cy1, cx2, cy2, ""])

        # save annotated frames if requested
        if args.save_annotated:
            cv2.imwrite(str(Path(args.out_dir)/f"frame_{frame_idx:06d}.jpg"), annotated)

        # draw & show
        dt = time.time() - prev; prev = time.time()
        fps = 1.0/dt if dt > 0 else 0.0
        if args.show:
            draw_fps(annotated, fps)
            cv2.imshow("ALPR (video/webcam)", annotated)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

        frame_idx += 1

    cap.release()
    if args.show:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
