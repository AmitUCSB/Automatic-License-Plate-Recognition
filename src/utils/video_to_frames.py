
import argparse
import os
import cv2

def main():
    p = argparse.ArgumentParser(description="Extract frames from a video every N frames")
    p.add_argument("--video", required=True, help="Path to input video")
    p.add_argument("--outdir", required=True, help="Output directory for frames")
    p.add_argument("--step", type=int, default=10, help="Save 1 frame every N frames (default: 10)")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    n = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % args.step == 0:
            out = os.path.join(args.outdir, f"{saved:06d}.jpg")
            cv2.imwrite(out, frame)
            saved += 1
        n += 1

    cap.release()
    print(f"[✓] Saved {saved} frames to {args.outdir}")

if __name__ == "__main__":
    main()
