# src/ocr_utils.py
import cv2, re, os
import numpy as np
from pathlib import Path

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False

PLATE_RE = re.compile(r"[A-Z0-9]{5,8}")  # tweak for your locale

def init_easyocr():
    if not EASYOCR_AVAILABLE:
        raise RuntimeError("EasyOCR not installed: pip install easyocr")
    # verbose=True shows first-time model downloads
    return easyocr.Reader(["en"], gpu=False, verbose=True)

def clean_plate_text(s: str) -> str:
    s = s.upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    if "O" in s and "0" not in s: s = s.replace("O","0")
    if "I" in s and "1" not in s: s = s.replace("I","1")
    return s

def preprocess_for_ocr(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0, (8,8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 5, 75, 75)
    th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return th

def ocr_once(reader, crop_bgr: np.ndarray):
    roi = preprocess_for_ocr(crop_bgr)
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
    results = reader.readtext(roi_rgb)  # [(bbox, text, conf), ...]
    if not results: return "", 0.0
    candidates = []
    for _, text, conf in results:
        t = clean_plate_text(text)
        if not t: continue
        bonus = 1.3 if PLATE_RE.fullmatch(t) else 1.0
        candidates.append((t, conf*bonus*max(1,len(t)), conf))
    if not candidates: return "", 0.0
    t, _, conf = max(candidates, key=lambda x: x[1])
    # prefer strict match if available
    if not PLATE_RE.fullmatch(t):
        alnums = [c[0] for c in candidates if len(c[0]) >= 4]
        t = max(alnums, key=len, default=t)
    return t, float(conf)

def plausible_plate(t: str) -> bool:
    t = t.upper()
    return 5 <= len(t) <= 8 and re.fullmatch(r"[A-Z0-9]+", t) is not None and any(c.isdigit() for c in t)

def tightened_crop(orig_img, x1, y1, x2, y2, shrink_x=0.10, shrink_y=0.10):
    """Shrink/expand the box per side; negatives expand."""
    H, W = orig_img.shape[:2]
    w, h = x2 - x1, y2 - y1
    nx1 = max(0, x1 + int(w * shrink_x))
    ny1 = max(0, y1 + int(h * shrink_y))
    nx2 = min(W, x2 - int(w * shrink_x))
    ny2 = min(H, y2 - int(h * shrink_y))
    if nx2 <= nx1 or ny2 <= ny1:
        nx1, ny1, nx2, ny2 = x1, y1, x2, y2
    return orig_img[ny1:ny2, nx1:nx2], (nx1, ny1, nx2, ny2)

def try_ocr_with_retries(reader, orig_img, x1, y1, x2, y2, base_sx, base_sy, save_dir: str|None=None, tag=""):
    """Try several shrink/expand variants; return (text, conf, (cx1,cy1,cx2,cy2))."""
    trials = [
        (base_sx, base_sy),
        (0.05, 0.05),
        (0.00, 0.00),
        (-0.03, -0.03),   # slight expansion
        (0.02, 0.02),
    ]
    best_text, best_conf, best_box = "", 0.0, (x1, y1, x2, y2)
    for idx, (sx, sy) in enumerate(trials):
        crop, box = tightened_crop(orig_img, x1, y1, x2, y2, sx, sy)
        text, conf = ocr_once(reader, crop)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(str(Path(save_dir)/f"retry_{tag}_{idx}_{sx}_{sy}.jpg"), crop)
        if plausible_plate(text):
            return text, conf, box
        if conf > best_conf:
            best_text, best_conf, best_box = text, conf, box
    return best_text, best_conf, best_box
