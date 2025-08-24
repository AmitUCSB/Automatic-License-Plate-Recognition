import cv2, re, os
import numpy as np
from pathlib import Path

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False

# Accept only letters+digits, typical US plate length (tweak for your locale)
PLATE_RE = re.compile(r"[A-Z0-9]{5,8}")
ALLOWED   = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# ------------------------- EasyOCR init -------------------------
def init_easyocr(gpu: bool = False, lang=("en",), verbose: bool = False):
    if not EASYOCR_AVAILABLE:
        raise RuntimeError("EasyOCR not installed: pip install easyocr==1.7.2")
    return easyocr.Reader(list(lang), gpu=gpu, verbose=verbose)

# ------------------------- Utilities ---------------------------
def clean_plate_text(s: str) -> str:
    s = s.upper()
    s = re.sub(r"[^A-Z0-9]", "", s)  # strip anything not alnum
    return s

def _ensure_3ch(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

# ------------------- Focus on characters band ------------------
def crop_to_char_band(plate_bgr: np.ndarray, debug: bool = False):
    """
    Find the horizontal band(s) with the strongest vertical-edge activity
    (i.e., where characters usually are) and crop to those band(s).

    Returns (crop, (y0,y1), debug_dict)
    """
    H, W = plate_bgr.shape[:2]
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)

    # Local contrast boost helps both light & dark plates
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    g = cv2.GaussianBlur(g, (5, 5), 0)

    # Vertical edges -> emphasize character strokes
    sobelx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    proj = np.abs(sobelx).sum(axis=1)  # sum across columns per row
    # Smooth projection
    proj_s = cv2.GaussianBlur(proj.reshape(-1, 1), (0, 0), 3).ravel()

    # Normalize
    if proj_s.max() > 1e-6:
        proj_s = (proj_s - proj_s.min()) / (proj_s.max() - proj_s.min() + 1e-6)
    else:
        proj_s = np.zeros_like(proj_s)

    # Threshold to find high-energy bands (possible single or double row)
    thr = 0.4 * proj_s.max()
    mask = proj_s >= thr

    # Group contiguous True runs
    bands = []
    i = 0
    while i < H:
        if mask[i]:
            j = i
            while j < H and mask[j]:
                j += 1
            bands.append((i, j))  # [i, j)
            i = j
        else:
            i += 1

    # If nothing reasonable, default to center 70%
    if not bands:
        y0 = int(0.15 * H)
        y1 = int(0.85 * H)
        crop = plate_bgr[y0:y1, :]
        dbg = {"proj": proj_s, "bands": [(y0, y1)], "used_default": True}
        return crop, (y0, y1), dbg if debug else {}

    # Filter out tiny bands (< 15% of height)
    bands = [(a, b) for (a, b) in bands if (b - a) >= int(0.15 * H)] or bands

    # If multiple bands (two-line plates), keep the 1 or 2 largest
    bands = sorted(bands, key=lambda ab: (ab[1] - ab[0]), reverse=True)[:2]

    # Merge close bands
    bands = sorted(bands)
    merged = []
    for (a, b) in bands:
        if not merged:
            merged.append([a, b])
        else:
            if a - merged[-1][1] < int(0.05 * H):  # bridge small gaps
                merged[-1][1] = b
            else:
                merged.append([a, b])
    bands = [(a, b) for a, b in merged]

    # Expand margins a bit
    margin = int(0.10 * H)
    y0 = max(0, min(a for a, _ in bands) - margin)
    y1 = min(H, max(b for _, b in bands) + margin)

    crop = plate_bgr[y0:y1, :]
    dbg = {"proj": proj_s, "bands": bands, "used_default": False}
    return crop, (y0, y1), dbg if debug else {}

# --------------------- Binarization helpers --------------------
def _binarize_both(gray: np.ndarray):
    """Return (bin_light_on_dark, bin_dark_on_light)."""
    # Adaptive handles uneven lighting; keep two polarities
    adp = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 9)
    adp_inv = 255 - adp

    # Mild morphology to clean noise
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    adp = cv2.morphologyEx(adp, cv2.MORPH_OPEN, k, iterations=1)
    adp_inv = cv2.morphologyEx(adp_inv, cv2.MORPH_OPEN, k, iterations=1)
    return adp, adp_inv

# ------------------------- OCR runner --------------------------
def read_plate_text(reader,
                    plate_bgr: np.ndarray,
                    try_char_band: bool = True,
                    decoder: str = "greedy",
                    min_len: int = 5,
                    save_debug_dir: str | None = None):
    """
    Run EasyOCR but *focus* on the characters band and restrict characters
    to letters+digits so we ignore logos, state names, and decorations.

    Returns: (best_text, best_conf, (y0,y1), debug_dict)
    """
    dbg = {}
    yband = (0, plate_bgr.shape[0])

    roi = plate_bgr.copy()
    if try_char_band:
        roi, yband, band_dbg = crop_to_char_band(roi, debug=True)
        dbg["band"] = band_dbg

    # Enhance
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    bin1, bin2 = _binarize_both(gray)

    candidates = [
        (roi, "roi"),
        (_ensure_3ch(bin1), "bin1"),
        (_ensure_3ch(bin2), "bin2"),
    ]

    best_text, best_conf, best_src = "", 0.0, None
    ocr_runs = []

    for img, tag in candidates:
        results = reader.readtext(
            img,
            allowlist=ALLOWED,
            detail=1,
            # A little stricter on linking low-confidence regions
            text_threshold=0.7,
            low_text=0.4,
            link_threshold=0.4,
            decoder=decoder,
        )
        # results: list of [bbox, text, conf]
        if not results:
            ocr_runs.append({"tag": tag, "raw": []})
            continue

        raw_text = "".join([r[1] for r in results])
        conf = float(np.mean([r[2] for r in results]))
        cleaned = clean_plate_text(raw_text)

        # Prefer strings that look like plates and are longer
        match = PLATE_RE.search(cleaned)
        candidate_text = match.group(0) if match else cleaned

        # Score: confidence + small bonus for length in [min_len, 8]
        length_bonus = min(max(len(candidate_text) - (min_len - 1), 0), 4) * 0.02
        score = conf + length_bonus

        ocr_runs.append({
            "tag": tag, "raw_text": raw_text, "cleaned": cleaned,
            "candidate": candidate_text, "conf": conf, "score": score
        })

        if score > best_conf:
            best_conf = score
            best_text = candidate_text
            best_src = tag

    dbg["ocr_runs"] = ocr_runs
    dbg["best_src"] = best_src

    if save_debug_dir:
        Path(save_debug_dir).mkdir(parents=True, exist_ok=True)
        base = os.path.join(save_debug_dir, "debug")
        # Save band projection visualization
        if "band" in dbg:
            proj = (dbg["band"]["proj"] * 255).astype(np.uint8)
            proj_img = np.repeat(proj.reshape(-1, 1), 200, axis=1)
            cv2.imwrite(base + "_proj.png", proj_img)
        cv2.imwrite(base + f"_roi.png", roi)
        cv2.imwrite(base + f"_bin1.png", _ensure_3ch(bin1))
        cv2.imwrite(base + f"_bin2.png", _ensure_3ch(bin2))

    return best_text, float(best_conf), yband, dbg

