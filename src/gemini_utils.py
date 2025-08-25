# src/gemini_utils.py
import os, re, json, cv2
import numpy as np

PLATE_RE = re.compile(r"[A-Z0-9]{2,8}")  # adjust for your locale (len/pattern)

def _clean_plate(s: str) -> str:
    s = (s or "").upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s

def _to_jpeg_bytes(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr)
    if not ok:
        raise RuntimeError("cv2.imencode('.jpg', ...) failed")
    return buf.tobytes()

def init_gemini(model_name: str = None):
    """
    Returns a configured Gemini GenerativeModel instance.
    Requires env var GOOGLE_API_KEY set.
    """
    import google.generativeai as genai  # lazy import
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY environment variable.")
    genai.configure(api_key=api_key)
    model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    return genai.GenerativeModel(model_name)

def read_plate_text(model, bgr_roi: np.ndarray, save_debug_dir: str = None,
                    system_hint: str = None, temperature: float = 0.0):
    """
    Calls Gemini on the ROI to read the license plate text.

    Returns:
      plate (str): cleaned A–Z0–9 text ("" if unreadable)
      conf (float): model-reported or heuristic confidence in [0,1]
      band (tuple[int,int]): (y0,y1) character band within ROI (we return full ROI)
      dbg (dict): raw fields for debugging
    """
    import google.generativeai as genai  # lazy import
    img_bytes = _to_jpeg_bytes(bgr_roi)

    prompt = system_hint or (
        "Read the license plate text in this image. "
        "Respond ONLY with compact JSON like "
        '{"plate":"ABC123","confidence":0.92}. '
        "Rules: Uppercase A–Z and digits 0–9 only (no spaces/dashes). "
        "If unreadable, return plate=\"\" and confidence=0.0."
    )

    response = model.generate_content(
        [{"text": prompt}, {"mime_type": "image/jpeg", "data": img_bytes}],
        generation_config={"temperature": temperature, "max_output_tokens": 64},
    )

    raw_text = getattr(response, "text", "") or ""
    if not raw_text:
        try:
            raw_text = response.candidates[0].content.parts[0].text
        except Exception:
            raw_text = ""

    # Try to parse JSON from the response
    plate, conf = "", 0.0
    try:
        jstart = raw_text.find("{")
        jend = raw_text.rfind("}") + 1
        payload = raw_text[jstart:jend] if jstart != -1 and jend > jstart else raw_text
        data = json.loads(payload)
        plate = _clean_plate(data.get("plate", ""))
        conf = float(data.get("confidence", 0.0) or 0.0)
    except Exception:
        # Fallback: extract any A–Z0–9 token from free text
        plate = _clean_plate(raw_text)

    # Post-validate
    m = PLATE_RE.search(plate)
    if m:
        plate = m.group(0)
    else:
        plate = ""

    # Heuristic confidence if model didn't provide one
    if plate and (conf <= 0.0 or conf > 1.0):
        conf = 0.7  # simple default; tune as you like

    # Save debug artifacts
    if save_debug_dir:
        os.makedirs(save_debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_debug_dir, "roi.jpg"), bgr_roi)
        with open(os.path.join(save_debug_dir, "gemini_raw.txt"), "w") as f:
            f.write(raw_text)

    band = (0, bgr_roi.shape[0])  # we don't get a char-band from Gemini; use full ROI
    return plate, float(conf), band, {"gemini_text": raw_text}
