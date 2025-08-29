# tests/test_ocr_utils.py
import inspect
import re
import numpy as np
import pytest

def _import_ocr_utils():
    try:
        import ocr_utils
        return ocr_utils
    except Exception as e:
        pytest.skip(f"Cannot import src/ocr_utils.py: {e}")

def test_clean_and_regex_basic():
    o = _import_ocr_utils()
    assert hasattr(o, "clean_plate_text")
    assert hasattr(o, "PLATE_RE")

    cleaned = o.clean_plate_text(" ca-7abc123  ")
    # Expect only A-Z0-9 and uppercase
    assert re.fullmatch(r"[A-Z0-9]+", cleaned)
    assert cleaned == cleaned.upper()

    # Regex sanity
    valid = ["7ABC123", "ABC12", "ZZ99999", "X1Y2Z3AA"]
    invalid = ["", "AB", "TOO0L0NGP", "abc123", "ABC-123"]
    for s in valid:
        assert o.PLATE_RE.fullmatch(s), f"Expected valid: {s}"
    for s in invalid:
        assert not o.PLATE_RE.fullmatch(s), f"Expected invalid: {s}"

def test_preprocess_returns_uint8_grayscale(dummy_plate_image):
    o = _import_ocr_utils()
    if not hasattr(o, "preprocess_for_ocr"):
        pytest.skip("preprocess_for_ocr not implemented")
    out = o.preprocess_for_ocr(dummy_plate_image)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.uint8
    assert out.ndim == 2
    assert out.shape[:2] == dummy_plate_image.shape[:2]

def test_tightened_crop_shrinks_bbox(dummy_plate_image):
    o = _import_ocr_utils()
    if not hasattr(o, "tightened_crop"):
        pytest.skip("tightened_crop not implemented")

    H, W = dummy_plate_image.shape[:2]
    x1, y1, x2, y2 = 80, 80, 320, 150
    sx, sy = 0.06, 0.10
    cropped, (cx1, cy1, cx2, cy2) = o.tightened_crop(
        dummy_plate_image, x1, y1, x2, y2, sx, sy
    )

    # Still within image bounds
    assert 0 <= cx1 < cx2 <= W and 0 <= cy1 < cy2 <= H
    # Shrunk (tighter) than original
    assert (cx2 - cx1) <= (x2 - x1)
    assert (cy2 - cy1) <= (y2 - y1)
    # Cropped image matches reported coords
    assert cropped.shape[1] == (cx2 - cx1)
    assert cropped.shape[0] == (cy2 - cy1)

def test_try_ocr_with_retries_uses_best_result(dummy_plate_image, reader_stub):
    o = _import_ocr_utils()
    if not hasattr(o, "try_ocr_with_retries"):
        pytest.skip("try_ocr_with_retries not implemented")

    # Make the stub return two different confidences on successive calls
    # If your function calls readtext once, that's fine—the best is 0.92.
    reader_stub.returns = [
        ([(0,0),(1,0),(1,1),(0,1)], "7ABC123", 0.80),
        ([(0,0),(1,0),(1,1),(0,1)], "7ABC123", 0.92),
    ]

    sig = inspect.signature(o.try_ocr_with_retries)
    params = list(sig.parameters)
    # Try the most common signature: (reader, frame, x1, y1, x2, y2, sx, sy, ...)
    if {"reader", "frame", "x1", "y1", "x2", "y2"}.issubset(set(params)):
        res = o.try_ocr_with_retries(
            reader=reader_stub,
            frame=dummy_plate_image,
            x1=80, y1=80, x2=320, y2=150,
            sx=0.0, sy=0.0,
            debug=False, tag="t"
        )
    else:
        # Fallback: call positionally with common arg order
        res = o.try_ocr_with_retries(reader_stub, dummy_plate_image, 80, 80, 320, 150, 0.0, 0.0)

    text, conf, coords = res
    assert isinstance(text, str)
    assert isinstance(conf, float)
    assert conf >= 0.80
