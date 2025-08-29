# tests/test_utils.py
import csv
import time
import pathlib
import pytest
import numpy as np

def _import_utils():
    try:
        import utils
        return utils
    except Exception as e:
        pytest.skip(f"Cannot import src/utils.py: {e}")

def test_timestamp_is_string():
    u = _import_utils()
    if not hasattr(u, "timestamp"):
        pytest.skip("timestamp not implemented")
    t = u.timestamp()
    assert isinstance(t, str) and len(t) >= 8

def test_save_csv_row_writes_line(tmp_csv):
    u = _import_utils()
    if not hasattr(u, "save_csv_row"):
        pytest.skip("save_csv_row not implemented")
    # Minimal columns—adjust/add fields as your function expects
    row = {"frame": 1, "text": "7ABC123", "conf": 0.91, "x1": 80, "y1": 80, "x2": 320, "y2": 150}
    u.save_csv_row(tmp_csv, row)
    assert tmp_csv.exists()
    with tmp_csv.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["text"] == "7ABC123"

def test_annotate_and_fps_do_not_crash(monkeypatch):
    u = _import_utils()
    for fn in ("annotate_text", "draw_fps"):
        if not hasattr(u, fn):
            pytest.skip(f"{fn} not implemented")
    try:
        import cv2
    except Exception:
        pytest.skip("OpenCV not installed")

    img = np.zeros((200, 300, 3), dtype=np.uint8)
    out = u.annotate_text(img.copy(), "7ABC123", (90, 120))
    assert isinstance(out, np.ndarray) and out.shape == img.shape

    # draw_fps should accept a float-ish fps and return an image without error
    out2 = u.draw_fps(img.copy(), fps=29.97)
    assert isinstance(out2, np.ndarray) and out2.shape == img.shape
