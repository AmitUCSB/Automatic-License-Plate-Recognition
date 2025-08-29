# tests/conftest.py
import os, sys, pathlib
import pytest
import numpy as np

# Ensure we can import from src/
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

@pytest.fixture
def dummy_plate_image():
    """Synthetic 240x400 RGB image with a 'plate' rectangle & text."""
    try:
        import cv2
    except Exception:
        pytest.skip("OpenCV not installed; skip image-dependent tests")

    img = np.full((240, 400, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (80, 80), (320, 150), (220, 220, 220), -1)  # plate bg
    cv2.putText(img, "7ABC123", (95, 135), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    return img

class _ReaderStub:
    """EasyOCR-like stub: readtext returns [(bbox, text, conf), ...]."""
    def __init__(self, returns=None):
        self.returns = returns or [([(0,0),(1,0),(1,1),(0,1)], "7ABC123", 0.85)]

    def readtext(self, image):
        return list(self.returns)

@pytest.fixture
def reader_stub():
    return _ReaderStub()

@pytest.fixture
def tmp_csv(tmp_path):
    return tmp_path / "out.csv"
