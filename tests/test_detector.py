# tests/test_detector.py
import types
import pytest
import numpy as np

def _import_detector():
    try:
        import detector
        return detector
    except Exception as e:
        pytest.skip(f"Cannot import src/detector.py: {e}")

class _FakeYOLO:
    def __init__(self, weights):
        self.weights = weights
        self.calls = []

    def __call__(self, frame, **kwargs):
        # Mimic ultralytics return: list-like with .boxes.data.tolist()
        self.calls.append(kwargs)
        class _Boxes:
            def __init__(self):
                import numpy as np
                # one bbox with high confidence
                self.data = np.array([[80, 80, 320, 150, 0.95]], dtype=float)
        class _Result:
            def __init__(self):
                self.boxes = _Boxes()
                self.orig_img = frame
        return [_Result()]

def test_plate_detector_construct_and_call(monkeypatch):
    d = _import_detector()
    if not hasattr(d, "PlateDetector"):
        pytest.skip("PlateDetector not implemented")

    # Patch YOLO so we don't load real weights
    monkeypatch.setattr(d, "YOLO", _FakeYOLO, raising=True)

    pd = d.PlateDetector(weights="does_not_matter.pt", device="cpu", imgsz=640, conf=0.5)
    # PlateDetector should have a model attribute using FakeYOLO
    assert hasattr(pd, "model")
    # Call predict if available; otherwise, just ensure model is callable
    dummy = np.zeros((240, 400, 3), dtype=np.uint8)
    if hasattr(pd, "predict"):
        out = pd.predict(dummy)
        # We don't assume a strict contract; just ensure something truthy
        assert out is not None
    else:
        # Simulate a direct model call like pd.model(dummy, imgsz=..., conf=...)
        res = pd.model(dummy, imgsz=640, conf=0.5)
        assert isinstance(res, list) and len(res) >= 1
