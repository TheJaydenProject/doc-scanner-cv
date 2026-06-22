import threading

import cv2
import numpy as np

# FSRCNN x3 lifts ~10px text toward the ~30px the OCR gate targets. The .pb is
# the canonical model OpenCV's dnn_superres expects (Saafke/FSRCNN_Tensorflow),
# committed under models/ like doc_classifier.onnx (it's ~40KB, far too small to
# justify the build-time fetch the 93MB EasyOCR weights use).
_MODEL_PATH = "models/fsrcnn/FSRCNN_x3.pb"
_FACTOR = 3

# Output grows by _FACTOR**2, so a large warped page would risk the CPU-only VPS.
# Above this input size we skip upscaling and OCR at native resolution: a large
# image already gives EasyOCR's recognizer enough pixels, and the 30px gate is a
# heuristic, not a hard requirement. Tune after observing VPS memory.
MAX_UPSCALE_INPUT_MP = 1.5

# Built once per process and reused, mirroring ocr._get_reader / classifier._get_session.
_sr: cv2.dnn_superres.DnnSuperResImpl | None = None
# dnn_superres networks are not guaranteed safe for concurrent upsample() calls
# across the 3-thread scan pool, so serialize. SR is fast and fires only on
# small-text scans, so contention is negligible.
_sr_lock = threading.Lock()


def _get_sr() -> cv2.dnn_superres.DnnSuperResImpl:
    global _sr
    if _sr is None:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(_MODEL_PATH)  # raises if the .pb is missing
        sr.setModel("fsrcnn", _FACTOR)
        _sr = sr
    return _sr


def upscale(image: np.ndarray) -> np.ndarray:
    """
    FSRCNN x3 super-resolution of a BGR image, for scans whose text is too small
    to OCR reliably at native resolution.

    Returns the image unchanged when it already exceeds MAX_UPSCALE_INPUT_MP — the
    factor**2 growth would risk the VPS, and a large image needs no enhancement.
    Thread-safe: dnn_superres is not guaranteed safe for concurrent upsample().
    """
    h, w = image.shape[:2]
    if (h * w) / 1_000_000 > MAX_UPSCALE_INPUT_MP:
        return image
    with _sr_lock:
        return _get_sr().upsample(image)


if __name__ == "__main__":
    # Self-check: x3 multiplies dimensions; oversized input is returned untouched.
    small = np.zeros((40, 60, 3), dtype=np.uint8)
    out = upscale(small)
    assert out.shape[:2] == (40 * _FACTOR, 60 * _FACTOR), out.shape

    side = int((MAX_UPSCALE_INPUT_MP * 1_000_000) ** 0.5) + 50
    big = np.zeros((side, side, 3), dtype=np.uint8)
    assert upscale(big).shape == big.shape

    print("superres self-check OK")
