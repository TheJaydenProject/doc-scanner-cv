import math
import threading

import cv2
import numpy as np

# FSRCNN super-resolution lifts small text toward the OCR gate. The .pb files are
# the canonical models OpenCV's dnn_superres expects (Saafke/FSRCNN_Tensorflow),
# committed under models/ like doc_classifier.onnx (each ~40KB, far too small to
# justify the build-time fetch the 93MB EasyOCR weights use). Only x2 and x3 are
# kept: x2 is enough once text is past half the gate, x3 covers the smaller end.
_MODEL_PATHS = {
    2: "models/fsrcnn/FSRCNN_x2.pb",
    3: "models/fsrcnn/FSRCNN_x3.pb",
}
_MIN_FACTOR = 2
_MAX_FACTOR = 3

# The median text height (px) we upscale toward. Mirrors documents.MIN_TEXT_HEIGHT_PX
# (the resolution gate); not imported to avoid a circular import. The factor is the
# smallest that lifts the measured median to at least this, clamped to {x2, x3}.
TARGET_TEXT_HEIGHT_PX = 30

# Output grows by factor**2, so a large warped page would risk the CPU-only VPS.
# Above this input size we skip upscaling and OCR at native resolution: a large
# image already gives EasyOCR's recognizer enough pixels, and the 30px gate is a
# heuristic, not a hard requirement. Tune after observing VPS memory.
MAX_UPSCALE_INPUT_MP = 1.5

# Upscale method. "fsrcnn" runs the learned SR network (sharper glyphs); "cubic"
# uses plain cv2.resize. We are using "fsrcnn" for better accuracy on small text.
UPSCALE_METHOD = "fsrcnn"

# One network per factor, built once and reused (mirrors ocr._get_reader /
# classifier._get_session). Populated and read only under _sr_lock.
_sr_by_factor: dict[int, "cv2.dnn_superres.DnnSuperResImpl"] = {}
# dnn_superres networks are not guaranteed safe for concurrent upsample() calls
# across the 3-thread scan pool, so serialize. SR is fast and fires only on
# small-text scans, so contention is negligible.
_sr_lock = threading.Lock()


def _choose_factor(median_text_height: float | None) -> int:
    """
    Smallest FSRCNN factor that lifts median_text_height to TARGET_TEXT_HEIGHT_PX,
    clamped to the {x2, x3} models we ship. Unknown/non-positive heights fall back
    to the strongest factor. Picking x2 when it suffices roughly halves the enlarged
    image's pixel count vs. always-x3, which is where the post-upscale OCR cost lives.
    """
    if not median_text_height or median_text_height <= 0:
        return _MAX_FACTOR
    needed = math.ceil(TARGET_TEXT_HEIGHT_PX / median_text_height)
    return max(_MIN_FACTOR, min(_MAX_FACTOR, needed))


def _get_sr(factor: int) -> "cv2.dnn_superres.DnnSuperResImpl":
    sr = _sr_by_factor.get(factor)
    if sr is None:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(_MODEL_PATHS[factor])  # raises if the .pb is missing
        sr.setModel("fsrcnn", factor)
        _sr_by_factor[factor] = sr
    return sr


def upscale(image: np.ndarray, median_text_height: float | None = None) -> np.ndarray:
    """
    Super-resolve a BGR scan whose text is too small to OCR reliably, using the
    smallest factor that lifts median_text_height to the OCR gate (see
    _choose_factor). When median_text_height is None the strongest factor is used.

    Returns the image unchanged when it already exceeds MAX_UPSCALE_INPUT_MP — the
    factor**2 growth would risk the VPS, and a large image needs no enhancement.
    FSRCNN is thread-safe via a lock; cubic resize is reentrant and needs none.
    """
    h, w = image.shape[:2]
    if (h * w) / 1_000_000 > MAX_UPSCALE_INPUT_MP:
        return image
    factor = _choose_factor(median_text_height)
    if UPSCALE_METHOD == "cubic":
        return cv2.resize(image, (w * factor, h * factor), interpolation=cv2.INTER_CUBIC)
    with _sr_lock:
        return _get_sr(factor).upsample(image)


if __name__ == "__main__":
    # Self-check: adaptive factor picks x2/x3 by measured height; oversized input
    # is returned untouched.
    small = np.zeros((40, 60, 3), dtype=np.uint8)
    assert _choose_factor(20) == 2 and _choose_factor(10) == 3 and _choose_factor(5) == 3
    assert _choose_factor(None) == _MAX_FACTOR

    out = upscale(small, median_text_height=20)  # 20*2 = 40 >= 30 -> x2
    assert out.shape[:2] == (80, 120), out.shape
    out = upscale(small, median_text_height=10)  # 10*3 = 30 -> x3
    assert out.shape[:2] == (120, 180), out.shape

    side = int((MAX_UPSCALE_INPUT_MP * 1_000_000) ** 0.5) + 50
    big = np.zeros((side, side, 3), dtype=np.uint8)
    assert upscale(big, median_text_height=10).shape == big.shape

    print("superres self-check OK")
