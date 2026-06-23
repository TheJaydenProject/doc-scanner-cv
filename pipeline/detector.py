import cv2
import numpy as np


def detect_text_regions(binarized: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """
    Runs MSER on the binarized document image and draws bounding boxes
    around detected text regions.

    Returns:
        annotated: BGR image with bounding boxes drawn (for display)
        regions:   list of (x, y, w, h) tuples for each detected region
    """
    mser = cv2.MSER_create()
    # detectRegions returns raw pixel sets; bboxes gives rectangles directly.
    _, bboxes = mser.detectRegions(binarized)

    annotated = cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)

    detections: list[tuple[int, int, int, int]] = []

    for (x, y, w, h) in bboxes:
        # Minimum area of 100px² filters single-pixel MSER artifacts reliably
        # without rejecting small punctuation on typical document resolutions.
        if w * h < 100:
            continue
        # Ruled/feint lines surface as very wide, very thin MSER regions —
        # real text characters and words don't reach a 15:1 width-to-height
        # ratio, and rule lines are only a few pixels tall.
        if w / max(h, 1) >= 15 or h <= 8:
            continue
        detections.append((int(x), int(y), int(w), int(h)))
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (37, 99, 235), 1)

    return annotated, detections
