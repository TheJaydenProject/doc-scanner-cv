import numpy as np
import cv2
import pytest
from pipeline.detector import detect_text_regions
from pipeline.classifier import classify_document

_CLASS_LABELS = {"printed", "handwritten", "mixed"}


def test_detect_text_regions_returns_annotated_and_list(document_image_bytes):
    from pipeline.scanner import run_pipeline, binarize_printed
    binarized = binarize_printed(run_pipeline(document_image_bytes))
    annotated, detections = detect_text_regions(binarized)

    # annotated must be a 3-channel image (BGR for display).
    assert len(annotated.shape) == 3
    assert annotated.shape[2] == 3

    # detections must be a list of 4-tuples.
    assert isinstance(detections, list)
    for region in detections:
        assert len(region) == 4


def test_detect_text_regions_filters_noise():
    """
    A uniform white image has no stable text regions — MSER should return
    an empty or near-empty list after the minimum area filter.
    """
    blank = np.ones((500, 500), dtype=np.uint8) * 255
    _, detections = detect_text_regions(blank)
    # Allow a small number of false positives from MSER on uniform input.
    assert len(detections) < 10


def test_classify_document_returns_valid_label():
    grey = np.ones((500, 500), dtype=np.uint8) * 200
    result = classify_document(grey)
    assert result["label"] in _CLASS_LABELS
    assert 0.0 <= result["confidence"] <= 1.0


def test_classify_document_handles_bgr_input():
    """Classifier must accept both grayscale and BGR without raising."""
    bgr = np.ones((300, 300, 3), dtype=np.uint8) * 128
    result = classify_document(bgr)
    assert result["label"] in _CLASS_LABELS
