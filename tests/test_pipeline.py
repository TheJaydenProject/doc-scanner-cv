import numpy as np
import cv2
import pytest
from pipeline.scanner import run_pipeline, binarize, ContourNotFoundError


def test_blank_image_returns_binarized_fallback(blank_image_bytes):
    # No contour found — pipeline falls back to full-frame binarization.
    result = run_pipeline(blank_image_bytes)
    assert len(result.shape) == 2
    assert set(np.unique(result)).issubset({0, 255})


def test_corrupt_bytes_raises_contour_error():
    with pytest.raises(ContourNotFoundError):
        run_pipeline(b"not an image")


def test_binarize_output_is_single_channel():
    colour = np.ones((300, 200, 3), dtype=np.uint8) * 128
    result = binarize(colour)
    assert len(result.shape) == 2


def test_binarize_output_only_contains_binary_values():
    colour = np.ones((300, 200, 3), dtype=np.uint8) * 128
    result = binarize(colour)
    unique_values = set(np.unique(result))
    assert unique_values.issubset({0, 255})
