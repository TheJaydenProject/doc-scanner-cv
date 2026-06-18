import numpy as np
import cv2
import pytest
from pipeline.scanner import (
    run_pipeline,
    binarize_printed,
    binarize_handwritten,
    ContourNotFoundError,
)


def test_blank_image_returns_unbinarized_fallback(blank_image_bytes):
    # No contour found — pipeline falls back to the raw, unbinarized BGR frame.
    result = run_pipeline(blank_image_bytes)
    assert len(result.shape) == 3
    assert result.shape[2] == 3


def test_corrupt_bytes_raises_contour_error():
    with pytest.raises(ContourNotFoundError):
        run_pipeline(b"not an image")


def test_binarize_printed_output_is_single_channel():
    colour = np.ones((300, 200, 3), dtype=np.uint8) * 128
    result = binarize_printed(colour)
    assert len(result.shape) == 2


def test_binarize_printed_output_only_contains_binary_values():
    colour = np.ones((300, 200, 3), dtype=np.uint8) * 128
    result = binarize_printed(colour)
    unique_values = set(np.unique(result))
    assert unique_values.issubset({0, 255})


def test_binarize_handwritten_output_is_single_channel():
    colour = np.ones((300, 200, 3), dtype=np.uint8) * 128
    result = binarize_handwritten(colour)
    assert len(result.shape) == 2


def test_binarize_handwritten_output_only_contains_binary_values():
    colour = np.ones((300, 200, 3), dtype=np.uint8) * 128
    result = binarize_handwritten(colour)
    unique_values = set(np.unique(result))
    assert unique_values.issubset({0, 255})
