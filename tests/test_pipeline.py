import numpy as np
import cv2
import pytest
from pipeline.scanner import run_pipeline, binarize, ContourNotFoundError


def test_blank_image_raises_contour_error(blank_image_bytes):
    with pytest.raises(ContourNotFoundError):
        run_pipeline(blank_image_bytes)


def test_binarize_output_is_single_channel():
    colour = np.ones((300, 200, 3), dtype=np.uint8) * 128
    result = binarize(colour)
    assert len(result.shape) == 2


def test_binarize_output_only_contains_binary_values():
    colour = np.ones((300, 200, 3), dtype=np.uint8) * 128
    result = binarize(colour)
    unique_values = set(np.unique(result))
    assert unique_values.issubset({0, 255})
