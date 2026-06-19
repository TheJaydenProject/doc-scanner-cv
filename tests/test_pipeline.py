import numpy as np
import cv2
import pytest
from pipeline.scanner import (
    run_pipeline,
    binarize_printed,
    binarize_handwritten,
    remove_ruled_lines,
    _quad_area_ratio,
    ContourNotFoundError,
)


def test_blank_image_returns_unbinarized_fallback(blank_image_bytes):
    # No contour found — pipeline falls back to the raw, unbinarized BGR frame.
    result, warped = run_pipeline(blank_image_bytes)
    assert len(result.shape) == 3
    assert result.shape[2] == 3
    assert warped is False


def test_flat_document_skips_warp(flat_document_image_bytes):
    # A contour is found, but it traces almost the entire frame — that's a flat
    # digital document, not a real boundary, so the warp (and its inset crop)
    # must be skipped entirely.
    result, warped = run_pipeline(flat_document_image_bytes)
    assert warped is False
    assert result.shape[:2] == (600, 800)


def test_small_stray_contour_skips_warp(small_blob_image_bytes):
    # A contour is found, but it's an implausibly small fraction of the
    # frame — a stray text blob or noise artifact, not a document boundary.
    result, warped = run_pipeline(small_blob_image_bytes)
    assert warped is False
    assert result.shape[:2] == (600, 800)


def test_quad_area_ratio_computes_fraction_of_frame():
    full_frame_quad = np.array([[0, 0], [99, 0], [99, 99], [0, 99]]).reshape(4, 1, 2)
    assert _quad_area_ratio(full_frame_quad, (100, 100)) == pytest.approx(0.98, abs=0.02)

    half_frame_quad = np.array([[0, 0], [49, 0], [49, 99], [0, 99]]).reshape(4, 1, 2)
    assert _quad_area_ratio(half_frame_quad, (100, 100)) == pytest.approx(0.5, abs=0.02)


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


def test_remove_ruled_lines_erases_tilted_ruled_line():
    # A continuous line tilted a few degrees off horizontal — minAreaRect
    # must measure its true thickness along its own axis, not be fooled
    # by the larger axis-aligned bounding box the tilt produces.
    binary = np.full((60, 400), 255, dtype=np.uint8)
    cv2.line(binary, (5, 30), (395, 25), 0, thickness=2)
    cleaned = remove_ruled_lines(binary)
    assert cleaned[27, 200] == 255


def test_remove_ruled_lines_preserves_disconnected_text_strokes():
    # Short, gapped dashes simulate handwritten glyphs roughly on one row.
    # Each dash is far shorter than the page-relative length floor, so it
    # must never be mistaken for a ruled-line segment.
    binary = np.full((60, 400), 255, dtype=np.uint8)
    for x in range(20, 380, 12):
        cv2.line(binary, (x, 30), (x + 6, 30), 0, thickness=2)
    cleaned = remove_ruled_lines(binary)
    assert np.sum(cleaned == 0) == np.sum(binary == 0)


def test_remove_ruled_lines_preserves_isolated_letter_stroke():
    # A lone tall, thin stroke (e.g. a lowercase "l" or "I" not joined to
    # neighbouring letters) looks identical to a short vertical rule by
    # thickness/aspect alone — only its short length relative to the page
    # should save it from erasure.
    binary = np.full((400, 60), 255, dtype=np.uint8)
    cv2.line(binary, (30, 100), (30, 118), 0, thickness=3)
    cleaned = remove_ruled_lines(binary)
    assert np.sum(cleaned == 0) == np.sum(binary == 0)


def test_remove_ruled_lines_erases_vertical_margin_border():
    # A near-full-height vertical border (e.g. a photographed page edge)
    # must be erased even though remove_horizontal_lines never looked for
    # vertical structure at all.
    binary = np.full((400, 300), 255, dtype=np.uint8)
    cv2.line(binary, (290, 5), (290, 395), 0, thickness=2)
    cleaned = remove_ruled_lines(binary)
    assert cleaned[200, 290] == 255
