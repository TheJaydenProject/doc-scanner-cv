import numpy as np
import cv2
import pytest
from pipeline.scanner import (
    run_pipeline,
    binarize_printed,
    binarize_handwritten,
    remove_ruled_lines,
    _quad_area_ratio,
    _edge_density_inside_vs_outside,
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


def test_remove_ruled_lines_erases_thick_page_edge_band():
    # A scan-shadow band along the page edge is far thicker than max_thickness
    # but spans the full height; the page-spanning-border branch erases it
    # where a fixed thickness cap alone would keep it.
    binary = np.full((400, 300), 255, dtype=np.uint8)
    cv2.rectangle(binary, (278, 0), (298, 399), 0, -1)  # 21px wide, full height
    cleaned = remove_ruled_lines(binary)
    assert cleaned[200, 288] == 255


def test_remove_ruled_lines_erases_collinear_margin_stub():
    # Three full-width rules confirm the page is ruled. A short fragment left
    # behind on one of those rows (where text broke the line apart) is too
    # short to be a rule on its own, but its alignment with a confirmed rule
    # gives it away and it must be erased.
    binary = np.full((200, 400), 255, dtype=np.uint8)
    cv2.line(binary, (5, 40), (395, 40), 0, thickness=2)
    cv2.line(binary, (5, 120), (395, 120), 0, thickness=2)
    cv2.line(binary, (5, 80), (150, 80), 0, thickness=2)  # broken, still a strong rule
    cv2.line(binary, (250, 80), (280, 80), 0, thickness=2)  # collinear stub, mid-page
    cleaned = remove_ruled_lines(binary)
    assert cleaned[80, 265] == 255


def test_remove_ruled_lines_keeps_margin_strokes_without_rules():
    # With no ruled lines on the page, the margin-stub heuristic must stay
    # disarmed so thin glyph strokes near the page edge survive untouched.
    binary = np.full((200, 400), 255, dtype=np.uint8)
    cv2.line(binary, (5, 100), (18, 100), 0, thickness=2)  # 13px, below the length floor
    cleaned = remove_ruled_lines(binary)
    assert np.sum(cleaned == 0) == np.sum(binary == 0)


def test_busy_report_skips_warp(busy_report_image_bytes):
    # Multiple comparably-sized internal blocks on a flat page with no real
    # boundary — the main failure mode this fix targets.  Every 4-point quad
    # has a similarly-sized rival, so Pass 1 rejects them all and the pipeline
    # returns the full, unwarped frame.
    result, warped = run_pipeline(busy_report_image_bytes)
    assert warped is False
    assert result.shape[:2] == (800, 1000)


def test_busy_report_with_boundary_warps_to_outer_rect(
    busy_report_with_real_boundary_image_bytes,
):
    # Same internal blocks, but a genuine outer boundary exists on a dark
    # background.  The outer boundary quad has no comparably-sized rival
    # (it's far larger than any internal block), so it passes the heuristics
    # and the pipeline warps to it.
    result, warped = run_pipeline(busy_report_with_real_boundary_image_bytes)
    assert warped is True
    # Warped result should be roughly the outer rectangle's dimensions
    # (approximately 740x560 minus inset crop), not the full 800x600 frame
    # or a small internal block crop.
    h, w = result.shape[:2]
    assert h > 400, f"warped height {h} unexpectedly small"
    assert w > 500, f"warped width {w} unexpectedly small"


def test_single_secondary_block_still_warps(single_secondary_block_image_bytes):
    # A real document with one embedded table/photo — the internal block is
    # clearly less than 50% of the outer boundary's area, so the outer
    # boundary passes the size-gap check and the document warps normally.
    result, warped = run_pipeline(single_secondary_block_image_bytes)
    assert warped is True
    h, w = result.shape[:2]
    assert h > 400, f"warped height {h} unexpectedly small"
    assert w > 500, f"warped width {w} unexpectedly small"


def test_edge_density_inside_vs_outside():
    # Hand-built edge map: 100x100, top half (rows 0-49) is all edge pixels,
    # bottom half (rows 50-99) is empty.  Quad covers the top-left quadrant
    # (50x50 = 2500 pixels, all edges).  Inside density should be 1.0;
    # outside density should be 2500 / 7500 ≈ 0.333 (the other 2500 edge
    # pixels in the top-right quadrant, spread over 7500 outside pixels).
    edges = np.zeros((100, 100), dtype=np.uint8)
    edges[0:50, :] = 255  # top half = edge pixels

    quad = np.array([[0, 0], [49, 0], [49, 49], [0, 49]]).reshape(4, 1, 2)
    inside_d, outside_d = _edge_density_inside_vs_outside(edges, quad)

    assert inside_d == pytest.approx(1.0, abs=0.05)
    assert outside_d == pytest.approx(0.333, abs=0.05)

