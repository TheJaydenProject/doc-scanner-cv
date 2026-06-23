import numpy as np

from pipeline import superres
from pipeline.superres import (
    MAX_UPSCALE_INPUT_MP,
    TARGET_TEXT_HEIGHT_PX,
    _choose_factor,
    upscale,
)


def test_choose_factor_lifts_text_toward_target():
    """Smallest factor that reaches the gate: x2 once text is past half the gate,
    x3 below, clamped to the shipped models for very small or unknown heights."""
    assert _choose_factor(TARGET_TEXT_HEIGHT_PX / 2) == 2  # exactly reaches target
    assert _choose_factor(20) == 2  # 20*2 = 40 >= 30
    assert _choose_factor(10) == 3  # 10*3 = 30 >= 30
    assert _choose_factor(4) == 3  # would want x8, clamped to the max we have
    assert _choose_factor(None) == 3  # unknown height -> strongest factor


def test_upscale_uses_adaptive_factor():
    """The enlarged dimensions follow the factor _choose_factor picked."""
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    assert upscale(img, median_text_height=20).shape[:2] == (80, 120)  # x2
    assert upscale(img, median_text_height=10).shape[:2] == (120, 180)  # x3


def test_upscale_skips_oversized_image():
    """
    Above MAX_UPSCALE_INPUT_MP the memory guard returns the image untouched
    rather than risking the factor**2 blow-up on a CPU-only VPS.
    """
    side = int((MAX_UPSCALE_INPUT_MP * 1_000_000) ** 0.5) + 50
    img = np.zeros((side, side, 3), dtype=np.uint8)
    assert upscale(img, median_text_height=10).shape == img.shape


def test_cubic_method_honours_factor(monkeypatch):
    """The cubic A/B path enlarges by the same adaptive factor as FSRCNN."""
    monkeypatch.setattr(superres, "UPSCALE_METHOD", "cubic")
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    assert upscale(img, median_text_height=20).shape[:2] == (80, 120)
