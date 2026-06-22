import numpy as np

from pipeline.superres import MAX_UPSCALE_INPUT_MP, _FACTOR, upscale


def test_upscale_multiplies_dimensions():
    """A small BGR image is enlarged by exactly the FSRCNN factor on both axes."""
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    out = upscale(img)
    assert out.shape[0] == 40 * _FACTOR
    assert out.shape[1] == 60 * _FACTOR
    assert out.shape[2] == 3


def test_upscale_skips_oversized_image():
    """
    Above MAX_UPSCALE_INPUT_MP the memory guard returns the image untouched
    rather than risking the factor**2 blow-up on a CPU-only VPS.
    """
    side = int((MAX_UPSCALE_INPUT_MP * 1_000_000) ** 0.5) + 50
    img = np.zeros((side, side, 3), dtype=np.uint8)
    out = upscale(img)
    assert out.shape == img.shape
