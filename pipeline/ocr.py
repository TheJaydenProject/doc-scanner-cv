import os
import shutil

import pytesseract
import numpy as np

# ponytail: Windows-only fallback for the default Tesseract install location.
# Only activates when tesseract isn't already on PATH. Linux/Docker unaffected.
if os.name == "nt" and not shutil.which("tesseract"):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_text(binarized_image: np.ndarray) -> str:
    # --psm 6 treats the input as a single uniform block of text, which fits
    # a tightly-cropped, binarized document page better than Tesseract's default
    # automatic page segmentation.
    raw_text = pytesseract.image_to_string(binarized_image, config="--psm 6")
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    return "\n".join(lines)
