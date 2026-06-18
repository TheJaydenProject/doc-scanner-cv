import os
import shutil

import pytesseract
import numpy as np

# ponytail: Windows-only fallback for the default Tesseract install location.
# Only activates when tesseract isn't already on PATH. Linux/Docker unaffected.
if os.name == "nt" and not shutil.which("tesseract"):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_text(binarized_image: np.ndarray, doc_type: str = "handwritten") -> str:
    # --psm 6 treats the input as a single uniform block of text, which fits
    # a tightly-cropped handwritten note. Printed pages can span multiple
    # paragraphs or columns, so --psm 3 (fully automatic page segmentation)
    # performs significantly better there.
    psm = "3" if doc_type == "printed" else "6"
    raw_text = pytesseract.image_to_string(binarized_image, config=f"--psm {psm}")
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    return "\n".join(lines)
