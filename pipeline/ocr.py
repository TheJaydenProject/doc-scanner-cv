import pytesseract
import numpy as np


def extract_text(binarized_image: np.ndarray) -> str:
    raw_text = pytesseract.image_to_string(binarized_image)
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    return "\n".join(lines)
