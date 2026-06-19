import os
import re
import shutil

import numpy as np
import pytesseract

# ponytail: Windows-only fallback for the default Tesseract install location.
# Only activates when tesseract isn't already on PATH. Linux/Docker unaffected.
if os.name == "nt" and not shutil.which("tesseract"):
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )


# Characters that are never part of real text content but commonly appear as
# ruled-line or binarization artifacts in Tesseract output.
_NOISE_CHARS = re.compile(r"^[\u2014\u2013\u2012\-~_\s]+$")
_LEADING_TRAILING_NOISE = re.compile(
    r"^[\u2014\u2013\u2012\-~_\s]+|[\u2014\u2013\u2012\-~_\s]+$"
)
# Multiple consecutive underscores between word characters are a Tesseract
# artifact caused by residual ruled-line pixels that the morphological removal
# missed; collapse them to a single space.
_UNDERSCORE_GAP = re.compile(r"(?<=\w)_+(?=\w)")


def _clean_ocr_text(text: str) -> str:
    """
    Post-processes raw Tesseract output to remove ruled-line noise artifacts.

    Three passes:
    1. Strip leading/trailing em-dashes, hyphens, tildes, and underscores
       (Tesseract reads surviving ruled-line fragments as these characters).
    2. Drop lines that contain no alphanumeric character at all — these are
       pure artifact lines with no real content.
    3. Collapse intra-word underscore runs to a single space — Tesseract
       inserts underscores where a ruled-line gap interrupted a word.
    """
    cleaned: list[str] = []
    for line in text.splitlines():
        line = _LEADING_TRAILING_NOISE.sub("", line).strip()
        if not line or _NOISE_CHARS.match(line):
            continue
        line = _UNDERSCORE_GAP.sub(" ", line)
        cleaned.append(line)
    return "\n".join(cleaned)


def extract_text(binarized_image: np.ndarray) -> str:
    # --psm 3 (fully automatic page segmentation) lets Tesseract detect
    # paragraphs, indentation, and isolated lines (date, subject, sign-off)
    # on its own. Layout complexity is independent of handwritten vs.
    # printed, so a single mode serves both document types.
    raw_text = pytesseract.image_to_string(binarized_image, config="--psm 3")
    return _clean_ocr_text(raw_text)
