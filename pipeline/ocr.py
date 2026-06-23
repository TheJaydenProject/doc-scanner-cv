import re

import easyocr
import numpy as np
import torch

# Single Gunicorn worker (-w 1) with a 3-thread ThreadPoolExecutor. Left
# unconfigured, torch's intra-op pool defaults to os.cpu_count() (4 on the VPS),
# so 3 concurrent readtext() calls would contend for 12 threads on 4 cores. Pin
# to 1 so the pool, not torch, owns the parallelism.
torch.set_num_threads(1)

_MODEL_DIR = "models/easyocr"

# Reader is built once per process and reused across requests. Construction
# imports the torch graph and loads two model files (~93 MB total), far too
# expensive to repeat per scan. Mirrors classifier._get_session().
_reader: easyocr.Reader | None = None


def _get_reader() -> easyocr.Reader:
    """
    Lazily build the EasyOCR reader.

    download_enabled=False: the detector/recognizer weights must already be on
    disk under _MODEL_DIR. They are fetched at build time by
    scripts/fetch_ocr_weights.sh (see README "OCR model weights"), not
    committed to git. If they are missing, the Reader raises instead of
    silently phoning home, which keeps the offline guarantee.
    """
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(
            ["en"],
            gpu=torch.cuda.is_available(),
            model_storage_directory=_MODEL_DIR,
            download_enabled=False,
            verbose=False,
        )
    return _reader


# Characters that are never part of real text content but commonly appear as
# ruled-line artifacts in OCR output. EasyOCR reads a ruled line running under
# handwriting as a run of underscores/dashes, so this post-pass still earns its
# keep even though OCR now consumes the non-binarized image.
_NOISE_CHARS = re.compile(r"^[—–‒\-~_\s]+$")
_LEADING_TRAILING_NOISE = re.compile(
    r"^[—–‒\-~_\s]+|[—–‒\-~_\s]+$"
)
# Multiple underscores between word characters are a ruled-line artifact:
# EasyOCR fills the gap a notebook rule leaves between words with underscores;
# collapse them to a single space.
_UNDERSCORE_GAP = re.compile(r"(?<=\w)_+(?=\w)")


def _clean_ocr_text(text: str) -> str:
    """
    Strips ruled-line noise artifacts from raw OCR output.

    Three passes:
    1. Strip leading/trailing em-dashes, hyphens, tildes, and underscores
       (surviving ruled-line fragments read as these characters).
    2. Drop lines with no alphanumeric character at all: pure artifact lines.
    3. Collapse intra-word underscore runs to a single space, inserted where a
       ruled-line gap interrupted a word.
    """
    cleaned: list[str] = []
    for line in text.splitlines():
        line = _LEADING_TRAILING_NOISE.sub("", line).strip()
        if not line or _NOISE_CHARS.match(line):
            continue
        line = _UNDERSCORE_GAP.sub(" ", line)
        cleaned.append(line)
    return "\n".join(cleaned)


def _lines_in_reading_order(detections: list) -> list[str]:
    """
    Reassemble EasyOCR's per-phrase boxes into reading order.

    readtext() returns boxes in the detector's own order, which is not reliably
    top-to-bottom (sign-off lines and right-shifted fragments come out
    scrambled). Group boxes into text rows by vertical centre, then order each
    row left-to-right. The row threshold scales with the median box height so it
    adapts to image resolution.

    Each detection is (bbox, text, confidence); bbox is four (x, y) points.
    """
    if not detections:
        return []

    boxes = []
    for bbox, text, _conf in detections:
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        boxes.append(
            {
                "y_center": (min(ys) + max(ys)) / 2,
                "height": max(ys) - min(ys),
                "left": min(xs),
                "text": text,
            }
        )

    heights = sorted(b["height"] for b in boxes)
    median_h = heights[len(heights) // 2]
    row_tol = max(median_h * 0.6, 1)

    boxes.sort(key=lambda b: b["y_center"])
    rows: list[dict] = []
    for b in boxes:
        # Compare to the row's top box (rows stay sorted ascending), which bounds
        # a row's vertical span to row_tol and stops slow drift from merging two
        # adjacent lines into one.
        if rows and b["y_center"] - rows[-1]["ref"] <= row_tol:
            rows[-1]["items"].append(b)
        else:
            rows.append({"ref": b["y_center"], "items": [b]})

    lines = []
    for row in rows:
        row["items"].sort(key=lambda b: b["left"])
        lines.append(" ".join(b["text"] for b in row["items"]))
    return lines


def extract_text(image: np.ndarray) -> str:
    """
    Run OCR on the warped, non-binarized document image: EasyOCR's CNNs read
    the natural grayscale gradient far better than a hard-binarized page.
    Returns cleaned text in reading order.
    """
    # batch_size > 1 groups detected text-region crops into fewer recognizer
    # forward passes instead of one per region; on a multi-line scan this cuts
    # wall-clock time noticeably even single-threaded (torch.set_num_threads(1)
    # above caps intra-op threads, not how many crops one pass can hold).
    detections = _get_reader().readtext(image, detail=1, batch_size=8)
    lines = _lines_in_reading_order(detections)
    return _clean_ocr_text("\n".join(lines))
