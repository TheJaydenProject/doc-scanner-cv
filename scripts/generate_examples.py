import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.classifier import classify_document
from pipeline.scanner import (
    ContourNotFoundError,
    binarize_handwritten,
    binarize_printed,
    run_pipeline,
)

EXAMPLES_DIR = "static/examples"
INPUT_EXTENSIONS = (".jpg", ".png")


def find_original(n):
    """Return the original image path for note `n`, trying each supported extension."""
    for ext in INPUT_EXTENSIONS:
        candidate = os.path.join(EXAMPLES_DIR, f"note{n}_original{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


n = 1
while True:
    input_path = find_original(n)
    if input_path is None:
        break

    output_path = os.path.join(EXAMPLES_DIR, f"note{n}_scanned.png")

    with open(input_path, "rb") as f:
        image_bytes = f.read()

    try:
        clean_image, _ = run_pipeline(image_bytes)
        doc_type = classify_document(clean_image)
        if doc_type["label"] == "printed":
            result = binarize_printed(clean_image)
        else:
            result = binarize_handwritten(clean_image)
        cv2.imwrite(output_path, result)
        print(f"OK: {input_path} -> {output_path} ({doc_type['label']})")
    except ContourNotFoundError as e:
        print(f"FAIL: {input_path} — {e}")

    n += 1

if n == 1:
    print(
        f"No files found. Drop note1_original.jpg or note1_original.png "
        f"into {EXAMPLES_DIR}/ and retry."
    )
