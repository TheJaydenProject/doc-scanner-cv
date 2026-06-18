import sys
import os
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.scanner import (
    run_pipeline,
    binarize_printed,
    binarize_handwritten,
    ContourNotFoundError,
)
from pipeline.classifier import classify_document

EXAMPLES_DIR = "static/examples"

n = 1
while True:
    input_path = os.path.join(EXAMPLES_DIR, f"note{n}_original.jpg")
    if not os.path.exists(input_path):
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
    print(f"No files found. Drop note1_original.jpg into {EXAMPLES_DIR}/ and retry.")
