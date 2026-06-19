import os
import shutil
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.classifier import classify_document
from pipeline.detector import detect_text_regions
from pipeline.scanner import (
    ContourNotFoundError,
    binarize_handwritten,
    binarize_printed,
    remove_ruled_lines,
    run_pipeline,
)

EXAMPLES_DIR = "static/examples"
INPUT_EXTENSIONS = (".jpg", ".png")
TARGET_FILES = ["note1", "note3"]


def find_original(base_name: str) -> str | None:
    """Return the original image path for `base_name`, trying each supported extension."""
    for ext in INPUT_EXTENSIONS:
        candidate = os.path.join(EXAMPLES_DIR, f"{base_name}_original{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


def generate_four_stage_examples() -> None:
    for base_name in TARGET_FILES:
        input_path = find_original(base_name)
        if input_path is None:
            print(f"SKIP: Original for {base_name} not found.")
            continue

        ext = os.path.splitext(input_path)[1]

        raw_path = os.path.join(EXAMPLES_DIR, f"{base_name}_raw{ext}")
        shutil.copy2(input_path, raw_path)

        with open(input_path, "rb") as f:
            image_bytes = f.read()

        try:
            clean_image, _ = run_pipeline(image_bytes)
            cv2.imwrite(os.path.join(EXAMPLES_DIR, f"{base_name}_warped.png"), clean_image)

            doc_type = classify_document(clean_image)

            if doc_type["label"] == "printed":
                binarized = binarize_printed(clean_image)
            else:
                binarized = binarize_handwritten(clean_image)

            # Matches api/documents.py: line removal runs unconditionally on
            # the binarized image, regardless of document type.
            cleaned = remove_ruled_lines(binarized)
            cv2.imwrite(os.path.join(EXAMPLES_DIR, f"{base_name}_binarized.png"), cleaned)

            annotated, _ = detect_text_regions(cleaned)
            cv2.imwrite(os.path.join(EXAMPLES_DIR, f"{base_name}_detected.png"), annotated)

            print(f"OK: Processed {base_name} ({doc_type['label']})")

        except ContourNotFoundError as e:
            print(f"FAIL: {input_path} — {e}")


if __name__ == "__main__":
    generate_four_stage_examples()
