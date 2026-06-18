import sys
import os
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.scanner import run_pipeline, ContourNotFoundError

EXAMPLES = [
    ("static/examples/note1_original.jpg", "static/examples/note1_scanned.png"),
    ("static/examples/note2_original.jpg", "static/examples/note2_scanned.png"),
]

for input_path, output_path in EXAMPLES:
    if not os.path.exists(input_path):
        print(f"SKIP: {input_path} not found")
        continue

    with open(input_path, "rb") as f:
        image_bytes = f.read()

    try:
        result = run_pipeline(image_bytes)
        cv2.imwrite(output_path, result)
        print(f"OK: {input_path} -> {output_path}")
    except ContourNotFoundError as e:
        print(f"FAIL: {input_path} — {e}")
