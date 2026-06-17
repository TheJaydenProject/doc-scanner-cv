from pipeline.scanner import run_pipeline
from pipeline.detector import detect_text_regions
from pipeline.classifier import classify_document
import cv2
import os

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open("test_images/sample_note1.jpg", "rb") as f:
    image_bytes = f.read()

# Phase 1 — scanner
binarized = run_pipeline(image_bytes)
cv2.imwrite(os.path.join(OUTPUT_DIR, "output.png"), binarized)
print(f"Phase 1: {OUTPUT_DIR}/output.png written.")

# Phase 1.5a — detector
annotated, detections = detect_text_regions(binarized)
cv2.imwrite(os.path.join(OUTPUT_DIR, "output_annotated.png"), annotated)
print(f"Phase 1.5a: {len(detections)} text regions detected. {OUTPUT_DIR}/output_annotated.png written.")

# Phase 1.5b — classifier
result = classify_document(binarized)
print(f"Phase 1.5b: {result['label']} (confidence: {result['confidence']:.2%})")
