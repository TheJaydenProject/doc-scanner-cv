from pipeline.scanner import run_pipeline, binarize_printed, binarize_handwritten
from pipeline.detector import detect_text_regions
from pipeline.classifier import classify_document
from pipeline.ocr import extract_text
import cv2
import os

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_PATH = "static/examples/note1_original.jpg"
if not os.path.exists(IMAGE_PATH):
    raise SystemExit(f"SKIP: {IMAGE_PATH} not found — drop a photo there first.")

with open(IMAGE_PATH, "rb") as f:
    image_bytes = f.read()

# Step 1 — scanner: perspective correction, inset crop, no binarization
# (skipped entirely for flat digital documents with no real boundary)
clean_image, warped = run_pipeline(image_bytes)
cv2.imwrite(os.path.join(OUTPUT_DIR, "output_clean.png"), clean_image)
print(f"Step 1: {OUTPUT_DIR}/output_clean.png written. (warped: {warped})")

# Step 2 — classifier runs on the clean image, before any binarization
result = classify_document(clean_image)
print(f"Step 2: {result['label']} (confidence: {result['confidence']:.2%}, source: {result['source']})")

# Step 3 — branch binarization by predicted type
if result["label"] == "printed":
    binarized = binarize_printed(clean_image)
else:
    binarized = binarize_handwritten(clean_image)
cv2.imwrite(os.path.join(OUTPUT_DIR, "output.png"), binarized)
print(f"Step 3: {OUTPUT_DIR}/output.png written.")

# Step 4 — detector + OCR
annotated, detections = detect_text_regions(binarized)
cv2.imwrite(os.path.join(OUTPUT_DIR, "output_annotated.png"), annotated)
print(f"Step 4: {len(detections)} text regions detected. {OUTPUT_DIR}/output_annotated.png written.")

text = extract_text(binarized)
print(f"Step 4: extracted {len(text)} characters.")
