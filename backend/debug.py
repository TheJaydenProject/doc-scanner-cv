import cv2
import numpy as np
import os
from pipeline.scanner import preprocess

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open("test_images/sample_note1.jpg", "rb") as f:
    img_bytes = f.read()

np_array = np.frombuffer(img_bytes, np.uint8)
image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
print(f"Image shape: {image.shape}")

blurred = preprocess(img_bytes)
edges = cv2.Canny(blurred, 30, 100)

contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
print(f"Total contours: {len(contours)}")
print(f"Top 10 areas: {[int(cv2.contourArea(c)) for c in contours[:10]]}")

# Draw top 10 contours on image
debug_img = image.copy()
cv2.drawContours(debug_img, contours[:10], -1, (0, 255, 0), 2)
cv2.imwrite(os.path.join(OUTPUT_DIR, "debug_contours.png"), debug_img)
cv2.imwrite(os.path.join(OUTPUT_DIR, "debug_edges.png"), edges)
print(f"Saved debug_contours.png and debug_edges.png to {OUTPUT_DIR}/")
