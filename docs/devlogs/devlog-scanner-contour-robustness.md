# doc-scanner-cv: Scanner Robustness + ONNX Opset Follow-Up

**Project:** doc-scanner-cv  
**Date:** 18 June 2026  
**Environment:** Windows 11, Python 3.x, `onnxruntime==1.24.4`, `torch` (latest), `opencv-python==4.13.0.92`  
**Files affected:** `pipeline/scanner.py`, `scripts/export_classifier.py`, `models/doc_classifier.onnx`, `debug.py`

---

## Overview

Two issues surfaced after the Bug 1 and Bug 2 fixes documented in `devlog-cv-pipeline-bugs.md`. The first was a hard failure — `ContourNotFoundError` raised on real photographed note paper despite the document being clearly visible in the image. The second was a consequence of bumping the ONNX opset further: the exporter switched to an external data format, producing two model files instead of one.

Both were resolved before wiring any Flask routes.

---

## Bug 3: ContourNotFoundError on Real Photographed Note Paper

**Symptom**

`tests/scratch_test.py` raised `ContourNotFoundError` when run against `test_images/sample_note1.jpg`. The document was clearly visible in the photo with good contrast between the paper and the desk surface.

```
ContourNotFoundError: No four-point document contour found.
```

**What was expected**

`find_document_contour()` would locate the document boundary and return a 4-point contour for the perspective transform.

**What actually happened**

The function iterated over `contours[:5]` and found no valid quadrilateral. The pipeline exited with `ContourNotFoundError` before the perspective transform could run.

**Root cause**

Two compounding problems in the original implementation:

First, `Canny(blurred, 75, 200)` thresholds were too conservative for real note paper. The notebook binding, shadows along the spine, and gradual edge lighting produced faint boundary edges that Canny missed entirely with high thresholds. The resulting edge map was fragmented — the document perimeter appeared as several disconnected arcs rather than a closed contour.

Second, the search was limited to `contours[:5]`. On a photograph with a busy background (wood grain, notebook lines visible through the paper), smaller contours from background texture ranked above the actual document boundary by area. Even when the document contour existed, it was often outside the top 5.

```python
# pipeline/scanner.py — before fix
def find_document_contour(blurred: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(blurred, 75, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:5]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            return approx

    raise ContourNotFoundError("No four-point document contour found.")
```

The `preprocess()` blur kernel `(5, 5)` was also insufficient. On high-resolution phone camera photos, paper grain survived the blur and fed noise into the Canny step.

**Diagnosis**

`debug.py` was written at the project root to visualise exactly what Canny and `findContours` were seeing:

```python
# debug.py — diagnostic script, not production code
import cv2, numpy as np, os
from pipeline.scanner import preprocess

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open("test_images/sample_note1.jpg", "rb") as f:
    img_bytes = f.read()

np_array = np.frombuffer(img_bytes, np.uint8)
image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
blurred = preprocess(img_bytes)
edges = cv2.Canny(blurred, 30, 100)

contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
print(f"Top 10 areas: {[int(cv2.contourArea(c)) for c in contours[:10]]}")

debug_img = image.copy()
cv2.drawContours(debug_img, contours[:10], -1, (0, 255, 0), 2)
cv2.imwrite(os.path.join(OUTPUT_DIR, "debug_contours.png"), debug_img)
cv2.imwrite(os.path.join(OUTPUT_DIR, "debug_edges.png"), edges)
```

`debug_edges.png` confirmed the document boundary was present in the Canny output as disconnected arcs, not a closed shape. With `(75, 200)` the edges were too sparse to form contours at all. Dropping to `(30, 100)` produced a dense enough edge map that the boundary became the dominant contour by area.

**Fix**

Four changes applied together:

**1. Blur kernel in `preprocess()` widened from `(5, 5)` to `(7, 7)`** to suppress more high-frequency paper grain before Canny.

**2. Canny thresholds lowered from `(75, 200)` to `(30, 100)`** to catch faint boundary edges on low-contrast photo backgrounds.

**3. Morphological closing added after Canny** to bridge the gaps between disconnected boundary arcs. An 11×11 rectangular kernel joins edge segments that are within ~5px of each other — large enough to bridge typical notebook shadow gaps, small enough not to merge distinct document and background edges.

```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
```

**4. Single-pass search replaced with a three-pass fallback chain:**

- **Pass 1** — same quad-fitting logic, now over `contours[:8]` instead of `[:5]`
- **Pass 2** — convex hull of the largest contour approximated at 3% epsilon. Recovers when paper grain fragments the boundary into multiple arcs that independently fail the quad test but together form a valid rectangle
- **Pass 3** — `minAreaRect` over all non-zero edge pixels. Fires when the boundary never closes into a contour at all (very low contrast between paper and background surface)

```python
# pipeline/scanner.py — after fix
def find_document_contour(blurred: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if not contours:
        raise ContourNotFoundError("No contours found in image.")

    # Pass 1: largest clean 4-point quad.
    for contour in contours[:8]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            return approx

    # Pass 2: convex hull of the largest contour.
    hull = cv2.convexHull(contours[0])
    hull_approx = cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True)
    if len(hull_approx) == 4:
        return hull_approx

    # Pass 3: minAreaRect over all edge pixels.
    edge_points = cv2.findNonZero(edges)
    if edge_points is not None:
        rect = cv2.minAreaRect(edge_points)
        box = cv2.boxPoints(rect)
        return np.intp(box).reshape(4, 1, 2)

    raise ContourNotFoundError("No four-point document contour found.")
```

Re-ran `tests/scratch_test.py`. `output/output.png` showed a correctly warped, flat document on both test images.

**Side effect on `test_pipeline.py`:** The blank-image test `test_blank_image_raises_contour_error` still passes. A blank white image produces zero Canny edges, so `contours` is empty and the new early-exit guard fires (`"No contours found in image."`) before any pass runs. The exception type is unchanged — `ContourNotFoundError` — so `pytest.raises(ContourNotFoundError)` still catches it.

---

## Bug 4: ONNX Export at Opset 18 Produces External Data Format

**Symptom**

After bumping `opset_version` from `14` to `18` (a further increment past the `14` fix in the previous devlog to use a more modern, stable operator set), running `scripts/export_classifier.py` produced two output files instead of one:

```
models/doc_classifier.onnx
models/doc_classifier.onnx.data
```

The `.onnx` file was unexpectedly small (~1KB). The `.data` file held the actual tensor weights.

**What was expected**

A single self-contained `models/doc_classifier.onnx` file containing both the graph and the weights.

**What actually happened**

`torch.onnx.export` serialised the model in ONNX external data format — graph definition in `.onnx`, tensor data in a separate `.data` file co-located alongside it. The original plan and git commit assumed a single file.

**Root cause**

ONNX external data format is triggered by PyTorch's exporter when the model exceeds an internal size threshold, or when the opset and exporter version combination defaults to it. With opset 18 and the version of PyTorch installed, the exporter chose external data format for MobileNetV2 even though the total weight size is modest (~14MB). There is no flag in `torch.onnx.export` to force inline format at this opset level without switching to a lower opset.

**Fix**

No code change — the external data format works transparently with ONNX Runtime. `InferenceSession` resolves the `.data` file automatically from the `.onnx` path as long as both files are co-located:

```python
# pipeline/classifier.py — unchanged; ONNX Runtime handles the .data file automatically
_session = ort.InferenceSession("models/doc_classifier.onnx", providers=["CPUExecutionProvider"])
```

The required change was operational: both files must be committed to Git and must travel together. If only `.onnx` is present at runtime, ONNX Runtime raises `InvalidGraph` on session creation.

Updated `.gitignore` to confirm neither file is excluded. Updated `build-plan.md` Dockerfile note to acknowledge both files are copied by `COPY . .`.

Added error handling to the export script so export failures are visible rather than silent:

```python
# scripts/export_classifier.py — after fix
try:
    torch.onnx.export(
        model, dummy_input, "models/doc_classifier.onnx",
        input_names=["input"], output_names=["output"],
        opset_version=18,
    )
    print("Exported models/doc_classifier.onnx")
except Exception as e:
    print(f"Export failed: {e}")
```

Re-ran `tests/scratch_test.py`. Classifier loaded without error and returned a label and confidence score.

---

## Impact on Build Plan

`build-plan.md` has been updated to reflect all four changes:

- `preprocess()` blur kernel documented as `(7, 7)`
- `find_document_contour()` listing replaced with the full 3-pass implementation
- `binarize()` note references this devlog and the previous one together
- Export script updated to opset 18 with try/except
- Folder structure updated to show both `models/doc_classifier.onnx` and `models/doc_classifier.onnx.data`
- Canonical Build Order verify step for Phase 1.5b updated to expect two model files

The `test_blank_image_raises_contour_error` test in `tests/test_pipeline.py` is unaffected. No other test changes required.

---

## Change Summary

| File | Change |
|---|---|
| `pipeline/scanner.py` | `preprocess()`: blur kernel `(5, 5)` → `(7, 7)` |
| `pipeline/scanner.py` | `find_document_contour()`: Canny `(75, 200)` → `(30, 100)`; added morphological closing `(11, 11)`; early-exit guard; Pass 1 scans `[:8]`; Pass 2 convex-hull fallback; Pass 3 `minAreaRect` fallback |
| `scripts/export_classifier.py` | `opset_version=14` → `opset_version=18`; wrapped export in `try/except` |
| `models/doc_classifier.onnx` | Re-exported at opset 18 — graph only (small) |
| `models/doc_classifier.onnx.data` | New file — tensor weights split out by external data format exporter |
| `debug.py` | New diagnostic script (committed); used to develop the 3-pass approach |
