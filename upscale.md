# Super-Resolution for Low-Resolution Scans

## Context & Goal

The resolution gate in `api/documents.py` measures the median height of MSER text-box detections to ensure scans are readable before passing them to the expensive OCR engine. Previously, any scan with a median text height below `MIN_TEXT_HEIGHT_PX` (15px) was rejected outright. 

To improve the success rate on low-resolution uploads, we introduced a **super-resolution enhancement stage**. Images with small text (median height < 30px) are now upscaled prior to OCR. To prevent upscaling noise or entirely unrecoverable text from wasting compute, a hard floor of 8px is enforced, below which the scan is still rejected.

```text
[Raw Warped Scan] -> [Binarize + Remove Lines + MSER Detect] -> [Resolution Gate]
                                                                       |
              +--------------------------------------------------------+--------------------------------------------------------+
              v (median height >= 30px)                                                                  v (8px <= median < 30px)
        [Direct OCR on warped image]                                                               [Upscale warped image]
              |                                                                                          |
              v                                                                                          v
        [Extracted text]                                                                       [OCR on upscaled image]
                                                                                                         |
                                                                                                         v (median < 8px)
                                                                                               [Reject: unrecoverable]
```

---

## Architectural Decisions

### 1. Bounded Whole-Page Upscale vs. Per-Crop Upscale

The textbook approach to super-resolution before OCR is to detect text lines, slice them into crops, upscale only the crops, and feed them individually to recognition. This was rejected for two reasons:

1. **MSER boxes are not recognition units:** MSER returns blob/character-level boxes used only for the resolution gate. EasyOCR's `readtext()` uses its own CRAFT detector for line-level context. Feeding MSER blob-crops into recognition strips the CRNN of context and degrades output quality.
2. **EasyOCR API constraints:** EasyOCR lacks a clean hook to intercept and upscale between its internal `detect()` (CRAFT) and `recognize()` (CRNN) stages. Building one requires a major rewrite of `pipeline/ocr.py` and abandoning existing reading-order reassembly logic.

**Decision:** Upscale the *entire* warped `clean_image` when the gate fires, then run `extract_text()` unchanged. CRAFT detects accurate line boxes on the larger image, and CRNN reads sharper glyphs.

### 2. Memory Bound (`MAX_UPSCALE_INPUT_MP`)

Upscaling multiplies the pixel count by `factor²`. A large, dense page upscaled 3x would grow 9x in memory, risking CPU exhaustion on low-spec VPS environments.

**Decision:** If the input image already exceeds `MAX_UPSCALE_INPUT_MP` (1.5 megapixels), upscaling is skipped. A large image already provides EasyOCR with sufficient pixels; the 30px gate is a heuristic, not a strict requirement for dense, high-resolution scans.

### 3. Adaptive Factor Selection

Instead of a fixed 3x upscale (which forces a 9x memory penalty even for 20px text), the pipeline dynamically selects the smallest factor required to lift the median text height to the 30px target.

- `_choose_factor(median_text_height)` clamps the factor between 2x and 3x.
- A scan with 16px text receives a 2x upscale (4x memory) instead of 3x (9x memory), halving the OCR processing cost for the most common threshold cases.

### 4. Upscale Method: Cubic vs. FSRCNN

Initially, the OpenCV `dnn_superres` module and an FSRCNN (`Fast Super-Resolution Convolutional Neural Network`) model were used to upscale the images.

**Decision:** After A/B testing, the upscaling method was reverted back to `FSRCNN` (Fast Super-Resolution Convolutional Neural Network) because the CNN forward pass provides better OCR accuracy with sharper glyph reconstruction. The `UPSCALE_METHOD` toggle in `pipeline/superres.py` defaults to `"fsrcnn"`. The cubic `cv2.resize` method is retained as a fallback option if a faster but softer scaling is required in the future.

---

## Implementation Details

### Pipeline Integration (`api/documents.py`)

The resolution gate evaluates the median text height:

1. **< 8px (`UPSCALE_FLOOR_PX`)**: Rejected immediately (`RESOLUTION_TOO_LOW`). Upscaling cannot reconstruct legible glyphs from sub-8px text.
2. **8px - 30px (`MIN_TEXT_HEIGHT_PX`)**: The image is routed to `upscale()`.
3. **> 30px**: Passes directly to OCR.

### The Upscaler (`pipeline/superres.py`)

- Exposes `upscale(image: np.ndarray, median_text_height: float | None = None) -> np.ndarray`.
- Skips images > 1.5MP.
- Applies the adaptive factor based on the measured median text height.
- Utilizes the FSRCNN models for sharp upscaling.

### Testing

Coverage is provided in `tests/test_superres.py`:
- `test_adaptive_factor_selection`: Verifies the correct 2x or 3x factor is chosen based on input height.
- `test_upscale_multiplies_dimensions`: Confirms the FSRCNN upscale applies the correct adaptive dimensions.
- `test_upscale_skips_large_images`: Validates the `MAX_UPSCALE_INPUT_MP` memory guard. 

*(Note: Test coverage includes both `"cubic"` and `"fsrcnn"` fallback paths).*
```
