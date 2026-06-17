# doc-scanner-cv: CV Pipeline Bug Report

**Project:** doc-scanner-cv  
**Date:** 18 June 2026  
**Environment:** Windows 11, Python 3.x, `onnxruntime==1.24.4`, `torch` (latest), `opencv-python==4.13.0.92`  
**Files affected:** `scripts/export_classifier.py`, `pipeline/scanner.py`

---

## Overview

Two sequential bugs were encountered during the initial CV pipeline setup — before any Flask routes were wired up. Both surfaced while manually testing the pipeline via `scratch_test.py`. Resolving the first revealed the second.

---

## Bug 1: ONNX Runtime C API Crash on Model Load

**Symptom**

The script crashed during classifier initialisation with:

```
Failed to convert the model to the target version 11 using the ONNX C API.
```

**What was expected**

`scripts/export_classifier.py` would export `doc_classifier.onnx` and `pipeline/classifier.py` would load it via ONNX Runtime without error.

**What actually happened**

ONNX Runtime rejected the exported model at load time. The `.onnx` file was produced without error by `torch.onnx.export`, but the resulting graph was incompatible with the target opset.

**Root cause**

`scripts/export_classifier.py` had `opset_version=11` hardcoded. The installed version of PyTorch emits MobileNetV2 graph operators that cannot be legally expressed in opset 11. ONNX Runtime attempted a downgrade conversion via the C API and failed — the operators have no valid opset 11 representation.

```python
# scripts/export_classifier.py — before fix
torch.onnx.export(
    model,
    dummy_input,
    "models/doc_classifier.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11,   # too old for current PyTorch MobileNetV2 operators
)
```

**Fix**

Updated `opset_version` to `14`. Opset 14 natively supports the modern PyTorch operators without any conversion step. Anything from 14 to 18 works — 14 is the minimum that resolves the incompatibility.

```python
# scripts/export_classifier.py — after fix
torch.onnx.export(
    model,
    dummy_input,
    "models/doc_classifier.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=14,
)
```

Deleted `models/doc_classifier.onnx` and re-ran the export script before retesting:

```powershell
Remove-Item models\doc_classifier.onnx
python scripts/export_classifier.py
```

Also update the `biome.json` schema reference if you bump the Biome version later — same class of silent version mismatch.

---

## Bug 2: MSER Bounding Boxes Blown Out by Binarization Noise

**Symptom**

`output_annotated.png` showed massive, chaotic bounding boxes covering the edges of the notebook and background texture. No individual text characters were isolated. `output.png` (the binarized image before detection) was heavily distorted with salt-and-pepper noise across the paper surface.

**What was expected**

`pipeline/detector.py` would draw tight bounding boxes around text glyphs on a clean binarized image.

**What actually happened**

MSER treated the paper grain and shadow noise as valid stable regions. Every patch of texture produced a detection, causing boxes to blow out across the entire image instead of isolating characters.

**Root cause**

`pipeline/scanner.py` `binarize()` lacked a smoothing pass before thresholding and used parameters far too sensitive for photographed note paper:

```python
# pipeline/scanner.py — before fix
def binarize(warped: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2   # block size 11, constant 2
    )
```

`adaptiveThreshold` with a block size of `11` evaluates each pixel against an 11×11 neighbourhood. On note paper this window is too small — it picks up individual paper fibres and micro-shadows as foreground. The constant `C=2` provides almost no suppression of low-contrast noise. MSER received a noisy binary image and had no way to distinguish paper grain from text strokes.

**Fix**

Two changes applied together:

Added a `GaussianBlur` pass before thresholding to suppress paper grain before the threshold step sees it:

```python
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
```

Increased the `adaptiveThreshold` block size to `51` and constant to `15`. The larger window forces the algorithm to evaluate each pixel against a much wider local region, making it ignore minor shadows and only trigger on high-contrast text strokes.

```python
# pipeline/scanner.py — after fix
def binarize(warped: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51, 15  # wider window + stronger suppression
    )
```

Re-ran `scratch_test.py` after saving. `output.png` was clean with no salt-and-pepper noise. `output_annotated.png` showed tight bounding boxes around individual text characters.

---

## Impact on Build Plan

Both fixes apply to Phase 1.5 of the build plan. Update `scripts/export_classifier.py` and `pipeline/scanner.py` as above before attempting any further pipeline testing.

The `test_binarize_output_only_contains_binary_values` test in `tests/test_pipeline.py` is unaffected — it tests pixel value range, not spatial quality. No test changes required.

---

## Change Summary

| File | Change |
|---|---|
| `scripts/export_classifier.py` | `opset_version=11` → `opset_version=14` |
| `models/doc_classifier.onnx` | Deleted and re-exported after opset fix |
| `pipeline/scanner.py` | `binarize()`: added `GaussianBlur(gray, (5, 5), 0)` pre-pass; `adaptiveThreshold` block size `11` → `51`, constant `2` → `15` |
