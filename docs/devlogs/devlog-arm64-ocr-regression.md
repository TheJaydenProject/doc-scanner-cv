# doc-scanner-cv: PyTorch ARM64 Performance Regression Bug Report

**Project:** doc-scanner-cv  
**Date:** 24 June 2026  
**Environment:** Linux ARM64 (Oracle Cloud VPS), `torch==2.2.1`, `easyocr==1.7.2`  
**Files affected:** `backend/requirements.txt`, `README.md`

---

## Overview

A severe performance regression was observed after deploying the application to an ARM64 (aarch64) production VPS. OCR scans that normally complete in 5-10 seconds were taking over 195 seconds (over 3 minutes) per image, and the resulting text output consisted of garbled, fragmented characters despite the perspective transform generating a clean, accurate crop of the document.

---

## Bug: PyTorch CPU Wheel Optimization on ARM64

**Symptom**

When a scan was submitted to the deployed VPS, the Celery worker took an abnormally long time to process the image. The worker logs showed:

```
[INFO/ForkPoolWorker-2] Job 041effa2-e310-4a06-bfb6-deaed3cf1b7b timing: upscale=2.3s ocr=195.0s
```

Despite EasyOCR identifying 581 text regions, the final OCR output consisted of just a few broken characters:

```
constituent engagement This report is intended to be a starting point
00510 2 0 .
All opportunities and issues of the conceptual equality help
```

**What was expected**

PyTorch should leverage ARM64 CPU optimizations (NEON/SVE kernels) to perform inference on the CRAFT and CRNN models in ~5-15 seconds, returning an accurate reading of the printed text.

**What actually happened**

In commit `f51cb0d`, the PyTorch version was downgraded from `2.12.1+cpu` to `2.2.1` to supposedly fix "ARM compatibility", and the `--extra-index-url` pointing to PyTorch's official CPU wheel repository was removed from `requirements.txt`.

Without the official index URL, pip defaulted to fetching the generic `torch==2.2.1` wheel from PyPI. PyTorch 2.2.1 (released early 2024) had notoriously poor CPU optimizations for ARM64 architectures. The lack of proper hardware acceleration forced the worker to perform inference using slow, unoptimized instruction sets, causing the 195-second execution time. Furthermore, the math operations in this unoptimized generic build produced severe numerical instability during EasyOCR's CRNN decoding step, resulting in garbled text.

**Root cause**

```text
# backend/requirements.txt — before fix
opencv-contrib-python-headless==4.9.0.80
easyocr==1.7.2
torch==2.2.1
torchvision==0.17.1
numpy<2
```

1. **Incorrect Version:** Downgrading to PyTorch 2.2.1 removed modern ARM64 optimizations.
2. **Missing Index:** Removing `--extra-index-url https://download.pytorch.org/whl/cpu` caused pip to pull the generic PyPI wheel instead of the specific CPU-optimized build compiled by the PyTorch team.

**Fix**

Reverted `backend/requirements.txt` to the exact versions that were known to work (`2.12.1+cpu`), and critically, restored the `--extra-index-url` flag so pip pulls the correct optimized wheel for the architecture.

```text
# backend/requirements.txt — after fix
--extra-index-url https://download.pytorch.org/whl/cpu

opencv-contrib-python-headless==4.13.0.92
easyocr==1.7.2
torch==2.12.1+cpu
torchvision==0.27.1+cpu
numpy==2.4.4
```

*Note: The `+cpu` specifier explicitly instructs pip to grab the CPU-only optimized wheel from the provided index URL, which is significantly smaller and optimized for server deployments without CUDA hardware.*

---

## Impact on Build Plan

The Dockerfile's layer caching logic for `torch` (which reads the `USE_GPU` flag) was unaffected, as it correctly reads `backend/requirements.txt` during the build process.

The `README.md` "Built With" table was updated to reflect the restored versions.

---

## Change Summary

| File | Change |
|---|---|
| `backend/requirements.txt` | Restored `--extra-index-url https://download.pytorch.org/whl/cpu`. Reverted `torch` to `2.12.1+cpu`, `torchvision` to `0.27.1+cpu`, `numpy` to `2.4.4`, and `opencv-contrib-python-headless` to `4.13.0.92`. |
| `README.md` | Updated version numbers in the Built With table. |
