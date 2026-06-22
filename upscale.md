# upscale.md — FSRCNN Super-Resolution for Low-Resolution Scans

## Goal

Today the resolution gate in `api/documents.py` **rejects** a scan when the median
MSER text-box height is below `MIN_TEXT_HEIGHT_PX` (currently 15px). We want to
replace that rejection with a **super-resolution enhancement stage**: small-text
images are upscaled with FSRCNN and then OCR'd, instead of being turned away. The
gate threshold is then raised to **30px**, so everything under 30px is routed
through enhancement rather than rejected.

```
[Raw Warped Scan] -> [Binarize + Remove Lines + MSER Detect] -> [Resolution Gate]
                                                                       |
              +--------------------------------------------------------+--------------------------------------------------------+
              v (median height >= 30px)                                                                  v (median height < 30px)
        [Direct OCR on warped image]                                                          [FSRCNN upscale of warped image]
              |                                                                                          |
              v                                                                                          v
        [Extracted text]                                                                       [OCR on upscaled image]
                                                                                                         |
                                                                                                         v (median < hard floor)
                                                                                               [Reject: unrecoverable]
```

---

## Step 0 — Two decisions (CONFIRMED)

> **Status: both decisions below are approved.** v1 uses bounded whole-page upscale
> with a single fixed FSRCNN x3 model and a hard reject floor. Recorded here as the
> agreed design, not open questions.

### Decision A — Whole-page upscale (CONFIRMED) vs. per-crop upscale

Your reference strategy says: detect boxes, slice crops, upscale only the crops,
feed crops to EasyOCR recognition. That is the textbook SR-before-OCR pattern, but
**it does not fit this codebase**, for two reasons:

1. **MSER boxes are not recognition units.** `detect_text_regions()` (MSER) returns
   blob/character-level boxes used only for the resolution gate and the
   `detection_count` stat. EasyOCR's `readtext()` runs its *own* CRAFT detector to
   find text *lines*, then its CRNN recognizer reads each whole line *with
   language-model context*. Feeding MSER blob-crops into recognition skips CRAFT and
   strips the CRNN of the context it needs — output gets worse, not better.

2. **EasyOCR has no clean "detect → upscale crops → recognize" hook.** Doing crops
   *properly* means: `reader.detect()` (CRAFT) to get line boxes -> upscale each line
   crop -> `reader.recognize()` on the grouped lines. That is a meaningful rewrite of
   `pipeline/ocr.py` and abandons the existing reading-order reassembly logic.

**Recommendation: bounded whole-page upscale.** When the gate fires, upscale the
whole warped `clean_image` with FSRCNN, then run the *existing* `extract_text()`
unchanged. CRAFT then runs on a sharp, larger image and naturally produces good
line boxes; the CRNN reads sharper glyphs. We bound memory by capping the output
size (see Decision B). This reuses the entire current OCR path and is the smallest
correct change.

**Why not crops first:** the memory win of crops is real, but it costs an OCR
rewrite and fights EasyOCR's design. We take whole-page now and keep the
**proper** crop path (CRAFT line detection -> upscale line crops -> recognize) as a
documented upgrade for if/when the bounded whole-page proves too heavy on the VPS.

### Decision B — Memory bound

FSRCNN output is `factor² ×` the input pixel count. A large warped page upscaled
3× can blow up a low-spec CPU VPS. The gate only fires on *small-text* images, which
are *usually* low-resolution to begin with, but not always (dense small print on a
big photo). So we cap it:

- `MAX_UPSCALE_INPUT_MP` — if the warped image already exceeds this, **skip**
  upscaling and OCR at native resolution (a large image already gives the CRNN
  plenty of pixels; the 30px gate is a heuristic, not a hard requirement).
- Start with FSRCNN **x3** only (lifts ~10px text to ~30px, the target). Adaptive
  factor selection (x2/x3/x4 by measured height) is a documented refinement, not v1.

Concretely for v1: load `FSRCNN_x3.pb`; upscale only when
`input_megapixels <= MAX_UPSCALE_INPUT_MP` (suggest ~1.5 MP, so x3 output ~13.5 MP).
Tune after observing VPS behavior.

### Decision status
- [x] Bounded whole-page upscale (not per-crop) for v1. **Confirmed.** Rationale:
      MSER boxes are blob-level gate/stat detections, not the text *lines* EasyOCR's
      CRNN recognizes; feeding blob-crops to recognition strips the CRNN of context
      and makes output worse. Upscaling the whole (bounded) page lets CRAFT find good
      line boxes on a sharp image and reuses the existing `extract_text()` unchanged.
      The proper crop path (CRAFT line-detect -> upscale line crops -> `recognize()`)
      stays documented as the memory-driven upgrade only.
- [x] Single fixed FSRCNN x3 model for v1. **Confirmed.** Adaptive x2/x3/x4 is a
      documented refinement, not v1.
- [x] Hard reject floor below `UPSCALE_FLOOR_PX` (default 8px). **Confirmed.** SR
      cannot reconstruct legible glyphs from ~4px text (3x -> 12px is still mush), so
      sub-floor text still fails cleanly with the existing resolution message instead
      of producing garbage OCR. Tunable.
- [ ] `MAX_UPSCALE_INPUT_MP` starting value (default 1.5) — tune after observing the
      VPS; not blocking.

---

## Step 1 — Dependency change (the critical one)

`cv2.dnn_superres` lives in OpenCV **contrib**, which is **not** in the currently
pinned `opencv-python-headless`. You must swap to the contrib headless build. You
**cannot** install both — contrib-headless is a strict superset.

**`requirements.txt`:**
```diff
- opencv-python-headless==4.13.0.92
+ opencv-contrib-python-headless==4.13.0.92
```

Notes:
- **Version verified (checked against PyPI).** `opencv-contrib-python-headless==4.13.0.92`
  is published and is in fact the latest release; `opencv-python-headless==4.13.0.92`
  (the current pin) also exists. The two wheels are released in lockstep with identical
  version numbers, so this is a clean same-version swap — the lowest-risk form of the
  change, no version drift between the package you remove and the one you add.
- **No new system libraries.** contrib-headless needs the same `libgl1` +
  `libglib2.0-0` already in the Dockerfile. `dnn_superres` ships inside the wheel
  (it has been part of opencv_contrib since 4.1 and is present in 4.13).
- This is a superset swap, so existing OpenCV calls behave identically. The only
  cost is a slightly larger install/image and longer Docker build.

Smoke test after install (locally, in `venv`):
```python
import cv2; print(cv2.dnn_superres.DnnSuperResImpl_create())  # must not raise
```

---

## Step 2 — Acquire the FSRCNN model

The canonical model OpenCV's `dnn_superres` expects is the TensorFlow `.pb` from the
Saafke/FSRCNN repo referenced in OpenCV's own docs.

- File: `FSRCNN_x3.pb` (~40 KB)
- Source: `https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x3.pb`
  (verify this resolves; it is the standard OpenCV `dnn_superres` FSRCNN model)

**Recommendation: commit it to the repo** at `models/fsrcnn/FSRCNN_x3.pb`.

Rationale: it is tiny (~40 KB) and matches how `models/doc_classifier.onnx` is already
handled (small models committed; only the 93 MB EasyOCR weights are fetched at build
time by `scripts/fetch_ocr_weights.sh`). Committing removes a build-time network
dependency and means **zero** changes to the Dockerfile, CI, or the deploy scp list
(`models` is already in the deploy `source:` and `COPY . .`).

Nothing excludes it (already confirmed):
- `.dockerignore` excludes only `models/easyocr/` — `models/fsrcnn/` ships.
- `.gitignore` excludes only `models/easyocr/` and has no `*.pb` rule — the file is
  tracked normally with a plain `git add`.

(Alternative, for consistency with the EasyOCR convention: add the curl line to
`scripts/fetch_ocr_weights.sh` and don't commit. More moving parts; only worth it if
you object to binary blobs in git. Not recommended for a 40 KB file.)

---

## Step 3 — New module `pipeline/superres.py`

Mirror the existing singleton pattern (`ocr._get_reader`, `classifier._get_session`).
`dnn_superres` networks are not guaranteed thread-safe across the 3-thread pool, so
guard `upsample()` with a lock. SR is fast and fires only on small-text scans, so
lock contention is negligible.

```python
import threading
import cv2
import numpy as np

_MODEL_PATH = "models/fsrcnn/FSRCNN_x3.pb"
_FACTOR = 3

# Skip upscaling above this input size — factor**2 growth would risk the VPS.
# A large image already gives EasyOCR enough pixels; see upscale.md Decision B.
MAX_UPSCALE_INPUT_MP = 1.5

_sr: "cv2.dnn_superres.DnnSuperResImpl | None" = None
_sr_lock = threading.Lock()


def _get_sr():
    global _sr
    if _sr is None:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(_MODEL_PATH)        # raises if the .pb is missing
        sr.setModel("fsrcnn", _FACTOR)
        _sr = sr
    return _sr


def upscale(image: np.ndarray) -> np.ndarray:
    """
    FSRCNN x3 super-resolution of a BGR image. Returns the image unchanged if it
    already exceeds MAX_UPSCALE_INPUT_MP (memory guard). Thread-safe via a lock
    because dnn_superres is not guaranteed safe for concurrent upsample() calls.
    """
    h, w = image.shape[:2]
    if (h * w) / 1_000_000 > MAX_UPSCALE_INPUT_MP:
        return image
    with _sr_lock:
        return _get_sr().upsample(image)
```

Self-check (CLAUDE.md "one runnable check"): a tiny synthetic image upscales to
`factor ×` its dimensions — see Step 6.

Optionally pre-warm in `app.py` next to `_get_reader()`. Not required: the 40 KB
load is sub-millisecond, so the first small-text scan can pay it. Leave it lazy for
v1 unless you want symmetry.

---

## Step 4 — Wire it into the pipeline (`api/documents.py`)

Replace the reject branch in `_run_scan_job` (currently lines ~119-136). Keep a
**hard floor**: FSRCNN cannot invent legible detail from near-nothing, so text below
a few pixels should still be rejected rather than silently producing garbage.

**Constant changes (top of file):**
```diff
- MIN_TEXT_HEIGHT_PX = 15
+ # Below this median MSER text-box height, route the scan through FSRCNN
+ # super-resolution before OCR instead of OCR'ing directly.
+ MIN_TEXT_HEIGHT_PX = 30
+ # Below this, even 3x upscaling can't reach a legible size, so still reject.
+ UPSCALE_FLOOR_PX = 8
```

**New import:**
```python
from pipeline.superres import upscale
```

**Branch rewrite (replacing the current reject block):**
```python
median_height = _median_text_height(detections)
if median_height is not None and median_height < UPSCALE_FLOOR_PX:
    logger.warning(
        "Job %s rejected: RESOLUTION_TOO_LOW (median text height %.1fpx, n=%d)",
        job_id, median_height, len(detections),
    )
    _job_store[job_id] = {
        "status": "failed",
        "error": (
            "Text is too small to scan accurately. Please capture a higher-"
            "resolution image or move the camera closer to the document."
        ),
    }
    return

if median_height is not None and median_height < MIN_TEXT_HEIGHT_PX:
    logger.info(
        "Job %s: upscaling (median text height %.1fpx < %dpx)",
        job_id, median_height, MIN_TEXT_HEIGHT_PX,
    )
    clean_image = upscale(clean_image)

text = extract_text(clean_image)
```

Notes:
- `clean_image` is the warped, non-binarized BGR image — same coordinate space the
  MSER boxes were measured in, and exactly what `extract_text()` already consumes.
- `detection_count` in the result stays the **pre-upscale** count (it's just a stat).
  Document this in a comment so nobody assumes it reflects the upscaled image.
- `classify_document()` runs before upscaling — fine, classification resizes to
  224×224 internally and is scale-insensitive.
- The error message and `RESOLUTION_ERROR_SIGNATURE` on the frontend
  (`ScanPanel.vue`) are unchanged — the floor reject reuses the same message, so the
  existing "clear the file selection on this error" UX still works.

---

## Step 5 — Frontend impact

**None required.** The reject path still exists (hard floor) with the same message,
so `RESOLUTION_ERROR_SIGNATURE` handling is intact. Upscaled scans simply succeed
where they used to fail. The processing time will be a bit higher for upscaled
scans, which the new percentage/ETA bar already absorbs (it scales off the historical
average and file size).

Optional polish (not v1): surface `doc_type_source`-style info that a scan was
"enhanced" so the user knows SR ran. Skip unless you want it.

---

## Step 6 — Tests

**New `tests/test_superres.py`:**
```python
import numpy as np
from pipeline.superres import upscale, MAX_UPSCALE_INPUT_MP, _FACTOR


def test_upscale_multiplies_dimensions():
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    out = upscale(img)
    assert out.shape[0] == 40 * _FACTOR
    assert out.shape[1] == 60 * _FACTOR


def test_upscale_skips_large_images():
    side = int((MAX_UPSCALE_INPUT_MP * 1_000_000) ** 0.5) + 50
    img = np.zeros((side, side, 3), dtype=np.uint8)
    out = upscale(img)
    assert out.shape == img.shape  # returned unchanged
```
This requires `models/fsrcnn/FSRCNN_x3.pb` present. Since it's committed (Step 2),
CI has it with no extra fetch step. If you choose the fetch alternative instead, add
the FSRCNN download to the CI "Fetch model weights" step.

**Existing tests:** `test_api.py`'s `test_median_text_height_*` still pass — heights
10-18 give median 14, and `14 < 30` still holds, so
`test_median_text_height_flags_low_resolution` is unaffected. No existing test
asserts end-to-end rejection on real small text, so bumping the constant breaks
nothing. Consider adding an API-level test that a small-text image now resolves
`complete` rather than `failed` (needs a fixture with sub-30px text).

Run: `pytest tests/ -v`.

---

## Step 7 — Docker / CI / Deploy

With the model committed and only `requirements.txt` swapped, the existing pipeline
handles everything automatically on push to `main`:

- **Dockerfile:** no change. `pip install -r requirements.txt` picks up
  contrib-headless; `COPY . .` includes `models/fsrcnn/`. Same `libgl1`/`libglib2.0-0`.
  Build will be somewhat slower / image somewhat larger (contrib wheel).
- **`.github/workflows/deploy.yml`:** no change. `models` is not in the scp `source:`
  list explicitly... **verify this** — the list is
  `"api,app.py,models.py,models,pipeline,scripts,..."` and it **does** include
  `models`, so `models/fsrcnn/` ships. The backend-check job installs from
  requirements (gets contrib-headless) and runs tests.
- **`docker-compose.yml`:** no change.

Action items (all pre-verified, listed for completeness):
- `models` is in the deploy `source:` list (`"api,app.py,models.py,models,pipeline,
  scripts,..."`), so `models/fsrcnn/` ships to the VPS.
- `.dockerignore` and `.gitignore` exclude only `models/easyocr/` — neither drops
  `*.pb` nor `models/fsrcnn/`.

---

## Step 8 — What you set up manually on the VPS

Because deploys are automated via GitHub Actions, **almost nothing is manual** — the
opencv swap and model both ride through CI/CD on push to `main`. The manual part is
**capacity verification**, since SR adds CPU and peak memory:

1. **Check RAM headroom.** SSH in and run `free -m`. FSRCNN x3 on a ~1.5 MP image
   needs transient memory for the upscaled buffers and feature maps (hundreds of MB
   peak, on top of EasyOCR/torch already resident). If free RAM is tight (e.g. a
   1 GB VPS), add a swapfile as a safety net:
   ```bash
   sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
   sudo mkswap /swapfile && sudo swapon /swapfile
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   ```
2. **Check disk headroom** for the larger contrib image: `df -h` and
   `docker system df`. Run `docker image prune` first if low.
3. **After the deploy completes**, verify the running container:
   ```bash
   cd ~/doc-scanner-cv
   docker compose exec doc-scanner-cv python -c \
     "import cv2; print(cv2.dnn_superres.DnnSuperResImpl_create())"
   docker compose exec doc-scanner-cv ls -la models/fsrcnn/
   ```
4. **Watch the first small-text scan.** Tail logs and watch memory live:
   ```bash
   docker compose logs -f doc-scanner-cv      # look for the "upscaling ..." log line
   docker stats doc-scanner-cv                # watch peak mem during a small-text scan
   ```
   If memory spikes near the limit, lower `MAX_UPSCALE_INPUT_MP` in
   `pipeline/superres.py` and redeploy.

That's the whole manual surface: verify RAM/disk, optionally add swap, confirm the
module + model are in the container, and watch the first enhanced scan. No manual
model download, no manual package install — CI does both.

---

## Step 9 — Rollback

Low-risk, fully revertible:
1. Revert `MIN_TEXT_HEIGHT_PX` to 15 and restore the reject branch (or just revert the
   commit) -> behavior returns to "reject small text."
2. The `requirements.txt` swap can stay (contrib is a superset, harmless), or revert
   to `opencv-python-headless` if you want to shrink the image again.
3. Push to `main`; CI redeploys automatically.

Keep the SR change in its own commit so a single `git revert` cleanly backs it out.

---

## Step 10 — Verification checklist

- [ ] `pip install -r requirements.txt` succeeds with contrib-headless; smoke test
      `cv2.dnn_superres.DnnSuperResImpl_create()` works locally.
- [ ] `models/fsrcnn/FSRCNN_x3.pb` committed and tracked.
- [ ] `pytest tests/ -v` green (incl. new `test_superres.py`).
- [ ] `ruff check .` clean.
- [ ] A genuinely small-text image that previously got rejected now returns
      `complete` with sensible text.
- [ ] A near-empty / sub-floor image still rejects with the resolution message.
- [ ] Frontend: small-text scan now completes; oversize/garbage still errors as before.
- [ ] VPS: `free -m` headroom OK (swap added if tight); first enhanced scan logs the
      "upscaling" line and memory stays under the limit.

---

## Risks & open questions

- **opencv-contrib-python-headless version availability** — RESOLVED. Verified on
  PyPI: `4.13.0.92` is published (latest), matching the current headless pin exactly.
  The remaining (small) confirmation is the local smoke test in Step 1 that
  `cv2.dnn_superres.DnnSuperResImpl_create()` constructs without error.
- **VPS RAM** — the entire reason your reference strategy pushed crops. Mitigated by
  `MAX_UPSCALE_INPUT_MP` + swap; if real-world images routinely exceed the cap *and*
  need enhancement, that's the trigger to build the proper crop-based path (Decision A
  upgrade: CRAFT line detect -> upscale line crops -> `reader.recognize()`).
- **Thread safety** — handled with a lock around `upsample()`; revisit only if SR
  becomes a throughput bottleneck (unlikely at max 3 concurrent, SR-on-subset).
- **Quality ceiling** — fixed x3 over-upscales 20-29px text and under-upscales
  sub-10px text. Adaptive factor (x2/x3/x4 by measured median) is the documented
  refinement once v1 is proven.
- **FSRCNN .pb URL drift** — if the Saafke raw URL moves, the model is committed so
  runtime is unaffected; only update the comment/source reference.
```
