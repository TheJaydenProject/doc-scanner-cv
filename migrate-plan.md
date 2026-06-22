# OCR Engine Migration: Tesseract to EasyOCR

## Summary

The OCR engine was replaced from Tesseract (via `pytesseract`) to EasyOCR (a CRAFT
text detector paired with a CRNN recognizer), running CPU-only on the production
VPS. The motivation was recognition quality on handwritten notes: Tesseract is a
character-segmentation engine and struggles with cursive and joined handwriting,
where a CNN-based recognizer like EasyOCR's is materially more accurate.

The investigation that preceded this change turned up a second, less expected
finding: the pipeline's existing binarization step, built to make Tesseract's job
easier, actively hurts EasyOCR. The two engines want different inputs. Working
that out changed not just which library is imported, but which image OCR reads
and, as a downstream consequence, which tabs the frontend results panel needed.
The sections below walk through that reasoning in the order it was discovered,
then record what was actually implemented and verified.

## Background

The application runs on an Oracle Cloud Ampere A1 VPS: 4 ARM (aarch64) OCPUs,
24 GB RAM, no GPU. Any OCR engine considered here had to run CPU-only on ARM,
fit comfortably in the Docker build the deploy pipeline already runs over SSH on
that same box, and keep per-scan latency well under the frontend's 60-second poll
timeout. These constraints ruled out heavier alternatives before quality was even
considered. A per-text-region transformer model (TrOCR was the candidate) was
rejected early: `pipeline/detector.py`'s MSER step can return well over 100 text
regions on a dense page, and running a transformer once per region, sequentially,
on an ARM CPU would take minutes per scan and blow through the poll budget.
EasyOCR runs its detector and recognizer once per page, not once per region, which
keeps it in a different performance class.

## Investigation

### Dependency and build feasibility

Before any quality testing, the basic question was whether EasyOCR's dependency
chain (`torch`, `torchvision`, `opencv`) would even install cleanly on aarch64 and
build inside the existing Docker pipeline without exhausting the VPS's memory
during the build. Both were tested directly on the production VPS rather than
inferred from documentation:

- `easyocr==1.7.2` resolves to prebuilt aarch64 wheels for its entire dependency
  tree, including `torch==2.12.1+cpu` and `torchvision==0.27.1+cpu`. No source
  builds were required, which would have been a much riskier and slower path on
  a 4-core ARM box.
- The full dependency set co-installs with `numpy==2.4.4`, the version already
  pinned for `onnxruntime` and `opencv`. No forced downgrade was needed, so there
  was no risk of rippling a numpy downgrade into the document classifier.
- A full Docker build (mirroring the real build the deploy workflow runs) completed
  on the VPS in 96.8 seconds with exit code 0. RAM and swap were watched during the
  build and stayed within bounds; no OOM kill occurred. Since the production deploy
  builds the image over SSH on this same box, this was the build path that actually
  mattered, and it was confirmed directly rather than assumed from a faster
  development machine.

One finding changed how the dependencies had to be installed: PyTorch's `+cpu`
wheels carry a local version tag that only exists on
`https://download.pytorch.org/whl/cpu`. A plain `pip install -r requirements.txt`
will not resolve them without that index supplied explicitly, so it had to be
added as an `--extra-index-url` line at the top of `requirements.txt` rather than
left as a Dockerfile-only flag, so every install path (Docker, CI, local
development) shares the one configuration.

The validated environment also turned out to be Python 3.12, not the 3.11 the
Dockerfile was pinned to at the time. Re-validating the spike against 3.11 was
considered, but shipping a stack whose pins were only ever tested on 3.12 was not
worth the risk for a one-line version bump, so the Dockerfile and CI workflow were
both moved to `python:3.12-slim`.

### Image-input quality comparison

This was the test that mattered most, and the one whose result changed the shape
of the migration. The pipeline existed for a Tesseract reading hard-binarized,
ruled-line-stripped images, on the reasoning that a clean black-and-white image
with no notebook lines gives a classical OCR engine the least to misread. The
question was whether that same preprocessing helps or hurts a CNN-based engine
like EasyOCR.

Three candidate inputs were run through `EasyOCR.Reader.readtext()` on real
handwritten and printed scans and compared against the existing Tesseract output:

- the fully processed `cleaned` image (binarized, then ruled lines erased)
- `binarized` (the same image before ruled-line removal)
- `clean_image`, the warped but otherwise untouched, non-binarized frame (the same
  image the document classifier already runs on)

The binarized inputs produced badly degraded text: "Applicaticn", "Te Cbss
Teacher", "Schoxl", "3025" are representative examples from a single scanned note.
The cause is structural rather than a tuning problem: hard binarization collapses
the grayscale gradient a CNN recognizer uses to distinguish a pen stroke from a
ruled line, and any ruled-line fragment that survives erasure (a common outcome on
notebook paper, where a line interrupted by a word leaves short remnants) sits
directly on top of the binarized glyphs with no intensity difference to separate
them. Tesseract's classical segmentation degrades far more gracefully on the same
artifact.

`clean_image`, the warped and non-binarized frame, won by a clear margin. The same
note that produced "Applicaticn" on the binarized input read correctly as "Date :
26 August 2025", "Subject: Application for", and "Respected Ma'am," on the natural
image, with only minor errors elsewhere. Measured latency for a full-page
`readtext()` call on this input was comfortably inside the 60-second poll budget;
later, end-to-end measurement on the deployed implementation (with the reader
pre-warmed at process start) put a single scan at roughly 5 to 9 seconds depending
on page density, well within that margin.

This result settled which image OCR should read (the natural, non-binarized
frame, not the pipeline's binarized output) and immediately raised a follow-on
question: if OCR no longer touches the binarized image, what is the binarize and
ruled-line removal code in the pipeline still for?

### Reading order

A secondary issue surfaced once `clean_image` was wired up against EasyOCR's
default output mode (`readtext(detail=0)`, plain strings with no position data):
sign-off lines and right-shifted fragments came back out of order, for example
the closing "Yours obediently, / Priya Mehta" of a note appearing displaced to the
end of the extracted text rather than where it physically sits on the page.
EasyOCR's detector returns boxes in an internal order that is not reliably
top-to-bottom.

The fix was to request `detail=1` (each detection's bounding box, text, and
confidence) and reconstruct reading order manually: boxes are grouped into rows by
vertical center, using a tolerance that scales with the page's median box height
so it adapts to image resolution, then each row is ordered left to right. This
resolved the ordering problem cleanly on every test image and is implemented in
`pipeline/ocr.py` as `_lines_in_reading_order`.

## Decisions

**Which image OCR reads.** `extract_text` now takes the warped, non-binarized
`clean_image`, the same frame the document classifier already runs on, rather than
the pipeline's `cleaned` (binarized and ruled-line-stripped) output. This is the
direct consequence of the quality comparison above.

**What the binarize and ruled-line pipeline is for now.** `pipeline/scanner.py`'s
`binarize_printed`, `binarize_handwritten`, and `remove_ruled_lines`, along with
`pipeline/detector.py`'s MSER text-region detection, are unchanged as code. What
changed is who consumes their output. Previously they fed both the resolution gate
and the OCR engine itself; now OCR bypasses them entirely, and they exist solely to
power two things that remain genuinely useful: the pre-OCR resolution gate
(`MIN_TEXT_HEIGHT_PX` in `api/documents.py`, which rejects a scan whose text is too
small to read before paying for an OCR pass) and the "Text Regions Detected" count
returned to the client. This is the one deliberate departure from treating the
migration as a pure recognition-engine swap, and it is confined to this single
consumer change rather than any change to the detection logic itself.

**Frontend simplification.** The results panel previously had five tabs: Raw,
Warped, Binarized, Detected, and Compare. The Detected and Compare tabs existed
specifically to show the MSER bounding-box overlay drawn on the binarized image.
Once OCR stopped reading that image, those two tabs lost their reason to exist as
the primary view, since the detection boxes were the only thing they uniquely
showed. They were removed, leaving Raw and Warped, with Warped as the default. The
"Text Regions Detected" stat card was kept (it only needs a count, not the boxes or
the binarized image), so `detection_count` stays in the `/scan` response while
`binarized_image_b64` and the raw `detections` list, which had no remaining
consumer, were dropped from the payload.

**Model weights are not committed to the repository.** EasyOCR needs two files at
runtime: the CRAFT detector (`craft_mlt_25k.pth`, approximately 79 MB) and the
English CRNN recognizer (`english_g2.pth`, approximately 14 MB). The project
already commits one binary model (`models/doc_classifier.onnx`, the document-type
classifier), so committing these as well was the initial approach. That was
reconsidered: unlike the classifier, which is a model this project trained,
EasyOCR's weights are an unmodified third-party artifact with a stable public
download location, and committing roughly 93 MB of binary data that never changes
adds permanent weight to git history for no benefit over fetching it from the
source. Instead, `scripts/fetch_ocr_weights.sh` downloads both files from
EasyOCR's own GitHub release assets, the same URLs `easyocr.Reader` itself uses
internally when downloads are enabled, and unzips them into `models/easyocr/`.
This script runs in three places: during the Docker build, in CI before the test
suite (since `app.py` pre-warms the reader at startup), and manually for local
development outside Docker. `pipeline/ocr.py` still loads the reader with
`download_enabled=False`, so the script is the only thing that ever reaches the
network for these files; the running application never does.

## Implementation

- `requirements.txt`: added the PyTorch CPU extra index URL, `easyocr==1.7.2`,
  `torch==2.12.1+cpu`, `torchvision==0.27.1+cpu`; replaced `opencv-python` with
  `opencv-python-headless` (EasyOCR requires the headless build; the two packages
  both provide the `cv2` module and cannot coexist; the codebase has no GUI calls
  such as `imshow`, so headless is a safe swap); removed `pytesseract`.
- `pipeline/ocr.py`: rewritten around a lazily constructed, module-level
  `easyocr.Reader` singleton (mirroring the existing pattern in
  `pipeline/classifier.py`), with `torch.set_num_threads(1)` set at import time.
  This is necessary, not just tidy: the app runs one Gunicorn worker with a
  3-thread pool for concurrent scans, and left unconfigured, torch's intra-op
  thread pool defaults to the machine's core count, so three concurrent OCR calls
  would contend for far more threads than the box has cores. `extract_text` now
  takes the warped image, requests `detail=1`, reconstructs reading order via
  `_lines_in_reading_order`, and runs the resulting text through the same
  ruled-line artifact cleanup (`_clean_ocr_text`) the Tesseract path used, since
  EasyOCR still reads a surviving ruled-line fragment under handwriting as a run
  of underscores or dashes.
- `api/documents.py`: the OCR call site now passes `clean_image` instead of
  `cleaned`; `binarized_image_b64` and `detections` were removed from the job
  result; `detection_count` and the resolution gate were left untouched.
- `app.py`: calls `pipeline.ocr._get_reader()` once at app-factory startup, so the
  torch import and model load happen at process boot rather than on the first
  scan after a deploy.
- `src/components/ResultPanel.vue`, `src/types.ts`, `src/style.css`: the Binarized,
  Detected, and Compare tabs and their supporting state (hover tracking, overlay
  SVG, compare-view layout) were removed; `ScanResult` no longer declares
  `binarized_image_b64` or `detections`.
- `scripts/fetch_ocr_weights.sh`: new script, shared by the Dockerfile and CI, that
  fetches and unzips the two model files.
- `Dockerfile`: base image moved to `python:3.12-slim`; `tesseract-ocr` removed
  from the apt install list; `unzip` added (needed by the weight fetch); the weight
  fetch script runs as its own build step before `COPY . .`.
- `.dockerignore` (new): excludes `models/easyocr/` from the build context, so a
  developer's local copy of the weights can never be mistaken for the build-time
  fetch that CI and the deploy pipeline actually rely on.
- `.gitignore`: added `models/easyocr/`.
- `.github/workflows/deploy.yml`: `backend-check` moved to Python 3.12, dropped
  `tesseract-ocr` from its apt step, added `unzip` and a step that runs the weight
  fetch script before the test suite, and added `scripts` to the deploy job's scp
  source list (the Dockerfile's weight-fetch step needs the script present on the
  VPS to build).
- Tests: `tests/test_api.py` and `tests/scratch_test.py` updated to match the new
  response shape and call site.
- Documentation: `README.md` updated (architecture walkthrough, tech table, a new
  "OCR model weights" section, prerequisites); `build-plan.md`'s historical
  "Why Tesseract stays the sole OCR engine" section was annotated as superseded
  rather than rewritten, so the original reasoning (which correctly rejected a
  per-region transformer) stays legible alongside the note explaining why its
  conclusion no longer holds.

## Verification

- `pytest tests/ -v`: 34 passed, including the async scan flow exercised against
  the real EasyOCR reader and the offline weight-loading path.
- `ruff check .`: clean.
- `npm run check` (Biome) and `npm run build` (`vue-tsc` plus `vite build`): clean.
- Manual end-to-end run against a real handwritten scan confirmed correct reading
  order and accurate text, consistent with the investigation findings above.
- App factory boot was confirmed to pre-warm the reader successfully with
  `download_enabled=False` against the locally fetched weights.

## Dependency reference

```
easyocr==1.7.2
torch==2.12.1+cpu
torchvision==0.27.1+cpu
opencv-python-headless==4.13.0.92
numpy==2.4.4
```

`--extra-index-url https://download.pytorch.org/whl/cpu` is required wherever
these install; it is set once at the top of `requirements.txt`.

## References

- EasyOCR: https://github.com/JaidedAI/EasyOCR
- EasyOCR on PyPI: https://pypi.org/project/easyocr/
- EasyOCR model hub: https://www.jaided.ai/easyocr/modelhub/
- CRAFT detector weights: https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip
- English CRNN recognizer weights: https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip
- PyTorch CPU wheel index: https://download.pytorch.org/whl/cpu
