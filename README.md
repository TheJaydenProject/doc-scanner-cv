# doc-scanner-cv

Computer vision pipeline that detects a document boundary in a photograph, dewraps it with a perspective transform, classifies its type, and extracts text via OCR -- served over a Flask REST API with a Vue 3 frontend.

![Python](https://img.shields.io/badge/python-3.11-blue)
![Vue](https://img.shields.io/badge/vue-3.4-brightgreen)
![OpenCV](https://img.shields.io/badge/opencv-4.13.0-orange)
![CI](https://github.com/TheJaydenProject/doc-scanner-cv/actions/workflows/deploy.yml/badge.svg)

---

## How It Works

1. **Preprocess**: The uploaded image is decoded, converted to grayscale, and smoothed with a 7x7 Gaussian blur to suppress edge noise before contour detection.
2. **Contour detection**: Three-pass strategy. Pass 1 approximates the top eight contours by area and returns the first clean four-point quad. Pass 2 applies a convex hull to the largest contour and approximates again. Pass 3 computes `minAreaRect` over all Canny edge pixels, covering low-contrast cases where the document boundary never closes into a contour.
3. **Boundary validation**: The detected quad's area is compared against the full frame (shoelace formula). A ratio outside `[0.15, 0.97]` means the contour isn't a real document edge — too small is a stray text blob or noise artifact, too large is the contour just tracing the image frame (e.g. a flat digital screenshot with no physical boundary). Either way, the perspective warp is skipped and the original frame passes through unchanged.
4. **Perspective transform**: Only run when the boundary passes validation. Corner points are sorted into `[top-left, top-right, bottom-right, bottom-left]` order using per-point coordinate sums and differences. `getPerspectiveTransform` and `warpPerspective` produce a flat, axis-aligned crop, then a symmetric 2.5% inset crop removes the binding/shadow sliver left at the contour edge.
5. **Document classification**: Runs on the clean image (warped, or the original frame if the warp was skipped) before any binarization. A bright-pixel heuristic checks for flat digital documents directly — if over 85% of pixels exceed luminance 240, the image is classified `printed` immediately, bypassing the ONNX model. Otherwise, a MobileNetV2 ONNX model (opset 18) classifies the image as `printed`, `handwritten`, or `mixed`. The `InferenceSession` is loaded once per process and reused across all requests to avoid the 100-300ms initialization cost.
6. **Binarize**: Branches by predicted type. `printed` uses Otsu's threshold (predictable bimodal contrast on flat ink); `handwritten`/`mixed` use adaptive Gaussian thresholding on the LAB lightness channel (block size 51, C=15), which tolerates uneven lighting and colored backgrounds better. Connected-component analysis then erases ruled/feint notebook lines and vertical margin borders: every blank line segment forms its own component regardless of length, and a component is erased only if it's long relative to the page in its dominant direction *and* thin in the perpendicular direction — a signature ordinary letters and joined cursive never share. (An earlier kernel/morphological approach required an unbroken ink run across ~20% of the page width, which missed lines broken by handwriting.)
7. **Text region detection**: MSER runs on the binarized, line-stripped image. Bounding boxes under 100 px² are discarded as noise, along with regions with a width:height ratio ≥ 15:1 or height ≤ 8px, both signatures of ruled-line fragments rather than glyphs. The binarize → line-removal → MSER chain exists solely to power the resolution gate (below) and a text-region count returned to the client; OCR itself never sees a binarized pixel.
8. **Resolution gate**: If MSER returned at least 5 boxes, their median height is computed. Below 5 boxes the sample is too volatile to judge, so the gate is skipped and OCR proceeds normally. Otherwise, a median under 30px routes the image through FSRCNN upscaling to enlarge small text for the OCR engine. If the text is below a hard floor of 8px, it is rejected outright with a friendly message asking the user to recapture at higher resolution.
9. **OCR**: EasyOCR (CRAFT text detector + CRNN recognizer, CPU-only) reads the warped, **non-binarized** image: its CNNs recover far more from the natural grayscale gradient than from a hard-binarized page, where surviving ruled lines fuse into the glyphs and collapse recognition. Detected word/phrase boxes are regrouped into reading order (clustered into rows by vertical centre, ordered left-to-right within each row), then a light post-pass strips ruled-line artifacts (leading/trailing dashes, intra-word underscores). The model weights are not committed to the repository; they are fetched at Docker build time and loaded with downloads disabled at runtime, so the running container never phones home (see "OCR model weights" below). The result, character count, word count, processing time, and the clean warped image are persisted to SQLite and returned to the client.

---

## Architecture

```
[Upload] JPEG / PNG (max 20 MB)
      |
  [Preprocess]      grayscale -> Gaussian blur (7x7)
      |
  [Contour]         Canny -> morphological close -> 3-pass quad extraction
      |
  [Boundary check]  area ratio outside [0.15, 0.97] -> skip warp (flat digital doc or stray contour)
      |
  [Perspective]     4-point warp -> axis-aligned crop + inset (skipped if boundary check failed)
      |
  [Classifier]      bright-pixel heuristic -> printed, else MobileNetV2 ONNX -> printed | handwritten | mixed
      |
  [Binarize]        Otsu (printed) | LAB-channel adaptive threshold (handwritten | mixed) -> connected-component line removal
      |
  [MSER Detector]   text region bounding boxes (noise/ruled-line fragments filtered)
      |
  [Resolution gate] median box height < 15px across >=5 boxes -> reject before OCR
      |
  [OCR]             EasyOCR (CRAFT + CRNN, CPU) on the warped image -> reading-order reconstruction -> extracted text
      |
  [SQLite]          ScanRecord (char_count, word_count, processing_time_ms)
      |
  [Response]        job_id -> poll -> { text, warped_image_b64, detection_count, doc_type, doc_type_source, ... }

Flask API           Flask + Gunicorn (1 worker, 3 concurrent scan threads)
Frontend            Vue 3 SPA (ScanPanel, ResultPanel, Dashboard with Chart.js)
Deployment          Docker + docker-compose + Cloudflare Tunnel
CI/CD               GitHub Actions: lint + type-check + test on push/PR -> auto-deploy to VPS via SSH on push to main
```

---

## Built With

| Component | Library | Version |
|-----------|---------|---------|
| Language | Python | 3.12 |
| Web framework | Flask | 3.1.3 |
| ORM | Flask-SQLAlchemy | 3.1.1 |
| Rate limiting | Flask-Limiter | 4.1.1 |
| Computer vision | OpenCV (headless) | 4.13.0 |
| ONNX inference | onnxruntime | 1.24.4 |
| OCR | EasyOCR (CRAFT + CRNN) | 1.7.2 |
| Deep learning runtime | PyTorch (CPU) | 2.12.1 |
| Image processing | Pillow | 12.1.1 |
| WSGI server | Gunicorn | 26.0.0 |
| Numerical | NumPy | 2.4.4 |
| Frontend | Vue 3 | 3.4.x |
| Charts | Chart.js | 4.5.x |
| Build tool | Vite | 5.0.x |
| Linter / formatter | Biome | 1.9.x |
| Test framework | pytest | 9.1.0 |

---

## Prerequisites

- Python 3.12 or later
- Node.js 20 or later and npm
- Docker and Docker Compose (for containerized deployment)

```bash
python --version     # must be 3.12+
node --version       # must be 20+
docker --version
```

The OCR engine (EasyOCR) needs no system package, but its model weights must be
present on disk before the app starts (see "OCR model weights" below). When running
outside Docker on Linux, OpenCV also needs two system libraries:

```bash
# Ubuntu / Debian (Linux only; not needed on macOS/Windows or via Docker)
sudo apt-get install libgl1 libglib2.0-0
```

---

## OCR model weights

EasyOCR needs two model files at runtime: the CRAFT text detector (`craft_mlt_25k.pth`,
~79 MB) and the English CRNN recognizer (`english_g2.pth`, ~14 MB). These are **not
committed to the repository**, both to keep ~93 MB of binary weights out of git
history and because they are a fixed, versioned external artifact rather than project
source. `pipeline/ocr.py` loads them with `download_enabled=False`, so the running
container never reaches out to the network on a request; the weights must already be
on disk under `models/easyocr/`.

`scripts/fetch_ocr_weights.sh` fetches both files from EasyOCR's own release assets
(the same URLs `easyocr.Reader(download_enabled=True)` would use internally) and
unzips them into `models/easyocr/`. It runs automatically:

- during the Docker build (`Dockerfile`), so the built image always ships with weights
- in CI (`.github/workflows/deploy.yml`, `backend-check`), before the test suite runs,
  since `app.py` pre-warms the reader at `create_app()` time

For local development outside Docker, run it once manually:

```bash
bash scripts/fetch_ocr_weights.sh
```

Sources:
- detector: https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip
- recognizer: https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip

---

## Installation

1. Clone the repository.

```bash
git clone https://github.com/TheJaydenProject/doc-scanner-cv.git
cd doc-scanner-cv
```

2. Create and activate a virtual environment.

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

3. Install Python dependencies.

```bash
pip install -r requirements.txt
```

4. Install Node dependencies and build the Vue frontend.

```bash
npm ci
npm run build
```

> `vite build` empties `static/` before writing the bundle, which deletes the committed `static/docs.html` and `static/examples/` images (the Docker build restores them automatically — see `Dockerfile`). If running locally outside Docker, restore them with `git checkout static/docs.html static/examples/`.

5. Start the development server.

```bash
python app.py
```

The application is available at `http://localhost:5000`.

**Docker (production)**

```bash
docker compose up --build
```

The image runs `vue-tsc`, `biome check`, and `vite build` during the Docker build step. All three must pass before the container starts serving. For the Cloudflare Tunnel to connect, set `CLOUDFLARE_TUNNEL_TOKEN` in the host environment before starting.

```bash
export CLOUDFLARE_TUNNEL_TOKEN=your_token_here
docker compose up -d
```

---

## Usage

Open `http://localhost:5000`, select a JPEG or PNG photograph of a document, and click Scan. The scan runs asynchronously; the UI polls for completion and displays the annotated result image, extracted text, document type classification (printed / handwritten / mixed), and per-scan metrics.

To call the API directly:

```bash
# Submit a scan
curl -X POST http://localhost:5000/api/documents/scan \
  -F "file=@document.jpg"
# {"job_id": "3f2a1b..."}

# Poll until complete
curl http://localhost:5000/api/documents/jobs/3f2a1b...
# {"status": "complete", "result": {...}}
```

---

## Configuration

| Variable | Type | Required | Description |
|----------|------|----------|-------------|
| `CLOUDFLARE_TUNNEL_TOKEN` | string | Production only | Authenticates the `cloudflared` container. Set in the host environment before `docker compose up`. |

All other settings (SQLite path, rate limits, max upload size, thread pool size) are hardcoded in `app.py` and `api/documents.py`. The database file is written to `instance/scans.db` and mounted to `./data/` in Docker so it persists across container restarts.

---

## API Reference

All endpoints are prefixed with `/api/documents`.

### POST /scan

Accepts `multipart/form-data`. Queues a background scan job and returns immediately.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | file | Yes | JPEG or PNG, max 20 MB |

| Status | Body | Description |
|--------|------|-------------|
| 202 | `{"job_id": "<uuid>"}` | Job accepted |
| 400 | `{"error": "..."}` | Missing file, wrong MIME type, or empty file |
| 429 | `{"error": "..."}` | 20/hour per-IP limit exceeded, duplicate in-flight scan from same IP, or global 3-concurrent cap reached |

---

### GET /jobs/:job_id

Poll for job status.

| Status | Body | Description |
|--------|------|-------------|
| 200 | `{"status": "processing"}` | Still running |
| 200 | `{"status": "complete", "result": {...}}` | Finished successfully |
| 200 | `{"status": "failed", "error": "..."}` | Pipeline error (e.g. no document contour found) or text resolution below the OCR-readable threshold |
| 404 | `{"error": "Job not found."}` | Unknown job ID |

Successful result shape:

```json
{
  "status": "complete",
  "result": {
    "text": "extracted text content",
    "char_count": 1042,
    "word_count": 183,
    "processing_time_ms": 312,
    "warped_image_b64": "<base64-encoded PNG, perspective-corrected>",
    "binarized_image_b64": "<base64-encoded PNG, thresholded>",
    "detections": [[12, 34, 18, 9]],
    "detection_count": 47,
    "doc_type": "printed",
    "doc_type_confidence": 0.9871,
    "doc_type_source": "heuristic"
  }
}
```

---

### GET /history

Returns the 50 most recent scan records ordered by `created_at` descending.

```json
[
  {
    "id": 14,
    "filename": "invoice.jpg",
    "char_count": 1042,
    "word_count": 183,
    "processing_time_ms": 312,
    "created_at": "2026-06-18T14:23:01.000000"
  }
]
```

---

### GET /metrics

Returns aggregate statistics and the 10 most recent records.

```json
{
  "total_scans": 42,
  "avg_processing_time_ms": 287,
  "avg_char_count": 894,
  "recent": [...]
}
```

---

## Tests

Three test modules cover API contract, scanner logic, and the detector/classifier.

| Module | What it tests |
|--------|---------------|
| `test_api.py` | Missing file, wrong MIME type, empty file, 202 shape with job_id, job not found, history list shape, metrics keys, async job resolves to `failed` on a blank image, async job resolves to `complete` with correct result keys on a synthetic document image, resolution-gate median height calculation (below sample-size threshold, at threshold, and below the 30px minimum) |
| `test_pipeline.py` | `ContourNotFoundError` raised on corrupt image bytes; blank image (no contour) falls back to the raw, unbinarized BGR frame with `warped=False`; a contour covering nearly the full frame (flat digital document) or an implausibly small fraction (stray text blob) both skip the warp; `_quad_area_ratio` shoelace math is correct on known quads; `binarize_printed` and `binarize_handwritten` each return a single-channel array containing only 0 and 255 |
| `test_detector.py` | `detect_text_regions` return types and shapes; noise bounding boxes filtered; `classify_document` returns a valid label string and the correct `source` (`model` vs `heuristic`); a near-white image is classified `printed` via the heuristic without running ONNX inference; BGR input handled correctly by classifier preprocessing |

Run the suite:

```bash
pytest tests/ -v
```

---

## Roadmap

- [x] 3-pass document contour detection with convex hull and minAreaRect fallbacks
- [x] Perspective transform with inset crop, early document classification, and type-branched binarization (Otsu for printed, LAB-channel adaptive threshold for handwritten/mixed)
- [x] MSER text region detection with bounding box annotation
- [x] ONNX document classifier (printed / handwritten / mixed)
- [x] Async scan jobs with in-memory job store (200-job LRU eviction)
- [x] Rate limiting (20/hour per IP, 1 concurrent per IP, 3 concurrent global)
- [x] SQLite persistence with scan history and aggregate metrics
- [x] Vue 3 SPA with Chart.js dashboard
- [x] Docker + Cloudflare Tunnel configuration
- [x] Examples gallery showing the 4-stage pipeline (raw, warped, binarized, detected) for a handwritten and a printed document
- [x] Static API documentation page
- [x] Live VPS deployment
- [x] Flat digital document handling: boundary-ratio check skips the perspective warp on implausible contours, a bright-pixel heuristic classifies flat documents as `printed` without the untrained ONNX head, and OCR page segmentation mode is chosen by document type
- [x] Connected-component ruled-line erasure to prevent OCR/MSER artifacts on ruled paper, tolerant of the slight tilt left by perspective warp and paper curl (replaced an earlier kernel/morphological approach that missed lines broken by handwriting)
- [x] Mobile-responsive layout for the Vue SPA interface
- [x] Pre-OCR resolution gate & Upscaling — routes scans whose median MSER text-box height falls below 30px through FSRCNN upscaling (minimum 5-box sample to avoid false positives), and outright rejects unrecoverable text < 8px, improving accuracy on low-resolution inputs.
- [x] GitHub Actions CI/CD — lint, type-check, and test on every push/PR; auto-deploys to the VPS via SSH on push to `main`

---

## Contributing

1. Fork the repository and create a branch: `git checkout -b feature/your-feature-name`
2. Make your changes and verify the test suite passes: `pytest tests/ -v`
3. Verify the frontend builds cleanly: `npm run build`
4. Open a pull request against `main` with a description of what changed and why.

Pushes to `main` run the full GitHub Actions pipeline (lint, type-check, test) and then auto-deploy to the production VPS if all checks pass — PRs run the same checks but never trigger deployment.

Bug reports and feature requests go in [GitHub Issues](https://github.com/TheJaydenProject/doc-scanner-cv/issues).

---

## Contact

[github.com/TheJaydenProject](https://github.com/TheJaydenProject)
