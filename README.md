# doc-scanner-cv

**[🔥 LIVE DEMO: https://doc-scanner.thejaydenproject.com/](https://doc-scanner.thejaydenproject.com/)**

Computer vision pipeline that detects a document boundary in a photograph, dewraps it with a perspective transform, classifies its type, and extracts text via OCR, served over a Flask REST API with a Vue 3 frontend.

![Python](https://img.shields.io/badge/python-3.12-blue)
![Vue](https://img.shields.io/badge/vue-3.4-brightgreen)
![OpenCV](https://img.shields.io/badge/opencv-4.9.0-orange)
![CI](https://github.com/TheJaydenProject/doc-scanner-cv/actions/workflows/deploy.yml/badge.svg)

---

## Contents

- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Built With](#built-with)
- [Prerequisites](#prerequisites)
- [OCR model weights](#ocr-model-weights)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Tests](#tests)
- [Roadmap](#roadmap)
- [Known Limitations](#known-limitations)
- [Contributing](#contributing)
- [Contact](#contact)

---

## How It Works

1. **Preprocess**: The uploaded image is decoded, converted to grayscale, and smoothed with a 7x7 Gaussian blur to suppress edge noise before contour detection.
2. **Contour detection**: Three-pass strategy. Pass 1 approximates the top eight contours by area and returns the first clean four-point quad. Pass 2 applies a convex hull to the largest contour and approximates again. Pass 3 computes `minAreaRect` over all Canny edge pixels, covering low-contrast cases where the document boundary never closes into a contour.
3. **Boundary validation**: The detected quad's area is compared against the full frame (shoelace formula). A ratio outside `[0.15, 0.97]` means the contour isn't a real document edge: too small is a stray text blob or noise artifact, too large is the contour just tracing the image frame (e.g. a flat digital screenshot with no physical boundary). Either way, the perspective warp is skipped and the original frame passes through unchanged.
4. **Perspective transform**: Only run when the boundary passes validation. Corner points are sorted into `[top-left, top-right, bottom-right, bottom-left]` order using per-point coordinate sums and differences. `getPerspectiveTransform` and `warpPerspective` produce a flat, axis-aligned crop, then a symmetric 2.5% inset crop removes the binding/shadow sliver left at the contour edge.
5. **Document classification**: Runs on the clean image (warped, or the original frame if the warp was skipped) before any binarization. A bright-pixel heuristic checks for flat digital documents directly: if over 85% of pixels exceed luminance 240, the image is classified `printed` immediately, bypassing the ONNX model. Otherwise, a MobileNetV2 ONNX model (opset 18) classifies the image as `printed`, `handwritten`, or `mixed`. The `InferenceSession` is loaded once per process and reused across all requests to avoid the 100-300ms initialization cost.
6. **Binarize**: Branches by predicted type. `printed` uses Otsu's threshold (predictable bimodal contrast on flat ink); `handwritten`/`mixed` use adaptive Gaussian thresholding on the LAB lightness channel (block size 51, C=15), which tolerates uneven lighting and colored backgrounds better. Connected-component analysis then erases ruled/feint notebook lines and vertical margin borders: every blank line segment forms its own component regardless of length, and a component is erased only if it's long relative to the page in its dominant direction *and* thin in the perpendicular direction, a signature ordinary letters and joined cursive never share. (An earlier kernel/morphological approach required an unbroken ink run across ~20% of the page width, which missed lines broken by handwriting.)
7. **Text region detection**: MSER runs on the binarized, line-stripped image. Bounding boxes under 100 px² are discarded as noise, along with regions with a width:height ratio ≥ 15:1 or height ≤ 8px, both signatures of ruled-line fragments rather than glyphs. The binarize → line-removal → MSER chain exists solely to power the resolution gate (below) and a text-region count returned to the client; OCR itself never sees a binarized pixel.
8. **Resolution gate**: If MSER returned at least 5 boxes, their median height is computed. Below 5 boxes the sample is too volatile to judge, so the gate is skipped and OCR proceeds normally. Otherwise, a median under 30px routes the image through FSRCNN upscaling to enlarge small text for the OCR engine. If the text is below a hard floor of 8px, it is rejected outright with a friendly message asking the user to recapture at higher resolution.
9. **OCR**: EasyOCR (CRAFT text detector + CRNN recognizer, CPU-only) reads the warped, **non-binarized** image: its CNNs recover far more from the natural grayscale gradient than from a hard-binarized page, where surviving ruled lines fuse into the glyphs and collapse recognition. Detected word/phrase boxes are regrouped into reading order (clustered into rows by vertical centre, ordered left-to-right within each row), then a light post-pass strips ruled-line artifacts (leading/trailing dashes, intra-word underscores). The model weights are not committed to the repository; they are fetched at Docker build time and loaded with downloads disabled at runtime, so the running container never phones home (see "OCR model weights" below).
10. **LLM Post-Correction (Optional)**: If `OPENROUTER_API_KEY` is configured, the raw OCR text is sent to DeepSeek V4 Flash via OpenRouter for context-aware spelling, casing, and punctuation correction, with a document-type-specific hint (printed vs. handwritten character confusions) added to the prompt; it never paraphrases or rewrites. If the key is unset, the completion is empty, or the request fails, it gracefully falls back to the raw text. The job's final state, including the text, character/word counts, processing time, and the warped image, is written to Redis and returned to the client on the next poll. A Celery task syncs the scan's summary stats into SQLite immediately afterward, with a 5-minute periodic sweep as a safety net for anything missed.

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
  [Resolution gate] median box height < 8px across >=5 boxes -> reject; < 30px -> FSRCNN upscale
      |
  [OCR]             EasyOCR (CRAFT + CRNN, CPU) on the warped image -> reading-order reconstruction -> LLM cleanup -> extracted text
      |
  [Redis]           job state (processing/complete/failed + result), 24h TTL
      |
  [SQLite]          synced from Redis on completion: ScanRecord (char_count, word_count, processing_time_ms)
      |
  [Response]        job_id -> poll -> { text, warped_image_b64, detection_count, doc_type, doc_type_source, ... }

Flask API           Flask + Gunicorn (4 workers); dispatches scans to Celery, never runs the pipeline inline
Job queue            Celery worker (separate container) + Redis (broker and job-state store)
Frontend             Vue 3 SPA (ScanPanel, ResultPanel, Dashboard with Chart.js)
Deployment           Docker + docker-compose (redis, app, worker, cloudflared) + Cloudflare Tunnel
CI/CD                GitHub Actions: ruff + biome lint, type-check, pytest on push/PR -> auto-deploy to VPS via SSH on push to main
```

---

## Built With

| Component | Library | Version |
|-----------|---------|---------|
| Language | Python | 3.12 |
| Web framework | Flask | 3.1.3 |
| ORM | Flask-SQLAlchemy | 3.1.1 |
| Rate limiting | Flask-Limiter | 4.1.1 |
| Task queue | Celery | 5.4.0 |
| Broker / job state | Redis | 5.0.4 (client), `redis:7-alpine` (server) |
| Computer vision | OpenCV (contrib, headless) | 4.9.0.80 |
| ONNX inference | onnxruntime | 1.24.4 |
| OCR | EasyOCR (CRAFT + CRNN) | 1.7.2 |
| Deep learning runtime | PyTorch (CPU) | 2.2.1 |
| Image processing | Pillow | 12.1.1 |
| WSGI server | Gunicorn | 26.0.0 |
| Numerical | NumPy | < 2.0 |
| Backend lint | Ruff | 0.4.4 (CI-pinned) |
| Frontend | Vue 3 | 3.4.x |
| Charts | Chart.js | 4.5.x |
| Build tool | Vite | 5.0.x |
| Linter / formatter (frontend) | Biome | 1.9.x |
| Test framework | pytest | 9.1.0 |

---

## Prerequisites

- Python 3.12 or later
- Node.js 20 or later and npm
- Docker and Docker Compose (for containerized deployment)
- Redis (only if running [Method 3](#installation) without Docker; Methods 1 and 2 provision it for you)

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

### Method 1: Docker (Recommended)

Using Docker is highly recommended because it manages the Redis broker and Celery worker for you across all operating systems.

1. Clone the repository.

```bash
git clone https://github.com/TheJaydenProject/doc-scanner-cv.git
cd doc-scanner-cv
```

2. Configure your environment variables.

Copy the example configuration file and add your OpenRouter API key (required for LLM OCR cleanup).

```bash
cp .env.example .env
```
*(Open the `.env` file in a text editor and set `OPENROUTER_API_KEY=your_key_here`)*

3. Start the full application stack.

We have automated startup scripts that will dynamically check if your system has an NVIDIA GPU with at least 2GB of VRAM and enable GPU acceleration.

**On Windows:**
```powershell
.\start_docker.ps1
```

**On Linux / macOS:**
```bash
chmod +x start_docker.sh
./start_docker.sh
```

*(Note: Add `-d` to the end of the script to run it silently in the background!)*

The application will be available at `http://localhost:5000`.

### Method 2: Native Windows (Low RAM Mode)

If you are on Windows and Docker/WSL is consuming too much memory, you can run the entire stack natively without Docker! We provide an automated script that downloads a lightweight portable Redis binary and opens the necessary terminals for you.

**On Windows:**
```powershell
.\start_native_stack.ps1
```
*(This will automatically open separate windows for your Frontend and Celery worker. When you are done, simply close the windows!)*

### Method 3: Local Development (Without Docker)

Use this method only if you need Hot Module Replacement (HMR) for frontend development or want to use a local GPU natively for OCR. You must have [Redis installed and running](https://redis.io/docs/install/install-redis/) on your machine (e.g. `redis-server`).

1. Clone the repository and navigate into it.

```bash
git clone https://github.com/TheJaydenProject/doc-scanner-cv.git
cd doc-scanner-cv
```

2. Install Python and Node dependencies.

```bash
pip install -r backend/requirements.txt

cd frontend
npm ci
cd ..
```

Method 3 serves the frontend via Vite's dev server (port 5173), not Flask's static folder, so there's no need to run `npm run build` here. (`vite.config.ts` outputs to `frontend/dist/`, not `backend/static/`, this only matters for the Docker image build, which `COPY`s the build output on top of the already-committed `backend/static/`, leaving `docs.html` and `examples/` untouched.)

3. Start the application stack. You will need to run these commands in separate terminal windows (with your virtual environment activated in each):

**Terminal 1 (Celery Worker):**
```bash
cd backend
# On Windows, you must use --pool=solo. On Linux/macOS, omit it.
celery -A tasks.celery_app worker -l info --pool=solo
```

**Terminal 2 (Frontend & Backend Dev Servers):**
```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:5173` with Hot Module Replacement, and the Flask API will run on `http://127.0.0.1:5000`.

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

# Cancel a still-running scan
curl -X DELETE http://localhost:5000/api/documents/jobs/3f2a1b...
# {"status": "ok"}
```

---

## Configuration

| Variable | Type | Required | Description |
|----------|------|----------|-------------|
| `OPENROUTER_API_KEY` | string | Optional | Enables LLM post-correction via OpenRouter to fix OCR spelling, casing, and punctuation errors. Falls back to raw OCR text if unset. |
| `REDIS_URL` | string | Optional | Connection string for the Celery broker and job-state store. Defaults to `redis://127.0.0.1:6379/0`; `docker-compose.yml` overrides this to `redis://redis:6379/0` for the app and worker containers. |
| `CLOUDFLARE_TUNNEL_TOKEN` | string | Production only | Authenticates the `cloudflared` container. Set in the host environment before `docker compose --profile prod up`. |
| `STORAGE_DIR` | string | Optional | Directory where an uploaded scan is written before the Celery worker picks it up. Defaults to `storage/` (relative to `backend/`); `docker-compose.yml` mounts this as a shared volume between the app and worker containers. |

Rate limiting, the concurrency caps, and the max upload size are hardcoded in `api/documents.py` and `tasks.py` (see [API Reference](#api-reference) below); SQLite path is hardcoded in `app.py`. The Celery worker's pool size (`--concurrency=3` in `docker-compose.yml`) is set to match `MAX_CONCURRENT_SCANS` in `tasks.py`, so there's no idle worker capacity the application-level cap would never dispatch to anyway. The database lives at `instance/scans.db` inside the container and is persisted via the named Docker volume `scans-db` (uploaded images persist via `shared-uploads`, shared between the app and worker containers), so both survive container restarts and rebuilds, including the ones `deploy.yml` runs on every push to `main` (it only ever runs `docker compose build` + `up`, never `down -v` or a volume prune).

---

## API Reference

All endpoints are prefixed with `/api/documents`.

### POST /scan

Accepts `multipart/form-data`. Saves the upload, dispatches a Celery task, and returns immediately.

Three independent limits guard this endpoint, all enforced in `api/documents.py`:

| Limit | Value | Enforced via |
|-------|-------|---------------|
| Per-IP request rate | 20 scans / hour | Flask-Limiter, backed by Redis storage so the count is shared across all 4 Gunicorn workers |
| Concurrent scans, globally | 3 at a time | `tasks.count_active_jobs()`, counting Redis job records with `status: "processing"` younger than 10 minutes |
| Concurrent scans, per IP | 1 at a time | Same helper; an IP with any non-stale `"processing"` job is rejected |

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | file | Yes | JPEG or PNG, max 20 MB |

| Status | Body | Description |
|--------|------|-------------|
| 202 | `{"job_id": "<uuid>"}` | Job accepted |
| 400 | `{"error": "..."}` | Missing file, wrong MIME type, or empty file |
| 429 | `{"error": "..."}` | One of the three limits above was hit (see message for which) |

---

### GET /jobs/:job_id

Poll for job status.

| Status | Body | Description |
|--------|------|-------------|
| 200 | `{"status": "processing"}` | Still running. May include `"stage": "upscaling"` while FSRCNN super-resolution is in progress. |
| 200 | `{"status": "complete", "result": {...}}` | Finished successfully |
| 200 | `{"status": "failed", "error": "..."}` | Pipeline error (e.g. no document contour found) or text resolution below the OCR-readable threshold |
| 200 | `{"status": "cancelled"}` | Job was cancelled via `DELETE` before it finished |
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
    "detection_count": 47,
    "doc_type": "handwritten",
    "doc_type_confidence": 0.9871,
    "doc_type_source": "model"
  }
}
```

---

### DELETE /jobs/:job_id

Cancels a scan. If the job is still processing, the Celery task is forcefully terminated (`SIGKILL`) via `celery_app.control.revoke()`. A job that has already completed or failed is left untouched, and an unknown job ID is a no-op; both cases still return `200`.

| Status | Body | Description |
|--------|------|-------------|
| 200 | `{"status": "ok"}` | Job cancelled, or no-op (already finished / unknown ID) |

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

Five test modules cover API contract, pipeline logic, the detector/classifier, super-resolution, and LLM cleanup. They require a local Redis instance (Celery tasks run synchronously in-process via `task_always_eager`, but job state still round-trips through a real Redis connection):

```bash
redis-server &           # or use Docker: docker run -p 6379:6379 redis:7-alpine
cd backend
pytest tests/ -v
```

| Module | What it tests |
|--------|---------------|
| `test_api.py` | Missing file, wrong MIME type, empty file, 202 shape with job_id, job not found, history list shape, metrics keys, async job resolves to `complete` via the blank-image fallback (no contour, unbinarized full frame), async job resolves to `failed` on corrupt/undecodable image bytes, async job resolves to `complete` with correct result keys on a synthetic document image, job cancellation (marks a processing job cancelled, never clobbers an already-finished job, no-ops on an unknown job id), resolution-gate median height calculation (below sample-size threshold, at threshold, and below the 30px minimum), concurrency caps (rejects the 4th global concurrent scan, rejects a 2nd concurrent scan from the same IP, allows a new scan once stale `"processing"` records age past `STALE_PROCESSING_TIMEOUT_S`) |
| `test_pipeline.py` | `ContourNotFoundError` raised on corrupt image bytes; blank image (no contour) falls back to the raw, unbinarized BGR frame with `warped=False`; a contour covering nearly the full frame (flat digital document) or an implausibly small fraction (stray text blob) both skip the warp; `_quad_area_ratio` shoelace math is correct on known quads; `binarize_printed` and `binarize_handwritten` each return a single-channel array containing only 0 and 255; `remove_ruled_lines` erases tilted/thick/page-spanning rules and collinear margin stubs while preserving disconnected handwriting strokes, isolated letter strokes, and margin glyphs on unruled pages |
| `test_detector.py` | `detect_text_regions` return types and shapes; noise bounding boxes filtered; `classify_document` returns a valid label string and the correct `source` (`model` vs `heuristic`); a near-white image is classified `printed` via the heuristic without running ONNX inference; BGR input handled correctly by classifier preprocessing |
| `test_superres.py` | `_choose_factor` picks the smallest FSRCNN factor (x2/x3) that lifts median text height to the 30px target, clamped to the shipped models; `upscale` applies that factor and leaves images above `MAX_UPSCALE_INPUT_MP` untouched; the `cubic` fallback path scales by the same factor as FSRCNN |
| `test_openrouter.py` | `correct_ocr_text` skips the API call on blank input or a missing `OPENROUTER_API_KEY`, returns the cleaned text on a successful response, and falls back to the raw OCR text on a request failure or an empty model completion |

---

## Roadmap

- [x] 3-pass document contour detection with convex hull and minAreaRect fallbacks
- [x] Perspective transform with inset crop, early document classification, and type-branched binarization (Otsu for printed, LAB-channel adaptive threshold for handwritten/mixed)
- [x] MSER text region detection with bounding box annotation
- [x] ONNX document classifier (printed / handwritten / mixed)
- [x] Async scan jobs via Celery + Redis (job state in Redis with a 24h TTL, synced into SQLite on completion); cancellable mid-run via `DELETE /jobs/:job_id`
- [x] Rate limiting (20 scans/hour per IP, max 3 concurrent scans globally, max 1 concurrent scan per IP), backed by Redis so the limits hold across all Gunicorn workers
- [x] SQLite persistence with scan history and aggregate metrics
- [x] Vue 3 SPA with Chart.js dashboard
- [x] Docker + Cloudflare Tunnel configuration
- [x] Examples gallery showing the 4-stage pipeline (raw, warped, binarized, detected) for a handwritten and a printed document
- [x] Static API documentation page
- [x] Live VPS deployment
- [x] Flat digital document handling: boundary-ratio check skips the perspective warp on implausible contours, a bright-pixel heuristic classifies flat documents as `printed` without the untrained ONNX head, and OCR page segmentation mode is chosen by document type
- [x] Connected-component ruled-line erasure to prevent OCR/MSER artifacts on ruled paper, tolerant of the slight tilt left by perspective warp and paper curl (replaced an earlier kernel/morphological approach that missed lines broken by handwriting)
- [x] Mobile-responsive layout for the Vue SPA interface
- [x] Pre-OCR resolution gate and upscaling: routes scans whose median MSER text-box height falls below 30px through FSRCNN upscaling (minimum 5-box sample to avoid false positives), and outright rejects unrecoverable text below 8px, improving accuracy on low-resolution inputs
- [x] GitHub Actions CI/CD: lint (ruff + biome), type-check, and test on every push/PR; auto-deploys to the VPS via SSH on push to `main`

---

## Known Limitations

- **The ONNX classifier head is untrained.** `scripts/export_classifier.py` attaches a freshly initialized `nn.Linear` head to an ImageNet-pretrained MobileNetV2 backbone and exports it immediately; there is no training step anywhere in the repository. `doc_type_source: "model"` predictions are consequently unreliable. The bright-pixel heuristic (`doc_type_source: "heuristic"`) is the only classification path with a meaningful confidence today. Fine-tuning the head on labeled samples is the main piece of unfinished work.
- **The concurrency caps have a small race window.** `count_active_jobs()` (`tasks.py`) counts in-flight jobs and `api/documents.py` then dispatches, as two separate steps rather than one atomic Redis operation. Two requests arriving within the same instant could both pass the check and briefly push concurrency one above `MAX_CONCURRENT_SCANS`. Given the cap is a conservative 3 on a 4-core VPS, an occasional off-by-one is an acceptable tradeoff against the complexity of a Lua-scripted atomic check-and-increment; revisit if the VPS ever shows memory pressure from it.

---

## Contributing

1. Fork the repository and create a branch: `git checkout -b feature/your-feature-name`
2. Make your changes and verify the backend: `cd backend && ruff check . && pytest tests/ -v` (requires a local Redis instance, see [Tests](#tests))
3. Verify the frontend: `cd frontend && npm run check && npm run build`
4. Open a pull request against `main` with a description of what changed and why.

Pushes to `main` run the full GitHub Actions pipeline (lint, type-check, test) and then auto-deploy to the production VPS if all checks pass. PRs run the same checks but never trigger deployment.

Bug reports and feature requests go in [GitHub Issues](https://github.com/TheJaydenProject/doc-scanner-cv/issues).

---

## Contact

[github.com/TheJaydenProject](https://github.com/TheJaydenProject)
