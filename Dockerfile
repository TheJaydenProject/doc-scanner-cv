FROM python:3.12-slim

# OpenCV system deps, Node.js (for tsc + biome at build time), and unzip
# (for the EasyOCR weight fetch below). opencv-python-headless still needs
# libgl1 + libglib2.0-0.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    unzip \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# EasyOCR model weights (CRAFT detector + English CRNN recognizer) are not
# committed to git: fetched once here, at build time, from EasyOCR's own
# release assets (see README "OCR model weights" for the rationale and links).
# The app loads them at runtime with download_enabled=False, so the running
# container never reaches out to the network on a request.
COPY scripts/fetch_ocr_weights.sh .
RUN bash fetch_ocr_weights.sh && rm fetch_ocr_weights.sh

# COPY . . includes both models/doc_classifier.onnx and models/doc_classifier.onnx.data —
# both must be committed. ONNX Runtime resolves the .data file from the .onnx path automatically.
COPY . .

# npm ci installs exact versions from package-lock.json (no network surprises).
# biome check runs lint + format validation; vue-tsc type-checks .vue files; vite build
# bundles the Vue SPA into static/. All three must pass before the container starts serving.
RUN npm ci && npm run check && npm run build

# Restore committed static assets wiped by Vite's emptyOutDir.
COPY static/examples static/examples
COPY static/docs.html static/docs.html

EXPOSE 5000

CMD ["python", "-m", "gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:create_app()"]
