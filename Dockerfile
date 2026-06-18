FROM python:3.11-slim

# Tesseract, OpenCV system deps, and Node.js (for tsc + biome at build time).
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
