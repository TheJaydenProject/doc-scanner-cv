# Stage 1: Builder for Frontend
FROM node:20 AS builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run check && npm run build

# Stage 2: Production for Backend
FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*
    
WORKDIR /app

COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/scripts/fetch_ocr_weights.sh ./backend/scripts/
RUN cd backend && bash scripts/fetch_ocr_weights.sh && rm scripts/fetch_ocr_weights.sh

# Copy backend python code and existing static assets (e.g. docs.html, examples/)
COPY backend/ ./backend/

# Copy ALL compiled assets (including favicon, manifest, and the assets/ folder)
COPY --from=builder /app/frontend/dist/ ./backend/static/

# Copy the Vite entry point to Flask's templates directory
COPY --from=builder /app/frontend/dist/index.html ./backend/templates/index.html

EXPOSE 5000

RUN addgroup --system celerygroup && adduser --system --ingroup celerygroup celeryworker
RUN chown -R celeryworker:celerygroup /app
USER celeryworker

WORKDIR /app/backend
CMD ["python", "-m", "gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:create_app()"]
