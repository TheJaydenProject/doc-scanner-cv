#!/usr/bin/env bash
# Fetches the EasyOCR detector + recognizer weights pipeline/ocr.py loads at
# runtime. Not committed to git (see README "OCR model weights"); both the
# Docker build and CI call this script so the URLs/md5sums live in one place.
# Source: EasyOCR's own model registry (venv/Lib/site-packages/easyocr/config.py).
set -euo pipefail

DEST="${1:-models/easyocr}"
mkdir -p "$DEST"

curl -fsSL -o /tmp/craft_mlt_25k.zip https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip
curl -fsSL -o /tmp/english_g2.zip https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip

unzip -oq /tmp/craft_mlt_25k.zip -d "$DEST"
unzip -oq /tmp/english_g2.zip -d "$DEST"

rm /tmp/craft_mlt_25k.zip /tmp/english_g2.zip
