import atexit
import base64
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import cv2
from flask import Blueprint, Flask, current_app, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from sqlalchemy import func

from models import ScanRecord, db
from pipeline.classifier import classify_document
from pipeline.detector import detect_text_regions
from pipeline.ocr import extract_text
from pipeline.scanner import (
    ContourNotFoundError,
    binarize_handwritten,
    binarize_printed,
    run_pipeline,
)

documents_bp = Blueprint("documents", __name__)

# Global rate limiter — init_app() called in app factory.
limiter = Limiter(key_func=get_remote_address, default_limits=[])

# 3 workers max. A 4th concurrent scan is rejected with 429.
# Keeps the CV pipeline from saturating the VPS CPU.
executor = ThreadPoolExecutor(max_workers=3)

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}
_MAX_CONCURRENT_GLOBAL = 3

# Tracks IPs with a scan currently in-flight.
_active_ips: set[str] = set()

# Explicit counter for in-flight jobs.
# Avoids accessing executor._work_queue, which is a private CPython API
# that is not guaranteed to exist across implementations or future versions.
_active_job_count: int = 0
_active_job_lock = Lock()

# In-memory job store. Maps job_id -> { status, result | error }.
# OrderedDict so eviction always removes the oldest entry.
# A production system would use Redis or a dedicated jobs table.
_job_store: OrderedDict = OrderedDict()
_JOB_STORE_MAX = 200
_JOB_STORE_EVICT = 50


def _evict_job_store() -> None:
    while len(_job_store) > _JOB_STORE_MAX:
        for _ in range(_JOB_STORE_EVICT):
            if _job_store:
                _job_store.popitem(last=False)


def _run_scan_job(
    app: Flask, job_id: str, image_bytes: bytes, filename: str, ip: str
) -> None:
    """
    Runs in a background thread via ThreadPoolExecutor.
    Receives the app instance directly to avoid creating a second app,
    then pushes an application context for DB access.
    Always removes the IP from _active_ips and decrements the job counter
    on completion or failure.
    """
    global _active_job_count
    with app.app_context():
        start = time.time()

        try:
            clean_image = run_pipeline(image_bytes)

            # Classify before any destructive pixel alteration — the clean,
            # warped image carries far more signal than a binarized one.
            doc_type = classify_document(clean_image)

            if doc_type["label"] == "printed":
                binarized = binarize_printed(clean_image)
            else:
                binarized = binarize_handwritten(clean_image)

            # detections are returned as raw coordinates so the frontend can draw
            # its own interactive overlay; the burned-in annotated image is unused.
            _, detections = detect_text_regions(binarized)

            text = extract_text(binarized)

            def encode_png(image) -> str:
                _, buffer = cv2.imencode(".png", image)
                return base64.b64encode(buffer).decode("utf-8")

            warped_image_b64 = encode_png(clean_image)
            binarized_image_b64 = encode_png(binarized)

            elapsed_ms = int((time.time() - start) * 1000)

            record = ScanRecord(
                filename=filename,
                char_count=len(text),
                word_count=len(text.split()),
                processing_time_ms=elapsed_ms,
            )
            db.session.add(record)
            db.session.commit()

            _job_store[job_id] = {
                "status": "complete",
                "result": {
                    "text": text,
                    "char_count": len(text),
                    "word_count": len(text.split()),
                    "processing_time_ms": elapsed_ms,
                    "warped_image_b64": warped_image_b64,
                    "binarized_image_b64": binarized_image_b64,
                    "detections": detections,
                    "detection_count": len(detections),
                    "doc_type": doc_type["label"],
                    "doc_type_confidence": doc_type["confidence"],
                },
            }

        except ContourNotFoundError as e:
            _job_store[job_id] = {"status": "failed", "error": str(e)}
        except Exception:
            _job_store[job_id] = {
                "status": "failed",
                "error": "Internal processing error.",
            }
        finally:
            # Always release the IP slot and decrement counter regardless of outcome.
            _active_ips.discard(ip)
            with _active_job_lock:
                _active_job_count -= 1


@documents_bp.route("/scan", methods=["POST"])
@limiter.limit("20 per hour")
def scan():
    global _active_job_count

    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]

    if file.mimetype not in ALLOWED_MIME_TYPES:
        return jsonify(
            {"error": "Unsupported file type. Only JPEG and PNG are accepted."}
        ), 400

    image_bytes = file.read()

    if len(image_bytes) == 0:
        return jsonify({"error": "Empty file."}), 400

    ip = get_remote_address()

    if ip in _active_ips:
        return jsonify(
            {"error": "A scan is already in progress for your IP. Please wait."}
        ), 429

    with _active_job_lock:
        if _active_job_count >= _MAX_CONCURRENT_GLOBAL:
            return jsonify(
                {"error": "Server is at capacity. Please try again shortly."}
            ), 429
        _active_job_count += 1

    job_id = str(uuid.uuid4())
    _job_store[job_id] = {"status": "processing"}
    _evict_job_store()
    _active_ips.add(ip)

    app = current_app._get_current_object()
    executor.submit(
        _run_scan_job, app, job_id, image_bytes, file.filename or "upload", ip
    )

    return jsonify({"job_id": job_id}), 202


@documents_bp.route("/jobs/<job_id>", methods=["GET"])
def get_job(job_id: str):
    job = _job_store.get(job_id)

    if job is None:
        return jsonify({"error": "Job not found."}), 404

    return jsonify(job), 200


@documents_bp.route("/history", methods=["GET"])
def history():
    records = ScanRecord.query.order_by(ScanRecord.created_at.desc()).limit(50).all()
    return jsonify(
        [
            {
                "id": r.id,
                "filename": r.filename,
                "char_count": r.char_count,
                "word_count": r.word_count,
                "processing_time_ms": r.processing_time_ms,
                "created_at": r.created_at.isoformat(),
            }
            for r in records
        ]
    ), 200


@documents_bp.route("/metrics", methods=["GET"])
def metrics():
    total = ScanRecord.query.count()

    if total == 0:
        return jsonify(
            {
                "total_scans": 0,
                "avg_processing_time_ms": 0,
                "avg_char_count": 0,
                "recent": [],
            }
        ), 200

    avg_time = db.session.query(func.avg(ScanRecord.processing_time_ms)).scalar()
    avg_chars = db.session.query(func.avg(ScanRecord.char_count)).scalar()
    recent = ScanRecord.query.order_by(ScanRecord.created_at.desc()).limit(10).all()

    return jsonify(
        {
            "total_scans": total,
            "avg_processing_time_ms": round(avg_time),
            "avg_char_count": round(avg_chars),
            "recent": [
                {
                    "filename": r.filename,
                    "char_count": r.char_count,
                    "processing_time_ms": r.processing_time_ms,
                    "created_at": r.created_at.isoformat(),
                }
                for r in recent
            ],
        }
    ), 200
