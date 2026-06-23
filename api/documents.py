import base64
import logging
import statistics
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
from pipeline.openrouter import correct_ocr_text
from pipeline.ocr import extract_text
from pipeline.scanner import (
    ContourNotFoundError,
    binarize_handwritten,
    binarize_printed,
    remove_ruled_lines,
    run_pipeline,
)
from pipeline.superres import upscale

documents_bp = Blueprint("documents", __name__)

logger = logging.getLogger(__name__)

# Global rate limiter — init_app() called in app factory.
limiter = Limiter(key_func=get_remote_address, default_limits=[])

# 3 workers max. A 4th concurrent scan is rejected with 429.
# Keeps the CV pipeline from saturating the VPS CPU.
executor = ThreadPoolExecutor(max_workers=3)

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}
_MAX_CONCURRENT_GLOBAL = 3

# Below this median MSER text-box height, OCR accuracy degrades sharply, so the
# scan is routed through FSRCNN super-resolution before OCR rather than read at
# native resolution (see pipeline/superres.py).
MIN_TEXT_HEIGHT_PX = 30
# Below this, even 3x upscaling can't reach a legible size (3x*8 = 24px is the
# practical floor; sub-8px text super-resolves to mush), so still reject outright.
UPSCALE_FLOOR_PX = 8
# Fewer detections than this and the median is too volatile (one short
# character or punctuation box can swing it) — skip the gate, let OCR run.
MIN_DETECTION_SAMPLE_SIZE = 5

# Tracks IPs with a scan currently in-flight.
_active_ips: set[str] = set()

# Explicit counter for in-flight jobs.
# Avoids accessing executor._work_queue, which is a private CPython API
# that is not guaranteed to exist across implementations or future versions.
_active_job_count: int = 0
_active_job_lock = Lock()

# job_id -> ip, for jobs whose IP/count slot hasn't been released yet.
# Cancellation (cancel_job) and natural completion (_run_scan_job's finally)
# race to release the same slot; popping this under _active_job_lock makes
# whichever happens first the sole releaser and the other a no-op, so the
# slot is never double-released (which would let _active_job_count drift
# negative and silently raise the real concurrency cap above _MAX_CONCURRENT_GLOBAL).
_job_ip: dict[str, str] = {}


def _release_job_slot(job_id: str) -> None:
    global _active_job_count
    with _active_job_lock:
        ip = _job_ip.pop(job_id, None)
        if ip is None:
            return
        _active_ips.discard(ip)
        _active_job_count -= 1


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


def _is_cancelled(job_id: str) -> bool:
    job = _job_store.get(job_id)
    return job is not None and job.get("status") == "cancelled"


def _median_text_height(
    detections: list[tuple[int, int, int, int]],
) -> float | None:
    """
    Median height of MSER text-box detections, or None if there are too
    few boxes for the median to be a reliable signal (MIN_DETECTION_SAMPLE_SIZE).
    """
    if len(detections) < MIN_DETECTION_SAMPLE_SIZE:
        return None
    return statistics.median(h for (_, _, _, h) in detections)


def _run_scan_job(
    app: Flask, job_id: str, image_bytes: bytes, filename: str
) -> None:
    """
    Runs in a background thread via ThreadPoolExecutor.
    Receives the app instance directly to avoid creating a second app,
    then pushes an application context for DB access.
    Always releases the job's IP/count slot (via _release_job_slot) on
    completion or failure.
    """
    with app.app_context():
        start = time.time()

        try:
            clean_image, _ = run_pipeline(image_bytes)

            # Classify before any destructive pixel alteration — the clean,
            # warped image carries far more signal than a binarized one.
            doc_type = classify_document(clean_image)

            if doc_type["label"] == "printed":
                binarized = binarize_printed(clean_image)
            else:
                binarized = binarize_handwritten(clean_image)

            # Ruled lines confuse MSER, so strip them before detection. The
            # binarize -> remove_ruled_lines -> MSER chain now serves only the
            # resolution gate and the detection-count stat; OCR reads the
            # non-binarized clean_image directly (migrate-plan D2/D3).
            cleaned = remove_ruled_lines(binarized)

            _, detections = detect_text_regions(cleaned)

            median_height = _median_text_height(detections)
            if median_height is not None and median_height < UPSCALE_FLOOR_PX:
                logger.warning(
                    "Job %s rejected: RESOLUTION_TOO_LOW (median text height %.1fpx, n=%d)",
                    job_id,
                    median_height,
                    len(detections),
                )
                _job_store[job_id] = {
                    "status": "failed",
                    "error": (
                        "Text is too small to scan accurately. Please capture a "
                        "higher-resolution image or move the camera closer to the document."
                    ),
                }
                return

            # Small-but-recoverable text: super-resolve the whole warped frame so
            # EasyOCR's detector/recognizer read sharper glyphs. upscale() no-ops on
            # images already large enough (its own memory guard), so this is safe to
            # call whenever the gate trips. detection_count below stays the
            # pre-upscale count — it's only a stat, not used downstream.
            upscale_secs = 0.0
            if median_height is not None and median_height < MIN_TEXT_HEIGHT_PX:
                logger.info(
                    "Job %s: upscaling (median text height %.1fpx < %dpx)",
                    job_id,
                    median_height,
                    MIN_TEXT_HEIGHT_PX,
                )
                _upscale_start = time.perf_counter()
                clean_image = upscale(clean_image, median_height)
                upscale_secs = time.perf_counter() - _upscale_start

            # The frontend gives up polling after a fixed budget and calls
            # DELETE /jobs/<id> when it does; checking here before the two
            # most expensive remaining stages stops a now-abandoned job from
            # burning more CPU/network than it has to.
            if _is_cancelled(job_id):
                return

            _ocr_start = time.perf_counter()
            text = extract_text(clean_image)
            ocr_secs = time.perf_counter() - _ocr_start
            # Splits the gate's added cost so the upscale-vs-OCR ratio is visible
            # in logs when tuning factor/method (see pipeline/superres.py).
            logger.info(
                "Job %s timing: upscale=%.1fs ocr=%.1fs", job_id, upscale_secs, ocr_secs
            )

            if _is_cancelled(job_id):
                return

            # Best-effort OpenRouter (DeepSeek V4 Flash) cleanup of OCR spelling/punctuation/casing.
            # No-ops (returns text unchanged) if the key is unset or the call
            # fails, so it never breaks a scan. doc_type tailors the prompt to
            # printed vs. handwritten register. All downstream counts/storage
            # use the corrected text.
            text = correct_ocr_text(text, doc_type["label"])

            def encode_png(image) -> str:
                _, buffer = cv2.imencode(".png", image)
                return base64.b64encode(buffer).decode("utf-8")

            warped_image_b64 = encode_png(clean_image)

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
                    "detection_count": len(detections),
                    "doc_type": doc_type["label"],
                    "doc_type_confidence": doc_type["confidence"],
                    "doc_type_source": doc_type["source"],
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
            # Releases the slot unless cancel_job() already did (see _job_ip).
            _release_job_slot(job_id)


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
    _job_ip[job_id] = ip

    app = current_app._get_current_object()
    executor.submit(_run_scan_job, app, job_id, image_bytes, file.filename or "upload")

    return jsonify({"job_id": job_id}), 202


@documents_bp.route("/jobs/<job_id>", methods=["GET"])
def get_job(job_id: str):
    job = _job_store.get(job_id)

    if job is None:
        return jsonify({"error": "Job not found."}), 404

    return jsonify(job), 200


@documents_bp.route("/jobs/<job_id>", methods=["DELETE"])
def cancel_job(job_id: str):
    """
    Cancellation: the frontend calls this on Stop, or when it gives up
    polling. Only flips a still-processing job to "cancelled" so
    _run_scan_job can bail at its next checkpoint; a job that already
    finished is left alone so this can never clobber a real result.

    Also releases the IP/count slot immediately rather than waiting for the
    background thread to notice the cancellation flag — otherwise a quick
    Stop-then-Scan from the same IP could still hit "a scan is already in
    progress" for however long the abandoned job takes to reach its next
    checkpoint.
    """
    job = _job_store.get(job_id)
    if job is not None and job.get("status") == "processing":
        job["status"] = "cancelled"
        _release_job_slot(job_id)
    return jsonify({"status": "ok"}), 200


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
