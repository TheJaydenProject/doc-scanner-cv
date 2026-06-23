import json
import logging
import os
import time
import uuid

from flask import Blueprint, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from sqlalchemy import func

from models import ScanRecord, db
from tasks import (
    MAX_CONCURRENT_SCANS,
    celery_app,
    count_active_jobs,
    get_job_state,
    redis_client,
    run_scan_job,
)

documents_bp = Blueprint("documents", __name__)
logger = logging.getLogger(__name__)

# Global rate limiter — init_app() called in app factory. Backed by Redis (not
# the default in-process memory store) so the 20/hour cap is shared across all
# Gunicorn workers instead of each worker enforcing its own separate count.
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],
    storage_uri=os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0"),
)

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}
STORAGE_DIR = os.environ.get("STORAGE_DIR", "storage")

# Ensure storage directory exists
os.makedirs(STORAGE_DIR, exist_ok=True)

@documents_bp.route("/scan", methods=["POST"])
@limiter.limit("20 per hour")
def scan():
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
    active_count, ip_active = count_active_jobs(ip)

    if ip_active:
        return jsonify(
            {"error": "You already have a scan in progress. Wait for it to finish, or cancel it, before starting another."}
        ), 429

    if active_count >= MAX_CONCURRENT_SCANS:
        return jsonify(
            {"error": "The server is at capacity. Please try again in a moment."}
        ), 429

    job_id = str(uuid.uuid4())
    file_path = os.path.join(STORAGE_DIR, f"{job_id}.png")

    # Save file to shared volume
    with open(file_path, "wb") as f:
        f.write(image_bytes)

    # Initialize job state in Redis. ip/created_at exist solely so
    # count_active_jobs() can attribute and age out in-flight jobs; they are
    # not read anywhere else and are dropped once the job reaches a terminal
    # state (the final update_job_state() call writes a fresh dict).
    initial_state = {
        "status": "processing",
        "filename": file.filename or "upload",
        "ip": ip,
        "created_at": time.time(),
    }
    redis_client.set(f"job:{job_id}", json.dumps(initial_state), ex=86400)

    # Dispatch Celery task
    task = run_scan_job.delay(job_id, file_path, file.filename or "upload")

    # Merge celery_task_id into whatever state is current rather than
    # overwriting outright: a very fast (or eager-mode) worker may have
    # already written a real "complete"/"failed" result by this point, and
    # blindly overwriting it with initial_state would clobber that result.
    current_state = get_job_state(job_id) or initial_state
    current_state["celery_task_id"] = task.id
    redis_client.set(f"job:{job_id}", json.dumps(current_state), ex=86400)

    return jsonify({"job_id": job_id}), 202


@documents_bp.route("/jobs/<job_id>", methods=["GET"])
def get_job(job_id: str):
    raw = redis_client.get(f"job:{job_id}")
    
    if not raw:
        return jsonify({"error": "Job not found."}), 404

    job = json.loads(raw)
    return jsonify(job), 200


@documents_bp.route("/jobs/<job_id>", methods=["DELETE"])
def cancel_job(job_id: str):
    """
    Cancellation: the frontend calls this on Stop, or when it gives up
    polling. Revokes the Celery task via SIGKILL if it's currently running.
    """
    raw = redis_client.get(f"job:{job_id}")
    
    if raw:
        job = json.loads(raw)
        if job.get("status") == "processing":
            # Forcefully terminate the worker if it's stuck in OCR
            task_id = job.get("celery_task_id")
            if task_id:
                celery_app.control.revoke(task_id, terminate=True, signal='SIGKILL')
            
            # Update Redis state
            job["status"] = "cancelled"
            redis_client.set(f"job:{job_id}", json.dumps(job), ex=86400)
            
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
