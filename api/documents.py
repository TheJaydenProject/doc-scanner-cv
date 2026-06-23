import json
import logging
import os
import uuid

from flask import Blueprint, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from sqlalchemy import func

from models import ScanRecord, db
from tasks import celery_app, run_scan_job, redis_client

documents_bp = Blueprint("documents", __name__)
logger = logging.getLogger(__name__)

# Global rate limiter — init_app() called in app factory.
limiter = Limiter(key_func=get_remote_address, default_limits=[])

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}
STORAGE_DIR = "/app/storage"

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

    job_id = str(uuid.uuid4())
    file_path = os.path.join(STORAGE_DIR, f"{job_id}.png")
    
    # Save file to shared volume
    with open(file_path, "wb") as f:
        f.write(image_bytes)

    # Initialize job state in Redis
    initial_state = {
        "status": "processing",
        "filename": file.filename or "upload"
    }
    redis_client.set(f"job:{job_id}", json.dumps(initial_state), ex=86400)

    # Dispatch Celery task
    task = run_scan_job.delay(job_id, file_path, file.filename or "upload")
    
    # Store Celery task_id so we can revoke it later
    initial_state["celery_task_id"] = task.id
    redis_client.set(f"job:{job_id}", json.dumps(initial_state), ex=86400)

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
