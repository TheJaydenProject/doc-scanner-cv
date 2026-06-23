import base64
import json
import logging
import os
import time
import statistics

import cv2
import redis
from celery import Celery

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
from models import ScanRecord, db
from celery.signals import worker_process_init

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants (Copied from api/documents.py)
MIN_TEXT_HEIGHT_PX = 30
UPSCALE_FLOOR_PX = 8
MIN_DETECTION_SAMPLE_SIZE = 5

# Initialize Celery
redis_url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
celery_app = Celery("tasks", broker=redis_url)

# Initialize standard Redis client for job state
redis_client = redis.Redis.from_url(redis_url, decode_responses=True)

flask_app = None

def get_flask_app():
    global flask_app
    if flask_app is None:
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
        from app import create_app
        flask_app = create_app()
    return flask_app

@worker_process_init.connect
def init_worker(**kwargs):
    app = get_flask_app()
    with app.app_context():
        from pipeline.ocr import _get_reader
        _get_reader()

def _median_text_height(detections: list[tuple[int, int, int, int]]) -> float | None:
    if len(detections) < MIN_DETECTION_SAMPLE_SIZE:
        return None
    return statistics.median(h for (_, _, _, h) in detections)

def update_job_state(job_id: str, state: dict):
    redis_client.set(f"job:{job_id}", json.dumps(state), ex=86400) # Expiry 1 day

def get_job_state(job_id: str) -> dict:
    raw = redis_client.get(f"job:{job_id}")
    return json.loads(raw) if raw else {}

@celery_app.task(bind=True)
def run_scan_job(self, job_id: str, file_path: str, filename: str):
    start = time.time()
    
    try:
        app = get_flask_app()
        with app.app_context():
            # Check if cancelled before even starting
            state = get_job_state(job_id)
            if state.get("status") == "cancelled":
                return

            with open(file_path, "rb") as f:
                image_bytes = f.read()

            clean_image, _ = run_pipeline(image_bytes)

            doc_type = classify_document(clean_image)

            if doc_type["label"] == "printed":
                binarized = binarize_printed(clean_image)
            else:
                binarized = binarize_handwritten(clean_image)

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
                update_job_state(job_id, {
                    "status": "failed",
                    "error": (
                        "Text is too small to scan accurately. Please capture a "
                        "higher-resolution image or move the camera closer to the document."
                    ),
                })
                return

            upscale_secs = 0.0
            if median_height is not None and median_height < MIN_TEXT_HEIGHT_PX:
                logger.info(
                    "Job %s: upscaling (median text height %.1fpx < %dpx)",
                    job_id,
                    median_height,
                    MIN_TEXT_HEIGHT_PX,
                )
                
                # Update state to upscaling, but check if cancelled
                state = get_job_state(job_id)
                if state.get("status") == "cancelled":
                    return
                state["stage"] = "upscaling"
                update_job_state(job_id, state)
                
                _upscale_start = time.perf_counter()
                clean_image = upscale(clean_image, median_height)
                upscale_secs = time.perf_counter() - _upscale_start

            if get_job_state(job_id).get("status") == "cancelled":
                return

            _ocr_start = time.perf_counter()
            text = extract_text(clean_image)
            ocr_secs = time.perf_counter() - _ocr_start
            
            logger.info("Job %s timing: upscale=%.1fs ocr=%.1fs", job_id, upscale_secs, ocr_secs)

            if get_job_state(job_id).get("status") == "cancelled":
                return

            text = correct_ocr_text(text, doc_type["label"])

            def encode_png(image) -> str:
                _, buffer = cv2.imencode(".png", image)
                return base64.b64encode(buffer).decode("utf-8")

            warped_image_b64 = encode_png(clean_image)
            elapsed_ms = int((time.time() - start) * 1000)

            # We no longer write to SQLite here to prevent DB corruption if the task is SIGKILLed mid-transaction.
            # State is strictly maintained in Redis. A separate process/sync job will reconcile completed 
            # Redis states back into the SQLite DB.

            update_job_state(job_id, {
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
            })
            
            # Instantly trigger the sync task to write this completed job to SQLite
            # so the scan history populates immediately!
            sync_redis_to_sqlite.delay()

    except ContourNotFoundError as e:
        update_job_state(job_id, {"status": "failed", "error": str(e)})
    except Exception:
        logger.exception("Internal error processing job %s", job_id)
        update_job_state(job_id, {
            "status": "failed",
            "error": "Internal processing error.",
        })
        # Cleanup temporary image file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError as e:
                logger.error("Failed to delete temp file %s: %s", file_path, e)

@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    # Run the sync task every 5 minutes
    sender.add_periodic_task(300.0, sync_redis_to_sqlite.s(), name='sync redis to sqlite every 5 min')

@celery_app.task
def sync_redis_to_sqlite():
    """
    Periodic task to scan Redis for completed jobs that haven't been synced to SQLite yet.
    """
    app = get_flask_app()
    with app.app_context():
        # Find all job keys
        job_keys = redis_client.keys("job:*")
        synced_count = 0
        
        for key in job_keys:
            raw = redis_client.get(key)
            if not raw:
                continue
                
            job = json.loads(raw)
            
            # Only sync completed jobs that haven't been synced yet
            if job.get("status") == "complete" and not job.get("synced_to_db"):
                result = job.get("result", {})
                
                # We need the original filename which we didn't explicitly store in the state before,
                # but we can grab it from the job object if we start storing it, or use a placeholder.
                filename = job.get("filename", "unknown")
                
                if "char_count" in result:
                    record = ScanRecord(
                        filename=filename,
                        char_count=result.get("char_count", 0),
                        word_count=result.get("word_count", 0),
                        processing_time_ms=result.get("processing_time_ms", 0),
                    )
                    db.session.add(record)
                    
                    # Mark as synced so we don't duplicate it next time
                    job["synced_to_db"] = True
                    redis_client.set(key, json.dumps(job), ex=86400)
                    synced_count += 1
                    
        if synced_count > 0:
            try:
                db.session.commit()
                logger.info("Successfully synced %d completed jobs to SQLite", synced_count)
            except Exception as e:
                db.session.rollback()
                logger.error("Failed to sync jobs to SQLite: %s", e)
