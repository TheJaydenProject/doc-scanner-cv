import atexit

import cv2
from flask import Flask, send_from_directory
from dotenv import load_dotenv

from models import db
from api.documents import documents_bp, executor, limiter

# Load environment variables from .env file for local development
load_dotenv()

def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", static_url_path="")
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///scans.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    # 20MB max. Stricter than default to protect VPS from large payload abuse.
    app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

    # OpenCV defaults its internal parallel-for to all logical cores, so a
    # single scan's cv2 calls (blur/canny/morphology/warp/FSRCNN) would each
    # try to use every core on top of the 3-thread scan pool. Pin to 1 so the
    # pool, not cv2, owns the parallelism (mirrors torch.set_num_threads(1) in
    # pipeline/ocr.py).
    cv2.setNumThreads(1)

    db.init_app(app)
    limiter.init_app(app)
    app.register_blueprint(documents_bp, url_prefix="/api/documents")

    @app.route("/")
    def index():
        return send_from_directory("static", "index.html")

    @app.route("/docs")
    def docs():
        return send_from_directory("static", "docs.html")

    with app.app_context():
        db.create_all()

    # Pre-warm the EasyOCR reader at boot (single gunicorn worker) so the first
    # scan after a deploy doesn't pay the torch import + ~100MB model load on the
    # request path (R4). Fails loud here if the offline weights are missing.
    from pipeline.ocr import _get_reader
    _get_reader()

    # Cleanly drain the thread pool on interpreter shutdown.
    # Without this, background threads may be killed mid-scan on Ctrl+C.
    atexit.register(executor.shutdown, wait=True)

    return app

if __name__ == "__main__":
    create_app().run(debug=True)