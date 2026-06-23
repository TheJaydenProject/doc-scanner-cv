

import cv2
from flask import Flask, send_from_directory
try:
    from dotenv import load_dotenv
    # Load environment variables from .env file for local development
    load_dotenv()
except ImportError:
    pass

from models import db
from api.documents import documents_bp, limiter
from sqlalchemy import event
from sqlalchemy.engine import Engine

@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if type(dbapi_connection).__name__ == "sqlite3.Connection" or "sqlite" in str(type(dbapi_connection)):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

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

    return app

if __name__ == "__main__":
    create_app().run(debug=True)