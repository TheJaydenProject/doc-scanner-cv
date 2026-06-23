

import cv2
import os
from flask import Flask, send_from_directory, render_template, jsonify
from jinja2.exceptions import TemplateNotFound
try:
    from dotenv import load_dotenv
    # Load environment variables from root .env file
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path)
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
    app = Flask(__name__, static_folder="static", template_folder="templates")
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

    # SPA Catch-All Route
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def catch_all(path: str):
        # 1. Prevent intercepting valid API calls
        if path.startswith("api/"):
            return jsonify({"error": "Endpoint not found"}), 404
            
        # 2. Serve static root files (e.g., favicon.ico, robots.txt)
        if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
            
        # 3. Fallback to Vue SPA router
        try:
            return render_template("index.html")
        except TemplateNotFound:
            return (
                "<h1>Development Mode</h1>"
                "<p>Frontend assets not built. Access the app via the Vite Dev Server (e.g., localhost:5173).</p>"
            ), 503

    @app.route("/docs")
    def docs():
        return send_from_directory("static", "docs.html")

    with app.app_context():
        db.create_all()

    return app

if __name__ == "__main__":
    create_app().run(debug=True)