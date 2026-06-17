import atexit
from flask import Flask
from models import db
from api.documents import documents_bp, executor, limiter

def create_app() -> Flask:
    app = Flask(__name__, static_folder="static")
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///scans.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    # 2MB max. Stricter than default to protect VPS from large payload abuse.
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

    db.init_app(app)
    limiter.init_app(app)
    app.register_blueprint(documents_bp, url_prefix="/api/documents")

    with app.app_context():
        db.create_all()

    # Cleanly drain the thread pool on interpreter shutdown.
    # Without this, background threads may be killed mid-scan on Ctrl+C.
    atexit.register(executor.shutdown, wait=True)

    return app

if __name__ == "__main__":
    create_app().run(debug=True)