import io
import numpy as np
import cv2
import pytest
from app import create_app


@pytest.fixture(autouse=True)
def reset_scan_state():
    """Reset module-level rate-limiting globals between tests for isolation."""
    import api.documents as doc_api
    doc_api._active_ips.clear()
    doc_api._job_store.clear()
    with doc_api._active_job_lock:
        doc_api._active_job_count = 0
    yield
    doc_api._active_ips.clear()
    doc_api._job_store.clear()
    with doc_api._active_job_lock:
        doc_api._active_job_count = 0


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    # Must push the app context before create_all so SQLAlchemy can
    # resolve the bound engine. Without this, tests that hit /history
    # or /metrics raise a "no such table" OperationalError.
    with app.app_context():
        from models import db
        db.create_all()
        with app.test_client() as c:
            yield c


@pytest.fixture
def blank_image_bytes() -> bytes:
    """A solid white image — guarantees ContourNotFoundError in the pipeline."""
    blank = np.ones((500, 500, 3), dtype=np.uint8) * 255
    _, buffer = cv2.imencode(".jpg", blank)
    return buffer.tobytes()


@pytest.fixture
def minimal_jpeg_bytes() -> bytes:
    """Smallest valid JPEG — passes MIME and size guards without running a real pipeline."""
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()


@pytest.fixture
def corrupt_image_bytes() -> bytes:
    """Raw garbage bytes — cv2.imdecode returns None, triggering ContourNotFoundError."""
    return b"not an image"


@pytest.fixture
def document_image_bytes() -> bytes:
    """
    Synthetic document image: white rectangle on a dark background.
    Provides enough contrast for the Canny + contour pipeline to find a
    four-point boundary without requiring a real photograph in the test suite.
    """
    canvas = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.rectangle(canvas, (80, 60), (720, 540), (255, 255, 255), -1)
    _, buffer = cv2.imencode(".jpg", canvas)
    return buffer.tobytes()
