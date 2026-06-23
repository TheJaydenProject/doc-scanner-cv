import numpy as np
import cv2
import pytest
from app import create_app
from tasks import celery_app, redis_client

# Run Celery tasks synchronously in-process — tests have no separate worker.
celery_app.conf.update(task_always_eager=True, task_eager_propagates=True)


@pytest.fixture(autouse=True)
def reset_scan_state():
    """Flush Redis job state between tests for isolation."""
    redis_client.flushdb()
    yield
    redis_client.flushdb()


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
def flat_document_image_bytes() -> bytes:
    """
    A white canvas with a thin border right at the edges — mimics a flat
    digital screenshot whose frame gets picked up as a contour even though
    it covers almost the entire image (no real document boundary).
    """
    canvas = np.ones((600, 800, 3), dtype=np.uint8) * 255
    cv2.rectangle(canvas, (2, 2), (797, 597), (0, 0, 0), 2)
    _, buffer = cv2.imencode(".jpg", canvas)
    return buffer.tobytes()


@pytest.fixture
def small_blob_image_bytes() -> bytes:
    """
    A white canvas with one small isolated rectangle — mimics a flat document
    where contour detection locks onto a stray text blob instead of a real
    document boundary (the contour ratio ends up implausibly small).
    """
    canvas = np.ones((600, 800, 3), dtype=np.uint8) * 255
    cv2.rectangle(canvas, (350, 280), (410, 320), (0, 0, 0), 2)
    _, buffer = cv2.imencode(".jpg", canvas)
    return buffer.tobytes()


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


@pytest.fixture
def busy_report_image_bytes() -> bytes:
    """
    A white canvas with 5 internal black-outlined rectangles of comparable
    size stacked top-to-bottom — mimics a dense, borderless printed page
    where contour detection used to lock onto one of the internal blocks
    (heading, paragraph, table) instead of a real page boundary that
    doesn't exist.  No outer boundary is present.

    The blocks are spaced far enough apart (> 11 px) that the 11×11
    morphological close cannot bridge them into a single merged contour,
    and each block is small enough that even if Pass 2/3 picks one up,
    the downstream area-ratio gate (MIN_CONTOUR_AREA_RATIO) rejects it.

    Expected: no candidate survives Pass 1 (every quad has a comparably-sized
    rival), run_pipeline returns warped=False with the full original frame.
    """
    canvas = np.ones((800, 1000, 3), dtype=np.uint8) * 255
    # Five roughly-equal-height horizontal blocks with ~35px vertical gaps,
    # each producing a clean 4-point contour.
    for i in range(5):
        y_top = 30 + i * 155
        y_bot = y_top + 120
        cv2.rectangle(canvas, (100, y_top), (900, y_bot), (0, 0, 0), 3)
    _, buffer = cv2.imencode(".jpg", canvas)
    return buffer.tobytes()


@pytest.fixture
def busy_report_with_real_boundary_image_bytes() -> bytes:
    """
    Same internal blocks as busy_report_image_bytes, but placed on a dark
    background with a genuine large outer boundary rectangle near the frame
    edges — mimics a photo of a busy printed report on a desk.

    Expected: find_document_contour returns the outer boundary quad (not any
    internal block), run_pipeline returns warped=True cropped to roughly the
    outer rectangle.
    """
    canvas = np.full((600, 800, 3), 40, dtype=np.uint8)  # dark background
    # Outer boundary — large white rectangle
    cv2.rectangle(canvas, (30, 20), (770, 580), (255, 255, 255), -1)
    # Internal blocks (same layout, drawn on the white interior)
    for i in range(5):
        y_top = 40 + i * 105
        y_bot = y_top + 90
        cv2.rectangle(canvas, (60, y_top), (740, y_bot), (0, 0, 0), 3)
    _, buffer = cv2.imencode(".jpg", canvas)
    return buffer.tobytes()


@pytest.fixture
def single_secondary_block_image_bytes() -> bytes:
    """
    A dark canvas with a large white document rectangle plus one secondary
    internal rectangle whose area is clearly less than 50% of the outer
    boundary — regression guard proving the size-gap threshold doesn't
    false-reject documents with one large embedded element (e.g. a table).

    Expected: still warps correctly to the full page.
    """
    canvas = np.full((600, 800, 3), 40, dtype=np.uint8)
    # Outer boundary
    cv2.rectangle(canvas, (50, 30), (750, 570), (255, 255, 255), -1)
    # One secondary block — roughly 35% of the outer boundary's area
    # Outer: ~700x540=378000; Inner: ~300x180=54000 => ratio ~14%
    cv2.rectangle(canvas, (200, 200), (500, 380), (0, 0, 0), 3)
    _, buffer = cv2.imencode(".jpg", canvas)
    return buffer.tobytes()

