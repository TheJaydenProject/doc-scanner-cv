import io
import time

import api.documents as doc_api
from api.documents import MIN_TEXT_HEIGHT_PX, _job_store, _median_text_height


def test_rejects_missing_file(client):
    res = client.post("/api/documents/scan")
    assert res.status_code == 400
    assert "No file provided" in res.get_json()["error"]


def test_rejects_wrong_mime_type(client):
    data = {"file": (io.BytesIO(b"fake"), "test.txt", "text/plain")}
    res = client.post("/api/documents/scan", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    assert "Unsupported file type" in res.get_json()["error"]


def test_rejects_empty_file(client):
    data = {"file": (io.BytesIO(b""), "empty.jpg", "image/jpeg")}
    res = client.post("/api/documents/scan", data=data, content_type="multipart/form-data")
    assert res.status_code == 400


def test_scan_returns_202_and_job_id(client, minimal_jpeg_bytes):
    data = {"file": (io.BytesIO(minimal_jpeg_bytes), "test.jpg", "image/jpeg")}
    res = client.post("/api/documents/scan", data=data, content_type="multipart/form-data")
    assert res.status_code == 202
    assert "job_id" in res.get_json()


def test_job_not_found(client):
    res = client.get("/api/documents/jobs/does-not-exist")
    assert res.status_code == 404


def test_history_returns_list(client):
    res = client.get("/api/documents/history")
    assert res.status_code == 200
    assert isinstance(res.get_json(), list)


def test_metrics_returns_expected_keys(client):
    res = client.get("/api/documents/metrics")
    data = res.get_json()
    assert "total_scans" in data
    assert "avg_processing_time_ms" in data
    assert "recent" in data


def test_async_job_completes_with_fallback_on_blank_image(client, blank_image_bytes):
    """
    Blank image has no detectable contour — pipeline falls back to full-frame
    binarization and returns a complete job rather than failing.
    """
    data = {"file": (io.BytesIO(blank_image_bytes), "blank.jpg", "image/jpeg")}
    submit_res = client.post("/api/documents/scan", data=data, content_type="multipart/form-data")
    assert submit_res.status_code == 202

    job_id = submit_res.get_json()["job_id"]

    for _ in range(20):
        poll_res = client.get(f"/api/documents/jobs/{job_id}")
        assert poll_res.status_code == 200
        job = poll_res.get_json()
        if job["status"] in ("complete", "failed"):
            assert job["status"] == "complete"
            assert "result" in job
            return
        time.sleep(0.5)

    raise AssertionError("Job did not resolve within 10 seconds.")


def test_async_job_resolves_failed_on_corrupt_bytes(client, corrupt_image_bytes):
    """
    Corrupt bytes cannot be decoded — pipeline raises ContourNotFoundError
    before the fallback, so the job must resolve as failed.
    """
    data = {"file": (io.BytesIO(corrupt_image_bytes), "corrupt.jpg", "image/jpeg")}
    submit_res = client.post("/api/documents/scan", data=data, content_type="multipart/form-data")
    assert submit_res.status_code == 202

    job_id = submit_res.get_json()["job_id"]

    for _ in range(20):
        poll_res = client.get(f"/api/documents/jobs/{job_id}")
        assert poll_res.status_code == 200
        job = poll_res.get_json()
        if job["status"] in ("complete", "failed"):
            assert job["status"] == "failed"
            assert "error" in job
            return
        time.sleep(0.5)

    raise AssertionError("Job did not resolve within 10 seconds.")


def test_async_job_resolves_complete_on_valid_image(client, document_image_bytes):
    """
    Submit a synthetic document image (white rect on dark background).
    Pipeline should detect the contour, extract text (possibly empty), and
    return status: complete with the expected result shape.
    """
    data = {"file": (io.BytesIO(document_image_bytes), "doc.jpg", "image/jpeg")}
    submit_res = client.post("/api/documents/scan", data=data, content_type="multipart/form-data")
    assert submit_res.status_code == 202

    job_id = submit_res.get_json()["job_id"]

    for _ in range(40):
        poll_res = client.get(f"/api/documents/jobs/{job_id}")
        assert poll_res.status_code == 200
        job = poll_res.get_json()
        if job["status"] in ("complete", "failed"):
            assert job["status"] == "complete"
            result = job["result"]
            assert "text" in result
            assert "char_count" in result
            assert "processing_time_ms" in result
            assert "warped_image_b64" in result
            assert "detection_count" in result
            assert result["processing_time_ms"] >= 0
            return
        time.sleep(0.5)

    raise AssertionError("Job did not resolve within 20 seconds.")


def test_cancel_marks_processing_job_cancelled(client):
    _job_store["job-x"] = {"status": "processing"}
    res = client.delete("/api/documents/jobs/job-x")
    assert res.status_code == 200
    assert _job_store["job-x"]["status"] == "cancelled"


def test_cancel_does_not_clobber_finished_job(client):
    """A job that already completed/failed must never be overwritten by a late cancel."""
    _job_store["job-y"] = {"status": "complete", "result": {"text": "done"}}
    res = client.delete("/api/documents/jobs/job-y")
    assert res.status_code == 200
    assert _job_store["job-y"]["status"] == "complete"


def test_cancel_unknown_job_is_a_noop(client):
    res = client.delete("/api/documents/jobs/does-not-exist")
    assert res.status_code == 200


def test_cancel_releases_ip_and_job_count_immediately(client):
    """
    Cancelling must free the slot right away rather than waiting for the
    background thread to notice — otherwise a quick Stop-then-Scan from the
    same IP would still be rejected with "a scan is already in progress".
    """
    doc_api._active_ips.add("9.9.9.9")
    doc_api._job_ip["job-z"] = "9.9.9.9"
    with doc_api._active_job_lock:
        doc_api._active_job_count = 1
    _job_store["job-z"] = {"status": "processing"}

    res = client.delete("/api/documents/jobs/job-z")

    assert res.status_code == 200
    assert "9.9.9.9" not in doc_api._active_ips
    assert doc_api._active_job_count == 0
    assert "job-z" not in doc_api._job_ip


def test_release_job_slot_is_idempotent(client):
    """
    Cancellation and the job's own natural-completion finally both call
    _release_job_slot for the same job_id. The second call must be a no-op —
    otherwise _active_job_count double-decrements and drifts negative,
    silently raising the real concurrency cap above _MAX_CONCURRENT_GLOBAL.
    """
    doc_api._active_ips.add("8.8.8.8")
    doc_api._job_ip["job-w"] = "8.8.8.8"
    with doc_api._active_job_lock:
        doc_api._active_job_count = 1

    doc_api._release_job_slot("job-w")
    doc_api._release_job_slot("job-w")

    assert doc_api._active_job_count == 0
    assert "8.8.8.8" not in doc_api._active_ips


def test_median_text_height_returns_none_below_sample_size():
    """Fewer than 5 boxes — median would be too volatile, so skip the gate."""
    detections = [(0, 0, 5, h) for h in (10, 12, 14, 16)]
    assert _median_text_height(detections) is None


def test_median_text_height_computes_median_at_sample_size():
    detections = [(0, 0, 5, h) for h in (10, 12, 14, 16, 18)]
    assert _median_text_height(detections) == 14


def test_median_text_height_flags_low_resolution():
    detections = [(0, 0, 5, h) for h in (10, 12, 14, 16, 18)]
    median = _median_text_height(detections)
    assert median is not None
    assert median < MIN_TEXT_HEIGHT_PX
