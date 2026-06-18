import io
import time


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
            assert "binarized_image_b64" in result
            assert "detections" in result
            assert result["processing_time_ms"] >= 0
            return
        time.sleep(0.5)

    raise AssertionError("Job did not resolve within 20 seconds.")
