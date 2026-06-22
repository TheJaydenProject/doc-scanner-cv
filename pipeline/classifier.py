from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

_CLASS_LABELS: list[str] = ["printed", "handwritten", "mixed"]

# Session is loaded once per process and reused across requests.
# ONNX Runtime session creation is expensive (~100-300ms); loading per-request
# would add measurable latency to every scan.
_session: ort.InferenceSession | None = None
_INPUT_NAME: str = ""


def _get_session(model_path: str = "models/doc_classifier.onnx") -> ort.InferenceSession:
    global _session, _INPUT_NAME
    if _session is None:
        # Defaults to os.cpu_count() intra-op threads, which would let a single
        # classify_document() call use every core on top of the 3-thread scan
        # pool. Pin to 1, mirroring torch.set_num_threads(1) in pipeline/ocr.py.
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        _session = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        _INPUT_NAME = _session.get_inputs()[0].name
    return _session


BRIGHT_PIXEL_THRESHOLD = 0.85
BRIGHT_LUMINANCE_CUTOFF = 240


def _is_likely_printed_document(image: np.ndarray) -> bool:
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bright_ratio = np.sum(gray > BRIGHT_LUMINANCE_CUTOFF) / gray.size
    return bright_ratio > BRIGHT_PIXEL_THRESHOLD


def _preprocess(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224))

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (resized.astype(np.float32) / 255.0 - mean) / std

    return np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]


def classify_document(image: np.ndarray) -> dict[str, object]:
    """
    Runs ONNX inference and returns the predicted class and confidence.

    A high proportion of near-white pixels strongly indicates a flat digital
    document (typed PDF/screenshot export) rather than a photographed page —
    that case is classified directly as printed, bypassing the ONNX model
    whose head is still randomly initialised and unreliable on these inputs.

    Returns:
        {
            "label": "printed" | "handwritten" | "mixed",
            "confidence": float,  # 0.0 - 1.0, softmax probability
            "source": "heuristic" | "model",
        }
    """
    if _is_likely_printed_document(image):
        return {"label": "printed", "confidence": 1.0, "source": "heuristic"}

    session = _get_session()
    input_tensor = _preprocess(image)
    logits: np.ndarray = session.run(None, {_INPUT_NAME: input_tensor})[0][0]

    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / exp_logits.sum()

    predicted_index = int(np.argmax(probabilities))
    return {
        "label": _CLASS_LABELS[predicted_index],
        "confidence": round(float(probabilities[predicted_index]), 4),
        "source": "model",
    }
