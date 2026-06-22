import json
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

# Gemini Flash is fast and inexpensive, and context-aware spelling/punctuation/
# casing correction is one of its most reliable uses. Change this id here if
# Google retires the model (verified live: the v1beta model path exists).
_MODEL = "gemini-2.5-flash"
_ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{_MODEL}:generateContent"
)
# Cleanup is the last stage of a scan that the frontend polls with a 60s budget,
# so cap the call well short of that; on timeout we fall back to the raw text.
_TIMEOUT_S = 15

_SYSTEM_PROMPT = (
    "You are a specialized OCR correction engine. Your strict mandate is to fix "
    "spelling, punctuation, and casing errors caused by poor OCR extraction. Do "
    "not rewrite the text, do not alter the author's original tone or vocabulary, "
    "and do not summarize. Output only the exact corrected text."
)


def correct_ocr_text(text: str) -> str:
    """
    Clean raw OCR output with Gemini Flash: spelling, punctuation, and casing
    only, never a rewrite or summary.

    This is a best-effort enhancement. If the API key is unset, the text is
    empty, or the request fails for any reason (network, non-200, blocked or
    malformed response), the original text is returned unchanged so a cleanup
    outage never costs the user their scan.
    """
    if not text or not text.strip():
        return text

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.info("GEMINI_API_KEY not set — skipping OCR cleanup.")
        return text

    body = json.dumps(
        {
            "system_instruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            # Deterministic correction, not creative rewriting.
            "generationConfig": {"temperature": 0},
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        _ENDPOINT,
        data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
            payload = json.loads(response.read())
        cleaned = payload["candidates"][0]["content"]["parts"][0]["text"].strip()
        # An empty completion (e.g. a safety block) is worse than the raw text.
        return cleaned or text
    except Exception as e:
        # Deliberately broad: any cleanup failure must degrade to the raw OCR
        # text, never propagate and fail the scan job.
        logger.warning("OCR cleanup via Gemini failed (%s); returning raw text.", e)
        return text


if __name__ == "__main__":
    # Self-check: the no-network paths return the input unchanged.
    assert correct_ocr_text("") == ""
    assert correct_ocr_text("   ") == "   "
    os.environ.pop("GEMINI_API_KEY", None)
    assert correct_ocr_text("teh cat") == "teh cat"
    print("gemini self-check OK")
