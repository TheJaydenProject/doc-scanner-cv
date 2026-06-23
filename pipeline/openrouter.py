import json
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

# DeepSeek V4 Flash via OpenRouter is fast and inexpensive, and context-aware
# spelling/punctuation/casing correction is one of its most reliable uses.
_MODEL = "deepseek/deepseek-v4-flash"
_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
# Cleanup is the last stage of a scan that the frontend polls with a 60s budget,
# so cap the call well short of that; on timeout we fall back to the raw text.
_TIMEOUT_S = 15

_BASE_PROMPT = (
    "You are a strict OCR post-correction engine. You receive raw text "
    "extracted by an OCR pipeline from a scanned document. Your ONLY task is "
    "to fix spelling errors where the OCR clearly misread characters to produce "
    "a non-word (e.g. 'teh' -> 'the', 'rnake' -> 'make').\n"
    "\n"
    "ABSOLUTE RULES:\n"
    "1. NEVER substitute a valid dictionary word for another valid dictionary word. "
    "If the text says 'destruction', keep 'destruction' (do not change to 'construction'). "
    "If the text says 'and', keep 'and' (do not change to 'or').\n"
    "2. NEVER alter punctuation. Do not swap periods for semicolons, colons, or commas. "
    "Do not delete or modify ellipses (...).\n"
    "3. NEVER alter line breaks. If there is a newline, keep it. Do not merge lines.\n"
    "4. NEVER alter capitalization. If a word starts with a capital or lowercase letter, leave it exactly as is.\n"
    "5. NEVER change contractions. Do not change 'it is' to 'it\\'s' or vice versa.\n"
    "6. Never add commentary, preambles, or markdown formatting.\n"
    "7. If a word is garbled, leave it as-is."
)

_HANDWRITTEN_HINT = (
    "\n\n"
    "This text is from a handwritten document. Handwriting OCR commonly "
    "confuses letter shapes; use context to fix obvious non-word misreads involving: "
    "a/o, e/c, u/v/n, m/n, rn/m, cl/d, g/y, b/h, l/t. "
    "Again, ONLY fix clear misreads that result in invalid words. DO NOT attempt to "
    "'improve' the grammar, punctuation, phrasing, or capitalization of the handwritten text."
)

_PRINTED_HINT = (
    "\n\n"
    "This text is from a printed document. Printed-font OCR commonly "
    "confuses these character pairs: 0/O, 1/l/I, 5/S, 8/B, rn/m, cl/d, vv/w. "
    "Use context to fix obvious non-word misreads involving these pairs. "
    "Be conservative with anything that looks like a deliberate field value "
    "(an account number, date, amount, or code)."
)

def _build_system_prompt(doc_type: str | None) -> str:
    if doc_type == "handwritten":
        return _BASE_PROMPT + _HANDWRITTEN_HINT
    if doc_type == "printed":
        return _BASE_PROMPT + _PRINTED_HINT
    return _BASE_PROMPT


def correct_ocr_text(text: str, doc_type: str | None = None) -> str:
    """
    Clean raw OCR output with DeepSeek V4 Flash (via OpenRouter): spelling,
    punctuation, and casing only, never a rewrite or summary.

    doc_type ("printed" or "handwritten", from classify_document()) tailors
    the system prompt to that register; omit it for the generic prompt.

    This is a best-effort enhancement. If the API key is unset, the text is
    empty, or the request fails for any reason (network, non-200, blocked or
    malformed response), the original text is returned unchanged so a cleanup
    outage never costs the user their scan.
    """
    if not text or not text.strip():
        return text

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.info("OPENROUTER_API_KEY not set — skipping OCR cleanup.")
        return text

    body = json.dumps(
        {
            "model": _MODEL,
            "messages": [
                {"role": "system", "content": _build_system_prompt(doc_type)},
                {"role": "user", "content": text},
            ],
            # Deterministic correction, not creative rewriting.
            "temperature": 0,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        _ENDPOINT,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
            payload = json.loads(response.read())
        cleaned = payload["choices"][0]["message"]["content"].strip()
        # An empty completion (e.g. a safety block) is worse than the raw text.
        return cleaned or text
    except Exception as e:
        # Deliberately broad: any cleanup failure must degrade to the raw OCR
        # text, never propagate and fail the scan job.
        logger.warning("OCR cleanup via OpenRouter failed (%s); returning raw text.", e)
        return text


if __name__ == "__main__":
    # Self-check: the no-network paths return the input unchanged.
    assert correct_ocr_text("") == ""
    assert correct_ocr_text("   ") == "   "
    os.environ.pop("OPENROUTER_API_KEY", None)
    assert correct_ocr_text("teh cat") == "teh cat"
    # Each doc_type gets its own tailored hint; unknown/missing stays generic.
    assert _build_system_prompt("handwritten") == _BASE_PROMPT + _HANDWRITTEN_HINT
    assert _build_system_prompt("printed") == _BASE_PROMPT + _PRINTED_HINT
    assert _build_system_prompt(None) == _BASE_PROMPT
    print("openrouter self-check OK")
