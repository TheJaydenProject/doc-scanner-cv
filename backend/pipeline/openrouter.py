import json
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

_MODEL = "deepseek/deepseek-v4-flash"
_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
_TIMEOUT_S = 15

# Refactored prompt: Allows context-driven spelling, casing, and punctuation healing 
# while maintaining strict structural parity.
_BASE_PROMPT = (
    "You are a strict OCR post-correction engine. Your single task is to fix artifacts "
    "introduced by text recognition errors (OCR misreads) using the surrounding context. "
    "You must return the corrected text, maintaining exact structural alignment with the input.\n"
    "\n"
    "ALLOWED CORRECTIONS:\n"
    "1. Fix broken punctuation caused by visual confusion (e.g., semicolons mistakenly swapped "
    "for commas or periods, missing periods at the end of paragraphs).\n"
    "2. Fix broken words, contractions, and spacing (e.g., 'today s' -> 'today\\'s', 'uS' -> 'us').\n"
    "3. Fix obvious spelling misreads that result in non-words.\n"
    "\n"
    "ABSOLUTE RESTRICTIONS:\n"
    "1. NEVER paraphrase, summarize, rewrite, or improve the author's original style or grammar.\n"
    "2. NEVER alter the line breaks, paragraph structure, or layout of the text.\n"
    "3. Do not add any preamble, markdown blocks, commentary, or explanations. Output ONLY the corrected text."
)

_HANDWRITTEN_HINT = (
    "\n\n"
    "This text is from a handwritten document. Handwriting OCR commonly "
    "confuses letter shapes; use context to fix obvious misreads involving: "
    "a/o, e/c, u/v/n, m/n, rn/m, cl/d, g/y, b/h, l/t."
)

_PRINTED_HINT = (
    "\n\n"
    "This text is from a printed document. Printed-font OCR commonly "
    "confuses these character pairs: 0/O, 1/l/I, 5/S, 8/B, rn/m, cl/d, vv/w. "
    "It also frequently misinterprets small punctuation marks (e.g., commas as semicolons). "
    "Be conservative with anything that looks like a deliberate serial number, code, or date."
)

def _build_system_prompt(doc_type: str | None) -> str:
    if doc_type == "handwritten":
        return _BASE_PROMPT + _HANDWRITTEN_HINT
    if doc_type == "printed":
        return _BASE_PROMPT + _PRINTED_HINT
    return _BASE_PROMPT


def correct_ocr_text(text: str, doc_type: str | None = None) -> str:
    """Clean raw OCR output with DeepSeek V4 Flash via OpenRouter.
    
    Maintains layout while correcting spelling, casing, and punctuation errors.
    Degrades gracefully to raw text on any failure condition.
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
            "temperature": 0,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        _ENDPOINT,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/TheJaydenProject/doc-scanner-cv",
            "X-Title": "Doc Scanner CV",
            "User-Agent": "DocScannerCV/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
            payload = json.loads(response.read())
        cleaned = payload["choices"][0]["message"]["content"].strip()
        return cleaned or text
    except Exception as e:
        logger.warning("OCR cleanup via OpenRouter failed (%s); returning raw text.", e)
        return text


if __name__ == "__main__":
    assert correct_ocr_text("") == ""
    assert correct_ocr_text("   ") == "   "
    os.environ.pop("OPENROUTER_API_KEY", None)
    assert correct_ocr_text("teh cat") == "teh cat"
    assert _build_system_prompt("handwritten") == _BASE_PROMPT + _HANDWRITTEN_HINT
    assert _build_system_prompt("printed") == _BASE_PROMPT + _PRINTED_HINT
    assert _build_system_prompt(None) == _BASE_PROMPT
    print("openrouter self-check OK")
