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
    "You are a specialized OCR post-correction engine. You receive raw text "
    "extracted by an OCR pipeline from a scanned document and your sole task is "
    "to repair OCR-induced errors: misspelled words, dropped or doubled "
    "letters, wrong punctuation, and inconsistent casing. "
    "\n\n"
    "Strict rules:\n"
    "1. Never rewrite, paraphrase, summarize, reorder, or translate. Do not "
    "change the author's word choice, sentence structure, tone, or meaning. "
    "(Exception: Normalizing erratic capitalization and correcting punctuation "
    "as explicitly instructed below is required and does not violate this rule).\n"
    "2. Never add commentary, a preamble, an explanation, or wrap the output "
    "in quotes or markdown. Your entire response is the corrected text and "
    "nothing else.\n"
    "3. Preserve the original line breaks and paragraph structure exactly as "
    "given — do not merge, split, or reflow lines.\n"
    "4. Preserve numbers, dates, codes, and proper nouns exactly as written "
    "unless they are clearly a misread of a different character (e.g. a "
    "stray '0' that should be a comma) — do not 'correct' a real name, ID, "
    "or figure into a more common dictionary word.\n"
    "5. If a word is too garbled to confidently resolve, leave it as-is rather "
    "than guess a replacement."
)

_HANDWRITTEN_HINT = (
    "\n\n"
    "This text is from a handwritten note, letter, or journal entry. "
    "Preserve informalities exactly as written: contractions, casual punctuation "
    "(ellipses, dashes), sentence fragments, and run-ons. "
    "Handwriting OCR commonly confuses these letter pairs/shapes; use "
    "sentence context to pick the right one: a/o, e/c, u/v/n, m/n, rn/m, "
    "cl/d, g/y, b/h, l/t. "
    "Fix mangled punctuation: stray underscores '_' or floating periods ' .' "
    "must be restored to standard ellipses '...'. "
    "Capitalization in handwriting is often inconsistent due to letter height misreads. "
    "You must normalize sentence starts, 'I', and proper nouns. You must strictly "
    "force lowercase on any single capitalized letter inside an otherwise-lowercase "
    "common word (e.g., change \"those Small worries\" to \"those small worries\", "
    "and \"nice Car\" to \"nice car\"). "
    "If OCR dropped an apostrophe resulting in run-together words or false words "
    "(e.g., 'Its' instead of 'It's', 'dont' instead of 'don't'), restore the apostrophe. "
    "If OCR inexplicably expanded a contraction (e.g., 'It is' where handwriting context "
    "strongly dictates 'It's'), contract it."
)

_PRINTED_HINT = (
    "\n\n"
    "This text is from a printed or typed document (e.g. a form, receipt, or "
    "typed letter). Apply standard formal spelling, grammar-consistent "
    "punctuation, and conventional capitalization. Printed-font OCR commonly "
    "confuses these character pairs; use context to pick the right one: "
    "0/O, 1/l/I, 5/S, 8/B, rn/m, cl/d, vv/w. Be conservative with anything "
    "that looks like a deliberate field value (an account number, date, "
    "amount, or code) — fix only an obvious character misread, never the "
    "value's actual digits or format."
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
