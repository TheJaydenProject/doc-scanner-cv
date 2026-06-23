import json
from unittest import mock

import pipeline.openrouter as openrouter


def test_empty_text_skips_api(monkeypatch):
    """Whitespace-only input is returned as-is without an API call."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    urlopen = mock.Mock()
    monkeypatch.setattr(openrouter.urllib.request, "urlopen", urlopen)

    assert openrouter.correct_ocr_text("   ") == "   "
    urlopen.assert_not_called()


def test_missing_key_returns_raw(monkeypatch):
    """With no key configured, cleanup is skipped and raw text passes through."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert openrouter.correct_ocr_text("teh cat sat") == "teh cat sat"


def test_happy_path_returns_cleaned(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    payload = {"choices": [{"message": {"content": "The cat sat."}}]}
    response = mock.MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(payload).encode()
    monkeypatch.setattr(
        openrouter.urllib.request, "urlopen", mock.Mock(return_value=response)
    )

    assert openrouter.correct_ocr_text("teh cat sat") == "The cat sat."


def test_api_failure_returns_raw(monkeypatch):
    """Any request failure degrades to the original OCR text, never raises."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        openrouter.urllib.request,
        "urlopen",
        mock.Mock(side_effect=Exception("network down")),
    )

    assert openrouter.correct_ocr_text("teh cat sat") == "teh cat sat"


def test_blank_completion_falls_back_to_raw(monkeypatch):
    """An empty model completion (e.g. a safety block) yields the raw text."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    payload = {"choices": [{"message": {"content": "   "}}]}
    response = mock.MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(payload).encode()
    monkeypatch.setattr(
        openrouter.urllib.request, "urlopen", mock.Mock(return_value=response)
    )

    assert openrouter.correct_ocr_text("teh cat sat") == "teh cat sat"


def test_drastically_short_completion_falls_back_to_raw(monkeypatch):
    """A completion that summarizes instead of correcting (much shorter than
    the input) is rejected in favor of the raw text."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    raw = "This is a long paragraph of OCR text that should be corrected verbatim. " * 5
    payload = {"choices": [{"message": {"content": "Short summary."}}]}
    response = mock.MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(payload).encode()
    monkeypatch.setattr(
        openrouter.urllib.request, "urlopen", mock.Mock(return_value=response)
    )

    assert openrouter.correct_ocr_text(raw) == raw
