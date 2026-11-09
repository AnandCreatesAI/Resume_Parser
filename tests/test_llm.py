"""Tests for the Ollama client. The local model is not called."""

import pytest
import requests

from app.exceptions import LLMResponseError, OllamaUnavailableError
from app.services.llm_service import generate_response


class _Response:
    def __init__(self, status_code=200, payload=None, invalid_json=False):
        self.status_code = status_code
        self._payload = {} if payload is None else payload
        self._invalid_json = invalid_json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        if self._invalid_json:
            raise ValueError("not json")
        return self._payload


def test_generate_response_uses_local_json_mode(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _Response(payload={"response": "{\"name\": null}"})

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "gemma3:1b")
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "45")
    monkeypatch.setattr("app.services.llm_service.requests.post", fake_post)

    text = generate_response("extract this resume")

    assert text == "{\"name\": null}"
    assert captured["url"] == "http://127.0.0.1:11434/api/generate"
    assert captured["json"]["model"] == "gemma3:1b"
    assert captured["json"]["stream"] is False
    assert captured["json"]["format"] == "json"
    assert captured["json"]["options"]["temperature"] == 0
    assert captured["timeout"] == 45
    assert "openai.com" not in captured["url"]


def test_connection_error_reports_ollama_unavailable(monkeypatch):
    def fake_post(url, json, timeout):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr("app.services.llm_service.requests.post", fake_post)

    with pytest.raises(OllamaUnavailableError, match="Ollama service is unavailable"):
        generate_response("hello")


def test_timeout_reports_ollama_unavailable(monkeypatch):
    def fake_post(url, json, timeout):
        raise requests.Timeout("slow")

    monkeypatch.setattr("app.services.llm_service.requests.post", fake_post)

    with pytest.raises(OllamaUnavailableError, match="too long"):
        generate_response("hello")


def test_missing_model_is_reported(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "missing-model")
    monkeypatch.setattr(
        "app.services.llm_service.requests.post",
        lambda url, json, timeout: _Response(status_code=404, payload={}),
    )

    with pytest.raises(OllamaUnavailableError, match="missing-model"):
        generate_response("hello")


def test_http_error_is_reported(monkeypatch):
    monkeypatch.setattr(
        "app.services.llm_service.requests.post",
        lambda url, json, timeout: _Response(status_code=500, payload={"error": "boom"}),
    )

    with pytest.raises(OllamaUnavailableError, match="returned an error"):
        generate_response("hello")


def test_empty_model_response(monkeypatch):
    monkeypatch.setattr(
        "app.services.llm_service.requests.post",
        lambda url, json, timeout: _Response(payload={"response": "   "}),
    )

    with pytest.raises(LLMResponseError, match="empty"):
        generate_response("hello")


def test_non_json_ollama_body(monkeypatch):
    monkeypatch.setattr(
        "app.services.llm_service.requests.post",
        lambda url, json, timeout: _Response(invalid_json=True),
    )

    with pytest.raises(LLMResponseError, match="unreadable"):
        generate_response("hello")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
