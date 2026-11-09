"""API test client isolated from the project database."""

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("OLLAMA_MODEL", "gemma3:1b")
    settings = replace(
        get_settings(),
        database_path=tmp_path / "resumes.db",
        upload_dir=tmp_path / "uploads",
        ollama_base_url="http://127.0.0.1:9",
        ollama_model="gemma3:1b",
    )
    application = create_app(settings)
    with TestClient(application) as client:
        yield client
