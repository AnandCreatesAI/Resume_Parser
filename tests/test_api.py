"""API tests. The local model is mocked."""

import json
from dataclasses import replace

from fastapi.testclient import TestClient

from app.config import get_settings
from app.exceptions import OllamaUnavailableError
from app.main import create_app
from tests.helpers import blank_pdf_bytes, docx_bytes, pdf_bytes

RESUME_LINES = [
    "Amina Rahman",
    "Email: amina@example.com",
    "Phone: 4155550134",
    "Location: San Francisco, CA",
    "Company: Northwind Analytics",
    "Role: Software Engineer",
    "Duration: 2020-2022",
    "Skills: Python, SQL, Communication",
]


def _model(monkeypatch, payload):
    def fake_generate(prompt: str) -> str:
        if isinstance(payload, str):
            return payload
        return json.dumps(payload)

    monkeypatch.setattr(
        "app.services.resume_extractor.generate_response",
        fake_generate,
    )


def _upload(client, filename: str, content: bytes, content_type: str):
    return client.post(
        "/api/upload",
        files={"file": (filename, content, content_type)},
    )


def test_upload_pdf_and_get_it_back(api_client, monkeypatch):
    _model(
        monkeypatch,
        {
            "candidate_name": "Amina Rahman",
            "email": "amina@example.com",
            "phone": "4155550134",
            "location": "San Francisco, CA",
            "sections": [
                {
                    "heading": "Selected Work",
                    "content": ["Northwind Analytics — Software Engineer — 2020-2022"],
                },
                {
                    "heading": "Technical Expertise",
                    "content": [],
                    "subsections": [
                        {"heading": "Languages", "content": ["Python", "SQL"]},
                        {"heading": "Working Style", "content": ["Communication"]},
                    ],
                },
            ],
        },
    )

    response = _upload(
        api_client,
        "candidate.pdf",
        pdf_bytes(RESUME_LINES),
        "application/pdf",
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["filename"] == "candidate.pdf"
    assert body["resume"]["candidate_name"] == "Amina Rahman"
    assert body["resume"]["email"] == "amina@example.com"
    assert [section["heading"] for section in body["resume"]["sections"]] == [
        "Selected Work",
        "Technical Expertise",
    ]
    expertise = body["resume"]["sections"][1]
    assert [item["heading"] for item in expertise["subsections"]] == [
        "Languages",
        "Working Style",
    ]

    loaded = api_client.get(f"/api/resume/{body['document_id']}")
    assert loaded.status_code == 200
    assert loaded.json()["document_id"] == body["document_id"]
    assert loaded.json()["resume"]["sections"][0]["content"] == [
        "Northwind Analytics — Software Engineer — 2020-2022"
    ]


def test_upload_docx(api_client, monkeypatch):
    _model(
        monkeypatch,
        {
            "candidate_name": "Amina Rahman",
            "email": "amina@example.com",
            "sections": [{"heading": "Toolbox", "content": ["Python"]}],
        },
    )

    response = _upload(
        api_client,
        "candidate.docx",
        docx_bytes(["Amina Rahman", "amina@example.com", "Skills: Python"]),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    assert response.status_code == 200
    assert response.json()["resume"]["email"] == "amina@example.com"
    assert response.json()["resume"]["sections"][0]["heading"] == "Toolbox"
    assert response.json()["resume"]["sections"][0]["content"] == ["Python"]


def test_unsupported_extension(api_client):
    response = _upload(api_client, "notes.txt", b"hello", "text/plain")

    assert response.status_code == 400
    assert "PDF and DOCX" in response.json()["detail"]


def test_empty_file(api_client):
    response = _upload(api_client, "empty.pdf", b"", "application/pdf")

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_invalid_pdf(api_client):
    response = _upload(
        api_client,
        "broken.pdf",
        b"%PDF-1.4\nthis is not a real pdf",
        "application/pdf",
    )

    assert response.status_code == 400
    assert "corrupted" in response.json()["detail"].lower()


def test_invalid_docx(api_client):
    response = _upload(api_client, "broken.docx", b"not-a-docx", "application/octet-stream")

    assert response.status_code == 400
    assert "corrupted" in response.json()["detail"].lower()


def test_empty_extracted_text_does_not_call_the_model(api_client, monkeypatch):
    called = {"count": 0}

    def fake_generate(prompt: str) -> str:
        called["count"] += 1
        return "{}"

    monkeypatch.setattr(
        "app.services.resume_extractor.generate_response",
        fake_generate,
    )

    response = _upload(api_client, "blank.pdf", blank_pdf_bytes(), "application/pdf")

    assert response.status_code == 422
    assert "readable text" in response.json()["detail"].lower()
    assert called["count"] == 0


def test_ollama_unavailable(api_client, monkeypatch):
    def fake_generate(prompt: str) -> str:
        raise OllamaUnavailableError("The local Ollama service is unavailable.")

    monkeypatch.setattr(
        "app.services.resume_extractor.generate_response",
        fake_generate,
    )

    response = _upload(
        api_client,
        "candidate.pdf",
        pdf_bytes(["Amina Rahman", "amina@example.com"]),
        "application/pdf",
    )

    assert response.status_code == 503
    assert "Ollama" in response.json()["detail"]


def test_invalid_model_json(api_client, monkeypatch):
    _model(monkeypatch, "this is not json")

    response = _upload(
        api_client,
        "candidate.pdf",
        pdf_bytes(["Amina Rahman", "amina@example.com"]),
        "application/pdf",
    )

    assert response.status_code == 422
    assert "JSON" in response.json()["detail"]


def test_missing_document_returns_404(api_client):
    response = api_client.get("/api/resume/does-not-exist")

    assert response.status_code == 404
    assert "document ID" in response.json()["detail"]


def test_duplicate_uploads_receive_different_ids(api_client, monkeypatch):
    _model(
        monkeypatch,
        {
            "candidate_name": "Amina Rahman",
            "email": "amina@example.com",
            "sections": [{"heading": "Toolbox", "content": ["Python"]}],
        },
    )
    content = pdf_bytes(["Amina Rahman", "amina@example.com", "Skills: Python"])

    first = _upload(api_client, "same.pdf", content, "application/pdf")
    second = _upload(api_client, "same.pdf", content, "application/pdf")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["document_id"] != second.json()["document_id"]
    assert api_client.get(f"/api/resume/{first.json()['document_id']}").status_code == 200
    assert api_client.get(f"/api/resume/{second.json()['document_id']}").status_code == 200


def test_oversized_upload(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "gemma3:1b")
    settings = replace(
        get_settings(),
        database_path=tmp_path / "resumes.db",
        upload_dir=tmp_path / "uploads",
        max_upload_bytes=32,
        max_upload_mb=1,
    )
    with TestClient(create_app(settings)) as client:
        response = _upload(
            client,
            "big.pdf",
            b"%PDF" + b"x" * 40,
            "application/pdf",
        )

    assert response.status_code == 413
    assert "upload limit" in response.json()["detail"]


def test_database_failure_on_lookup(api_client, monkeypatch):
    from app.exceptions import DatabaseError

    def fail(document_id: str):
        raise DatabaseError()

    monkeypatch.setattr(api_client.app.state.repository, "get", fail)

    response = api_client.get("/api/resume/any-id")

    assert response.status_code == 500
    assert "database" in response.json()["detail"].lower()
