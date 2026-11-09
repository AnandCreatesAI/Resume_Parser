"""Tests for SQLite persistence."""

import pytest

from app.database.repository import ResumeRepository
from app.exceptions import DatabaseError
from app.models import ResumeData


def _resume() -> ResumeData:
    return ResumeData.model_validate(
        {
            "candidate_name": "Amina Rahman",
            "email": "amina@example.com",
            "sections": [
                {
                    "heading": "Publications",
                    "content": ["Rahman, A. Warehouse events. 2022."],
                }
            ],
        }
    )


def test_resume_survives_a_new_connection(tmp_path):
    database_path = tmp_path / "resumes.db"
    repository = ResumeRepository(database_path)
    saved = repository.create(
        document_id="doc-1",
        filename="resume.pdf",
        stored_path=tmp_path / "resume.pdf",
        resume=_resume(),
    )

    reopened = ResumeRepository(database_path)
    loaded = reopened.get("doc-1")

    assert loaded is not None
    assert loaded["document_id"] == saved["document_id"]
    assert loaded["filename"] == "resume.pdf"
    assert loaded["status"] == "completed"
    assert loaded["resume"]["candidate_name"] == "Amina Rahman"
    assert loaded["resume"]["email"] == "amina@example.com"
    assert loaded["resume"]["sections"][0]["heading"] == "Publications"
    assert loaded["resume"]["sections"][0]["content"] == ["Rahman, A. Warehouse events. 2022."]
    assert loaded["created_at"]


def test_duplicate_uploads_keep_separate_ids(tmp_path):
    repository = ResumeRepository(tmp_path / "resumes.db")
    first = repository.create(
        document_id="doc-1",
        filename="resume.pdf",
        stored_path=tmp_path / "one.pdf",
        resume=_resume(),
    )
    second = repository.create(
        document_id="doc-2",
        filename="resume.pdf",
        stored_path=tmp_path / "two.pdf",
        resume=_resume(),
    )

    assert first["document_id"] != second["document_id"]
    assert repository.get("doc-1")["document_id"] == "doc-1"
    assert repository.get("doc-2")["document_id"] == "doc-2"


def test_missing_document_returns_none(tmp_path):
    repository = ResumeRepository(tmp_path / "resumes.db")

    assert repository.get("missing") is None


def test_directory_path_raises_database_error(tmp_path):
    bad_path = tmp_path / "not-a-database"
    bad_path.mkdir()

    with pytest.raises(DatabaseError):
        ResumeRepository(bad_path)
