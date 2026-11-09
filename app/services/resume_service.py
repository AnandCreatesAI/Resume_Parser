"""Upload pipeline: validate, extract text, call the model, and store the result."""

import uuid
from pathlib import Path

from loguru import logger

from app.config import Settings
from app.database.repository import ResumeRepository
from app.exceptions import (
    EmptyFileError,
    EmptyResumeTextError,
    FileTooLargeError,
    InvalidFileError,
    ResumeNotFoundError,
    UnsupportedFileTypeError,
)
from app.services.parser import extract_resume_text, is_docx_bytes
from app.services.resume_extractor import extract_resume_data


class ResumeService:
    """Coordinates file handling, extraction, and persistence."""

    def __init__(self, settings: Settings, repository: ResumeRepository) -> None:
        self.settings = settings
        self.repository = repository

    def process_upload(self, filename: str | None, content: bytes) -> dict:
        original_name = _display_name(filename)
        extension = self._validate(original_name, content)
        document_id = uuid.uuid4().hex
        stored_path = self._write(document_id, extension, content)
        logger.info(
            "Stored upload document_id={} filename={} bytes={}",
            document_id,
            original_name,
            len(content),
        )

        try:
            text = extract_resume_text(str(stored_path))
            if not text.strip():
                raise EmptyResumeTextError()
            logger.info(
                "Extracted resume text document_id={} characters={}",
                document_id,
                len(text),
            )
            resume = extract_resume_data(text)
            record = self.repository.create(
                document_id=document_id,
                filename=original_name,
                stored_path=stored_path,
                resume=resume,
            )
        except Exception:
            stored_path.unlink(missing_ok=True)
            logger.error(
                "Extraction failed document_id={} filename={}",
                document_id,
                original_name,
            )
            raise

        logger.info("Extraction completed document_id={}", document_id)
        return record

    def get_resume(self, document_id: str) -> dict:
        record = self.repository.get(document_id)
        if record is None:
            logger.info("Resume lookup missed document_id={}", document_id)
            raise ResumeNotFoundError()
        logger.info("Resume lookup succeeded document_id={}", document_id)
        return record

    def _validate(self, filename: str, content: bytes) -> str:
        extension = Path(filename).suffix.lower()
        if extension not in {".pdf", ".docx"}:
            logger.warning("Rejected unsupported file filename={}", filename)
            raise UnsupportedFileTypeError()
        if not content:
            logger.warning("Rejected empty file filename={}", filename)
            raise EmptyFileError()
        if len(content) > self.settings.max_upload_bytes:
            logger.warning(
                "Rejected oversized file filename={} bytes={}",
                filename,
                len(content),
            )
            raise FileTooLargeError(self.settings.max_upload_mb)
        if extension == ".pdf" and not content.startswith(b"%PDF"):
            logger.warning("Rejected invalid PDF filename={}", filename)
            raise InvalidFileError("The PDF file is invalid or corrupted.")
        if extension == ".docx" and not is_docx_bytes(content):
            logger.warning("Rejected invalid DOCX filename={}", filename)
            raise InvalidFileError("The DOCX file is invalid or corrupted.")
        return extension

    def _write(self, document_id: str, extension: str, content: bytes) -> Path:
        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)
        stored_path = self.settings.upload_dir / f"{document_id}{extension}"
        stored_path.write_bytes(content)
        return stored_path


def _display_name(filename: str | None) -> str:
    name = Path(filename or "").name.strip()
    if not name or name in {".", ".."}:
        return "resume"
    return name[:180]
