"""SQLite storage for processed resumes.

Each upload receives its own document id, including repeat uploads of the
same file. The extracted resume is stored as JSON and survives restarts.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from app.exceptions import DatabaseError
from app.models import ResumeData


class ResumeRepository:
    """Small repository around a single SQLite file."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.init_db()

    def init_db(self) -> None:
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS resumes (
                        document_id TEXT PRIMARY KEY,
                        original_filename TEXT NOT NULL,
                        stored_path TEXT NOT NULL,
                        extracted_json TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
        except sqlite3.Error as exc:
            logger.exception("Database initialization failed path={}", self.database_path)
            raise DatabaseError() from exc
        logger.info("Database ready path={}", self.database_path)

    def create(
        self,
        *,
        document_id: str,
        filename: str,
        stored_path: Path,
        resume: ResumeData,
        status: str = "completed",
    ) -> dict:
        created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        payload = resume.model_dump(mode="json")
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO resumes (
                        document_id,
                        original_filename,
                        stored_path,
                        extracted_json,
                        status,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        filename,
                        str(stored_path),
                        json.dumps(payload),
                        status,
                        created_at,
                    ),
                )
        except sqlite3.Error as exc:
            logger.exception("Failed to save resume document_id={}", document_id)
            raise DatabaseError() from exc

        logger.info("Saved resume document_id={} status={}", document_id, status)
        return self._record(document_id, status, filename, created_at, payload)

    def get(self, document_id: str) -> dict | None:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT document_id, original_filename, extracted_json, status, created_at
                    FROM resumes
                    WHERE document_id = ?
                    """,
                    (document_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            logger.exception("Failed to read resume document_id={}", document_id)
            raise DatabaseError() from exc

        if row is None:
            return None
        try:
            payload = json.loads(row["extracted_json"])
        except json.JSONDecodeError as exc:
            logger.error("Stored resume JSON is unreadable document_id={}", document_id)
            raise DatabaseError() from exc
        return self._record(
            row["document_id"],
            row["status"],
            row["original_filename"],
            row["created_at"],
            payload,
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _record(
        document_id: str,
        status: str,
        filename: str,
        created_at: str,
        payload: dict,
    ) -> dict:
        return {
            "document_id": document_id,
            "status": status,
            "filename": filename,
            "created_at": created_at,
            "resume": payload,
        }
