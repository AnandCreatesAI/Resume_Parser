"""Runtime settings loaded from the environment and the local .env file."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _integer(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        parsed = int(raw)
    except ValueError:
        logger.warning(
            "Environment variable {}={!r} is not an integer. Using {}.",
            name,
            raw,
            default,
        )
        return default
    if parsed < minimum:
        logger.warning(
            "Environment variable {}={} is below {}. Using {}.",
            name,
            parsed,
            minimum,
            default,
        )
        return default
    return parsed


@dataclass(frozen=True)
class Settings:
    """Configuration for the local model, storage, and upload limits."""

    ollama_base_url: str
    ollama_model: str
    ollama_timeout_seconds: int
    max_upload_mb: int
    max_upload_bytes: int
    database_path: Path
    upload_dir: Path
    log_level: str
    api_base_url: str


def get_settings() -> Settings:
    """Read settings each time so tests and process restarts pick up changes.

    `load_dotenv` does not override variables that are already set, so an
    explicit environment value always wins over `.env`.
    """

    load_dotenv(PROJECT_ROOT / ".env")

    max_upload_mb = _integer("MAX_UPLOAD_SIZE_MB", 10)
    return Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        ollama_model=os.getenv("OLLAMA_MODEL", "gemma3:1b"),
        ollama_timeout_seconds=_integer("OLLAMA_TIMEOUT_SECONDS", 180),
        max_upload_mb=max_upload_mb,
        max_upload_bytes=max_upload_mb * 1024 * 1024,
        database_path=_resolve_path(os.getenv("DATABASE_PATH", "data/resumes.db")),
        upload_dir=_resolve_path(os.getenv("UPLOAD_DIR", "data/uploads")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        api_base_url=os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/"),
    )
