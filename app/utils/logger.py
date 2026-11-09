"""Application logging.

Resume text, model prompts, and extracted contact details are not written to
the log. Messages should name the operation, document id, and error type.
"""

import sys
from pathlib import Path

from loguru import logger

from app.config import PROJECT_ROOT, get_settings

LOG_DIR = PROJECT_ROOT / "logs"
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"


def setup_logger() -> None:
    """Configure stderr and a rotating log file. Safe to call more than once."""

    settings = get_settings()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level=settings.log_level, format=LOG_FORMAT)
    logger.add(
        Path(LOG_DIR) / "app.log",
        level=settings.log_level,
        rotation="5 MB",
        retention="7 days",
        encoding="utf-8",
        format=LOG_FORMAT,
    )
    logger.info("Logger ready level={} file={}", settings.log_level, LOG_DIR / "app.log")
