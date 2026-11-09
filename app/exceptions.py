"""Errors that map to a clear API response."""


class ResumeParserError(Exception):
    """Base error with a user-facing message and an HTTP status code."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class UnsupportedFileTypeError(ResumeParserError):
    """Raised when the upload is not a PDF or DOCX file."""

    def __init__(self) -> None:
        super().__init__("Only PDF and DOCX resumes are supported.", 400)


class EmptyFileError(ResumeParserError):
    """Raised when the uploaded file contains no bytes."""

    def __init__(self) -> None:
        super().__init__("The uploaded file is empty.", 400)


class FileTooLargeError(ResumeParserError):
    """Raised when the upload exceeds the configured size limit."""

    def __init__(self, limit_mb: int) -> None:
        super().__init__(
            f"The file is larger than the {limit_mb} MB upload limit.",
            413,
        )


class InvalidFileError(ResumeParserError):
    """Raised when a PDF or DOCX file cannot be opened."""

    def __init__(self, message: str) -> None:
        super().__init__(message, 400)


class EmptyResumeTextError(ResumeParserError):
    """Raised when the file opens but contains no readable text."""

    def __init__(self) -> None:
        super().__init__(
            "No readable text could be extracted from this file. "
            "Scanned image-only PDFs are not supported.",
            422,
        )


class OllamaUnavailableError(ResumeParserError):
    """Raised when the local Ollama service cannot complete a request."""

    def __init__(self, message: str) -> None:
        super().__init__(message, 503)


class LLMResponseError(ResumeParserError):
    """Raised when the model response is empty or not valid JSON."""

    def __init__(self, message: str) -> None:
        super().__init__(message, 422)


class ResumeNotFoundError(ResumeParserError):
    """Raised when no stored resume matches the document ID."""

    def __init__(self) -> None:
        super().__init__("No resume was found for that document ID.", 404)


class DatabaseError(ResumeParserError):
    """Raised when SQLite cannot save or load a resume."""

    def __init__(self) -> None:
        super().__init__(
            "The resume could not be saved or loaded because the database failed.",
            500,
        )
