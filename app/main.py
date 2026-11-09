"""FastAPI application for uploading and retrieving parsed resumes."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger

from app.config import Settings, get_settings
from app.database.repository import ResumeRepository
from app.exceptions import ResumeParserError
from app.models import ErrorResponse, StoredResume
from app.services.resume_service import ResumeService
from app.utils.logger import setup_logger


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the API with its own settings, database, and service."""

    setup_logger()
    settings = settings or get_settings()
    repository = ResumeRepository(settings.database_path)
    service = ResumeService(settings, repository)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(
            "Application started model={} database={}",
            app.state.settings.ollama_model,
            app.state.settings.database_path,
        )
        yield
        logger.info("Application shutdown")

    app = FastAPI(
        title="AI Resume Parser",
        description=(
            "Upload a PDF or DOCX resume. The API extracts text, asks a local "
            "Ollama model for the resume's own sections, and stores the result "
            "in SQLite."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.repository = repository
    app.state.service = service

    @app.exception_handler(ResumeParserError)
    async def handle_parser_error(request: Request, exc: ResumeParserError) -> JSONResponse:
        logger.error(
            "API error method={} path={} status={} message={}",
            request.method,
            request.url.path,
            exc.status_code,
            exc.message,
        )
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled API error method={} path={}",
            request.method,
            request.url.path,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected error occurred while processing the request."},
        )

    @app.post(
        "/api/upload",
        response_model=StoredResume,
        summary="Upload a resume and extract structured data",
        responses={
            400: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    def upload_resume(file: UploadFile = File(...)) -> dict:
        # Sync handler so the Ollama call runs in FastAPI's threadpool.
        content = file.file.read()
        return service.process_upload(file.filename, content)

    @app.get(
        "/api/resume/{document_id}",
        response_model=StoredResume,
        summary="Retrieve a previously parsed resume",
        responses={
            404: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    def get_resume(document_id: str) -> dict:
        return service.get_resume(document_id)

    return app


app = create_app()
