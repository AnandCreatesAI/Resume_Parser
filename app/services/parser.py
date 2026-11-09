"""Extract plain text from PDF and DOCX resumes."""

import zipfile
from io import BytesIO
from pathlib import Path

import pymupdf
from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from loguru import logger

from app.exceptions import EmptyFileError, InvalidFileError, UnsupportedFileTypeError


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF. Image-only pages return no text."""

    path = Path(file_path)
    _reject_empty_file(path)
    if not path.read_bytes().startswith(b"%PDF"):
        logger.warning("PDF validation failed: missing PDF header")
        raise InvalidFileError("The PDF file is invalid or corrupted.")

    try:
        with pymupdf.open(path) as pdf:
            if pdf.is_encrypted:
                logger.warning("PDF validation failed: file is encrypted")
                raise InvalidFileError("Encrypted PDF files are not supported.")
            if pdf.page_count == 0:
                logger.warning("PDF validation failed: document has no pages")
                raise InvalidFileError("The PDF file is invalid or corrupted.")
            pages = [_page_text(page) for page in pdf]
    except InvalidFileError:
        raise
    except Exception as exc:
        logger.warning("PDF text extraction failed: {}", type(exc).__name__)
        raise InvalidFileError("The PDF file is invalid or corrupted.") from exc

    return "\n".join(pages).strip()


def extract_text_from_docx(file_path: str) -> str:
    """Extract paragraph and table text from a DOCX resume."""

    path = Path(file_path)
    _reject_empty_file(path)
    if not is_docx_bytes(path.read_bytes()):
        logger.warning("DOCX validation failed: file is not a Word document")
        raise InvalidFileError("The DOCX file is invalid or corrupted.")

    try:
        document = Document(path)
    except (PackageNotFoundError, zipfile.BadZipFile, ValueError) as exc:
        logger.warning("DOCX text extraction failed: {}", type(exc).__name__)
        raise InvalidFileError("The DOCX file is invalid or corrupted.") from exc

    parts: list[str] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            parts.append(text)

    for table in document.tables:
        for row in table.rows:
            cells = []
            seen: set[str] = set()
            for cell in row.cells:
                text = " ".join(cell.text.split()).strip()
                if text and text not in seen:
                    seen.add(text)
                    cells.append(text)
            if cells:
                parts.append(" | ".join(cells))

    return "\n".join(parts).strip()


def extract_resume_text(file_path: str) -> str:
    """Extract resume text based on the file extension."""

    extension = Path(file_path).suffix.lower()
    if extension == ".pdf":
        return extract_text_from_pdf(file_path)
    if extension == ".docx":
        return extract_text_from_docx(file_path)

    logger.warning("Unsupported resume extension={}", extension or "<none>")
    raise UnsupportedFileTypeError()


def is_docx_bytes(content: bytes) -> bool:
    """Return True when the bytes are a DOCX package."""

    if not content.startswith(b"PK"):
        return False
    try:
        with zipfile.ZipFile(BytesIO(content)) as archive:
            names = set(archive.namelist())
    except zipfile.BadZipFile:
        return False
    return "word/document.xml" in names


def _page_text(page) -> str:
    """Return page text in visual order, joining wrapped lines inside a block.

    PDFs often store blocks in drawing order, not top-to-bottom reading order.
    """

    raw_blocks = page.get_text("dict").get("blocks", [])
    blocks: list[tuple[float, float, float, float, list[tuple[str, float]]]] = []
    for block in raw_blocks:
        if block.get("type") != 0:
            continue
        lines: list[tuple[float, float, str, float]] = []
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
            if text:
                bbox = line.get("bbox", (0, 0, 0, 0))
                lines.append((bbox[1], bbox[0], text, bbox[2]))
        if not lines:
            continue
        lines.sort()
        x0, y0, x1, y1 = block["bbox"]
        blocks.append((y0, x0, y1, x1, [(text, x1_line) for _y, _x, text, x1_line in lines]))

    if not blocks:
        return page.get_text().strip()

    parts = [
        _join_wrapped_lines(lines, block_x1)
        for _y0, _x0, _y1, block_x1, lines in _reading_order(blocks)
    ]
    return "\n".join(part for part in parts if part).strip()


def _reading_order(
    blocks: list[tuple[float, float, float, float, list[tuple[str, float]]]],
) -> list[tuple[float, float, float, float, list[tuple[str, float]]]]:
    """Sort left-to-right within overlapping rows, then top-to-bottom."""

    def row_top(
        block: tuple[float, float, float, float, list[tuple[str, float]]],
    ) -> float:
        y0, _x0, y1, _x1, _lines = block
        top = y0
        for other_y0, _ox, other_y1, _ox1, _olines in blocks:
            if other_y0 < y1 and y0 < other_y1:
                top = min(top, other_y0)
        return top

    return sorted(blocks, key=lambda block: (round(row_top(block), 1), block[1]))


def _join_wrapped_lines(lines: list[tuple[str, float]], block_x1: float) -> str:
    joined: list[tuple[str, float]] = []
    for text, x1 in lines:
        if joined and _block_line_continues(joined[-1][0], text, joined[-1][1], block_x1):
            previous, _ = joined[-1]
            if previous.endswith("-"):
                merged = previous + text
            else:
                merged = f"{previous.rstrip()} {text}"
            joined[-1] = (merged, x1)
        else:
            joined.append((text, x1))
    return "\n".join(text for text, _x1 in joined)


def _block_line_continues(previous: str, nxt: str, previous_x1: float, block_x1: float) -> bool:
    if not previous or not nxt:
        return False
    if previous[-1] in ".!?:)":
        return False
    if nxt.endswith(":") and len(nxt.split()) <= 6:
        return False
    letters = [character for character in nxt if character.isalpha()]
    if letters and all(character.isupper() for character in letters) and len(nxt.split()) <= 8:
        return False
    if "|" in nxt:
        return False
    if nxt[:1].islower() or nxt[:1] == "(":
        return True
    return block_x1 - previous_x1 <= 40


def _reject_empty_file(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        logger.warning("Text extraction rejected an empty file")
        raise EmptyFileError()
