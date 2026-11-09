"""Tests for PDF and DOCX text extraction."""

from pathlib import Path

import pytest

from app.exceptions import EmptyFileError, InvalidFileError, UnsupportedFileTypeError
from app.services.parser import extract_resume_text
from tests.helpers import blank_pdf_bytes, docx_bytes, pdf_bytes


def test_valid_pdf_text(tmp_path):
    path = tmp_path / "resume.pdf"
    path.write_bytes(
        pdf_bytes(
            [
                "Amina Rahman",
                "amina@example.com",
                "Backend engineer",
            ]
        )
    )

    text = extract_resume_text(str(path))

    assert "Amina Rahman" in text
    assert "amina@example.com" in text
    assert "Backend engineer" in text


def test_valid_docx_text_includes_tables(tmp_path):
    path = tmp_path / "resume.docx"
    path.write_bytes(
        docx_bytes(
            ["Amina Rahman", "amina@example.com"],
            table_rows=[("Company", "Northwind Analytics")],
        )
    )

    text = extract_resume_text(str(path))

    assert "Amina Rahman" in text
    assert "amina@example.com" in text
    assert "Northwind Analytics" in text


def test_unsupported_extension(tmp_path):
    path = tmp_path / "resume.txt"
    path.write_text("Amina Rahman", encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError):
        extract_resume_text(str(path))


def test_empty_file(tmp_path):
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"")

    with pytest.raises(EmptyFileError):
        extract_resume_text(str(path))


def test_invalid_pdf(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4\nthis is not a real pdf")

    with pytest.raises(InvalidFileError):
        extract_resume_text(str(path))


def test_invalid_docx(tmp_path):
    path = tmp_path / "broken.docx"
    path.write_bytes(b"this is not a word document")

    with pytest.raises(InvalidFileError):
        extract_resume_text(str(path))


def test_pdf_with_no_text_returns_empty_string(tmp_path):
    path = tmp_path / "blank.pdf"
    path.write_bytes(blank_pdf_bytes())

    assert extract_resume_text(str(path)) == ""


def test_pdf_text_follows_visual_order_not_draw_order(tmp_path):
    import pymupdf

    document = pymupdf.open()
    page = document.new_page()
    page.insert_textbox(pymupdf.Rect(50, 420, 500, 500), "LATER ROLE\nSecond workplace")
    page.insert_textbox(pymupdf.Rect(50, 80, 500, 160), "EARLIER ROLE\nFirst workplace")
    page.insert_textbox(pymupdf.Rect(50, 260, 500, 340), "MIDDLE ROLE\nBetween workplaces")
    path = tmp_path / "drawn-backwards.pdf"
    path.write_bytes(document.tobytes())
    document.close()

    text = extract_resume_text(str(path))
    earlier = text.find("EARLIER ROLE")
    middle = text.find("MIDDLE ROLE")
    later = text.find("LATER ROLE")

    assert earlier != -1 and middle != -1 and later != -1
    assert earlier < middle < later


def test_pdf_joins_wrapped_lines_inside_a_block(tmp_path):
    import pymupdf

    document = pymupdf.open()
    page = document.new_page()
    page.insert_textbox(
        pymupdf.Rect(50, 80, 280, 220),
        "Designed reporting dashboards used by six business teams and reduced review time.",
    )
    path = tmp_path / "wrapped.pdf"
    path.write_bytes(document.tobytes())
    document.close()

    text = extract_resume_text(str(path))
    compact = " ".join(text.split())

    assert "Designed reporting dashboards" in compact
    assert "reduced review time." in compact


def test_sample_resumes_when_present():
    pdf_path = Path("data/uploads/sample_resume.pdf")
    docx_path = Path("data/uploads/sample_resume.docx")
    if pdf_path.exists():
        pdf_text = extract_resume_text(str(pdf_path))
        assert "rahul@gmail.com" in pdf_text.lower()
    if docx_path.exists():
        docx_text = extract_resume_text(str(docx_path))
        assert "abc technologies" in docx_text.lower()
        assert "python" in docx_text.lower()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
