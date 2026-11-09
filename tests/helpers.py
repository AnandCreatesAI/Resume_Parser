"""Builders for resume files used by the tests."""

import io

import pymupdf
from docx import Document


def pdf_bytes(lines: list[str]) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    y_position = 72
    for line in lines:
        page.insert_text((72, y_position), line)
        y_position += 22
    payload = document.tobytes()
    document.close()
    return payload


def blank_pdf_bytes() -> bytes:
    document = pymupdf.open()
    document.new_page()
    payload = document.tobytes()
    document.close()
    return payload


def docx_bytes(paragraphs: list[str], table_rows: list[tuple[str, str]] | None = None) -> bytes:
    document = Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    if table_rows:
        table = document.add_table(rows=len(table_rows), cols=2)
        for index, (left, right) in enumerate(table_rows):
            table.rows[index].cells[0].text = left
            table.rows[index].cells[1].text = right
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()
