"""Recruiter dashboard for the local resume parser.

This UI talks to FastAPI over HTTP. It does not import the extraction code.
"""

import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = int(os.getenv("API_TIMEOUT_SECONDS", "210"))


def main() -> None:
    """Render the recruiter dashboard for upload and retrieval."""

    st.set_page_config(page_title="AI Resume Parser", layout="wide")
    _apply_styles()
    st.title("AI Resume Parser")
    st.caption(
        "Upload a PDF or DOCX resume. FastAPI extracts the text, a local Gemma model "
        "structures it, and the result is stored in SQLite."
    )

    with st.sidebar:
        st.header("Connection")
        st.write(f"API: `{API_BASE_URL}`")
        st.write("Model calls stay on this machine through Ollama.")
        st.divider()
        st.header("What gets extracted")
        st.markdown(
            "Every heading in the file is kept under its original name, "
            "including nested subsections."
        )

    upload_column, retrieve_column = st.columns(2, gap="large")
    with upload_column:
        st.subheader("Upload a resume")
        uploaded = st.file_uploader(
            "PDF or DOCX file",
            type=["pdf", "docx"],
            accept_multiple_files=False,
        )
        extract_clicked = st.button("Upload and Extract", type="primary")
    with retrieve_column:
        st.subheader("Retrieve a previous result")
        document_id = st.text_input("Document ID", placeholder="Paste a document ID")
        retrieve_clicked = st.button("Retrieve")

    if extract_clicked:
        if uploaded is None:
            st.warning("Choose a PDF or DOCX file first.")
        else:
            _upload(uploaded)

    if retrieve_clicked:
        if not document_id.strip():
            st.warning("Enter a document ID.")
        else:
            _retrieve(document_id.strip())

    result = st.session_state.get("result")
    if result:
        _render_result(result)


def _upload(uploaded: Any) -> None:
    with st.spinner("Extracting the resume with the local model. This can take a minute."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/upload",
                files={
                    "file": (
                        uploaded.name,
                        uploaded.getvalue(),
                        uploaded.type or "application/octet-stream",
                    )
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.Timeout:
            st.error("The API did not finish before the client timed out.")
            return
        except requests.RequestException:
            st.error(f"Cannot reach the API at {API_BASE_URL}. Start FastAPI and try again.")
            return
    _store_response(response)


def _retrieve(document_id: str) -> None:
    with st.spinner("Loading the saved resume."):
        try:
            response = requests.get(
                f"{API_BASE_URL}/api/resume/{document_id}",
                timeout=30,
            )
        except requests.RequestException:
            st.error(f"Cannot reach the API at {API_BASE_URL}. Start FastAPI and try again.")
            return
    _store_response(response)


def _store_response(response: requests.Response) -> None:
    if response.ok:
        st.session_state["result"] = response.json()
        st.session_state["error"] = None
        return
    st.session_state["result"] = None
    st.error(_error_detail(response))


def _render_result(result: dict) -> None:
    resume = result.get("resume") or {}
    st.success("Resume extracted.")
    meta_left, meta_right = st.columns(2)
    meta_left.text(f"File: {result.get('filename') or 'Not available'}")
    meta_right.text(f"Processed: {result.get('created_at') or 'Not available'}")
    st.write("**Document ID**")
    st.code(result.get("document_id") or "", language=None)

    name = resume.get("candidate_name") or resume.get("name")
    if name:
        st.subheader(name)
    contact = [
        value
        for value in (resume.get("email"), resume.get("phone"), resume.get("location"))
        if value
    ]
    if contact:
        st.write(" · ".join(contact))

    sections = resume.get("sections") or []
    if not sections:
        st.caption("No sections were found in the resume.")
    for section in sections:
        _render_section(section, level=3)


def _render_section(section: dict, level: int) -> None:
    heading = section.get("heading")
    content = [str(line) for line in section.get("content") or [] if line]
    subsections = section.get("subsections") or []
    if heading:
        st.markdown(f"{'#' * min(level, 6)} {heading}")
    if content:
        st.markdown("\n".join(f"- {line}" for line in content))
    for child in subsections:
        if isinstance(child, dict):
            _render_section(child, level + 1)


def _error_detail(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return "The API returned an unexpected response."
    detail = body.get("detail", "The request failed.")
    if isinstance(detail, list):
        messages = []
        for item in detail:
            if isinstance(item, dict):
                messages.append(item.get("msg", str(item)))
            else:
                messages.append(str(item))
        return "; ".join(messages)
    return str(detail)


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; max-width: 1180px; }
        [data-testid="stCaption"] { font-size: 0.95rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


main()
