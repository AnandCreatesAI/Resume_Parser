"""Pydantic models for a resume made of arbitrary sections.

A resume can use any headings. Sections keep the heading text the model
returned, their own lines, and any nested subsections. Placeholder text such
as "Not specified" is treated as missing.
"""

from typing import Any, Optional

from email_validator import EmailNotValidError, validate_email
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_BULLETS = set("•·●▪◦‣⁃")

PLACEHOLDERS = {
    "n/a",
    "na",
    "n.a",
    "n.a.",
    "none",
    "null",
    "nil",
    "unknown",
    "unspecified",
    "unavailable",
    "not specified",
    "not available",
    "not mentioned",
    "not provided",
    "not found",
    "not applicable",
    "no information",
    "no data",
    "-",
    "--",
    "—",
    "tbd",
}

_HEADER_FIELDS = {"candidate_name", "name", "email", "phone", "location", "sections"}


def clean_optional_text(value: Any) -> Optional[str]:
    """Return a stripped string, or null for blanks and placeholders."""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float) and value.is_integer():
        value = str(int(value))
    elif isinstance(value, (int, float)):
        value = str(value)
    if not isinstance(value, str):
        return None

    text = " ".join(value.replace("\x00", " ").split()).strip()
    while text and text[0] in _BULLETS:
        text = text[1:].strip()
    if not text:
        return None
    if text.lower().rstrip(".") in PLACEHOLDERS:
        return None
    return text


def content_lines(value: Any) -> list[str]:
    """Turn a string, list, or record into content lines without renaming them."""

    if value is None:
        return []
    if isinstance(value, str):
        return [text for line in value.splitlines() if (text := clean_optional_text(line))]
    if isinstance(value, list):
        lines: list[str] = []
        for item in value:
            if isinstance(item, dict):
                pieces = _record_lines(item)
                if pieces:
                    lines.append(" — ".join(pieces))
                continue
            if isinstance(item, str) and "\n" in item:
                lines.extend(content_lines(item))
                continue
            text = clean_optional_text(item)
            if text:
                lines.append(text)
        return lines
    if isinstance(value, dict):
        return _record_lines(value)
    text = clean_optional_text(value)
    return [text] if text else []


def _record_lines(record: dict) -> list[str]:
    lines: list[str] = []
    for key, child in record.items():
        if isinstance(child, list):
            lines.extend(content_lines(child))
            continue
        if isinstance(child, dict):
            nested = content_lines(child)
            if nested:
                lines.append(f"{key}: {'; '.join(nested)}")
            continue
        text = clean_optional_text(child)
        if text:
            lines.append(text)
    return lines


class ResumeSection(BaseModel):
    """One heading, the lines under it, and any nested headings."""

    model_config = ConfigDict(extra="ignore")

    heading: Optional[str] = None
    content: list[str] = Field(default_factory=list)
    subsections: list["ResumeSection"] = Field(default_factory=list)

    @field_validator("heading", mode="before")
    @classmethod
    def _clean_heading(cls, value: Any) -> Optional[str]:
        return clean_optional_text(value)

    @field_validator("content", mode="before")
    @classmethod
    def _clean_content(cls, value: Any) -> list[str]:
        return content_lines(value)

    @field_validator("subsections", mode="before")
    @classmethod
    def _clean_subsections(cls, value: Any) -> list[Any]:
        return coerce_sections(value)

    @model_validator(mode="after")
    def _drop_empty_children(self) -> "ResumeSection":
        self.subsections = [item for item in self.subsections if not item.is_empty()]
        return self

    def is_empty(self) -> bool:
        return self.heading is None and not self.content and not self.subsections


class ResumeData(BaseModel):
    """Header details plus every section found in the resume."""

    model_config = ConfigDict(extra="ignore")

    candidate_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    sections: list[ResumeSection] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _prepare(cls, value: Any) -> Any:
        if isinstance(value, ResumeData):
            return value
        if not isinstance(value, dict):
            return value

        data = dict(value)
        if not clean_optional_text(data.get("candidate_name")):
            data["candidate_name"] = data.get("name")

        raw_sections = data.get("sections")
        if raw_sections not in (None, "", []):
            data["sections"] = coerce_sections(raw_sections)
            return data

        promoted: list[Any] = []
        for key, item in data.items():
            if key in _HEADER_FIELDS:
                continue
            promoted.extend(section_from_value(str(key), item))
        data["sections"] = promoted
        return data

    @field_validator("candidate_name", "phone", "location", mode="before")
    @classmethod
    def _clean_header_fields(cls, value: Any) -> Optional[str]:
        return clean_optional_text(value)

    @field_validator("email", mode="before")
    @classmethod
    def _clean_email(cls, value: Any) -> Optional[str]:
        text = clean_optional_text(value)
        if text is None:
            return None
        try:
            result = validate_email(text, check_deliverability=False)
        except EmailNotValidError:
            logger.warning("Discarded an email value that failed format validation")
            return None
        return result.normalized

    @model_validator(mode="after")
    def _drop_empty_sections(self) -> "ResumeData":
        self.sections = [item for item in self.sections if not item.is_empty()]
        return self


class StoredResume(BaseModel):
    """A processed resume returned by the API."""

    document_id: str
    status: str
    filename: str
    created_at: str
    resume: ResumeData


class ErrorResponse(BaseModel):
    """Error body used in the API documentation."""

    detail: str


def coerce_sections(value: Any) -> list[Any]:
    """Accept a list, one section, or a mapping of heading to content."""

    if value is None or value == "":
        return []
    if isinstance(value, ResumeSection):
        return [value]
    if isinstance(value, str):
        return [{"heading": None, "content": [value], "subsections": []}]
    if isinstance(value, dict):
        if _looks_like_section(value):
            return [value]
        sections: list[Any] = []
        for key, item in value.items():
            sections.extend(section_from_value(str(key), item))
        return sections
    if isinstance(value, list):
        sections = []
        for item in value:
            if isinstance(item, (dict, ResumeSection)):
                sections.append(item)
            elif isinstance(item, str):
                sections.append({"heading": None, "content": [item], "subsections": []})
        return sections
    return []


def section_from_value(heading: str, value: Any) -> list[dict]:
    """Keep a model field as a section instead of dropping an unknown key."""

    if value is None or value == "" or value == [] or value == {}:
        return []
    if isinstance(value, dict) and _looks_like_section(value):
        return [value]
    if isinstance(value, dict):
        content: list[str] = []
        subsections: list[Any] = []
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                subsections.extend(section_from_value(str(key), child))
                continue
            text = clean_optional_text(child)
            if text:
                content.append(text)
        if not content and not subsections:
            return []
        return [{"heading": heading, "content": content, "subsections": subsections}]
    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        if all(_looks_like_section(item) for item in value):
            return [{"heading": heading, "content": [], "subsections": value}]
        subsections = []
        for item in value:
            lines = content_lines(item)
            if lines:
                subsections.append({"heading": None, "content": lines, "subsections": []})
        if not subsections:
            return []
        return [{"heading": heading, "content": [], "subsections": subsections}]
    lines = content_lines(value)
    if not lines:
        return []
    return [{"heading": heading, "content": lines, "subsections": []}]


def _looks_like_section(value: dict) -> bool:
    return any(key in value for key in ("heading", "content", "subsections"))


ResumeSection.model_rebuild()
