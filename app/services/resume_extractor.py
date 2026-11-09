"""Turn resume text into structured data with the local Gemma model."""

import json
import re

from loguru import logger
from pydantic import ValidationError

from app.exceptions import EmptyResumeTextError, LLMResponseError
from app.models import ResumeData, ResumeSection
from app.services.llm_service import generate_response

MAX_PROMPT_CHARACTERS = 20000

INSTRUCTIONS = """
You extract one resume into JSON.
Return JSON only. Do not add Markdown or commentary.

The resume can use any headings, in any order. Do not assume it has Skills,
Experience, Education, or Certifications.

Rules:
- Copy only information that is written in the resume. Do not paraphrase or summarize.
- If a value is missing, use null for a string and [] for a list.
- Never write "Not specified", "N/A", "Unknown", or any other placeholder.
- candidate_name is the person's name. Use null when no name is written.
- sections must include every heading in the resume. Do not skip a later heading.
- Copy each heading exactly. Do not rename, merge, or reclassify it.
- A subheading is its own short line, sometimes indented, with the following lines belonging to it. Put that line in subsections, and put its lines in that subsection. Use the same structure for deeper levels.
- Do not leave a subheading line inside the parent content. Do not move a subsection to a different parent.
- Do not invent a heading. Text with no heading uses "heading": null.
- Do not invent sections or lines that are not written.
- Copy a sentence once when the resume states it once. Repeat a line only when the resume itself repeats it.
- Include every role, date, school, list, and later heading. Do not stop after the first role or the first list.
- A heading on its own line is its own section. Do not put that heading, or the lines under it, into the previous section.
- Skip bullet symbols. Keep the words that follow them.
- Keep the resume's wording, lists, dates, and descriptions.

JSON shape:
{
  "candidate_name": null,
  "email": null,
  "phone": null,
  "location": null,
  "sections": [
    {
      "heading": null,
      "content": [],
      "subsections": [
        {"heading": null, "content": [], "subsections": []}
      ]
    }
  ]
}
""".strip()


def extract_resume_data(resume_text: str) -> ResumeData:
    """Stage 1 keeps the full resume text. Stage 2 lets Gemma structure it.

    After the model returns JSON, lines that are still in the source but missing
    from the structured result are recovered. Known headings keep their lines.
    Anything whose parent is unclear is kept with heading set to null.
    """

    resume_text = _prepare_text(resume_text)
    if not resume_text.strip():
        raise EmptyResumeTextError()

    parsed: ResumeData | None = None
    chunks = _text_chunks(resume_text, MAX_PROMPT_CHARACTERS)
    if len(chunks) > 1:
        logger.info("Resume split into model calls chunks={}", len(chunks))
    for chunk in chunks:
        piece = _extract_chunk(chunk)
        parsed = piece if parsed is None else _merge_extractions(parsed, piece)

    parsed = _align_with_source(parsed, resume_text)
    logger.info(
        "Extracted sections count={} candidate_name={}",
        len(parsed.sections),
        bool(parsed.candidate_name),
    )
    return parsed


def build_prompt(resume_text: str) -> str:
    """Build the extraction prompt for one resume or one slice of a long resume."""

    return f"{INSTRUCTIONS}\n\nResume:\n\"\"\"\n{resume_text.strip()}\n\"\"\""


def _extract_chunk(resume_text: str) -> ResumeData:
    raw_response = generate_response(build_prompt(resume_text))
    try:
        payload = parse_model_json(raw_response)
    except ValueError as exc:
        logger.error("Model JSON parsing failed")
        raise LLMResponseError("The model returned invalid JSON.") from exc

    try:
        return ResumeData.model_validate(payload)
    except ValidationError as exc:
        logger.error("Resume schema validation failed errors={}", exc.error_count())
        raise LLMResponseError(
            "The model response did not match the resume schema."
        ) from exc


def _merge_extractions(base: ResumeData, extra: ResumeData) -> ResumeData:
    if not base.candidate_name:
        base.candidate_name = extra.candidate_name
    if not base.email:
        base.email = extra.email
    if not base.phone:
        base.phone = extra.phone
    if not base.location:
        base.location = extra.location
    base.sections.extend(extra.sections)
    return base


def _text_chunks(text: str, limit: int) -> list[str]:
    """Split a long resume on line boundaries so the tail is still sent."""

    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.splitlines():
        extra = len(line) + (1 if current else 0)
        if current and size + extra > limit:
            chunks.append("\n".join(current))
            current = []
            size = 0
        current.append(line)
        size += extra
    if current:
        chunks.append("\n".join(current))
    return chunks or [text]


def parse_model_json(response: str) -> dict:
    """Parse a model response that may be wrapped in a Markdown fence."""

    if not response or not response.strip():
        raise ValueError("Model response is empty")

    cleaned = re.sub(r"```(?:json)?", "", response, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("Model response did not contain a JSON object") from None
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("Model response contained invalid JSON") from exc

    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ValueError("Model response was not a JSON object")
    return payload


def _prepare_text(text: str) -> str:
    """Drop empty bullet markers and join a wrapped line to the sentence it continues."""

    if not text:
        return ""
    lines: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or all(character in "•·●▪◦‣⁃" for character in stripped):
            continue
        while stripped and stripped[0] in "•·●▪◦‣⁃":
            stripped = stripped[1:].strip()
        if not stripped:
            continue
        if lines and _continues_line(lines[-1], stripped):
            lines[-1] = f"{lines[-1].rstrip()} {stripped}"
            continue
        lines.append(stripped)
    return "\n".join(lines)


def _continues_line(previous: str, nxt: str) -> bool:
    if not previous or not nxt or _heading_line(previous) or _heading_line(nxt):
        return False
    if previous[-1] in ".!?:;":
        return False
    return nxt[:1].islower()


def _align_with_source(resume: ResumeData, source_text: str) -> ResumeData:
    """Keep Gemma contact fields, then recover any source lines the JSON omitted."""

    _fill_contact(resume, source_text)
    outlined = _sections_from_source(source_text, resume.candidate_name)
    if outlined:
        resume.sections = outlined
    else:
        source_flat = _flat(source_text)
        for section in resume.sections:
            _dedupe_section(section, source_flat)
    _recover_missing_lines(resume, source_text)
    resume.sections = [section for section in resume.sections if not section.is_empty()]
    return resume


def _fill_contact(resume: ResumeData, source_text: str) -> None:
    """Copy name, email, phone, and location from the document when omitted."""

    if not resume.email:
        match = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", source_text, re.I)
        if match:
            resume.email = match.group(0)
    if not resume.phone:
        match = re.search(r"\+\d{1,3}[\s-]?\d[\d\s-]{7,}", source_text)
        if not match:
            match = re.search(
                r"(?i)\b(?:phone|tel|mobile)\s*[:.]?\s*(\+?\d[\d\s-]{7,}\d)",
                source_text,
            )
        if match:
            group = match.group(0 if match.lastindex is None else 1)
            resume.phone = re.sub(r"\s+", " ", group).strip()
    _fill_location(resume, source_text)
    if resume.candidate_name:
        return
    for line in source_text.splitlines():
        text = line.strip()
        if not text:
            continue
        if "@" in text or text[:1].isdigit() or text.startswith("+"):
            continue
        if _heading_line(text) and "&" not in text:
            resume.candidate_name = text
            return


def _fill_location(resume: ResumeData, source_text: str) -> None:
    """Copy a written location. Do not invent a city when none is present."""

    if resume.location:
        return
    match = re.search(r"(?i)\blocation\s*[:]\s*([^|\n]+)", source_text)
    if match:
        text = match.group(1).strip(" ,;:-")
        if text and not _looks_like_url_or_contact(text):
            resume.location = text
            return
    for line in source_text.splitlines():
        if "@" not in line:
            continue
        for part in reversed([piece.strip() for piece in line.split("|") if piece.strip()]):
            cleaned = part
            for value in (resume.email, resume.phone):
                if value:
                    cleaned = re.sub(re.escape(value), " ", cleaned, flags=re.I)
            cleaned = re.sub(
                r"(?i)\b(email|phone|tel|mobile|location)\s*[:.]?\s*",
                " ",
                cleaned,
            )
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" |,;:-")
            if (
                cleaned
                and "," in cleaned
                and any(character.isalpha() for character in cleaned)
                and not _looks_like_url_or_contact(cleaned)
            ):
                resume.location = cleaned
                return


def _looks_like_url_or_contact(text: str) -> bool:
    lowered = text.casefold()
    return "@" in text or "http" in lowered or "www." in lowered or "linkedin" in lowered


def _recover_missing_lines(resume: ResumeData, source_text: str) -> None:
    """Put omitted source lines back under their heading, or keep them unclassified."""

    blobs = _output_blobs(resume)
    current: ResumeSection | None = None
    unclassified: list[str] = []
    for line in source_text.splitlines():
        text = line.strip()
        if not text:
            continue
        if _heading_line(text):
            found = _section_with_heading(resume.sections, text)
            if found is not None:
                current = found
                continue
            unique = _without_extracted_contact(text, resume)
            if unique is None or _line_is_represented(unique, blobs, resume):
                continue
            unclassified.append(unique)
            blobs.append(_flat(unique))
            current = None
            continue
        unique = _without_extracted_contact(text, resume)
        if unique is None or _line_is_represented(unique, blobs, resume):
            continue
        if current is None:
            unclassified.append(unique)
            blobs.append(_flat(unique))
            continue
        current.content.append(unique)
        blobs.append(_flat(unique))

    if not unclassified:
        return
    if resume.sections and resume.sections[0].heading is None:
        existing = [_flat(line) for line in resume.sections[0].content]
        extra = [line for line in unclassified if _flat(line) not in existing]
        resume.sections[0].content = extra + resume.sections[0].content
        return
    resume.sections.insert(0, ResumeSection(heading=None, content=unclassified))


def _output_blobs(resume: ResumeData) -> list[str]:
    blobs = [
        _flat(value)
        for value in (
            resume.candidate_name,
            resume.email,
            resume.phone,
            resume.location,
        )
        if value
    ]

    def walk(section: ResumeSection) -> None:
        if section.heading:
            blobs.append(_flat(section.heading))
        blobs.extend(_flat(line) for line in section.content)
        for child in section.subsections:
            walk(child)

    for section in resume.sections:
        walk(section)
    return blobs


def _section_with_heading(sections: list[ResumeSection], heading: str) -> ResumeSection | None:
    target = _flat(heading)
    for section in sections:
        if section.heading and _flat(section.heading) == target:
            return section
        found = _section_with_heading(section.subsections, heading)
        if found is not None:
            return found
    return None


def _line_is_represented(line: str, blobs: list[str], resume: ResumeData) -> bool:
    key = re.sub(r":$", "", _flat(line)).strip()
    if not key:
        return True
    if any(key == blob or key in blob for blob in blobs if blob):
        return True
    if ":" in line:
        value = _flat(line.split(":", 1)[1])
        if value and any(value == blob or value in blob for blob in blobs if blob):
            return True
        words = [word for word in re.findall(r"[a-z0-9]+", value) if len(word) > 2]
        joined = " ".join(blobs)
        if words and all(word in joined for word in words):
            return True
    leftover = key
    for value in (resume.email, resume.phone, resume.location, resume.candidate_name):
        if value:
            leftover = leftover.replace(_flat(value), " ")
    leftover = re.sub(r"\b(email|phone|tel|mobile|location)\b", " ", leftover)
    leftover = re.sub(r"[^a-z0-9]+", " ", leftover).strip()
    if leftover != key and not leftover:
        return True
    return False


def _without_extracted_contact(line: str, resume: ResumeData) -> str | None:
    """Drop phone, email, and other header values already stored on the resume."""

    text = line
    for value in (resume.email, resume.phone, resume.location, resume.candidate_name):
        if value:
            text = re.sub(re.escape(value), " ", text, flags=re.I)
    if _flat(text) == _flat(line):
        return line
    text = re.sub(r"(?i)\b(email|phone|tel|mobile|location)\s*[:.]?\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" |,;:-")
    return text or None


def _dedupe_section(section: ResumeSection, source_flat: str) -> None:
    section.content = _drop_repeated_lines(section.content, source_flat)
    for child in section.subsections:
        _dedupe_section(child, source_flat)


def _sections_from_source(source_text: str, candidate_name: str | None) -> list[ResumeSection]:
    """Build sections from heading lines in the resume, not from a heading name list."""

    lines = _content_start(source_text, candidate_name)
    if not lines:
        return []
    return _parse_sections(lines, parent_indent=-1, nested=False)


def _content_start(source_text: str, candidate_name: str | None) -> list[str]:
    lines = [line.strip() for line in source_text.splitlines() if line.strip()]
    name = _flat(candidate_name or "")
    index = 0
    while index < len(lines):
        if name and _flat(lines[index]) == name:
            index += 1
            continue
        if _heading_line(lines[index]):
            break
        index += 1
    return lines[index:]


def _parse_sections(
    lines: list[str],
    parent_indent: int,
    nested: bool,
) -> list[ResumeSection]:
    spans = _structure_nodes(lines, parent_indent, require_repeat=nested)
    if not spans:
        return []
    source_flat = _flat("\n".join(lines))
    sections: list[ResumeSection] = []
    for start, end in spans:
        body = lines[start + 1 : end]
        section = ResumeSection(
            heading=lines[start].strip(),
            content=[line.strip() for line in body if line.strip()],
            subsections=[],
        )
        _nest_subsections(
            section,
            body,
            source_flat,
            _indent(lines[start]),
            require_repeat=True,
        )
        sections.append(section)
    return sections


def _drop_repeated_lines(lines: list[str], source_flat: str) -> list[str]:
    """Keep a repeated line only as many times as the resume states it."""

    phrases = [_flat(line) for line in lines]
    kept_counts: dict[str, int] = {}
    kept: list[bool] = [False] * len(lines)
    order = sorted(range(len(lines)), key=lambda index: len(phrases[index]), reverse=True)

    for index in order:
        phrase = phrases[index]
        if not phrase:
            continue
        explained = 0
        for kept_phrase, count in kept_counts.items():
            if phrase != kept_phrase and phrase in kept_phrase and kept_phrase in source_flat:
                explained += count * kept_phrase.count(phrase)
        source_count = source_flat.count(phrase)
        if phrase in source_flat:
            allowed = max(source_count - explained, 0)
        else:
            allowed = 1 if kept_counts.get(phrase, 0) == 0 else 0
        if kept_counts.get(phrase, 0) < allowed:
            kept[index] = True
            kept_counts[phrase] = kept_counts.get(phrase, 0) + 1

    return [line for index, line in enumerate(lines) if kept[index]]


def _nest_subsections(
    section: ResumeSection,
    block: list[str],
    source_flat: str,
    parent_indent: int,
    *,
    require_repeat: bool,
) -> None:
    nodes = _structure_nodes(block, parent_indent, require_repeat=require_repeat)
    if not nodes:
        _nest_colon_fields(section)
        return

    existing = {_flat(item.heading) for item in section.subsections if item.heading}
    rebuilt: list[str] = []
    index = 0
    lines = section.content
    while index < len(lines):
        node = _node_at(lines[index], nodes, block)
        if node is None or _flat(lines[index]) in existing:
            rebuilt.append(lines[index])
            index += 1
            continue
        heading_index, end = node
        body = [line.strip() for line in block[heading_index + 1 : end] if line.strip()]
        body_keys = [_flat(line) for line in body]
        taken: list[str] = []
        cursor = index + 1
        while cursor < len(lines) and body_keys and _flat(lines[cursor]) == body_keys[0]:
            taken.append(lines[cursor])
            body_keys.pop(0)
            cursor += 1
        child = ResumeSection(
            heading=lines[index].strip(),
            content=_drop_repeated_lines(taken, source_flat),
            subsections=[],
        )
        section.subsections.append(child)
        existing.add(_flat(child.heading or ""))
        _nest_subsections(
            child,
            block[heading_index + 1 : end],
            source_flat,
            _indent(block[heading_index]),
            require_repeat=False,
        )
        index = cursor
    section.content = _drop_repeated_lines(rebuilt, source_flat)
    _nest_colon_fields(section)


def _structure_nodes(
    lines: list[str],
    parent_indent: int,
    *,
    require_repeat: bool,
) -> list[tuple[int, int]]:
    heading_indexes = [index for index, line in enumerate(lines) if _heading_line(line)]
    if not heading_indexes:
        return []

    spans: list[tuple[int, int]] = []
    covered: set[int] = set()
    for position, index in enumerate(heading_indexes):
        end = heading_indexes[position + 1] if position + 1 < len(heading_indexes) else len(lines)
        body = lines[index + 1 : end]
        if body and all(not _heading_line(line) for line in body if line.strip()):
            spans.append((index, end))
            covered.add(index)

    levels = [_heading_level(lines[index]) for index in heading_indexes]
    mixed = 0 in levels and 1 in levels
    if mixed:
        title_leaves = [span for span in spans if _heading_level(lines[span[0]]) != 0]
        cap_leaves = [span for span in spans if _heading_level(lines[span[0]]) == 0]
        cap_indexes = [
            index for index in heading_indexes if _heading_level(lines[index]) == 0
        ]
        wrap_caps = len(cap_indexes) >= 2
        rebuilt: list[tuple[int, int]] = list(title_leaves)
        for index in cap_indexes:
            next_cap = next((other for other in cap_indexes if other > index), len(lines))
            owned = [span for span in title_leaves if index < span[0] < next_cap]
            if wrap_caps and owned:
                rebuilt.append((index, next_cap))
                continue
            own_leaf = next((span for span in cap_leaves if span[0] == index), None)
            if own_leaf is not None:
                rebuilt.append(own_leaf)
                continue
            position = heading_indexes.index(index)
            end = (
                heading_indexes[position + 1]
                if position + 1 < len(heading_indexes)
                else len(lines)
            )
            rebuilt.append((index, end))
        spans = rebuilt
        covered = {span[0] for span in spans}
    else:
        changed = True
        while changed:
            changed = False
            for index in heading_indexes:
                if index in covered:
                    continue
                next_uncovered = next(
                    (other for other in heading_indexes if other > index and other not in covered),
                    len(lines),
                )
                owned = [span for span in spans if index < span[0] < next_uncovered]
                if owned:
                    spans.append((index, next_uncovered))
                    covered.add(index)
                    changed = True

    top = []
    for span in spans:
        if any(other[0] < span[0] and other[1] >= span[1] for other in spans if other != span):
            continue
        top.append(span)
    top.sort()
    if _accepted(top, lines, parent_indent, require_repeat=require_repeat):
        return top
    return []


def _accepted(
    spans: list[tuple[int, int]],
    lines: list[str],
    parent_indent: int,
    *,
    require_repeat: bool,
) -> bool:
    if not spans:
        return False
    if not require_repeat:
        return True
    if len(spans) >= 2:
        return True
    indent = _indent(lines[spans[0][0]])
    return indent > parent_indent and indent > 0


def _node_at(
    line: str,
    nodes: list[tuple[int, int]],
    block: list[str],
) -> tuple[int, int] | None:
    target = _flat(line)
    for start, end in nodes:
        if _flat(block[start]) == target:
            return start, end
    return None


def _heading_level(line: str) -> int:
    """0 for all-capital headings, 1 for title-case headings."""

    letters = [character for character in line if character.isalpha()]
    if letters and all(character.isupper() for character in letters):
        return 0
    return 1


def _heading_line(line: str) -> bool:
    text = line.strip()
    if not text or any(marker in text for marker in ":;,"):
        return False
    if text.count("|") > 1:
        return False
    if any(character.isdigit() for character in text):
        return False
    if text[-1] in ".!?":
        return False
    core = re.sub(r"\s*\([^)]*\)\s*$", "", text).strip() or text
    connectors = {"&", "/", "+", "|"}
    words = [word for word in core.split() if word not in connectors]
    if not 1 <= len(words) <= 6:
        return False
    if not all(word[:1].isupper() for word in words):
        return False
    return any(character.isalpha() for character in core)


def _colon_field(line: str) -> tuple[str, str] | None:
    """Split a title-case 'Label:' or 'Label: value' line used inside a section."""

    if ":" not in line:
        return None
    label, rest = line.split(":", 1)
    label = label.strip()
    if not label or not _heading_line(label) or _heading_level(label) != 1:
        return None
    return label, rest.strip()


def _nest_colon_fields(section: ResumeSection) -> None:
    """Turn repeated title-case labels into subsections, keeping source order."""

    indexes = [index for index, line in enumerate(section.content) if _colon_field(line)]
    if len(indexes) < 2:
        return
    rebuilt = section.content[: indexes[0]]
    extra: list[ResumeSection] = []
    for position, start in enumerate(indexes):
        end = indexes[position + 1] if position + 1 < len(indexes) else len(section.content)
        heading, rest = _colon_field(section.content[start]) or ("", "")
        body: list[str] = []
        if rest:
            body.append(rest)
        body.extend(section.content[start + 1 : end])
        extra.append(ResumeSection(heading=heading, content=body, subsections=[]))
    section.content = rebuilt
    section.subsections = extra + section.subsections


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()
