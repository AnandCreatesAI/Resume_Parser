"""Manual check against the local Ollama model.

Automated tests mock the model. Run this file when Ollama is up:

    python tests/manual_ollama_integration.py

It extracts both sample resumes and fails if the response cannot be parsed
into the resume schema. Differences in wording are not treated as failures.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models import ResumeData
from app.services.parser import extract_resume_text
from app.services.resume_extractor import extract_resume_data


def _check(path: Path) -> ResumeData:
    source = extract_resume_text(str(path))
    print(f"\n===== {path.name} =====")
    print("--- source ---")
    print(source)
    result = extract_resume_data(source)
    if not isinstance(result, ResumeData):
        raise TypeError("Extraction did not return resume data")
    print("--- extracted ---")
    print(result.model_dump_json(indent=2))
    return result


def main() -> int:
    paths = [
        ROOT / "data" / "uploads" / "sample_resume.docx",
        ROOT / "data" / "uploads" / "sample_resume.pdf",
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        print("Sample resumes are missing:")
        for path in missing:
            print(path)
        return 2

    try:
        for path in paths:
            _check(path)
    except Exception as exc:
        print(f"Integration run failed: {type(exc).__name__}: {exc}")
        return 2

    print("\nLocal Ollama extraction returned valid resume JSON.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
