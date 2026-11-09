"""Tests for dynamic section extraction.

The model is mocked. These tests check that headings are kept as written,
including custom names and nested subsections.
"""

import json

import pytest

from app.exceptions import EmptyResumeTextError, LLMResponseError
from app.models import ResumeData
from app.services.resume_extractor import build_prompt, extract_resume_data, parse_model_json


def _use_model(monkeypatch, payload):
    def fake_generate(prompt: str) -> str:
        assert "Resume:" in prompt
        assert "Do not invent a heading" in prompt
        if isinstance(payload, str):
            return payload
        return json.dumps(payload)

    monkeypatch.setattr(
        "app.services.resume_extractor.generate_response",
        fake_generate,
    )


def _headings(resume: ResumeData) -> list[str | None]:
    return [section.heading for section in resume.sections]


def test_markdown_fence_is_parsed():
    payload = parse_model_json(
        """```json
        {"candidate_name": "Amina Rahman", "sections": []}
        ```"""
    )

    assert payload["candidate_name"] == "Amina Rahman"


def test_invalid_model_json(monkeypatch):
    _use_model(monkeypatch, "I could not extract the resume.")

    with pytest.raises(LLMResponseError, match="invalid JSON"):
        extract_resume_data("Amina Rahman\namina@example.com")


def test_empty_resume_text_does_not_call_the_model(monkeypatch):
    def fail_if_called(prompt: str) -> str:
        raise AssertionError("The model should not be called for empty text")

    monkeypatch.setattr(
        "app.services.resume_extractor.generate_response",
        fail_if_called,
    )

    with pytest.raises(EmptyResumeTextError):
        extract_resume_data("   ")


def test_standard_headings_keep_their_names_and_lines(monkeypatch):
    _use_model(
        monkeypatch,
        {
            "candidate_name": "Amina Rahman",
            "email": "amina@example.com",
            "sections": [
                {
                    "heading": "Professional Summary",
                    "content": [
                        "Backend engineer focused on billing systems.",
                        "Builds developer tools for finance teams.",
                    ],
                },
                {
                    "heading": "Work Experience",
                    "content": [
                        "Northwind Analytics — Software Engineer — Jan 2020 - Mar 2022",
                        "Built billing APIs for subscription renewals.",
                    ],
                },
                {
                    "heading": "Education",
                    "content": ["BSc Computer Science — State University — 2018"],
                },
            ],
        },
    )

    result = extract_resume_data("Amina Rahman\nProfessional Summary\nWork Experience\nEducation\n")

    assert result.candidate_name == "Amina Rahman"
    assert _headings(result) == ["Professional Summary", "Work Experience", "Education"]
    assert result.sections[0].content == [
        "Backend engineer focused on billing systems.",
        "Builds developer tools for finance teams.",
    ]
    assert "Built billing APIs for subscription renewals." in result.sections[1].content


def test_custom_headings_are_not_dropped_or_renamed(monkeypatch):
    _use_model(
        monkeypatch,
        {
            "candidate_name": "Leila Okonkwo",
            "sections": [
                {"heading": "Career Objective", "content": ["Move into climate-risk modeling."]},
                {
                    "heading": "Publications",
                    "content": ["Okonkwo, L. Flood maps for coastal cities. 2023."],
                },
                {"heading": "Awards", "content": ["Harbor City Research Prize, 2022"]},
                {"heading": "Languages", "content": ["English", "Yoruba", "French"]},
            ],
        },
    )

    result = extract_resume_data("Leila Okonkwo\nCareer Objective\nPublications\nAwards\nLanguages\n")

    assert _headings(result) == ["Career Objective", "Publications", "Awards", "Languages"]
    assert result.sections[1].content == ["Okonkwo, L. Flood maps for coastal cities. 2023."]
    assert "Experience" not in _headings(result)
    assert "Skills" not in _headings(result)


def test_nested_subsections_stay_with_their_parent(monkeypatch):
    _use_model(
        monkeypatch,
        {
            "candidate_name": "Noah Berg",
            "sections": [
                {
                    "heading": "Technical Expertise",
                    "content": [],
                    "subsections": [
                        {"heading": "Programming Languages", "content": ["Python", "Java"]},
                        {"heading": "Cloud Technologies", "content": ["AWS", "Azure"]},
                    ],
                }
            ],
        },
    )

    result = extract_resume_data(
        "Noah Berg\nTechnical Expertise\nProgramming Languages\nCloud Technologies\n"
    )

    assert _headings(result) == ["Technical Expertise"]
    parent = result.sections[0]
    assert parent.content == []
    assert [item.heading for item in parent.subsections] == [
        "Programming Languages",
        "Cloud Technologies",
    ]
    assert parent.subsections[0].content == ["Python", "Java"]
    assert parent.subsections[1].content == ["AWS", "Azure"]


def test_missing_traditional_sections_are_not_invented(monkeypatch):
    _use_model(
        monkeypatch,
        {
            "candidate_name": "Priya Nair",
            "email": "priya.nair@example.com",
            "sections": [
                {
                    "heading": "Volunteer Projects",
                    "content": ["Weekend coding club for public libraries."],
                }
            ],
        },
    )

    result = extract_resume_data("Priya Nair\nVolunteer Projects\n")

    assert _headings(result) == ["Volunteer Projects"]
    assert result.email == "priya.nair@example.com"


def test_unusual_heading_order_is_preserved(monkeypatch):
    _use_model(
        monkeypatch,
        {
            "candidate_name": "Sam Ito",
            "sections": [
                {"heading": "Gallery Shows", "content": ["North Hall, March 2024"]},
                {"heading": "Studio Practice", "content": ["Oil and charcoal studies."]},
                {
                    "heading": "Teaching",
                    "content": [],
                    "subsections": [
                        {"heading": "Weekend Workshops", "content": ["Life drawing for beginners."]}
                    ],
                },
            ],
        },
    )

    result = extract_resume_data("Sam Ito\nGallery Shows\nStudio Practice\nTeaching\n")

    assert _headings(result) == ["Gallery Shows", "Studio Practice", "Teaching"]
    assert result.sections[2].subsections[0].heading == "Weekend Workshops"
    assert result.sections[2].subsections[0].content == ["Life drawing for beginners."]


def test_text_without_a_heading_does_not_get_one(monkeypatch):
    _use_model(
        monkeypatch,
        {
            "candidate_name": "Rita Cole",
            "sections": [
                {
                    "heading": None,
                    "content": ["Available for contract work from June 2026."],
                }
            ],
        },
    )

    result = extract_resume_data("Rita Cole\nAvailable for contract work from June 2026.\n")

    assert result.sections[0].heading is None
    assert result.sections[0].content == ["Available for contract work from June 2026."]


def test_lines_the_model_omits_are_recovered_under_their_heading(monkeypatch):
    source = """
Nia Shah
FIELD NOTES
Kept the catalog current.
Wrote finding aids for visiting collections.
EXHIBITIONS
Harbor Hall, 2024
"""
    _use_model(
        monkeypatch,
        {
            "candidate_name": "Nia Shah",
            "sections": [
                {"heading": "FIELD NOTES", "content": ["Kept the catalog current."]},
                {"heading": "EXHIBITIONS", "content": ["Harbor Hall, 2024"]},
            ],
        },
    )

    result = extract_resume_data(source)

    notes = next(section for section in result.sections if section.heading == "FIELD NOTES")
    assert "Wrote finding aids for visiting collections." in notes.content


def test_unplaced_preamble_is_kept_without_inventing_a_heading(monkeypatch):
    source = """
Nia Shah
Available for contract catalog work.
nia.shah@example.com
FIELD NOTES
Kept the catalog current.
"""
    _use_model(
        monkeypatch,
        {
            "candidate_name": "Nia Shah",
            "email": "nia.shah@example.com",
            "sections": [{"heading": "FIELD NOTES", "content": ["Kept the catalog current."]}],
        },
    )

    result = extract_resume_data(source)

    assert result.sections[0].heading is None
    assert result.sections[0].content == ["Available for contract catalog work."]
    assert result.email == "nia.shah@example.com"
    assert _headings(result)[1] == "FIELD NOTES"


def test_location_is_copied_from_a_labeled_field_and_a_contact_line(monkeypatch):
    labeled = """
Nia Shah
Email: nia.shah@example.com | Phone: +91 8085489720 | Location: Harbor City, India
FIELD NOTES
Kept the catalog current.
"""
    piped = """
Nia Shah
nia.shah@example.com | +91 8085489720 | Harbor City, India
FIELD NOTES
Kept the catalog current.
"""
    payload = {
        "candidate_name": "Nia Shah",
        "email": "nia.shah@example.com",
        "phone": "+91 8085489720",
        "sections": [{"heading": "FIELD NOTES", "content": ["Kept the catalog current."]}],
    }
    _use_model(monkeypatch, payload)
    labeled_result = extract_resume_data(labeled)
    _use_model(monkeypatch, payload)
    piped_result = extract_resume_data(piped)
    _use_model(monkeypatch, payload)
    missing_result = extract_resume_data(
        "Nia Shah\nnia.shah@example.com\nFIELD NOTES\nKept the catalog current.\n"
    )

    assert labeled_result.location == "Harbor City, India"
    assert piped_result.location == "Harbor City, India"
    assert missing_result.location is None


def test_omitted_heading_like_list_is_kept_unclassified(monkeypatch):
    source = """
Nia Shah
FIELD NOTES
Kept the catalog current.
Spoken Languages
Hausa
Yoruba
French
"""
    _use_model(
        monkeypatch,
        {
            "candidate_name": "Nia Shah",
            "sections": [{"heading": "FIELD NOTES", "content": ["Kept the catalog current."]}],
        },
    )

    result = extract_resume_data(source)

    unclassified = next(section for section in result.sections if section.heading is None)
    joined = "\n".join(unclassified.content)
    assert "Spoken Languages" in joined
    assert "Hausa" in joined
    assert "Yoruba" in joined
    assert "French" in joined
    assert "Experience" not in _headings(result)


def test_title_case_sections_after_one_caps_heading_stay_siblings(monkeypatch):
    source = """
Nia Shah
FIELD NOTES
Kept the catalog current.
Studio Methods
Harbor Studio
Built the weekly intake sheet.
Spoken Languages
English
Yoruba
French
Publications
Okonkwo, L. Flood maps for coastal cities. 2023.
Awards
Harbor City Research Prize, 2022
"""
    _use_model(
        monkeypatch,
        {
            "candidate_name": "Nia Shah",
            "sections": [{"heading": "FIELD NOTES", "content": ["Kept the catalog current."]}],
        },
    )

    result = extract_resume_data(source)

    assert _headings(result) == [
        None,
        "FIELD NOTES",
        "Harbor Studio",
        "Publications",
        "Awards",
    ]
    notes = next(section for section in result.sections if section.heading == "FIELD NOTES")
    assert notes.content == ["Kept the catalog current."]
    assert notes.subsections == []
    studio = next(section for section in result.sections if section.heading == "Harbor Studio")
    assert studio.content == ["Built the weekly intake sheet."]
    publications = next(section for section in result.sections if section.heading == "Publications")
    assert publications.content == ["Okonkwo, L. Flood maps for coastal cities. 2023."]
    awards = next(section for section in result.sections if section.heading == "Awards")
    assert awards.content == ["Harbor City Research Prize, 2022"]
    unclassified = next(section for section in result.sections if section.heading is None)
    joined = "\n".join(unclassified.content)
    assert "Studio Methods" in joined
    assert "Spoken Languages" in joined
    assert "English" in joined
    assert "Yoruba" in joined
    assert "French" in joined
    assert "Experience" not in _headings(result)


def test_colon_fields_already_extracted_are_not_copied_again(monkeypatch):
    source = """
Ada West
Email: ada.west@example.com
Skills: Python, SQL
Keeps the lab notebook.
"""
    _use_model(
        monkeypatch,
        {
            "candidate_name": "Ada West",
            "email": "ada.west@example.com",
            "sections": [
                {"heading": "Notes", "content": ["Keeps the lab notebook."]},
                {"heading": "Skills", "content": ["Python", "SQL"]},
            ],
        },
    )

    result = extract_resume_data(source)

    unclassified = [section for section in result.sections if section.heading is None]
    joined = " ".join(line for section in unclassified for line in section.content)
    assert "Python" not in joined
    assert "ada.west@example.com" not in joined


def test_prompt_does_not_whitelist_section_names():
    prompt = build_prompt("Jordan Lee\nField Notes\n")

    assert "any headings" in prompt
    assert "Do not invent a heading" in prompt
    assert "Projects" not in prompt
    assert "Technical Skills" not in prompt


def test_model_fields_outside_sections_are_kept(monkeypatch):
    _use_model(
        monkeypatch,
        {
            "name": "Amina Rahman",
            "email": "amina@example.com",
            "Professional Highlights": ["Reduced invoice exceptions by 18 percent."],
            "Training": ["Incident response tabletop, 2024"],
        },
    )

    result = extract_resume_data("Amina Rahman\nProfessional Highlights\nTraining\n")

    assert result.candidate_name == "Amina Rahman"
    assert _headings(result) == ["Professional Highlights", "Training"]
    assert result.sections[0].content == ["Reduced invoice exceptions by 18 percent."]


def test_a_sentence_copied_twice_is_kept_once_when_the_resume_says_it_once(monkeypatch):
    source = """
Ida Ferenc
Field Notes
Studies how small museums catalog traveling exhibits. Writes finding aids for visiting collections.
"""
    repeated = "Writes finding aids for visiting collections."
    full = (
        "Studies how small museums catalog traveling exhibits. "
        "Writes finding aids for visiting collections."
    )
    _use_model(
        monkeypatch,
        {
            "candidate_name": "Ida Ferenc",
            "sections": [{"heading": "Field Notes", "content": [full, repeated]}],
        },
    )

    result = extract_resume_data(source)

    assert result.sections[0].content == [full]


def test_a_line_the_resume_repeats_is_kept(monkeypatch):
    source = """
Ida Ferenc
Notes
Check the loan folder.
Check the loan folder.
"""
    _use_model(
        monkeypatch,
        {
            "candidate_name": "Ida Ferenc",
            "sections": [
                {
                    "heading": "Notes",
                    "content": ["Check the loan folder.", "Check the loan folder."],
                }
            ],
        },
    )

    result = extract_resume_data(source)

    assert result.sections[0].content == [
        "Check the loan folder.",
        "Check the loan folder.",
    ]


def test_later_headings_are_not_repeated_inside_an_earlier_section(monkeypatch):
    source = """
Mina Alvarez
Field Notes
Studies how small museums catalog traveling exhibits. Writes finding aids for visiting collections.

Exhibitions
River House Gallery — cataloged 40 works, 2024
Lantern Annex — built a loan checklist, 2023

Studio Methods
Pigments
Mineral watercolor, iron gall ink
Mounting
Hinged mats, cotton corners
"""
    sentence = (
        "Studies how small museums catalog traveling exhibits. "
        "Writes finding aids for visiting collections."
    )
    _use_model(
        monkeypatch,
        {
            "candidate_name": "Mina Alvarez",
            "sections": [
                {
                    "heading": "Field Notes",
                    "content": [
                        sentence,
                        "Exhibitions",
                        "River House Gallery — cataloged 40 works, 2024",
                        "Lantern Annex — built a loan checklist, 2023",
                        "Studio Methods",
                        "Pigments",
                        "Mineral watercolor, iron gall ink",
                        "Mounting",
                        "Hinged mats, cotton corners",
                    ],
                },
                {
                    "heading": "Exhibitions",
                    "content": [
                        "River House Gallery — cataloged 40 works, 2024",
                        "Lantern Annex — built a loan checklist, 2023",
                    ],
                },
                {
                    "heading": "Studio Methods",
                    "content": [
                        "Pigments",
                        "Mineral watercolor, iron gall ink",
                        "Mounting",
                        "Hinged mats, cotton corners",
                    ],
                },
            ],
        },
    )

    result = extract_resume_data(source)

    notes, exhibitions, methods = result.sections
    assert notes.content == [sentence]
    assert notes.subsections == []
    assert exhibitions.content == [
        "River House Gallery — cataloged 40 works, 2024",
        "Lantern Annex — built a loan checklist, 2023",
    ]
    assert [item.heading for item in methods.subsections] == ["Pigments", "Mounting"]


def test_flattened_subheadings_are_nested_from_the_line_layout(monkeypatch):
    source = """
Mina Alvarez
Studio Methods
Pigments
Mineral watercolor, iron gall ink
Mounting
Hinged mats, cotton corners
"""
    _use_model(
        monkeypatch,
        {
            "candidate_name": "Mina Alvarez",
            "sections": [
                {
                    "heading": "Studio Methods",
                    "content": [
                        "Pigments",
                        "Mineral watercolor, iron gall ink",
                        "Mounting",
                        "Hinged mats, cotton corners",
                    ],
                }
            ],
        },
    )

    result = extract_resume_data(source)

    parent = result.sections[0]
    assert parent.content == []
    assert [item.heading for item in parent.subsections] == ["Pigments", "Mounting"]
    assert parent.subsections[0].content == ["Mineral watercolor, iron gall ink"]
    assert parent.subsections[1].content == ["Hinged mats, cotton corners"]


def test_deeply_nested_headings_stay_nested(monkeypatch):
    source = """
Owen Blake
Workshop
Blades
Steel Types
Carbon and stainless stock.
Handles
Oak and ash.
Clamps
Bar Clamps
Sized for travel crates.
Corner Jigs
Birch blocks.
"""
    _use_model(
        monkeypatch,
        {
            "candidate_name": "Owen Blake",
            "sections": [
                {
                    "heading": "Workshop",
                    "content": [
                        "Blades",
                        "Steel Types",
                        "Carbon and stainless stock.",
                        "Handles",
                        "Oak and ash.",
                        "Clamps",
                        "Bar Clamps",
                        "Sized for travel crates.",
                        "Corner Jigs",
                        "Birch blocks.",
                    ],
                }
            ],
        },
    )

    result = extract_resume_data(source)

    workshop = result.sections[0]
    assert [item.heading for item in workshop.subsections] == ["Blades", "Clamps"]
    blades, clamps = workshop.subsections
    assert [item.heading for item in blades.subsections] == ["Steel Types", "Handles"]
    assert blades.subsections[0].content == ["Carbon and stainless stock."]
    assert blades.subsections[1].content == ["Oak and ash."]
    assert [item.heading for item in clamps.subsections] == ["Bar Clamps", "Corner Jigs"]
    assert clamps.subsections[0].content == ["Sized for travel crates."]


def test_one_title_line_is_not_turned_into_a_subsection(monkeypatch):
    source = """
Ada West
Selected Work
Northwind Analytics
Built billing APIs for subscription renewals.
"""
    _use_model(
        monkeypatch,
        {
            "candidate_name": "Ada West",
            "sections": [
                {
                    "heading": "Selected Work",
                    "content": [
                        "Northwind Analytics",
                        "Built billing APIs for subscription renewals.",
                    ],
                }
            ],
        },
    )

    result = extract_resume_data(source)

    assert result.sections[0].subsections == []
    assert result.sections[0].content == [
        "Northwind Analytics",
        "Built billing APIs for subscription renewals.",
    ]


def test_empty_bullet_markers_are_dropped_and_the_text_is_kept(monkeypatch):
    source = """
Nia Shah
WORK HISTORY
Analyst — Northline | 2020 – 2022
•
Wrote weekly status notes.
•
Kept the incident log.
"""
    _use_model(
        monkeypatch,
        {
            "candidate_name": "Nia Shah",
            "sections": [
                {
                    "heading": "WORK HISTORY",
                    "content": [
                        "Analyst — Northline | 2020 – 2022",
                        "•",
                        "Wrote weekly status notes.",
                        "•",
                        "Kept the incident log.",
                    ],
                }
            ],
        },
    )

    result = extract_resume_data(source)

    assert result.sections[0].content == [
        "Analyst — Northline | 2020 – 2022",
        "Wrote weekly status notes.",
        "Kept the incident log.",
    ]


def test_a_skipped_second_role_and_school_are_restored(monkeypatch):
    source = """
Nia Shah
nia.shah@example.com
PROFESSIONAL EXPERIENCE
Analyst — Harbor Metrics | March 2022 – Present
•
Designed the weekly operations report.
•
Checked the source tables before each review.
Researcher — Lumen Field | August 2019 – February 2022
•
Queried sales records to find delayed orders.
•
Built the Monday briefing for the field team.
EDUCATION
Master of Letters — Harbor Institute, 2019
Bachelor of Arts — Lumen College, 2017
"""
    _use_model(
        monkeypatch,
        {
            "candidate_name": "Nia Shah",
            "email": "nia.shah@example.com",
            "sections": [
                {
                    "heading": "PROFILE",
                    "content": ["Notes."],
                    "subsections": [
                        {
                            "heading": "PROFESSIONAL EXPERIENCE",
                            "content": [
                                "Analyst — Harbor Metrics | March 2022 – Present",
                                "Designed the weekly operations report.",
                            ],
                        }
                    ],
                },
                {
                    "heading": "EDUCATION",
                    "content": ["Master of Letters — Harbor Institute, 2019"],
                },
            ],
        },
    )

    result = extract_resume_data(source)

    assert _headings(result) == ["PROFESSIONAL EXPERIENCE", "EDUCATION"]
    experience, education = result.sections
    assert experience.content == [
        "Analyst — Harbor Metrics | March 2022 – Present",
        "Designed the weekly operations report.",
        "Checked the source tables before each review.",
        "Researcher — Lumen Field | August 2019 – February 2022",
        "Queried sales records to find delayed orders.",
        "Built the Monday briefing for the field team.",
    ]
    assert education.content == [
        "Master of Letters — Harbor Institute, 2019",
        "Bachelor of Arts — Lumen College, 2017",
    ]


def test_later_capital_headings_stay_separate_sections(monkeypatch):
    source = """
Nia Shah
TECHNICAL SKILLS
Python, SQL, Power BI
CORE STRENGTHS
Writing, Mentoring
CERTIFICATIONS
Warehouse Safety — Harbor Institute, 2021
Records Keeping — Lumen School, 2020
"""
    _use_model(
        monkeypatch,
        {
            "candidate_name": "Nia Shah",
            "sections": [
                {
                    "heading": "TECHNICAL SKILLS",
                    "content": [
                        "Python, SQL, Power BI",
                        "Core Strengths: Writing, Mentoring",
                        "Certifications:",
                        "Warehouse Safety — Harbor Institute, 2021",
                        "Records Keeping — Lumen School, 2020",
                    ],
                }
            ],
        },
    )

    result = extract_resume_data(source)

    assert _headings(result) == ["TECHNICAL SKILLS", "CORE STRENGTHS", "CERTIFICATIONS"]
    skills, strengths, certifications = result.sections
    assert skills.content == ["Python, SQL, Power BI"]
    assert strengths.content == ["Writing, Mentoring"]
    assert certifications.content == [
        "Warehouse Safety — Harbor Institute, 2021",
        "Records Keeping — Lumen School, 2020",
    ]


def test_a_long_resume_sends_the_later_heading_to_the_model(monkeypatch):
    monkeypatch.setattr("app.services.resume_extractor.MAX_PROMPT_CHARACTERS", 80)
    source = (
        "Nia Shah\nOverview\n"
        + ("Kept this opening sentence. " * 12)
        + "\nLater Notes\nFinal line stays."
    )
    prompts: list[str] = []

    def fake_generate(prompt: str) -> str:
        prompts.append(prompt)
        if "Later Notes" in prompt:
            return json.dumps(
                {"sections": [{"heading": "Later Notes", "content": ["Final line stays."]}]}
            )
        if "Overview" in prompt:
            return json.dumps(
                {
                    "candidate_name": "Nia Shah",
                    "sections": [{"heading": "Overview", "content": ["Kept this opening sentence."]}],
                }
            )
        return json.dumps({"sections": []})

    monkeypatch.setattr("app.services.resume_extractor.generate_response", fake_generate)

    result = extract_resume_data(source)

    assert len(prompts) > 1
    assert any("Later Notes" in prompt for prompt in prompts)
    assert _headings(result) == ["Overview", "Later Notes"]
    assert result.sections[1].content == ["Final line stays."]


def test_ampersand_headings_keep_document_order(monkeypatch):
    source = """
Nia Shah
FIELD NOTES
Kept the catalog current.
SKILLS & TOOLING
Python, SQL
KEY WORK & IMPACT
Harbor Ledger (Risk Automation)
Objective: Built the weekly risk file.
Lane Mapper (Trade Routing)
Objective: Ranked shipping lanes.
EDUCATION
Master of Letters — Harbor Institute, 2019
"""
    _use_model(
        monkeypatch,
        {
            "candidate_name": "Nia Shah",
            "sections": [{"heading": "FIELD NOTES", "content": ["Kept the catalog current."]}],
        },
    )

    result = extract_resume_data(source)
    assert _headings(result) == [
        "FIELD NOTES",
        "SKILLS & TOOLING",
        "KEY WORK & IMPACT",
        "EDUCATION",
    ]
    work = result.sections[2]
    assert [item.heading for item in work.subsections] == [
        "Harbor Ledger (Risk Automation)",
        "Lane Mapper (Trade Routing)",
    ]
    assert work.subsections[0].content == ["Objective: Built the weekly risk file."]
    assert result.sections[3].content == ["Master of Letters — Harbor Institute, 2019"]


def test_extracted_phone_and_email_are_not_repeated_unclassified(monkeypatch):
    source = """
Nia Shah
Catalog researcher
Phone : +91 8085489720 Email: nia.shah@example.com LinkedIn: www.linkedin.com/in/nia-shah
FIELD NOTES
Kept the catalog current.
"""
    _use_model(
        monkeypatch,
        {
            "candidate_name": "Nia Shah",
            "email": "nia.shah@example.com",
            "phone": "+91 8085489720",
            "sections": [{"heading": "FIELD NOTES", "content": ["Kept the catalog current."]}],
        },
    )

    result = extract_resume_data(source)
    unclassified = next(section for section in result.sections if section.heading is None)
    joined = "\n".join(unclassified.content)

    assert result.email == "nia.shah@example.com"
    assert result.phone == "+91 8085489720"
    assert "Catalog researcher" in joined
    assert "LinkedIn: www.linkedin.com/in/nia-shah" in joined
    assert "+91 8085489720" not in joined
    assert "nia.shah@example.com" not in joined
    assert joined.count("Catalog researcher") == 1


def test_pipe_separated_and_title_case_roles_nest_under_experience(monkeypatch):
    source = """
Nia Shah
PROFESSIONAL EXPERIENCE
Catalog Engineer | Records Systems
Harbor Ledger | March 2022 – Present
Designed the weekly operations report.
Data Steward
Lumen Field | August 2019 – February 2022
Queried sales records to find delayed orders.
EDUCATION
Master of Letters — Harbor Institute, 2019
"""
    _use_model(
        monkeypatch,
        {
            "candidate_name": "Nia Shah",
            "sections": [{"heading": "PROFESSIONAL EXPERIENCE", "content": []}],
        },
    )

    result = extract_resume_data(source)
    experience = next(section for section in result.sections if section.heading == "PROFESSIONAL EXPERIENCE")

    assert [item.heading for item in experience.subsections] == [
        "Catalog Engineer | Records Systems",
        "Data Steward",
    ]
    assert experience.subsections[0].content == [
        "Harbor Ledger | March 2022 – Present",
        "Designed the weekly operations report.",
    ]
    assert experience.subsections[1].content == [
        "Lumen Field | August 2019 – February 2022",
        "Queried sales records to find delayed orders.",
    ]
    assert "Catalog Engineer | Records Systems" not in experience.content
    assert "Data Steward" not in experience.content


def test_project_colon_fields_nest_contribution_bullets(monkeypatch):
    source = """
Nia Shah
FIELD NOTES
Kept the catalog current.
KEY WORK & IMPACT
Harbor Ledger (Risk Automation)
Objective: Built the weekly risk file.
Technologies: Python, SQL
Key Contributions:
Wrote the intake sheet.
Checked the source tables.
Impact: Cut review time.
Lane Mapper (Trade Routing)
Objective: Ranked shipping lanes.
EDUCATION
Master of Letters — Harbor Institute, 2019
"""
    _use_model(
        monkeypatch,
        {
            "candidate_name": "Nia Shah",
            "sections": [
                {
                    "heading": "KEY WORK & IMPACT",
                    "content": [
                        "Harbor Ledger (Risk Automation)",
                        "Objective: Built the weekly risk file.",
                        "Wrote the intake sheet.",
                        "Impact: Cut review time.",
                        "Wrote the intake sheet.",
                    ],
                }
            ],
        },
    )

    result = extract_resume_data(source)
    work = next(section for section in result.sections if section.heading == "KEY WORK & IMPACT")
    project = next(
        item for item in work.subsections if item.heading == "Harbor Ledger (Risk Automation)"
    )
    other = next(item for item in work.subsections if item.heading == "Lane Mapper (Trade Routing)")

    assert project.content == []
    assert [item.heading for item in project.subsections] == [
        "Objective",
        "Technologies",
        "Key Contributions",
        "Impact",
    ]
    assert project.subsections[0].content == ["Built the weekly risk file."]
    assert project.subsections[1].content == ["Python, SQL"]
    assert project.subsections[2].content == [
        "Wrote the intake sheet.",
        "Checked the source tables.",
    ]
    assert project.subsections[3].content == ["Cut review time."]
    assert other.content == ["Objective: Ranked shipping lanes."]
    joined = "\n".join(
        [*project.content, *[line for child in project.subsections for line in child.content]]
    )
    assert joined.count("Wrote the intake sheet.") == 1
    assert joined.count("Cut review time.") == 1


def test_a_single_role_is_not_replaced_with_an_unwritten_second_job(monkeypatch):
    source = """
Arjun Rao
arjun.rao.demo@example.com | +91 9988776655 | Hyderabad, India
PROFILE
Data Analyst with 5 years of experience transforming operational data into business insights. Skilled in Python, SQL, dashboard development, and stakeholder communication.
PROFESSIONAL EXPERIENCE
Lead Data Analyst — Meridian Analytics Pvt. Ltd. | March 2022 – Present
•
Designed and maintained reporting dashboards used by six business teams.
•
Automated recurring data quality checks with Python, reducing manual review time by 30%.
•
Presented monthly performance trends to product and operations stakeholders.
EDUCATION
Master of Science in Data Analytics — Deccan Institute of Technology, 2019
Bachelor of Commerce — Osmania City College, 2017
TECHNICAL SKILLS
Python, SQL, Power BI, Excel, Pandas, Data Visualization
CORE STRENGTHS
Analytical Thinking, Communication, Collaboration, Attention to Detail
CERTIFICATIONS
Microsoft Power BI Data Analyst — DataSkills Institute, 2021
SQL for Business Intelligence — LearnHub, 2020
"""
    _use_model(
        monkeypatch,
        {
            "candidate_name": "Arjun Rao",
            "email": "arjun.rao.demo@example.com",
            "phone": "+91 9988776655",
            "location": "Hyderabad, India",
            "sections": [
                {
                    "heading": "PROFESSIONAL EXPERIENCE",
                    "content": [
                        "Lead Data Analyst — Meridian Analytics Pvt. Ltd. | March 2022 – Present",
                        "Data Analyst — CedarBridge Services | August 2019 – February 2022",
                    ],
                },
                {
                    "heading": "TECHNICAL SKILLS",
                    "content": [
                        "Python, SQL, Power BI, Excel, Pandas, Data Visualization",
                        "Core Strengths: Analytical Thinking, Communication, Collaboration, Attention to Detail",
                    ],
                },
            ],
        },
    )

    result = extract_resume_data(source)
    experience = next(section.content for section in result.sections if section.heading == "PROFESSIONAL EXPERIENCE")
    joined = "\n".join(experience)

    assert _headings(result) == [
        "PROFILE",
        "PROFESSIONAL EXPERIENCE",
        "EDUCATION",
        "TECHNICAL SKILLS",
        "CORE STRENGTHS",
        "CERTIFICATIONS",
    ]
    assert "CedarBridge" not in joined
    assert "Lead Data Analyst — Meridian Analytics Pvt. Ltd. | March 2022 – Present" in experience
    assert result.sections[2].content == [
        "Master of Science in Data Analytics — Deccan Institute of Technology, 2019",
        "Bachelor of Commerce — Osmania City College, 2017",
    ]
    assert result.sections[3].content == [
        "Python, SQL, Power BI, Excel, Pandas, Data Visualization"
    ]
    assert result.sections[4].heading == "CORE STRENGTHS"
    assert result.sections[5].heading == "CERTIFICATIONS"


def test_invalid_email_is_dropped_instead_of_failing_the_resume():
    resume = ResumeData.model_validate(
        {
            "candidate_name": "Amina Rahman",
            "email": "not-an-email",
            "sections": [{"heading": "Notes", "content": ["On call this quarter."]}],
        }
    )

    assert resume.candidate_name == "Amina Rahman"
    assert resume.email is None
    assert resume.sections[0].heading == "Notes"


def test_placeholder_lines_and_empty_sections_are_removed():
    resume = ResumeData.model_validate(
        {
            "candidate_name": "N/A",
            "sections": [
                {"heading": "Not specified", "content": ["Unknown"]},
                {"heading": "Notes", "content": ["Not specified", "Keeps the lab notebook."]},
                {"heading": None, "content": [], "subsections": []},
            ],
        }
    )

    assert resume.candidate_name is None
    assert _headings(resume) == ["Notes"]
    assert resume.sections[0].content == ["Keeps the lab notebook."]
