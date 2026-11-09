# AI Resume Parser

Local resume parser for PDF and DOCX files. FastAPI accepts an upload, extracts the text, asks a local Gemma 3 model through Ollama for JSON, validates that JSON, stores it in SQLite, and serves it to a Streamlit dashboard.

No paid model API is used.

## Features

- PDF text extraction with PyMuPDF and DOCX text extraction with python-docx, including tables
- Structured extraction with local Ollama (`gemma3:1b` by default)
- JSON mode, timeout handling, and cleanup of Markdown fences
- Pydantic validation that keeps every heading the model returns, including nested subsections
- Placeholder text such as "Not specified" is removed during validation
- `POST /api/upload` and `GET /api/resume/{document_id}`
- SQLite storage that survives restarts, with a new document ID for every upload
- Streamlit dashboard that reads the API over HTTP
- Pytest coverage with the model mocked, plus a manual script that calls Ollama

## Architecture

```text
PDF / DOCX
    -> FastAPI  POST /api/upload
    -> file validation and safe storage
    -> text extraction (full document, visual reading order)
    -> Ollama Gemma 3  POST /api/generate
    -> JSON parse + Pydantic ResumeData
    -> recover source lines the model omitted
    -> SQLite
    -> JSON response
    -> Streamlit dashboard
```

```text
resume-parser/
├── app/
│   ├── main.py                  # FastAPI routes
│   ├── config.py                # environment settings
│   ├── models.py                # Pydantic schemas
│   ├── exceptions.py
│   ├── database/
│   │   └── repository.py        # SQLite repository
│   ├── services/
│   │   ├── parser.py            # PDF and DOCX text
│   │   ├── llm_service.py       # Ollama client
│   │   ├── resume_extractor.py  # prompt, JSON, validation
│   │   └── resume_service.py    # upload pipeline
│   └── utils/
│       └── logger.py
├── ui/
│   └── streamlit_app.py
├── tests/
│   ├── test_parser.py
│   ├── test_llm.py
│   ├── test_extractor.py
│   ├── test_api.py
│   ├── test_database.py
│   └── manual_ollama_integration.py
├── data/
│   ├── resumes.db               # created on startup
│   └── uploads/
└── logs/
    └── app.log
```

## Technology stack

- Python 3.13
- FastAPI and Uvicorn
- Streamlit
- Pydantic and email-validator
- PyMuPDF and python-docx
- Requests to the local Ollama API
- SQLite
- Loguru
- Pytest and HTTPX

## Prerequisites

- macOS on Apple Silicon, or another machine that can run Ollama
- Python 3.13
- Ollama installed and running at `http://localhost:11434`
- About 1 GB free for the `gemma3:1b` model weights

## Ollama and the Gemma model

Install Ollama from [https://ollama.com/download](https://ollama.com/download), then start it and pull the model:

```bash
ollama serve
ollama pull gemma3:1b
ollama list
```

`ollama list` should show `gemma3:1b`. The app reads the model name from `OLLAMA_MODEL`, so a different local model can be selected without code changes. This project was verified with `gemma3:1b`.

## Python environment

From the project root `/Users/anandagnihotri/projects/resume-parser`:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The virtual environment in this workspace is already created at `venv/`.

## Environment variables

Copy the example file if `.env` is missing:

```bash
cp .env.example .env
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama server |
| `OLLAMA_MODEL` | `gemma3:1b` | Model name passed to `/api/generate` |
| `OLLAMA_TIMEOUT_SECONDS` | `180` | Request timeout for generation |
| `MAX_UPLOAD_SIZE_MB` | `10` | Upload size limit |
| `DATABASE_PATH` | `data/resumes.db` | SQLite file |
| `UPLOAD_DIR` | `data/uploads` | Stored resumes |
| `API_BASE_URL` | `http://127.0.0.1:8000` | Address the Streamlit app calls |
| `API_TIMEOUT_SECONDS` | `210` | Streamlit HTTP timeout |
| `LOG_LEVEL` | `INFO` | Loguru level |

`load_dotenv` does not override variables that are already set in the shell.

## Database

No separate database setup is required. On startup the API creates `data/resumes.db` and the `resumes` table if they do not exist.

Each row stores the document ID, original filename, stored file path, extracted JSON, status, and UTC timestamp. Uploading the same file twice creates two rows.

## Run the API

```bash
source venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Run the dashboard

In a second terminal:

```bash
source venv/bin/activate
streamlit run ui/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501). The dashboard uploads and retrieves resumes through the API. It does not call the parser directly.

## API

### `POST /api/upload`

Multipart form field: `file`. Accepts `.pdf` and `.docx`.

```bash
curl -X POST http://127.0.0.1:8000/api/upload \
  -F "file=@data/uploads/sample_resume.docx"
```

The stored resume keeps the candidate's own headings. A section with no heading uses `null`. Nested headings stay in `subsections`. The section list below is the local `gemma3:1b` result for `data/uploads/sample_resume.docx`.

```json
{
  "document_id": "a406c333b1664cef91983a361131593f",
  "status": "completed",
  "filename": "sample_resume.docx",
  "created_at": "2026-11-09T18:25:33+00:00",
  "resume": {
    "candidate_name": "Rahul Sharma",
    "email": "rahul@gmail.com",
    "phone": "9876543210",
    "location": null,
    "sections": [
      {
        "heading": "Experience",
        "content": ["3 years", "ABC Technologies"],
        "subsections": []
      },
      {
        "heading": "Role",
        "content": ["Python Developer"],
        "subsections": []
      },
      {
        "heading": "Skills",
        "content": ["Python", "Django", "FastAPI", "SQL"],
        "subsections": []
      }
    ]
  }
}
```

Status codes:

| Code | When |
| --- | --- |
| 200 | Extraction stored |
| 400 | Unsupported type, empty file, or corrupt PDF/DOCX |
| 413 | File larger than `MAX_UPLOAD_SIZE_MB` |
| 422 | No readable text, invalid model JSON, or schema failure |
| 503 | Ollama is down, timed out, or the model is missing |
| 500 | Database or unexpected failure |

### `GET /api/resume/{document_id}`

```bash
curl http://127.0.0.1:8000/api/resume/a406c333b1664cef91983a361131593f
```

Returns the same payload that was stored. An unknown ID returns `404`.

## Testing

Automated tests mock Ollama. They do not need the model running.

```bash
source venv/bin/activate
pytest
```

Covered cases:

- Valid PDF and DOCX, including DOCX tables
- Unsupported extension, empty file, corrupt PDF, corrupt DOCX, and a PDF with no text
- Ollama connection failure, timeout, missing model, HTTP error, and empty response
- Invalid model JSON and Markdown fences
- Standard headings, custom headings, nested subsections, and unusual heading order
- Sections that the resume does not contain are not invented
- Placeholder text and empty sections removed
- SQLite read after a new connection, separate IDs for duplicate saves, and a bad database path
- Upload, retrieve, unknown ID, duplicate uploads, oversize upload, and a database error on read

The last local run was `68 passed`.

Manual check against the running Ollama model:

```bash
python tests/manual_ollama_integration.py
```

This calls Gemma on both sample resumes and fails only when the response cannot be parsed into the resume schema.

## Known limitations

Gemma fills name, email, phone, location, and sections for resumes that do not use heading lines. When the document has heading lines with their own content, those headings and the lines under them are taken from the document so a later section is not dropped or merged. There is no fixed list of heading names. Gemma 3 1B can still misread contact fields. Placeholder strings such as "Not specified" are still cleared by the schema.

- Scanned PDFs with no text layer are rejected. There is no OCR.
- Legacy `.doc` files and encrypted PDFs are rejected.
- The API has no authentication. It is meant to run locally.
- Generation time depends on the machine. On this M1 with 8 GB of RAM, the short sample resumes returned in about 4 to 6 seconds.

## Troubleshooting

**`The local Ollama service is unavailable`**

Start Ollama and confirm `curl http://localhost:11434/api/tags` lists `gemma3:1b`. Pull it with `ollama pull gemma3:1b` if it is missing.

**Streamlit says it cannot reach the API**

Start Uvicorn first and check that `API_BASE_URL` matches that address.

**`No readable text could be extracted`**

The file opened, but it has no text layer. Export a text-based PDF or DOCX.

**Upload returns 400 for a real PDF**

The file may be encrypted or truncated. Try opening it locally and exporting a new copy.

**Port already in use**

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
lsof -nP -iTCP:8501 -sTCP:LISTEN
```

Stop the existing process or choose another port. If the API port changes, set `API_BASE_URL` to the same port before starting Streamlit.

Logs are written to `logs/app.log`. They record document IDs, file names, sizes, and error types. They do not record the resume text.
