# Loan Pledge Slip OCR API

FastAPI service that reads gold-loan pledge slips (JPG, PNG, PDF) with PaddleOCR and
extracts fields such as customer name, bank, loan amount, dates, interest rate, tenure
and weight. Values that are not clearly read are marked for review instead of being
guessed: a wrong value must never be marked as extracted.

## Requirements

- Python 3.11.5 or newer, 64-bit
- About 2 GB of RAM per running worker, and about 3 GB of disk space for the packages
- Internet access on the first run (the OCR models are downloaded once)
- Keep the project outside OneDrive or other sync folders (they can damage the large
  OCR library files in the virtual environment)

## Setup

```
python -m venv venv311
venv311\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env`. Set `CORS_ORIGINS` to the address(es) of the frontend that calls this API,
separated by commas.


### Handwriting re-read (TrOCR)

The TrOCR packages (`torch`, `transformers`) are installed with the rest. The model is
off by default. To turn it on, set `ENABLE_TROCR=true` in `.env`. This uses more RAM.

## Run

``` 
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Wait for `Application startup complete`. Then open:

- `http://<server-ip>:8000/docs` for the interactive API documentation
- `http://<server-ip>:8000/api/v1/health` for a status check

For development with automatic reload, run `uvicorn app.main:app --reload` instead.

Each worker loads its own OCR model, so `--workers 2` doubles the memory use and lets two
slips be read at the same time.

## API

### `POST /api/v1/pledge-slips/upload`

Multipart form with one field named `file` (JPG, JPEG, PNG or PDF, up to 15 MB).

The response contains:

- `extractedData`: the fields. Each one has `value`, `rawText`, `unit`, `status`,
  `ocrConfidence`, `matchConfidence`, `note` and `evidence`.
- `ocr`: the raw OCR text (`rawText`) and the text lines with their boxes (`lines`).
- `review`: which mandatory fields are missing, uncertain or invalid, and whether a person
  needs to check the slip (`required`).
- `document`: file name, type and page information.
- `requestId`: identifier of the request, also in the `X-Request-ID` response header.

The 13 mandatory fields are: `customerName`, `bankName`, `loanAccountNumber`,
`pledgeNumber`, `loanAmount`, `loanDate`, `dueDate`, `branchName`, `phoneNumber`,
`interestRate`, `tenure`, `netWeight`, `address`. The list is set by `MANDATORY_FIELDS`
in `.env`.

### Field status

| Status | Meaning |
|---|---|
| `extracted` | Read clearly and passed all checks. Safe to use. |
| `uncertain` | A value was found but something is doubtful (low OCR confidence, weak label match). A person should check it. |
| `invalid` | A label was found but its value could not be read as the expected type. |
| `missing` | Not found on the slip. |

Only `extracted` values should be used without checking.

### `GET /api/v1/health`

Service status, and whether the OCR engine has loaded.

### Errors

Errors are returned as JSON with an `error` object (`code` and `message`). Common codes:
`file_too_large` (413), unsupported or invalid file (415, 422), `ocr_unavailable` (503),
`server_busy` (503).

## How a slip is processed

```
upload -> file checks -> waiting queue -> preprocessing (rotate, deskew, resize)
       -> PaddleOCR (text lines with boxes) -> optional handwriting re-read
       -> field extraction (label + value rules, tables)
       -> validation (types, amount in words, cross-checks) -> review summary -> JSON
```

Nothing is stored: each upload is processed in memory and the result is returned
immediately.

## Busy server

One slip is read at a time per worker. Extra uploads wait in a queue
(`OCR_QUEUE_MAX`, `OCR_QUEUE_TIMEOUT`). When the queue is full, or a request waits too long,
the API answers `503 server_busy`.

## Configuration

All settings are in `.env`. See `.env.example` for the full list. The most important:

| Setting | Meaning |
|---|---|
| `CORS_ORIGINS` | Frontend addresses allowed to call the API |
| `MAX_UPLOAD_MB` | Maximum upload size |
| `OCR_MAX_CONCURRENCY` | Slips read at the same time per worker (keep 1) |
| `OCR_QUEUE_MAX` | Uploads allowed to wait per worker |
| `OCR_QUEUE_TIMEOUT` | Seconds an upload may wait before `server_busy` |
| `FIELD_MIN_OCR_CONFIDENCE` | Below this OCR confidence a value becomes `uncertain` |
| `FIELD_MIN_MATCH_CONFIDENCE` | Below this label-match confidence a value becomes `uncertain` |
| `MANDATORY_FIELDS` | Fields reported in the review summary |

## Tests

```
python -m pytest -q
```

## Project layout

```
app/
  main.py                 application setup, CORS, startup
  api/routes.py           upload and health endpoints
  core/                   settings, error handling, extraction rules (labels, aliases)
  schemas/                response models
  services/               file checks, preprocessing, OCR, field extraction,
                          validation, review summary
  utils/                  request id helper
tests/                    automated tests and fixtures
```

## Privacy

Pledge slips contain personal data (names, account numbers, amounts). Before real use:
protect the endpoint (login or API key), use HTTPS, and decide whether logs may contain
extracted text.