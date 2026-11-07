# Walkthrough of the working application

**`walkthrough.mp4`** (2 min 49 s, 1440 × 900, H.264, about 6 MB) is a recorded walkthrough of the whole system running
live in the compose stack: the Next.js UI, the FastAPI API, the worker with RapidOCR and a local Qwen3-4B model on
llama.cpp, and PostgreSQL with pgvector. `walkthrough.webm` is the same recording as Playwright wrote it (VP9, about
29 MB); the MP4 is converted from it and plays everywhere, including QuickTime and phones. It was recorded on
2026-11-05 against the final application code (stack images built at `c3e0dfc`, a point in the development history; no application code has changed since).

Nothing is mocked or edited: Playwright drives Chromium (slowMo 250 ms) through the real UI, and captions are drawn
into the page to say what each step shows. A ring marks the mouse pointer, and an amber outline marks the element being
discussed. There is no voice-over.

The sample files come from the labelled evaluation set in `backend/eval/dataset`:

| File | What it is |
|---|---|
| `pdf/t1_digital_03.pdf` | Digital PDF invoice PFI-25824 from Pinecrest Fasteners Inc., CAD 180.04 |
| `images/t2_scan_light_01.jpg` | Scanned invoice (JPEG), read by OCR |
| `ledgers/xv_ledger.xlsx` | Excel receivables ledger, 20 rows on two sheets. Row PFI-25824 was keyed as 423.47 / 430.39 |
| `corrupt/truncated.pdf` | A PDF cut short: accepted, then fails safely |
| `corrupt/empty.pdf` | 0 bytes: refused before upload |

## What it shows (timestamps)

The timestamps are the ones burned into the captions. They run about one second behind the video player's clock.

| Time | Step |
|---|---|
| 00:00 | Title card |
| 00:05 | **Dashboard**: live totals (documents, in progress, records to review, failures, open checks, processing time) and the local LLM model |
| 00:10 | **Upload a mixed batch**: digital PDF, scanned JPEG, Excel ledger, truncated PDF and empty file in one multi-file upload |
| 00:15 | **One outcome per file**: the empty file is refused at once with its reason; the other four are queued, each with a link |
| 00:21 | **Processing in the background**: the documents table updates live with each stage (text layer or OCR, LLM extraction, validation, cross-checks, indexing) |
| 00:24 | **Seconds later**: the PDF needs review, the ledger gave 20 records, the truncated PDF failed safely with its reason, and the scan is still in OCR and extraction |
| 00:32 | **The extracted invoice**: every field shows its source (rules, LLM or both) and a confidence score; the record has an overall score |
| 00:38 | **Evidence**: the total's evidence tooltip quotes the source text ("Balance Due $180.04 CAD", page 1) |
| 00:44 | **Validation**: one error (the record differs from its ledger row) and one warning (the rules and the LLM read the supplier name differently) |
| 00:51 | **Cross-verification, PDF vs Excel ledger**: same invoice number, but subtotal and total differ by 250.35; the PDF agrees with the printed invoice, so the ledger row was keyed wrong |
| 00:59 | **The ledger row**: PFI-25824 is sheet Domestic, row 6 of `xv_ledger.xlsx` |
| 01:06 | **Correct the two amounts** (173.12 and 180.04); the edited fields are marked with the values they replace |
| 01:11 | **Say why**: a comment for the audit trail |
| 01:16 | **Saved and re-validated on the server**: the values are now manual (100 %), and the toast reads "Checked again: every check passes". The mismatch is gone from both records |
| 01:24 | **Approve**: the record is marked reviewed, with who approved it and when |
| 01:31 | **History and audit trail**: processing events, plus the corrections and the approval (field, before, after, comment, who, when) |
| 01:39 | **Back on the PDF**: the cross-check against the ledger row now matches |
| 01:43 | **Approve the PDF invoice**: its only warning was the supplier name, and the name kept (Pinecrest Fasteners Inc.) is the one printed on the invoice. The document becomes Completed (01:51) |
| 01:54 | **The scanned JPEG**: completed. Its Text tab shows the OCR text with method and OCR confidence (01:58) |
| 02:05 | **Keyword search**: exact words and numbers across PDF text, OCR text and ledger rows, with the matched words highlighted |
| 02:14 | **Semantic search** for "cutting discs for grinders": the Pinecrest invoice ranks first without the supplier being named |
| 02:21 | **Compare**: the PDF invoice on one side, its Excel ledger row on the other |
| 02:27 | **Cross-verification view, field by field**: a tick on every field after the correction |
| 02:34 | Closing card |

## Screenshots

`screenshots/` holds the key steps, captured during the same run:

1. `01-upload-results.png`: per-file upload outcomes, with the empty file rejected and its reason
2. `02-processing-live.png`: live stage progress in the documents table
3. `03-fields-confidence-evidence.png`: extracted fields with source, confidence and the evidence tooltip
4. `04-validation-issues.png`: validation issues grouped by severity (one error, one warning)
5. `05-cross-verification-mismatch.png`: PDF vs Excel ledger with the subtotal and total mismatches
6. `06-correction-saved.png`: corrected amounts (manual, 100 %) and the "Checked again: every check passes" toast
7. `07-history-audit.png`: processing history and the correction and approval audit trail
8. `08-search-keyword.png`: keyword search with highlighted matches
9. `09-search-semantic.png`: semantic search ranking the invoice first
10. `10-compare-pdf-vs-ledger.png`: field-by-field comparison, all matching

## Re-recording

With the compose stack running (UI on http://127.0.0.1:3000):

```bash
cd frontend
npx playwright install chromium   # once
npm run test:walkthrough          # writes video/walkthrough.webm and video/screenshots/
```

The MP4 is converted from the WebM with ffmpeg (any build with libx264):

```bash
ffmpeg -i video/walkthrough.webm -c:v libx264 -preset slow -crf 22 -pix_fmt yuv420p -movflags +faststart -an video/walkthrough.mp4
```

The run prints the caption timeline (in video time) that the table above is taken from.

The spec is `frontend/e2e/walkthrough.spec.ts`. It asserts every state it narrates, so it fails rather than recording a
broken story. Before recording, it deletes earlier copies of its four sample files (matched by SHA-256) and waits for
the processing queue to be empty. The documents it uploads are kept as its end state. Processing times vary with the
GPU: the scan's OCR and extraction took 14–95 s across runs. The story does not wait for the scan until it needs it.
