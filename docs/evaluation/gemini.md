# Evaluation: Google Gemini (`gemini-3-flash-preview`)

**Status: not measured.** There is no `hybrid-gemini` results file, so this page has no accuracy numbers. After the
key's free-tier quota proved too small for a full run (below), the project owner decided a Gemini evaluation is not
needed. The provider itself is implemented and unit-tested, and a live request was checked. The results for the
local Qwen3-4B model are in [qwen3-4b.md](qwen3-4b.md), and the overview is [../EVALUATION.md](../EVALUATION.md).
*How to finish the run* below says how to measure it later; this page is then regenerated with `python -m eval.report`.
Commit IDs quoted in the documents (such as `7907c58`) name the point in the development history at which a measurement was taken; the published repository holds the finished work as a single commit, so they identify runs rather than browsable commits.

## Why there are no numbers yet

The key used here is a free-tier Google AI Studio key. For `gemini-3-flash-preview` it allows **20 requests per day**
(the 429 response names the quota `GenerateRequestsPerDayPerProjectPerModel-FreeTier`, limit 20). A full run over the
labeled dataset needs about 170 requests, one per invoice that goes through the LLM: 169 records in the Qwen run took
the hybrid path, while the 213 spreadsheet rows never use the LLM. That is about nine days of free quota.

## What was tried (2026-11-05)

| Step | Code | Outcome | Log |
| :--- | :--- | :--- | :--- |
| Key check | `5afbd28` | One request through the app's Gemini provider answered correctly in 1.9 s | – |
| Attempt 1 | `5afbd28` | Stopped by hand after 19 of 164 files: 4 requests failed with HTTP 429 `RESOURCE_EXHAUSTED` and fell back to rules (the provider then gave up after about 6 s of retries) | `backend/eval/results/logs/discarded/hybrid-gemini-attempt1-rate-limited.log` |
| Attempt 2 | `944891e` | Stopped by hand after 16 files: retries now waited about a minute, but the 429s were the daily quota (retry delay about 20 h) | `backend/eval/results/logs/discarded/hybrid-gemini-attempt2-daily-quota.log` |

Neither attempt is reported. Records that fell back to rules would have been scored as Gemini's, which would describe
the quota, not the model.

What changed in the product as a result (`31d2252`): the Gemini provider follows the delay the 429 asks for. It waits
and retries when that delay is a minute or less, and falls back to rules at once when the quota is spent, logging
"quota exhausted, retry after N h". Before, it gave up on the first 429; the interim version waited about a minute
for nothing.

## How to finish the run

Any one of these makes a full run possible:

1. **Enable billing** on the Google Cloud project that owns the key (paid tier). The limits become much higher, and
   Google does not use paid-tier content to improve its products. A full run is about 170 requests.
2. **Use a Gemini model with a larger free quota**: set `GEMINI_MODEL` and check that model's limits on the AI Studio
   rate-limit page first. The results then describe that model, not `gemini-3-flash-preview`.

Then, from `backend/` (the key is read from the environment or the root `.env`):

    python -m eval.run_eval --dataset eval/dataset --config hybrid-gemini --out eval/results/hybrid-gemini.json
    python -m eval.report eval/results/rules-postfix.json eval/results/hybrid-gemini.json \
      --title "Evaluation: Google Gemini (gemini-3-flash-preview)" --notes eval/notes/gemini.md \
      --out ../docs/evaluation/gemini.md

`rules-postfix` is the right baseline: it was measured at `e922bae`, and no commit since then changes rules extraction.
The run log goes to `eval/results/logs/hybrid-gemini.log`, next to the Qwen runs' logs.
