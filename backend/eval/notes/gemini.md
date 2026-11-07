## Run notes (hand-written; not produced by the generator)

- **Baseline.** `rules-postfix` was measured at `e922bae`; no commit since changes rules extraction, so it is the
  rules-only baseline for the Gemini run (the hybrid result is the rules result merged with Gemini's answer).
- **Earlier attempts (not reported).** Two `hybrid-gemini` attempts on 2026-11-05 were stopped by hand, because the
  free-tier key's quota made records fall back to rules: about 4 of the first 20 files at `5afbd28` (the provider then
  gave up after about 6 s of retries), and the daily quota of 20 requests for `gemini-3-flash-preview` at `944891e`.
  Their logs are in `eval/results/logs/discarded/`. Since `31d2252` the provider waits out a short 429 and falls back
  at once when the quota is spent, so any fallback in this run is counted in the summary's last row.
- **Data.** The dataset is synthetic, so sending it to Google raises no privacy concern; real invoices should go only
  through a key whose Google Cloud project has billing enabled (Google does not use paid-tier content to improve its
  products).
- **What the run does not show.** It covers this one model and the key's tier at the time of the run; Gemini preview
  models change without notice, so the numbers hold for the date in *Environment*.
- **Commit IDs.** IDs such as `7907c58` name the point in the project's development history at which a run was measured (the results files record them under `run.commit`). The published repository holds that work as a single commit, so the IDs identify runs; they are not browsable commits.
