## Reports per model

This overview puts every run side by side. Each model also has its own report, next to the same rules-only baseline,
with its own run notes and logs:

- **Qwen3-4B** (local, llama.cpp): [evaluation/qwen3-4b.md](evaluation/qwen3-4b.md), measured (frozen and post-fix runs).
- **Google Gemini** (`gemini-3-flash-preview`): [evaluation/gemini.md](evaluation/gemini.md), **not measured**. The
  free-tier key allows 20 requests per day for this model, and a full run needs about 170; two attempts were stopped
  and are not reported (their logs are in `eval/results/logs/discarded/`), and the owner then decided a Gemini
  evaluation is not needed.

## Run notes (hand-written; not produced by the generator)

- **Frozen code (primary results).** The `rules` and `hybrid-local-4b` runs used commit `7907c58` from a separate clean worktree, so no in-progress edits were measured. Before freezing, a 3-file end-to-end sanity probe was run that included 2 held-out files. It exposed one extraction bug: on GST invoices, "Date of Supply" is read as the invoice date. That bug was deliberately **not** fixed before these runs. Fixes made after them are listed in `docs/TEST_RESULTS.md` and are not reflected in these two columns.
- **Discarded attempts.** Two earlier `hybrid-local-4b` attempts were stopped and are not reported. In the first, the Docker-hosted llama.cpp container was killed for lack of memory, and 80 records fell back to rules. In the second, the Docker Desktop engine hung. The reported run used the same llama.cpp release (b11580, CUDA build) running natively on Windows, with the same model file and settings. It had 0 LLM fallbacks (see the last row of the summary).
- **Contention.** During the `rules` run (07:28–07:43 UTC), a separate OCR-benchmark process was competing for CPU on the same machine. Timings for the scanned files, including `scale/scanned_100_pages.pdf`, are therefore pessimistic. Accuracy is not affected.
- **Other models.** The 9B model (`hybrid-local-9b`) was not run, by the project owner's decision; Gemini was not measured (see *Reports per model*). The system works with any OpenAI-compatible local model, or with Claude or Gemini (see README › Use your own model). A stronger model generally improves extraction on hard layouts, but it cannot fix OCR misreads, and spreadsheet parsing never uses the LLM.
- **Where time goes on the 1000-page PDF.** Indexing (chunking plus CPU embeddings) is the slowest stage: about 4 pages/s in the frozen `rules` run and about 7 pages/s in `rules-postfix`, after the indexing speed-up in `29118a7` (the whole file took 265 s, then 160 s; the frozen run also had the CPU contention noted above). Parsing runs at roughly 250–280 pages/s and extraction at roughly 75–85 pages/s.
- **Post-fix runs (`rules-postfix`, `hybrid-local-4b-postfix`).** The same dataset, configurations and model, re-run at commit `e922bae` (all final-review fixes, including the "Date of Supply" fix) from a clean worktree. **These columns are not a blind measurement:** at least one of the fixes ("Date of Supply") was found on a held-out file, so the held-out split is no longer fully held out for this code. Read them as "the shipped code on this dataset" and the frozen columns as the fairer estimate. One change between the runs matters for the LLM column: the input budget now follows the model's context window (8192 tokens per slot, minus 4096 reserved for output and 1014 for the prompt, × 3 characters per token = 9,246 characters) instead of the fixed 20,000 characters, so every request fits the 4B server's 8192-token slot.
- **Logs.** Each reported run's log is named under *Run logs* above. The discarded attempts (two `hybrid-local-4b`, two `hybrid-gemini`) are kept in `eval/results/logs/discarded/` (`hybrid-local-4b-attempt1-docker-oom.log`, `hybrid-local-4b-attempt2-docker-hang.log`).
- **Commit IDs.** IDs such as `7907c58` name the point in the project's development history at which a run was measured (the results files record them under `run.commit`). The published repository holds that work as a single commit, so the IDs identify runs; they are not browsable commits.
