"""Processing pipeline: ``intake`` (documents and jobs), ``processor`` (``process_job``: parse -> extract ->
validate -> index -> persist), ``progress`` (stages, throttled progress and heartbeats, history events) and
``review`` (record issues, duplicates and cross-verification, review and document status)."""
