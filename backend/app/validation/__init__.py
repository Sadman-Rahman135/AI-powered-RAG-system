"""Validation of extracted invoice records.

- ``rules``: field, amount, date and arithmetic checks on one record, plus document-level issues;
- ``duplicates``: the same invoice in another document;
- ``cross_verify``: field-by-field comparison of two records of the same invoice;
- ``review``: the decision whether a record needs a human look.
"""
