"""The review decision: which records a person has to look at."""

from __future__ import annotations

from collections.abc import Sequence

from app.domain.schemas import ReviewStatus, Severity, ValidationIssue

__all__ = ["review_status_for"]

_NEEDS_REVIEW = frozenset({Severity.ERROR, Severity.WARNING})


def review_status_for(issues: Sequence[ValidationIssue]) -> ReviewStatus:
    """``NEEDS_REVIEW`` if any issue is an error or a warning, else ``AUTO_APPROVED``.

    ``REVIEWED`` is never returned: only a human approving or correcting a record sets it.
    """
    if any(issue.severity in _NEEDS_REVIEW for issue in issues):
        return ReviewStatus.NEEDS_REVIEW
    return ReviewStatus.AUTO_APPROVED
