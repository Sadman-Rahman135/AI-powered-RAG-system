"""One value per field from the label candidates, with arithmetic used as evidence (never as a source).

- Text and date fields: the most specific label wins (label pattern priority), then confidence, then the first
  occurrence.
- Totals: grand total < amount due < balance due < total incl. tax < invoice total < plain total, then the one
  nearest the bottom; a zero or post-payment "balance due" never beats a printed total.
- Tax: an aggregate line ("Total Tax") if printed, else the sum of the distinct tax lines (CGST + SGST, GST + PST).
- If ``subtotal - discount + shipping + tax = total`` (within 0.02) for some combination of the candidates, that
  combination is chosen and its amounts are raised to 0.95 confidence. A missing value is never computed.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from itertools import product

from app.extraction.rules_parts import vocab
from app.extraction.rules_parts.kv import Candidate
from app.extraction.rules_parts.layout import TextLine

TOLERANCE = Decimal("0.02")
ARITHMETIC_CONFIDENCE = 0.95
_PAYMENT_LINE = re.compile(
    r"\b(?:amount\s+paid|paid(?:\s+to\s+date)?|payments?\s+(?:received|made|applied)|deposit|less\s+payments?|"
    r"prepaid|advance\s+paid)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class Pick:
    """A chosen value with the candidate(s) that support it."""

    value: object
    confidence: float
    members: list[Candidate]
    notes: list[str] = field(default_factory=list)


def _single(c: Candidate) -> Pick:
    return Pick(c.value, c.confidence, [c], list(c.notes))


def by_field(candidates: Iterable[Candidate]) -> dict[str, list[Candidate]]:
    groups: dict[str, list[Candidate]] = defaultdict(list)
    for c in candidates:
        groups[c.field].append(c)
    return groups


def pick_labelled(cands: Sequence[Candidate]) -> Pick | None:
    """Most specific label, then highest confidence, then first in reading order."""
    if not cands:
        return None
    best = min(cands, key=lambda c: (c.label.priority, -c.confidence, c.line))
    return _single(best)


# ------------------------------------------------------------------------------------------------- money


def _total_key(c: Candidate, payments: bool) -> tuple[int, int, int]:
    rank = vocab.total_rank(c.label, c.after)
    due_like = rank in (1, 2)
    demote = due_like and (c.value == 0 or payments)
    return (1 if demote else 0, rank, -c.line)


def _ordered(cands: Sequence[Candidate], name: str, payments: bool) -> list[Candidate]:
    if name == "total":
        return sorted(cands, key=lambda c: _total_key(c, payments))
    return sorted(cands, key=lambda c: (c.label.priority, -c.line))


def _distinct(cands: Sequence[Candidate], limit: int) -> list[Candidate]:
    seen: set[object] = set()
    out: list[Candidate] = []
    for c in cands:
        if c.value not in seen:
            seen.add(c.value)
            out.append(c)
        if len(out) == limit:
            break
    return out


def tax_options(cands: Sequence[Candidate]) -> list[Pick]:
    """Ways to read the tax amount, preferred first: the aggregate line, the sum of distinct tax lines, each line."""
    aggregates = sorted((c for c in cands if vocab.is_aggregate_tax(c.label)), key=lambda c: -c.line)
    components: list[Candidate] = []
    seen: set[tuple[str, object, object]] = set()
    for c in sorted((c for c in cands if not vocab.is_aggregate_tax(c.label)), key=lambda c: c.line):
        key = (vocab.tax_kind(c.label.text), c.rate, c.value)
        if key not in seen:
            seen.add(key)
            components.append(c)
    options: list[Pick] = [_single(c) for c in aggregates]
    if len(components) > 1:
        total = sum((Decimal(str(c.value)) for c in components), Decimal("0"))
        for c in components:  # a line equal to the sum of the others is their total ("VAT 30 + VAT 10 ... VAT 40")
            rest = sum((Decimal(str(o.value)) for o in components if o is not c), Decimal("0"))
            if len(components) > 2 and abs(Decimal(str(c.value)) - rest) <= TOLERANCE:
                options.append(_single(c))
        notes = sorted({n for c in components for n in c.notes})
        options.append(Pick(total, min(c.confidence for c in components), components, notes))
    options.extend(_single(c) for c in components)
    return options


def _value(pick: Pick | None) -> Decimal:
    return Decimal(str(pick.value)) if pick is not None else Decimal("0")


def _boost(pick: Pick | None) -> None:
    if pick is not None:
        pick.confidence = max(pick.confidence, ARITHMETIC_CONFIDENCE)


@dataclass(slots=True)
class MoneyPicks:
    subtotal: Pick | None = None
    discount: Pick | None = None
    shipping: Pick | None = None
    tax: Pick | None = None
    total: Pick | None = None
    items_consistent: bool = False  # the line items add up (to the subtotal, or with tax etc. to the total)

    def as_dict(self) -> dict[str, Pick]:
        picks = {"subtotal": self.subtotal, "discount": self.discount, "shipping": self.shipping, "tax": self.tax}
        picks["total"] = self.total
        return {k: v for k, v in picks.items() if v is not None}


def _balances(s: Pick, d: Pick | None, h: Pick | None, t: Pick | None, g: Pick) -> bool:
    return abs(_value(s) - abs(_value(d)) + _value(h) + _value(t) - _value(g)) <= TOLERANCE


def pick_money(
    groups: dict[str, list[Candidate]], lines: Sequence[TextLine], items_sum: Decimal | None
) -> MoneyPicks:
    payments = any(_PAYMENT_LINE.search(ln.text) for ln in lines)
    ordered = {name: _ordered(groups.get(name, []), name, payments) for name in vocab.MONEY_FIELDS if name != "tax"}
    subtotals = [_single(c) for c in _distinct(ordered["subtotal"], 3)]
    totals = [_single(c) for c in _distinct(ordered["total"], 4)]
    taxes = tax_options(groups.get("tax", []))[:6]
    discount = _single(ordered["discount"][0]) if ordered["discount"] else None
    shipping = _single(ordered["shipping"][0]) if ordered["shipping"] else None
    picks = MoneyPicks(
        subtotal=subtotals[0] if subtotals else None,
        discount=discount,
        shipping=shipping,
        tax=taxes[0] if taxes else None,
        total=totals[0] if totals else None,
    )
    tax_choices: list[Pick | None] = [*taxes, None] if taxes else [None]
    for g, s, t in product(totals, subtotals, tax_choices):
        if _balances(s, discount, shipping, t, g):
            picks.subtotal, picks.total = s, g
            if t is not None:
                picks.tax = t
            for p in (s, t, g, discount, shipping):
                _boost(p)
            break
    if items_sum is not None:
        _items_evidence(picks, items_sum, tax_choices, totals)
    return picks


def _items_evidence(picks: MoneyPicks, items_sum: Decimal, taxes: list[Pick | None], totals: list[Pick]) -> None:
    """Line items that add up to the subtotal (or, with no subtotal printed, with tax etc. to the total)."""
    if picks.subtotal is not None:
        if abs(items_sum - _value(picks.subtotal)) <= TOLERANCE:
            picks.items_consistent = True
            _boost(picks.subtotal)
        return
    base = Pick(items_sum, 0.0, [])
    for g, t in product(totals, taxes):
        if _balances(base, picks.discount, picks.shipping, t, g):
            picks.items_consistent = True
            picks.total = g
            if t is not None:
                picks.tax = t
            for p in (t, g, picks.discount, picks.shipping):
                _boost(p)
            return


def pick_tax_rate(tax: Pick | None, rate_cands: Sequence[Candidate], bases: Sequence[Decimal]) -> Pick | None:
    """The percentage printed on the tax line when exactly one tax line carries one; otherwise a rate-only line
    ("Tax rate: 20%") when there is exactly one such rate. ``rate x base = tax`` for one of the taxable ``bases``
    raises its confidence."""
    pick: Pick | None = None
    if tax is not None:
        rated = [c for c in tax.members if c.rate is not None]
        if len(rated) == 1:
            c = rated[0]
            pick = Pick(c.rate, c.confidence, [c], list(c.notes))
        elif rated:
            return None
    if pick is None:
        rates = {c.rate if c.rate is not None else c.value for c in rate_cands}
        if len(rates) != 1:
            return None
        c = rate_cands[0]
        pick = Pick(next(iter(rates)), c.confidence, [c], list(c.notes))
    if tax is not None:
        rate = Decimal(str(pick.value))
        for base in bases:
            expected = base * rate / Decimal(100)
            if abs(expected - _value(tax)) <= max(TOLERANCE, abs(expected) * Decimal("0.001")):
                _boost(pick)
                break
    return pick
