"""Merge the rules result with the (grounded) LLM result, reconciling amounts arithmetically.

Per field
    agree        the value, source ``rules+llm``, confidence ``min(0.99, max + 0.1)``.
    one missing  the other side's value and meta, unchanged.
    disagree     money: the combination of candidates (from both sides) for which
                 ``subtotal - discount + shipping + tax = total`` holds (tolerance 0.02) and whose subtotal matches the
                 line items; failing that the more confident candidates. Names: the LLM. Invoice number and dates: the
                 rules when they are trusted (confidence at least 0.8, or a labelled date capped at 0.7 only because
                 its day/month order is ambiguous). The tax rate: the rules when they read it on the tax line.
                 Anything else: the more confident side (ties: rules).
                 A disagreement is flagged ``extraction_disagreement`` and its confidence multiplied by 0.7 - but when
                 the winner is printed verbatim in the source text, not below the review threshold: one disagreement
                 is one warning, not a second ``LOW_CONFIDENCE`` for the same doubt.
    rules guess  a party name the rules only guessed (no label, confidence <= 0.65) against an LLM name is not a
                 disagreement: the LLM's value and confidence, with the ``rules_guess_overruled`` note.
Tax rate: a rate only the LLM reported is kept only when it explains the tax (rate x subtotal, after discount, with
shipping); "9 %" next to CGST 9 % + SGST 9 % is a component rate, not the invoice's, and goes to ``rejected``.
Line items: the list whose amounts add up to the chosen subtotal (without a subtotal: to total - tax - shipping +
discount). When neither does, an LLM list with an amount that is not printed inside the item table (between the
rules' table header and the totals block) loses to the rules list; without the source text, so does any LLM list
when no subtotal is printed. Otherwise the longer list.

The merge only ever picks among values the two extractors produced, so it cannot introduce a value that is not in
the document: the rules read the text directly and the LLM values arrive grounded.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from app.domain.schemas import (
    DATE_FIELDS,
    INVOICE_FIELDS,
    MONEY_FIELDS,
    DocumentType,
    ExtractionResult,
    ExtractionSource,
    ExtractionWarning,
    FieldMeta,
    InvoiceData,
    LineItem,
    MetaNote,
)
from app.extraction.grounding import SourceIndex
from app.extraction.normalize import names_match, normalize_invoice_number
from app.extraction.rules_parts.kv import AMBIGUOUS_CAP

Side = Literal["rules", "llm"]

AGREEMENT_BOOST = 0.1
MAX_AGREED_CONFIDENCE = 0.99
DISAGREEMENT_FACTOR = 0.7
RULES_TRUSTED_AT = 0.8  # rules confidence from which they win invoice-number and date disagreements
RULES_GUESS_AT_MOST = 0.65  # rules confidence of a label-free guess (the supplier fallback is 0.5-0.65)
RULES_GUESS_OVERRULED = "rules_guess_overruled"  # FieldMeta note: the LLM replaced a name the rules only guessed
NEUTRAL_CONFIDENCE = 0.5  # for a value that arrives without a FieldMeta
AGREEMENT_TOLERANCE = Decimal("0.005")  # two readings of one printed amount are equal to half a cent
DEFAULT_TOLERANCE = Decimal("0.02")  # arithmetic checks (AMOUNT_TOLERANCE)
DEFAULT_REVIEW_THRESHOLD = 0.6  # REVIEW_CONFIDENCE_THRESHOLD
TAX_RATE_RELATIVE_TOLERANCE = Decimal("0.005")  # tax rounded per line or per document (as validation allows)

_NAME_FIELDS = ("supplier_name", "customer_name")
_RULES_PREFERRED_FIELDS = ("invoice_number", *DATE_FIELDS)
_SOURCES = {"rules": ExtractionSource.RULES, "llm": ExtractionSource.LLM}
_TOTALS_BLOCK = ("subtotal", "tax", "total")  # rules evidence lines that end the item table


@dataclass(frozen=True, slots=True)
class _Candidate:
    """One side's value for a field, with the meta it came with."""

    value: Any
    meta: FieldMeta
    side: Side

    @property
    def confidence(self) -> float:
        return self.meta.confidence


@dataclass(frozen=True, slots=True)
class _Policy:
    """What the merge of one record knows besides the two results."""

    review_threshold: float
    index: SourceIndex | None  # the source text, when the caller has it (verbatim checks, the item table region)
    rules_preferred: frozenset[str]  # fields the rules win when they are trusted


def _present(name: str, value: Any) -> bool:
    if name == "line_items":
        return bool(value)
    if name == "document_type":
        return value is not DocumentType.UNKNOWN
    return value is not None


def _candidate(result: ExtractionResult, name: str, side: Side) -> _Candidate | None:
    value = getattr(result.data, name)
    if not _present(name, value):
        return None
    meta = result.meta.get(name) or FieldMeta(confidence=NEUTRAL_CONFIDENCE, source=_SOURCES[side])
    return _Candidate(value, meta, side)


def _with_note(notes: list[str], note: MetaNote | str) -> list[str]:
    """``notes`` plus ``note`` as a new list (never the list of an input)."""
    text = note.value if isinstance(note, MetaNote) else note
    return list(notes) if text in notes else [*notes, text]


def _agree(name: str, a: Any, b: Any) -> bool:
    if name == "invoice_number":
        return normalize_invoice_number(a) == normalize_invoice_number(b)
    if name in _NAME_FIELDS:
        return names_match(a, b)
    if name in MONEY_FIELDS or name == "tax_rate":
        if name == "discount":  # printed "-10.00" or "10.00": the sign convention is not a disagreement
            a, b = abs(a), abs(b)
        return abs(a - b) <= AGREEMENT_TOLERANCE
    return bool(a == b)  # dates, currency, document type


def _more_confident(rules: _Candidate, llm: _Candidate) -> _Candidate:
    return rules if rules.confidence >= llm.confidence else llm


def _agreed_meta(rules: _Candidate, llm: _Candidate) -> FieldMeta:
    """Meta of a value both sides found: the more confident side's evidence and page, boosted confidence."""
    winner = _more_confident(rules, llm)
    notes = list(winner.meta.notes)
    for note in (*rules.meta.notes, *llm.meta.notes):
        notes = _with_note(notes, note)
    return winner.meta.model_copy(
        update={
            "source": ExtractionSource.HYBRID,
            "confidence": min(MAX_AGREED_CONFIDENCE, max(rules.confidence, llm.confidence) + AGREEMENT_BOOST),
            "notes": notes,
        }
    )


def _verbatim(name: str, winner: _Candidate, index: SourceIndex | None) -> bool:
    """The winning value is printed as such in the source: its evidence occurs there (line items: every amount)."""
    if index is None:
        return False
    if name == "line_items":
        amounts = [item.amount for item in winner.value if item.amount is not None]
        return bool(amounts) and all(index.supports_money(amount) for amount in amounts)
    evidence = winner.meta.evidence
    return evidence is not None and evidence != "" and index.locate(evidence) is not None


def _flagged_meta(name: str, winner: _Candidate, policy: _Policy) -> FieldMeta:
    """Meta of the value chosen in a disagreement: flagged, confidence times 0.7 - floored at the review threshold
    (never above its own confidence) when the winner is printed verbatim, so the flag is the only warning."""
    confidence = winner.confidence * DISAGREEMENT_FACTOR
    if _verbatim(name, winner, policy.index):
        confidence = max(confidence, min(winner.confidence, policy.review_threshold))
    return winner.meta.model_copy(
        update={"confidence": confidence, "notes": _with_note(winner.meta.notes, MetaNote.EXTRACTION_DISAGREEMENT)}
    )


def _rules_trusted(name: str, rules: _Candidate) -> bool:
    if rules.confidence >= RULES_TRUSTED_AT:
        return True
    # a labelled date the rules capped only because its day/month order is ambiguous still names the right date
    ambiguous = MetaNote.AMBIGUOUS_DATE_ORDER.value in rules.meta.notes
    return name in DATE_FIELDS and ambiguous and rules.confidence >= AMBIGUOUS_CAP


def _pick_winner(
    name: str, rules: _Candidate, llm: _Candidate, money_side: str | None, policy: _Policy
) -> _Candidate:
    if name in MONEY_FIELDS:
        return rules if money_side == "rules" else llm
    if name in _NAME_FIELDS:
        return llm
    if name in policy.rules_preferred and _rules_trusted(name, rules):
        return rules
    return _more_confident(rules, llm)


def _resolve(
    name: str, rules: _Candidate | None, llm: _Candidate | None, money_side: str | None, policy: _Policy
) -> tuple[Any, FieldMeta] | None:
    """The merged value and meta of one field, or ``None`` when neither side has a value."""
    if rules is None or llm is None:
        only = rules if rules is not None else llm
        return None if only is None else (only.value, only.meta.model_copy(deep=True))
    if _agree(name, rules.value, llm.value):
        return _more_confident(rules, llm).value, _agreed_meta(rules, llm)
    if name in _NAME_FIELDS and rules.confidence <= RULES_GUESS_AT_MOST < llm.confidence:
        # the rules only guessed (no label); the LLM read a grounded name: a replacement, not a disagreement
        meta = llm.meta.model_copy(update={"notes": _with_note(llm.meta.notes, RULES_GUESS_OVERRULED)})
        return llm.value, meta
    winner = _pick_winner(name, rules, llm, money_side, policy)
    return winner.value, _flagged_meta(name, winner, policy)


def _rate_on_tax_line(rules: ExtractionResult) -> bool:
    """The rules read the tax rate on the line that states the tax amount ("VAT 20 %  40.00")."""
    rate, tax = rules.meta.get("tax_rate"), rules.meta.get("tax")
    if rules.data.tax_rate is None or rules.data.tax is None or rate is None or tax is None:
        return False
    rate_line, tax_lines = rate.evidence or "", tax.evidence or ""
    return bool(rate_line) and rate_line in tax_lines


# ------------------------------------------------------------------------------------ money reconciliation


@dataclass(frozen=True, slots=True)
class _Pick:
    """One candidate value of a money field in the combinatorial search."""

    side: str  # "rules" / "llm" for a disputed field; "both" / "none" otherwise
    value: Decimal | None
    confidence: float


def _items_total(items: list[LineItem]) -> Decimal | None:
    amounts = [item.amount for item in items if item.amount is not None]
    return sum(amounts, Decimal(0)) if amounts else None


def _sums_to(items: list[LineItem], subtotal: Decimal | None, tolerance: Decimal) -> bool:
    total = _items_total(items)
    return subtotal is not None and total is not None and abs(total - subtotal) <= tolerance


def _equation_holds(values: dict[str, Decimal | None], tolerance: Decimal) -> bool:
    subtotal, total = values["subtotal"], values["total"]
    if subtotal is None or total is None:
        return False  # not enough amounts to check anything
    zero = Decimal(0)  # a field neither side printed counts as nothing
    expected = subtotal - abs(values["discount"] or zero) + (values["shipping"] or zero) + (values["tax"] or zero)
    return abs(expected - total) <= tolerance


def _money_options(rules: ExtractionResult, llm: ExtractionResult) -> tuple[list[list[_Pick]], set[str]]:
    """Per money field the candidate values to choose from, and the names of the fields the sides dispute."""
    options: list[list[_Pick]] = []
    disputed: set[str] = set()
    for name in MONEY_FIELDS:
        from_rules, from_llm = _candidate(rules, name, "rules"), _candidate(llm, name, "llm")
        if from_rules is not None and from_llm is not None:
            if _agree(name, from_rules.value, from_llm.value):
                winner = _more_confident(from_rules, from_llm)
                options.append([_Pick("both", winner.value, max(from_rules.confidence, from_llm.confidence))])
            else:
                options.append(
                    [
                        _Pick("rules", from_rules.value, from_rules.confidence),
                        _Pick("llm", from_llm.value, from_llm.confidence),
                    ]
                )
                disputed.add(name)
        elif from_rules is not None:
            options.append([_Pick("rules", from_rules.value, from_rules.confidence)])
        elif from_llm is not None:
            options.append([_Pick("llm", from_llm.value, from_llm.confidence)])
        else:
            options.append([_Pick("none", None, 0.0)])
    return options, disputed


def _reconcile_money(rules: ExtractionResult, llm: ExtractionResult, tolerance: Decimal) -> dict[str, str]:
    """For every money field the two sides disagree about, which side supplies it.

    All combinations of candidates are scored (at most 2^5): arithmetic consistency first, then whether the subtotal
    equals the sum of a line-item list, then total confidence, then the number of values taken from the rules.
    """
    options, disputed = _money_options(rules, llm)
    if not disputed:
        return {}
    item_lists = [result.data.line_items for result in (rules, llm)]

    def score(combo: tuple[_Pick, ...]) -> tuple[bool, bool, float, int]:
        values = {name: pick.value for name, pick in zip(MONEY_FIELDS, combo, strict=True)}
        return (
            _equation_holds(values, tolerance),
            any(_sums_to(items, values["subtotal"], tolerance) for items in item_lists),
            round(sum(pick.confidence for pick in combo), 6),
            sum(pick.side == "rules" for pick in combo),
        )

    best = max(itertools.product(*options), key=score)  # ties keep the first combination: rules before llm
    return {name: pick.side for name, pick in zip(MONEY_FIELDS, best, strict=True) if name in disputed}


def _unexplained_tax_rate(values: dict[str, Any], tolerance: Decimal) -> bool:
    """The tax rate does not reproduce the tax from any plausible base (subtotal, after discount, with shipping)."""
    rate, tax, subtotal = values.get("tax_rate"), values.get("tax"), values.get("subtotal")
    if rate is None or tax is None or subtotal is None:
        return False  # nothing to check it against
    after_discount = subtotal - abs(values.get("discount") or Decimal(0))
    bases = (subtotal, after_discount, after_discount + (values.get("shipping") or Decimal(0)))
    allowed = max(tolerance, abs(tax) * TAX_RATE_RELATIVE_TOLERANCE)
    return all(abs(rate / 100 * base - tax) > allowed for base in bases)


# ----------------------------------------------------------------------------------------------- line items


def _same_amounts(a: list[LineItem], b: list[LineItem]) -> bool:
    left = sorted(item.amount for item in a if item.amount is not None)
    right = sorted(item.amount for item in b if item.amount is not None)
    if not left or len(left) != len(right):
        return False
    return all(abs(x - y) <= AGREEMENT_TOLERANCE for x, y in zip(left, right, strict=True))


def _richness(items: list[LineItem]) -> int:
    return sum(cell is not None for item in items for cell in (item.description, item.quantity, item.unit_price))


def _copied(items: list[LineItem]) -> list[LineItem]:
    return [item.model_copy() for item in items]


def _items_target(values: dict[str, Any]) -> Decimal | None:
    """What the items should add up to: the subtotal, else total - tax - shipping + discount (evidence only)."""
    if values.get("subtotal") is not None:
        return values["subtotal"]
    if values.get("total") is None:
        return None
    zero = Decimal(0)
    return values["total"] - (values.get("tax") or zero) - (values.get("shipping") or zero) + abs(
        values.get("discount") or zero
    )


def _item_region(rules: ExtractionResult, index: SourceIndex) -> SourceIndex:
    """The source text of the item table: from the rules' table header to the first totals line after it."""
    text = index.text
    header = rules.meta.get("line_items")
    start = (index.locate(header.evidence) if header is not None and header.evidence else None) or 0
    ends = [
        position
        for name in _TOTALS_BLOCK
        if (meta := rules.meta.get(name)) is not None
        and meta.evidence
        and (position := index.locate(meta.evidence)) is not None
        and position > start
    ]
    return SourceIndex(text[start : min(ends, default=len(text))], default_dollar=index.default_dollar)


def _printed_in_table(items: list[LineItem], region: SourceIndex) -> bool:
    return all(region.supports_money(item.amount) for item in items if item.amount is not None)


def _resolve_line_items(
    rules: _Candidate | None,
    llm: _Candidate | None,
    values: dict[str, Any],
    rules_result: ExtractionResult,
    tolerance: Decimal,
    policy: _Policy,
) -> tuple[list[LineItem], FieldMeta] | None:
    if rules is None or llm is None:
        only = rules if rules is not None else llm
        return None if only is None else (_copied(only.value), only.meta.model_copy(deep=True))
    if _same_amounts(rules.value, llm.value):
        richest = max((rules, llm), key=lambda c: (_richness(c.value), c.confidence, c.side == "rules"))
        return _copied(richest.value), _agreed_meta(rules, llm)
    target = _items_target(values)
    matching = [c for c in (rules, llm) if _sums_to(c.value, target, tolerance)]
    if matching:
        winner = max(matching, key=lambda c: (len(c.value), c.confidence, c.side == "rules"))
    elif policy.index is not None and not _printed_in_table(llm.value, _item_region(rules_result, policy.index)):
        winner = rules  # an LLM row whose amount is only printed outside the item table
    elif policy.index is None and values.get("subtotal") is None:
        winner = rules  # nothing vouches for the LLM's extra or different rows
    else:
        winner = max((rules, llm), key=lambda c: (len(c.value), c.confidence, c.side == "rules"))
    return _copied(winner.value), _flagged_meta("line_items", winner, policy)


# ------------------------------------------------------------------------------------------------- the merge


def _rejection_meta(name: str, rules: ExtractionResult, llm: ExtractionResult) -> FieldMeta | None:
    """The meta of a field nobody could supply, if one side rejected a value for it (so validation can report it)."""
    for result in (llm, rules):
        meta = result.meta.get(name)
        if meta is not None and MetaNote.UNGROUNDED_VALUE_REJECTED.value in meta.notes:
            return meta.model_copy(deep=True)
    return None


def _merge_rejected(rules: dict[str, Any], llm: dict[str, Any]) -> dict[str, Any]:
    rejected = copy.deepcopy(rules)
    for key, value in copy.deepcopy(llm).items():
        if isinstance(rejected.get(key), list) and isinstance(value, list):
            rejected[key] = [*rejected[key], *value]
        else:
            rejected[key] = value
    return rejected


def merge_results(
    rules: ExtractionResult | None,
    llm: ExtractionResult | None,
    *,
    tolerance: Decimal = DEFAULT_TOLERANCE,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    source: str | SourceIndex | None = None,
    default_dollar: str = "USD",
) -> ExtractionResult:
    """Combine the rules and LLM results of one document into a ``hybrid`` result.

    A missing side returns a copy of the other unchanged (its method stays ``rules`` / ``llm``). Inputs are never
    mutated. ``tolerance`` is the absolute amount tolerance of the arithmetic checks (``AMOUNT_TOLERANCE``),
    ``review_threshold`` the ``REVIEW_CONFIDENCE_THRESHOLD``. ``source`` - the text both sides read - enables the
    verbatim checks (disagreement floor) and the item-table check; without it neither applies.
    """
    if rules is None or llm is None:
        only = rules if rules is not None else llm
        return ExtractionResult(method="hybrid") if only is None else only.model_copy(deep=True)

    index = SourceIndex(source, default_dollar=default_dollar) if isinstance(source, str) else source
    preferred = {*_RULES_PREFERRED_FIELDS, *(("tax_rate",) if _rate_on_tax_line(rules) else ())}
    policy = _Policy(review_threshold, index, frozenset(preferred))
    money_sides = _reconcile_money(rules, llm, tolerance)
    values: dict[str, Any] = {}
    meta: dict[str, FieldMeta] = {}
    for name in INVOICE_FIELDS:
        if name == "line_items":
            continue  # judged against the reconciled subtotal below
        resolved = _resolve(
            name, _candidate(rules, name, "rules"), _candidate(llm, name, "llm"), money_sides.get(name), policy
        )
        if resolved is not None:
            values[name], meta[name] = resolved
        elif (rejection := _rejection_meta(name, rules, llm)) is not None:
            meta[name] = rejection

    rejected = _merge_rejected(rules.rejected, llm.rejected)
    if _candidate(rules, "tax_rate", "rules") is None and _unexplained_tax_rate(values, tolerance):
        rejected["tax_rate"] = values.pop("tax_rate")  # only the LLM read it, and it is not the invoice's rate
        meta.pop("tax_rate", None)

    items = _resolve_line_items(
        _candidate(rules, "line_items", "rules"), _candidate(llm, "line_items", "llm"), values, rules, tolerance, policy
    )
    if items is not None:
        values["line_items"], meta["line_items"] = items
    elif (rejection := _rejection_meta("line_items", rules, llm)) is not None:
        meta["line_items"] = rejection

    warnings = list(dict.fromkeys([*rules.warnings, *llm.warnings]))
    if values.get("line_items"):
        warnings = [w for w in warnings if w != ExtractionWarning.NO_LINE_ITEMS_TABLE.value]

    merged = ExtractionResult(
        data=InvoiceData(**values),
        meta=meta,
        method="hybrid",
        warnings=warnings,
        rejected=rejected,
        llm_provider=llm.llm_provider,
        llm_model=llm.llm_model,
        prompt_version=llm.prompt_version,
    )
    merged.compute_overall_confidence()
    return merged
