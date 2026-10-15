import re
from dataclasses import dataclass
from datetime import date
from difflib import get_close_matches

from app.core.extraction_rules import (
    FIELD_RULES,
    ITEM_FIELD_KINDS,
    LENDER_FIELD_FALLBACKS,
    LOAN_PER_GRAM_RANGE,
    MIN_NET_TO_GROSS_RATIO,
    VALUE_MINIMUMS,
)

from app.schemas.pledge_slip import Evidence, ExtractedData, ExtractedField, FieldStatus, IdentifierField, OrnamentItem
from app.services.field_extractor import Candidate, TableResult

EDGE_CHARS = " :;-=.|,_"
FIX_TABLE = {"O": "0", "o": "0", "D": "0", "Q": "0", "l": "1", "I": "1"}
DATE_SEP_RE = re.compile(r"^(\d{1,2})\s*[./\-|\\\s]\s*(\d{1,2})\s*[./\-|\\\s]\s*(\d{2,4})$")
TENURE_RE = re.compile(r"(\d{1,3})\s*[.\-]?\s*(months?|mths?|mon|days?|years?|yrs?)\b", re.I)
MONTHLY_RE = re.compile(r"\bp\s*\.?\s*m\b\.?|\bper\s*month\b|/\s*(?:month|mon|m)\b|\bmonthly\b", re.I)
ANNUAL_RE = re.compile(r"\bp\s*\.?\s*a\b\.?|\bper\s*annum\b|/\s*(?:annum|year|yr)\b|\bannual(?:ly)?\b|\byearly\b", re.I)
PERCENT_SIGN_RE = re.compile(r"%|\bper\s*cent\b|\bpct\b", re.I)
AMOUNT_RE = re.compile(
    r"(?P<int>\d+|\d{1,3}(?:[.,]\d{2})*[.,]\d{3}|\d{1,3}(?:[.,]\d{3})+)(?:\.(?P<dec>\d{1,2}))?"
)


@dataclass(frozen=True)
class Thresholds:
    min_ocr_confidence: float = 0.90
    min_match_confidence: float = 0.75


@dataclass
class Parsed:
    value: str | int | float | None = None
    unit: str | None = None
    substituted: bool = False
    error: str | None = None
    note: str | None = None


def _fix_digits(text: str, pipes: bool = True) -> tuple[str, bool]:
    table = dict(FIX_TABLE)
    if pipes:
        table["|"] = "1"
    fixed = "".join(table.get(c, c) for c in text)
    return fixed, fixed != text


def _parse_text(raw: str) -> Parsed:
    s = " ".join(raw.split()).strip(EDGE_CHARS)
    if sum(c.isalpha() for c in s) < 2:
        return Parsed(error="No readable text.")
    if re.search(r"[A-Za-z]{3,}[._][A-Za-z]{3,}", s):
        return Parsed(value=s, note="Punctuation inside a word; two words may have been joined or misread.")
    return Parsed(value=s)


def _parse_id(raw: str) -> Parsed:
    s = re.sub(r"[^A-Za-z0-9/-]", "", raw).strip("/-")
    if len(re.sub(r"[^A-Za-z0-9]", "", s)) < 4:
        return Parsed(error="Identifier is too short.")
    if sum(c.isdigit() for c in s) < 2:
        return Parsed(error="Identifier must contain digits.")
    return Parsed(value=s)


def _parse_amount_number(raw: str) -> Parsed:
    s = re.sub(r"(?i)\brs\b\.?|\binr\b|₹|/-", " ", raw)
    s, sub = _fix_digits(s)
    s = re.sub(r"\s", "", s).rstrip(".")
    m = AMOUNT_RE.fullmatch(s)
    if not m:
        return Parsed(error="Not a valid amount.", substituted=sub)
    digits = re.sub(r"[.,]", "", m.group("int"))
    if len(digits) > 1 and digits.startswith("0"):
        return Parsed(error="Amount starts with 0; a digit may be missing.", substituted=sub)
    if "." in m.group("int"):
        sub = True
    v = float(f"{digits}.{m.group('dec')}") if m.group("dec") else float(digits)
    if v <= 0:
        return Parsed(error="Amount must be greater than zero.", substituted=sub)
    if v > 1e10:
        return Parsed(error="Amount is implausibly large.", substituted=sub)
    return Parsed(value=int(v) if v.is_integer() else v, unit="INR", substituted=sub)


NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
SCALE_WORDS = {
    "thousand": 1000, "lakh": 100000, "lakhs": 100000, "lac": 100000, "lacs": 100000,
    "crore": 10000000, "crores": 10000000,
}
WORD_VOCAB = [*NUMBER_WORDS, *SCALE_WORDS, "hundred"]
AMOUNT_TAIL_RE = re.compile(r"\(|\brupees\b|\bonly\b", re.I)
WEIGHT_HEAD_RE = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*(?:grams?|gms?|gm|g)\b[\s.)]*(.*)$", re.I)


def _words_to_number(text: str) -> int | None:
    total = current = 0
    seen = False
    for token in re.findall(r"[a-z]+", text.lower()):
        word = token if token in WORD_VOCAB else None
        if word is None and len(token) >= 5:
            close = get_close_matches(token, WORD_VOCAB, n=1, cutoff=0.82)
            word = close[0] if close else None
        if word is None:
            continue
        seen = True
        if word in NUMBER_WORDS:
            current += NUMBER_WORDS[word]
        elif word == "hundred":
            current = max(current, 1) * 100
        else:
            total += max(current, 1) * SCALE_WORDS[word]
            current = 0
    return total + current if seen else None


def _parse_amount(raw: str) -> Parsed:
    m = AMOUNT_TAIL_RE.search(raw)
    if not m or m.start() == 0 or not any(c.isdigit() for c in raw[: m.start()]):
        return _parse_amount_number(raw)
    parsed = _parse_amount_number(raw[: m.start()])
    if parsed.error:
        return parsed
    words = _words_to_number(raw[m.start():])
    if words is None:
        parsed.note = "Amount in words could not be read."
    elif words != parsed.value:
        parsed.note = f"Amount in digits does not match the amount in words ({words:,})."
    return parsed


def _parse_weight(raw: str) -> Parsed:
    m = WEIGHT_HEAD_RE.match(raw)
    if m and m.group(2).strip():
        v = float(m.group(1).replace(",", "."))
        if not 0 < v <= 5000:
            return Parsed(error="Weight is outside the plausible range.")
        return Parsed(value=round(v, 3), unit="g", note="Extra text after the weight was ignored.")
    s = re.sub(r"(?i)\s*(?:grams?|gms?|gm|g)\.?\s*\)?\s*$", "", raw.strip())
    s, sub = _fix_digits(s)
    s = s.replace(" ", "").replace(",", ".").rstrip(".")
    if not re.fullmatch(r"\d+(?:\.\d+)?", s):
        return Parsed(error="Not a valid weight.", substituted=sub)
    v = float(s)
    if not 0 < v <= 5000:
        return Parsed(error="Weight is outside the plausible range.", substituted=sub)
    return Parsed(value=round(v, 3), unit="g", substituted=sub)


def _parse_quantity(raw: str) -> Parsed:
    s, sub = _fix_digits(raw.strip().rstrip("."))
    s = s.replace(" ", "")
    if not re.fullmatch(r"\d{1,3}", s) or int(s) < 1:
        return Parsed(error="Not a valid quantity.", substituted=sub)
    return Parsed(value=int(s), substituted=sub)


def _parse_percent(raw: str) -> Parsed:
    monthly = MONTHLY_RE.search(raw) is not None
    annual = ANNUAL_RE.search(raw) is not None
    has_sign = PERCENT_SIGN_RE.search(raw) is not None
    s = PERCENT_SIGN_RE.sub(" ", ANNUAL_RE.sub(" ", MONTHLY_RE.sub(" ", raw)))
    s, sub = _fix_digits(s)
    s = s.replace(" ", "").rstrip(".")
    if not re.fullmatch(r"\d{1,2}(?:\.\d{1,2})?", s):
        return Parsed(error="Not a valid percentage.", substituted=sub)
    v = float(s)
    if not 0 < v <= 60:
        return Parsed(error="Interest rate is outside the plausible range.", substituted=sub)
    if monthly:
        return Parsed(value=v, unit="% p.m.", substituted=sub, note="Rate is per month, not per annum.")
    note = None if has_sign or annual else "No % sign or rate period next to the value."
    return Parsed(value=v, unit="%", substituted=sub, note=note)


def _parse_date(raw: str) -> Parsed:
    s, sub = _fix_digits(raw.strip(), pipes=False)
    s = s.strip(" .:,")
    m = DATE_SEP_RE.match(s)
    if m:
        d, mo, y = m.groups()
    else:
        digits = re.sub(r"\D", "", s)
        if len(digits) in (6, 8):
            d, mo, y = digits[:2], digits[2:4], digits[4:]
            sub = True
        else:
            return Parsed(error="Unrecognized date format.", substituted=sub)
    year = int(y)
    if year < 100:
        year += 2000
    try:
        value = date(year, int(mo), int(d))
    except ValueError:
        return Parsed(error="Not a valid calendar date.", substituted=sub)
    if not 1990 <= year <= 2100:
        return Parsed(error="Year is outside the plausible range.", substituted=sub)
    return Parsed(value=value.isoformat(), substituted=sub)


def _parse_phone(raw: str) -> Parsed:
    s, sub = _fix_digits(raw)
    digits = re.sub(r"\D", "", s)
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    if not re.fullmatch(r"[6-9]\d{9}", digits):
        return Parsed(error="Not a valid 10-digit mobile number.", substituted=sub)
    return Parsed(value=digits, substituted=sub)


def _parse_tenure(raw: str) -> Parsed:
    m = TENURE_RE.search(raw)
    if not m:
        return Parsed(error="Tenure must include a number and a unit (days, months or years).")
    unit_raw = m.group(2).lower()
    if unit_raw.startswith("day"):
        unit = "days"
    elif unit_raw.startswith(("year", "yr")):
        unit = "years"
    else:
        unit = "months"
    return Parsed(value=int(m.group(1)), unit=unit)


def _parse_purity(raw: str) -> Parsed:
    m = re.search(r"(\d{2})\s*(?:kt|k|ct|carat|karat)\b", raw, re.I)
    if m and 8 <= int(m.group(1)) <= 24:
        return Parsed(value=int(m.group(1)), unit="K")
    m = re.search(r"\b(\d{3})\b", raw)
    if m and 375 <= int(m.group(1)) <= 999:
        return Parsed(value=int(m.group(1)), unit="fineness")
    m = re.fullmatch(r"\s*(14|16|18|20|21|22|23|24)\s*", raw)
    if m:
        return Parsed(value=int(m.group(1)), unit="K", note="Carat unit was not written; assumed K.")
    return Parsed(error="Not a recognizable purity.")


PARSERS = {
    "text": _parse_text,
    "id": _parse_id,
    "amount": _parse_amount,
    "weight": _parse_weight,
    "quantity": _parse_quantity,
    "percent": _parse_percent,
    "date": _parse_date,
    "phone": _parse_phone,
    "tenure": _parse_tenure,
    "purity": _parse_purity,
}


def parse_value(kind: str, raw: str) -> Parsed:
    return PARSERS[kind](raw or "")

def build_field(kind: str, cand: Candidate | None, th: Thresholds) -> ExtractedField:
    if cand is None:
        return ExtractedField()
    parsed = parse_value(kind, cand.raw)
    confidences = [line.confidence for line in cand.lines if line.confidence is not None]
    ocr_conf = round(min(confidences), 4) if confidences else None
    if cand.ocr_confidence is not None:
        ocr_conf = round(cand.ocr_confidence, 4)
    match_conf = round(min(1.0, cand.match_confidence * (0.9 if parsed.substituted else 1.0)), 3)

    evidence: list[Evidence] = []
    seen: set[int] = set()
    for line in ([cand.label] if cand.label else []) + cand.lines:
        if line.index not in seen:
            seen.add(line.index)
            evidence.append(Evidence(page=line.page, lineIndex=line.index, text=line.text))

    if parsed.error:
        note = f"{parsed.error} {cand.note}" if cand.note else parsed.error
        return ExtractedField(
            value=None, rawText=cand.raw, status=FieldStatus.INVALID,
            ocrConfidence=ocr_conf, matchConfidence=match_conf, note=note, evidence=evidence,
        )

    reasons = [cand.note] if cand.note else []
    if parsed.note:
        reasons.append(parsed.note)
    if ocr_conf is not None and ocr_conf < th.min_ocr_confidence:
        reasons.append("Low OCR confidence.")
    if match_conf < th.min_match_confidence:
        reasons.append("Weak label match.")
    if parsed.substituted:
        reasons.append("OCR characters were corrected or the format was inferred.")
    return ExtractedField(
        value=parsed.value,
        rawText=cand.raw,
        unit=parsed.unit,
        status=FieldStatus.UNCERTAIN if reasons else FieldStatus.EXTRACTED,
        ocrConfidence=ocr_conf,
        matchConfidence=match_conf,
        note=" ".join(reasons) or None,
        evidence=evidence,
    )

def _num(f: ExtractedField) -> float | None:
    if f.status in (FieldStatus.EXTRACTED, FieldStatus.UNCERTAIN) and isinstance(f.value, (int, float)):
        return float(f.value)
    return None


def _flag(f: ExtractedField, note: str) -> None:
    f.status = FieldStatus.INVALID
    f.note = f"{f.note} {note}" if f.note else note


def _downgrade(f: ExtractedField, note: str) -> None:
    if f.status not in (FieldStatus.EXTRACTED, FieldStatus.UNCERTAIN):
        return
    f.status = FieldStatus.UNCERTAIN
    f.note = f"{f.note} {note}" if f.note else note


def cross_check(data: ExtractedData) -> list[str]:
    warnings: list[str] = []

    ratio_pct = int(MIN_NET_TO_GROSS_RATIO * 100)
    for i, item in enumerate(data.ornamentDetails):
        g, n = _num(item.grossWeight), _num(item.netWeight)
        if g is not None and n is not None and n > g:
            _flag(item.netWeight, "Net weight is greater than gross weight.")
            warnings.append(f"Item {i + 1}: net weight is greater than gross weight.")
        elif g and n is not None and n / g < MIN_NET_TO_GROSS_RATIO:
            note = f"Net weight is under {ratio_pct}% of gross weight; a decimal point may have been missed."
            _downgrade(item.grossWeight, note)
            _downgrade(item.netWeight, note)
            warnings.append(f"Item {i + 1}: net weight is under {ratio_pct}% of gross weight; one of them is probably misread.")
    g, n = _num(data.grossWeight), _num(data.netWeight)
    if g is not None and n is not None and n > g:
        _flag(data.netWeight, "Net weight is greater than gross weight.")
        warnings.append("Total net weight is greater than total gross weight.")

    for key, label in (("grossWeight", "gross weight"), ("netWeight", "net weight")):
        total = _num(getattr(data, key))
        items = [getattr(it, key) for it in data.ornamentDetails]
        if total is None or not items:
            continue
        values = [_num(f) for f in items]
        if any(v is None for v in values):
            warnings.append(f"Some item {label} values are missing or unreadable, so the total {label} could not be cross-checked.")
            continue
        s = round(sum(values), 3)
        if abs(s - total) > max(0.5, 0.01 * total):
            warnings.append(f"Item {label}s add up to {s:g} g, but the total shows {total:g} g.")

    ld, dd = data.loanDate, data.dueDate
    if (
        ld.status in (FieldStatus.EXTRACTED, FieldStatus.UNCERTAIN)
        and dd.status in (FieldStatus.EXTRACTED, FieldStatus.UNCERTAIN)
        and isinstance(ld.value, str)
        and isinstance(dd.value, str)
        and dd.value < ld.value
    ):
        warnings.append("Due date is earlier than the loan date.")

    for key, minimum in VALUE_MINIMUMS.items():
        f = getattr(data, key)
        v = _num(f)
        if v is not None and v < minimum:
            _flag(f, f"Amount is below the plausible minimum of {minimum:g}.")
            warnings.append(f"{key} is below the plausible minimum of {minimum:g}.")

    amount, net = _num(data.loanAmount), _num(data.netWeight)
    if net is None and data.ornamentDetails:
        nets = [_num(it.netWeight) for it in data.ornamentDetails]
        if all(n is not None for n in nets):
            net = sum(nets)
    if amount is not None and net:
        low, high = LOAN_PER_GRAM_RANGE
        per_gram = amount / net
        if not low <= per_gram <= high:
            note = f"Loan amount is {per_gram:,.0f} per gram of net weight, outside the plausible {low:g}-{high:g} range."
            _downgrade(data.loanAmount, note)
            _downgrade(data.netWeight, note)
            warnings.append(f"Loan amount per gram of net weight ({per_gram:,.0f}) is outside the plausible range.")

    return warnings
def _derive_tenure(data: ExtractedData) -> ExtractedField | None:
    ld, dd = data.loanDate, data.dueDate
    usable = (FieldStatus.EXTRACTED, FieldStatus.UNCERTAIN)
    if ld.status not in usable or dd.status not in usable:
        return None
    if not isinstance(ld.value, str) or not isinstance(dd.value, str):
        return None
    start, end = date.fromisoformat(ld.value), date.fromisoformat(dd.value)
    if end <= start:
        return None
    months = (end.year - start.year) * 12 + end.month - start.month
    if end.day + 3 < start.day:
        months -= 1
    value, unit = (months, "months") if months >= 1 else ((end - start).days, "days")
    return ExtractedField(
        value=value,
        unit=unit,
        status=FieldStatus.UNCERTAIN,
        note="Derived from the loan date and due date; not read from the slip.",
        evidence=[*ld.evidence, *dd.evidence],
    )
def _apply_lender_fallbacks(data: ExtractedData) -> None:
    bank = data.bankName.value if isinstance(data.bankName.value, str) else None
    for target, source in LENDER_FIELD_FALLBACKS.get(bank, {}).items():
        t, s = getattr(data, target), getattr(data, source)
        if t.status == FieldStatus.MISSING and s.status != FieldStatus.MISSING:
            copied = s.model_copy(deep=True)
            mapped = f"Mapped from {source} for {bank}."
            copied.note = f"{copied.note} {mapped}" if copied.note else mapped
            setattr(data, target, copied)

def build_extracted_data(
    fields: dict[str, Candidate],
    table: TableResult,
    th: Thresholds,
    others: list[tuple[str, Candidate]] | None = None,
) -> tuple[ExtractedData, list[str]]:
    data = ExtractedData()
    for rule in FIELD_RULES:
        setattr(data, rule.name, build_field(rule.kind, fields.get(rule.name), th))

    _apply_lender_fallbacks(data)

    if data.tenure.status == FieldStatus.MISSING:
        derived = _derive_tenure(data)
        if derived:
            data.tenure = derived

    for key in ("grossWeight", "netWeight"):
        if getattr(data, key).status == FieldStatus.MISSING and key in table.totals:
            setattr(data, key, build_field("weight", table.totals[key], th))

    for row in table.rows:
        item = OrnamentItem(description=build_field("text", row.description, th))
        for key, cand in row.cells.items():
            if key in ITEM_FIELD_KINDS:
                setattr(item, key, build_field(ITEM_FIELD_KINDS[key], cand, th))
        data.ornamentDetails.append(item)

    for label, cand in others or []:
        built = build_field("id", cand, th)
        data.otherIdentifiers.append(IdentifierField(label=label, **built.model_dump()))

    return data, [*table.warnings, *cross_check(data)]  