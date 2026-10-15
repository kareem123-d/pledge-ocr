import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from statistics import median

from app.core.extraction_rules import (
    FIELD_RULES,
    IGNORED_LABEL_ALIASES,
    TABLE_COLUMNS,
    TABLE_END_ALIASES,
    TABLE_IGNORED_COLUMNS,
    TABLE_TOTAL_ALIASES,
    FieldRule,
)
from app.schemas.pledge_slip import OcrLine

FUZZY_MIN = 0.85
SHORT_ALIAS_LEN = 4
CONTAINS_MIN_LEN = 8
EDGE_CHARS = " :;-=.|,_"
SERIAL_RE = re.compile(r"^\s*\d{1,2}\s*[.)]?\s*$")
OF_CONTINUATION_RE = re.compile(r"(?:of|and|or|is|are|was|were|to|for|with|which|where|will|shall|has|have|if|by|as)\b", re.I)
KNOWN_VALUE_HEADER_LINES = 8


def compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


@dataclass(frozen=True)
class Geo:
    line: OcrLine
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def h(self) -> float:
        return max(1.0, self.y1 - self.y0)


@dataclass
class Candidate:
    raw: str
    lines: list[OcrLine]
    match_confidence: float
    label: OcrLine | None = None
    ocr_confidence: float | None = None
    note: str | None = None
    spans_columns: bool = False


@dataclass
class TableRow:
    description: Candidate | None
    cells: dict[str, Candidate] = field(default_factory=dict)


@dataclass
class TableResult:
    rows: list[TableRow] = field(default_factory=list)
    totals: dict[str, Candidate] = field(default_factory=dict)
    header_indices: set[int] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)


def to_geo(line: OcrLine) -> Geo | None:
    if not line.box:
        return None
    xs = [p[0] for p in line.box]
    ys = [p[1] for p in line.box]
    return Geo(line, min(xs), min(ys), max(xs), max(ys))


def overlap_ratio(a: Geo, b: Geo) -> float:
    overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
    return overlap / min(a.h, b.h) if overlap > 0 else 0.0


def match_label(text: str, aliases, reject_of: bool = False) -> tuple[float, int] | None:
    line_c = compact(text)
    best: tuple[bool, int, float, int] | None = None
    for alias in aliases:
        a = compact(alias)
        if not a:
            continue
        if len(a) <= SHORT_ALIAS_LEN:
            if not re.match(rf"^\W*{re.escape(alias)}(?![A-Za-z0-9])", text, re.I):
                continue
            hit = (True, len(a), 1.0, len(a))
        elif line_c.startswith(a):
            hit = (True, len(a), 1.0, len(a))
        else:
            score = SequenceMatcher(None, line_c[: len(a)], a).ratio()
            if score >= FUZZY_MIN and len(line_c) >= len(a) and line_c[len(a) - 2 : len(a)] == a[-2:]:
                hit = (False, len(a), score, len(a))
            else:
                idx = line_c.find(a)
                if idx <= 0 or len(a) < CONTAINS_MIN_LEN:
                    continue
                hit = (False, len(a), 0.9, idx + len(a))
        if reject_of and OF_CONTINUATION_RE.match(text_after(text, hit[3])):
            continue
        if best is None or hit[:3] > best[:3]:
            best = hit
    return (best[2], best[3]) if best else None


def text_after(text: str, alnum_count: int) -> str:
    seen = 0
    for i, ch in enumerate(text):
        if ch.isascii() and ch.isalnum():
            if seen == alnum_count:
                return text[i:].strip().strip(EDGE_CHARS).strip()
            seen += 1
    return ""

def _is_fused(text: str, alnum_count: int) -> bool:
    seen = 0
    for i, ch in enumerate(text):
        if ch.isascii() and ch.isalnum():
            if seen == alnum_count:
                prev = text[i - 1] if i > 0 else ""
                return prev.isascii() and prev.isalnum()
            seen += 1
    return False


def _label_owners(lines: list[OcrLine], exclude: set[int]) -> dict[int, tuple[str, float, int]]:
    owners: dict[int, tuple[str, float, int]] = {}
    for line in lines:
        if line.index in exclude:
            continue
        best: tuple[str, float, int] | None = None
        ignored = match_label(line.text, IGNORED_LABEL_ALIASES)
        if ignored:
            best = ("", ignored[0], ignored[1])
        for rule in FIELD_RULES:
            m = match_label(line.text, rule.aliases, reject_of=True)
            if m and (best is None or (m[0], m[1]) > (best[1], best[2])):
                best = (rule.name, m[0], m[1])
        if best:
            owners[line.index] = best
    return owners

def _right_of(ref: Geo, geos: list[Geo], flags: set[int]) -> list[Geo]:
    cands = sorted(
        (
            g for g in geos
            if g.line.page == ref.line.page
            and g.line.index != ref.line.index
            and g.x0 >= ref.x1 - ref.h
            and overlap_ratio(ref, g) >= 0.5
        ),
        key=lambda g: g.x0,
    )
    
        # Boxes stacked in the same column (skewed photo rows) are alternatives, not a
    # continuation: keep only the one that lines up best with the label.
    kept: list[Geo] = []
    for g in cands:
        clash = next(
            (k for k in kept if min(g.x1, k.x1) - max(g.x0, k.x0) > 0.5 * min(g.x1 - g.x0, k.x1 - k.x0)),
            None,
        )
        if clash is None:
            kept.append(g)
        elif overlap_ratio(ref, g) > overlap_ratio(ref, clash):
            kept[kept.index(clash)] = g
    out: list[Geo] = []
    prev = ref
    for g in kept:
        if g.line.index in flags:
            break
        if out and g.x0 - prev.x1 > 4 * max(prev.h, g.h):
            break
        out.append(g)
        prev = g
    return out


def _below(ref: Geo, geos: list[Geo], flags: set[int], limit: int) -> list[Geo]:
    cands = sorted(
        (
            g for g in geos
            if g.line.page == ref.line.page
            and g.line.index != ref.line.index
            and g.y0 >= ref.y1 - 0.3 * ref.h
            and g.x0 < ref.x1
            and g.x1 > ref.x0
        ),
        key=lambda g: g.cy,
    )
    out: list[Geo] = []
    prev = ref
    for g in cands:
        if len(out) >= limit or g.line.index in flags:
            break
        if g.y0 - prev.y1 > 1.2 * prev.h:
            break
        out.append(g)
        prev = g
    return out


def _value_for_label(
    line: OcrLine,
    score: float,
    end: int,
    rule: FieldRule,
    by_index: dict[int, OcrLine],
    geos: list[Geo],
    geo_by_index: dict[int, Geo],
    flags: set[int],
) -> Candidate | None:
    ref = geo_by_index.get(line.index)
    inline = text_after(line.text, end)
    first: Geo | None = None

    if compact(inline):
        cand = Candidate(inline, [line], 0.95 * score, label=line)
        if _is_fused(line.text, end):
            cand.note = "Value is joined to its label in the OCR text; it may be misread."
        first = ref
    elif ref is None:
        nxt = by_index.get(line.index + 1)
        if nxt is None or nxt.page != line.page or nxt.index in flags:
            return None
        return Candidate(nxt.text, [nxt], 0.6 * score, label=line)
    else:
        right = _right_of(ref, geos, flags)
        if right:
            cand = Candidate(
                " ".join(g.line.text for g in right), [g.line for g in right], 0.85 * score, label=line
            )
            first = right[0]
        else:
            below = _below(ref, geos, flags, rule.max_value_lines)
            if not below:
                return None
            return Candidate(
                ", ".join(g.line.text for g in below), [g.line for g in below], 0.65 * score, label=line
            )

    if rule.max_value_lines > 1 and first is not None:
        extra = _below(first, geos, flags, rule.max_value_lines - 1)
        if extra:
            cand.raw = ", ".join([cand.raw, *(g.line.text for g in extra)])
            cand.lines = cand.lines + [g.line for g in extra]
            cand.match_confidence *= 0.9
    return cand
    
def _embedded_candidate(rule: FieldRule, lines: list[OcrLine], owners: dict[int, tuple[str, float, int]]) -> Candidate | None:
    for line in lines:
        owner = owners.get(line.index)
        if owner is None or owner[0] == rule.name:
            continue
        line_c = compact(line.text)
        for alias in rule.aliases:
            a = compact(alias)
            if len(a) < CONTAINS_MIN_LEN:
                continue
            idx = line_c.find(a, owner[2])
            if idx < 0:
                continue
            value = text_after(line.text, idx + len(a))
            if sum(c.isalnum() for c in value) < 2:
                continue
            note = "This label shares an OCR line with another label; check the value."
            return Candidate(value, [line], 0.8, label=line, note=note)
    return None


def _pattern_value(rule: FieldRule, lines: list[OcrLine], flags: set[int]) -> Candidate | None:
    for line in lines:
        if line.index in flags:
            continue
        for pattern in rule.patterns:
            m = re.search(pattern, line.text, re.I)
            if m:
                return Candidate(m.group(1).strip(), [line], rule.pattern_confidence)
    return None
def _known_value(rule: FieldRule, lines: list[OcrLine]) -> Candidate | None:
    hits: list[tuple[OcrLine, str]] = []
    for line in lines:
        line_c = compact(line.text)
        best: tuple[int, str] | None = None
        for value in rule.known_values:
            v = compact(value)
            if v and v in line_c and (best is None or len(v) > best[0]):
                best = (len(v), value)
        if best:
            hits.append((line, best[1]))
    if not hits:
        return None
    first_page = lines[0].page
    header = {ln.index for ln in [ln for ln in lines if ln.page == first_page][:KNOWN_VALUE_HEADER_LINES]}
    line, value = hits[0]
    if len({v for _, v in hits}) > 1:
        return Candidate(value, [line], 0.6, note="Several different lender names appear on the slip.")
    if line.index not in header:
        return Candidate(value, [line], 0.6, note="Lender name was found outside the slip header.")
    return Candidate(value, [line], 0.9)

def extract_fields(lines: list[OcrLine], exclude: set[int] | None = None) -> dict[str, Candidate]:
    exclude = exclude or set()
    owners = _label_owners(lines, exclude)
    flags = set(owners) | set(exclude) | {
        line.index for line in lines
        if match_label(line.text, TABLE_TOTAL_ALIASES) or match_label(line.text, TABLE_END_ALIASES)
    }
    geo_by_index: dict[int, Geo] = {}
    for line in lines:
        g = to_geo(line)
        if g:
            geo_by_index[line.index] = g
    geos = list(geo_by_index.values())
    by_index = {line.index: line for line in lines}

    results: dict[str, Candidate] = {}
    for rule in FIELD_RULES:
        best: Candidate | None = None
        for idx, (name, score, end) in owners.items():
            if name != rule.name:
                continue
            cand = _value_for_label(by_index[idx], score, end, rule, by_index, geos, geo_by_index, flags)
            if cand and (best is None or cand.match_confidence > best.match_confidence):
                best = cand
        fallbacks = (
            _known_value(rule, lines) if rule.known_values else None,
            _pattern_value(rule, lines, flags) if rule.patterns else None,
            _embedded_candidate(rule, lines, owners),
        )
        for fallback in fallbacks:
            if fallback and (best is None or fallback.match_confidence > best.match_confidence):
                best = fallback
        if best:
            results[rule.name] = best
    return results


def _header_hit(text: str, aliases) -> bool:
    c = compact(text)
    for alias in aliases:
        a = compact(alias)
        if not a:
            continue
        if len(a) <= SHORT_ALIAS_LEN:
            if c == a:
                return True
        elif c.startswith(a) or SequenceMatcher(None, c[: len(a)], a).ratio() >= FUZZY_MIN:
            return True
    return False

def _is_header_cell(text: str, aliases) -> bool:
    return not any(c.isdigit() for c in text) and _header_hit(text, aliases)


def _cell(geos: list[Geo], spanning: set[int] | None = None) -> Candidate:
    ordered = sorted(geos, key=lambda g: g.x0)
    cand = Candidate(" ".join(g.line.text for g in ordered), [g.line for g in ordered], 0.85)
    if spanning and any(g.line.index in spanning for g in ordered):
        cand.note = "Value box spans more than one table column; values may be merged."
        cand.spans_columns = True
    return cand


def extract_table(lines: list[OcrLine]) -> TableResult:
    result = TableResult()
    geos = [g for g in (to_geo(line) for line in lines) if g]
    anchors = sorted(
        (
            g for g in geos
            if _is_header_cell(g.line.text, TABLE_COLUMNS["grossWeight"])
            or _is_header_cell(g.line.text, TABLE_COLUMNS["netWeight"])
        ),
        key=lambda g: (g.line.page, g.cy),
    )
    for anchor in anchors:
        page = anchor.line.page
        page_geos = [g for g in geos if g.line.page == page]
        h = min(anchor.h, median(g.h for g in page_geos))
        band = [g for g in page_geos if abs(g.cy - anchor.cy) <= 1.5 * h]
        header_indices: set[int] = set()
        columns: dict[str, float] = {}
        has_description = False
        for name, aliases in TABLE_COLUMNS.items():
            for g in band:
                if _is_header_cell(g.line.text, aliases):
                    header_indices.add(g.line.index)
                    if name == "description":
                        has_description = True
                    else:
                        columns.setdefault(name, g.cx)
                    break
        if len(columns) >= 2 or (columns and has_description):
            break
    else:
        return result
    for g in band:
        if g.line.index not in header_indices and _is_header_cell(g.line.text, TABLE_IGNORED_COLUMNS):
            header_indices.add(g.line.index)
            columns[f"_ignored{g.line.index}"] = g.cx
    result.header_indices = header_indices

    header_bottom = max(g.y1 for g in page_geos if anchor.y0 - h <= g.cy <= anchor.cy + 1.6 * h)
    header_geos = [g for g in page_geos if g.line.index in header_indices]
    left, right = min(g.x0 for g in header_geos) - h, max(g.x1 for g in header_geos) + h
    result.header_indices |= {
        g.line.index for g in page_geos
        if anchor.y0 <= g.cy <= header_bottom and left <= g.cx <= right
        and not any(c.isdigit() for c in g.line.text)
    }
    below_header = sorted((g for g in page_geos if g.cy > header_bottom), key=lambda g: g.cy)
    total_geo = next((g for g in below_header if _header_hit(g.line.text, TABLE_TOTAL_ALIASES)), None)
    end_geo = next((g for g in below_header if _header_hit(g.line.text, TABLE_END_ALIASES)), None)
    bottoms = [g.cy - 0.3 * g.h for g in (total_geo, end_geo) if g]
    bottom = min(bottoms) if bottoms else float("inf")

    total_row = {
        g.line.index for g in page_geos
        if total_geo is not None and g is not total_geo and overlap_ratio(g, total_geo) >= 0.4
    }
    region = [
        g for g in page_geos
        if header_bottom < g.cy < bottom
        and g.line.index not in result.header_indices
        and g.line.index not in total_row
    ]

    centers = sorted(columns.values())
    spacing = min((b - a for a, b in zip(centers, centers[1:])), default=3 * h)
    boundary = centers[0] - 0.5 * spacing

    def nearest_column(g: Geo) -> str:
        return min(columns, key=lambda n: abs(columns[n] - g.cx))

    spanning = {
        g.line.index for g in page_geos
        if sum(1 for c in columns.values() if g.x0 <= c <= g.x1) >= 2
    }
    desc_geos = sorted(
        (g for g in region if g.cx < boundary and not SERIAL_RE.match(g.line.text)),
        key=lambda g: g.cy,
    )
    rows: list[list[Geo]] = []
    for g in desc_geos:
        for row in rows:
            if any(overlap_ratio(g, r) >= 0.4 for r in row):
                row.append(g)
                break
        else:
            rows.append([g])

    row_cells: list[dict[str, list[Geo]]] = [{} for _ in rows]
    unassigned = 0
    for v in (g for g in region if g.cx >= boundary):
        best_i, best_ratio = None, 0.4
        for i, row in enumerate(rows):
            ratio = max(overlap_ratio(v, r) for r in row)
            if ratio >= best_ratio:
                best_i, best_ratio = i, ratio
        if best_i is None:
            unassigned += 1
            continue
        row_cells[best_i].setdefault(nearest_column(v), []).append(v)

    for row, cells in zip(rows, row_cells):
        kept = {col: gs for col, gs in cells.items() if not col.startswith("_")}
        if not kept:
            continue
        result.rows.append(
            TableRow(description=_cell(row), cells={col: _cell(gs, spanning) for col, gs in kept.items()})
        )
    if unassigned:
        result.warnings.append(f"{unassigned} table value(s) could not be matched to an item row.")

    if total_geo:
        result.header_indices.add(total_geo.line.index)
        totals: dict[str, list[Geo]] = {}
        for v in page_geos:
            if v is total_geo or v.cx < boundary or overlap_ratio(v, total_geo) < 0.4:
                continue
            totals.setdefault(nearest_column(v), []).append(v)
        result.totals = {col: _cell(gs, spanning) for col, gs in totals.items() if not col.startswith("_")}

    return result
GENERIC_ID_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z /.&'-]{0,40}?)\s*\b(?:no|number|num|id|code|ref)\b\.?\s*[:#\-]?\s*(.*)$",
    re.I,
)


def extract_other_identifiers(lines: list[OcrLine], exclude: set[int] | None = None) -> list[tuple[str, Candidate]]:
    exclude = exclude or set()
    owners = _label_owners(lines, exclude)
    flags = set(owners) | set(exclude)
    geo_by_index: dict[int, Geo] = {}
    for line in lines:
        g = to_geo(line)
        if g:
            geo_by_index[line.index] = g
    geos = list(geo_by_index.values())

    found: list[tuple[str, Candidate]] = []
    for line in lines:
        if line.index in flags:
            continue
        m = GENERIC_ID_RE.match(line.text)
        if not m:
            continue
        label = line.text[: m.start(2)].strip(" :#-.")
        value = m.group(2).strip().strip(EDGE_CHARS)
        if sum(c.isdigit() for c in value) >= 3:
            found.append((label, Candidate(value, [line], 0.7, label=line)))
            continue
        ref = geo_by_index.get(line.index)
        right = _right_of(ref, geos, flags) if ref else []
        text = " ".join(g.line.text for g in right)
        if sum(c.isdigit() for c in text) >= 3:
            found.append((label, Candidate(text, [g.line for g in right], 0.6, label=line)))
    return found