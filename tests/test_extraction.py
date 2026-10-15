import pytest

from app.schemas.pledge_slip import FieldStatus, OcrLine
from app.services.field_extractor import extract_fields, extract_other_identifiers, extract_table
from app.services.field_validator import Thresholds, build_extracted_data, parse_value
from app.services.review_service import build_review

TH = Thresholds()
MANDATORY = ["customerName", "loanAccountNumber", "loanDate", "loanAmount", "grossWeight", "netWeight"]

class Page:
    def __init__(self):
        self.lines: list[OcrLine] = []

    def add(self, text, x0, y0, x1, conf=0.98):
        y1 = y0 + 20
        self.lines.append(
            OcrLine(
                page=1, index=len(self.lines), text=text, confidence=conf,
                box=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], source="ocr",
            )
        )


def run(lines):
    table = extract_table(lines)
    fields = extract_fields(lines, exclude=table.header_indices)
    others = extract_other_identifiers(lines, exclude=table.header_indices)
    return build_extracted_data(fields, table, TH, others)


def slip(amount="14,00,000", loan_date="31/12/2025", amount_conf=0.98, necklace_nw="70", total_gw="137.2"):
    p = Page()
    p.add("Canara Bank", 360, 100, 650)
    p.add("Green City Branch, Visakhapatnam. Ph: 0891-2741899", 80, 170, 600)
    p.add("Name of the Borrower N. Sankunthala", 30, 250, 450)
    p.add("Customer ID", 30, 280, 150)
    p.add("336303209", 250, 278, 430)
    p.add("Loan Account No.", 30, 305, 195)
    p.add("1661 1681 9189", 320, 303, 530)
    p.add("Amount of Loan", 30, 330, 180)
    p.add(amount, 245, 328, 370, conf=amount_conf)
    p.add("Date of Loan", 30, 355, 150)
    p.add(loan_date, 240, 353, 400)
    p.add("Date of Close: 30/06/2026", 405, 352, 660)
    p.add("Sl.", 35, 380, 70)
    p.add("Description of Gold Items", 150, 380, 355)
    p.add("Qty.", 435, 382, 475)
    p.add("GW", 505, 378, 550)
    p.add("NW", 600, 378, 640)
    p.add("1.", 35, 420, 60)
    p.add("Haram", 95, 420, 155)
    p.add("2.", 35, 445, 60)
    p.add("Necklace", 95, 445, 175)
    p.add("3", 435, 443, 460)
    p.add("90", 500, 443, 540)
    p.add(necklace_nw, 590, 443, 635)
    p.add("3.", 35, 470, 60)
    p.add("Bangles", 95, 470, 165)
    p.add("4", 435, 468, 460)
    p.add("42", 505, 468, 540)
    p.add("41", 595, 468, 630)
    p.add("16", 55, 495, 85)
    p.add("Others", 115, 495, 170)
    p.add("Lockets", 225, 495, 320)
    p.add("3", 425, 493, 450)
    p.add("5.2", 500, 493, 545)
    p.add("4", 600, 493, 620)
    p.add("Total Weight", 280, 520, 390)
    p.add("10", 433, 518, 463)
    p.add(total_gw, 495, 518, 550)
    p.add("115", 585, 518, 650)
    p.add("Signature of the Borrower", 70, 560, 260)
    return p.lines


def test_header_fields():
    data, warnings = run(slip())
    assert data.customerName.value == "N. Sankunthala"
    assert data.customerName.status == FieldStatus.EXTRACTED
    assert data.customerId.value == "336303209"
    assert data.loanAccountNumber.value == "166116819189"
    assert data.pledgeNumber.status == FieldStatus.MISSING
    assert data.bankName.value == "Canara Bank"
    assert data.tenure.value == 6 and data.tenure.unit == "months"
    assert data.tenure.status == FieldStatus.UNCERTAIN
    assert data.loanAmount.value == 1400000 and data.loanAmount.unit == "INR"
    assert data.loanDate.value == "2025-12-31"
    assert data.dueDate.value == "2026-06-30"
    assert data.branchName.value == "Green City"
    assert data.phoneNumber.status == FieldStatus.MISSING
    assert warnings == []


def test_ornament_table_and_totals():
    data, _ = run(slip())
    items = data.ornamentDetails
    assert [i.description.value for i in items] == ["Necklace", "Bangles", "Others Lockets"]
    assert [i.quantity.value for i in items] == [3, 4, 3]
    assert [i.grossWeight.value for i in items] == [90, 42, 5.2]
    assert [i.netWeight.value for i in items] == [70, 41, 4]
    assert data.grossWeight.value == 137.2 and data.grossWeight.unit == "g"
    assert data.netWeight.value == 115


def test_evidence_points_to_ocr_lines():
    lines = slip()
    data, _ = run(lines)
    evidence = data.customerId.evidence
    assert [e.text for e in evidence] == ["Customer ID", "336303209"]
    assert all(lines[e.lineIndex].text == e.text for e in evidence)


def test_unreadable_values_are_invalid_not_guessed():
    data, _ = run(slip(amount="00000", loan_date="312202"))
    assert data.loanAmount.status == FieldStatus.INVALID
    assert data.loanAmount.value is None
    assert data.loanAmount.rawText == "00000"
    assert data.loanDate.status == FieldStatus.INVALID
    assert data.loanDate.value is None


def test_low_ocr_confidence_is_uncertain():
    data, _ = run(slip(amount_conf=0.6))
    assert data.loanAmount.value == 1400000
    assert data.loanAmount.status == FieldStatus.UNCERTAIN


def test_net_greater_than_gross_is_flagged():
    data, warnings = run(slip(necklace_nw="95"))
    assert data.ornamentDetails[0].netWeight.status == FieldStatus.INVALID
    assert any("net weight is greater than gross weight" in w for w in warnings)


def test_total_mismatch_is_warned():
    _, warnings = run(slip(total_gw="150"))
    assert any("add up to 137.2 g" in w for w in warnings)


def test_text_lines_without_boxes():
    texts = ["Customer Name: Ravi Kumar", "Loan Amount", "50,000", "Mobile No: +91 98765 43210"]
    lines = [OcrLine(page=1, index=i, text=t, source="pdf_text") for i, t in enumerate(texts)]
    data, _ = run(lines)
    assert data.customerName.value == "Ravi Kumar"
    assert data.customerId.status == FieldStatus.MISSING
    assert data.loanAmount.value == 50000
    assert data.loanAmount.status == FieldStatus.UNCERTAIN
    assert data.phoneNumber.value == "9876543210"


def test_lone_branch_word_is_not_a_branch_name():
    p = Page()
    p.add("Branch", 557, 936, 611)
    p.add("Signature of the Borrower", 265, 974, 475)
    data, _ = run(p.lines)
    assert data.branchName.status == FieldStatus.MISSING


def test_review_summary():
    data, warnings = run(slip(amount="00000"))
    review = build_review(data, MANDATORY, warnings)
    assert review.required is True
    assert "loanAmount" in review.invalidFields
    assert "phoneNumber" in review.missingFields
    assert review.missingMandatoryFields == []


@pytest.mark.parametrize(
    "kind,raw,value,unit",
    [
        ("amount", "Rs. 1,40,000/-", 140000, "INR"),
        ("date", "31/12/2025", "2025-12-31", None),
        ("date", "31-12-25", "2025-12-31", None),
        ("date", "31|12|2025", "2025-12-31", None),
        ("weight", "15.3 g", 15.3, "g"),
        ("weight", "198.", 198, "g"),
        ("weight", "15,3", 15.3, "g"),
        ("quantity", "3", 3, None),
        ("phone", "98765 43210", "9876543210", None),
        ("percent", "9.5 % p.a.", 9.5, "%"),
        ("tenure", "6 Months", 6, "months"),
        ("purity", "22K", 22, "K"),
        ("purity", "916", 916, "fineness"),
        ("id", "1661 1681 9189", "166116819189", None),
    ],
)
def test_parse_valid(kind, raw, value, unit):
    parsed = parse_value(kind, raw)
    assert parsed.error is None
    assert parsed.value == value
    assert parsed.unit == unit


@pytest.mark.parametrize(
    "kind,raw",
    [
        ("amount", "00000"),
        ("date", "312202"),
        ("date", "31/02/2025"),
        ("date", "3o06 202"),
        ("weight", "H2"),
        ("weight", "OE"),
        ("weight", "1/"),
        ("phone", "12345"),
        ("tenure", "6"),
        ("text", "."),
    ],
)
def test_parse_invalid(kind, raw):
    assert parse_value(kind, raw).error


def test_digit_correction_is_flagged():
    parsed = parse_value("weight", "2O.5")
    assert parsed.value == 20.5
    assert parsed.substituted is True
def test_bank_name_from_known_lenders():
    p = Page()
    p.add("UNION BANK OF INDIA", 100, 50, 500)
    p.add("Name: Ravi", 30, 100, 300)
    data, _ = run(p.lines)
    assert data.bankName.value == "Union Bank of India"
    assert data.customerName.value == "Ravi"


def test_nominee_is_not_customer_name():
    p = Page()
    p.add("Name of the Nominee: Lakshmi", 30, 100, 400)
    data, _ = run(p.lines)
    assert data.customerName.status == FieldStatus.MISSING


def test_labelled_mandatory_fields():
    p = Page()
    p.add("Branch: Gajuwaka", 30, 100, 300)
    p.add("Pledge No: PL-2045", 30, 130, 300)
    p.add("Loan A/c No: 1234 5678", 30, 160, 300)
    p.add("Rate of Interest: 9.25%", 30, 190, 300)
    p.add("Mobile No: 98765 43210", 30, 220, 300)
    p.add("Loan Due Date: 30/06/2026", 30, 250, 300)
    data, _ = run(p.lines)
    assert data.branchName.value == "Gajuwaka"
    assert data.pledgeNumber.value == "PL-2045"
    assert data.loanAccountNumber.value == "12345678"
    assert data.interestRate.value == 9.25
    assert data.phoneNumber.value == "9876543210"
    assert data.dueDate.value == "2026-06-30"
def test_identifier_types_are_kept_separate():
    texts = [
        "Loan A/c No: 1001 2002",
        "Loan Application No: APP-55",
        "Pledge Ticket No: PT 7788",
        "Loan No: LN-901",
        "Receipt No: R-3344",
        "CIF No: 556677",
        "UTR No: SBIN0012345",
        "Bag No: B-1204",
        "Branch Code: 0123",
    ]
    p = Page()
    for i, text in enumerate(texts):
        p.add(text, 30, 100 + 30 * i, 400)
    data, _ = run(p.lines)
    assert data.loanAccountNumber.value == "10012002"
    assert data.loanApplicationNumber.value == "APP-55"
    assert data.pledgeNumber.value == "PT7788"
    assert data.loanNumber.value == "LN-901"
    assert data.receiptNumber.value == "R-3344"
    assert data.customerId.value == "556677"
    assert data.transactionReference.value == "SBIN0012345"
    assert data.packetNumber.value == "B-1204"
    assert data.branchCode.value == "0123"
    assert data.branchName.status == FieldStatus.MISSING
    assert data.otherIdentifiers == []


def test_loan_number_is_not_treated_as_account_number():
    p = Page()
    p.add("Loan No: 4512 7788", 30, 100, 300)
    data, _ = run(p.lines)
    assert data.loanNumber.value == "45127788"
    assert data.loanAccountNumber.status == FieldStatus.MISSING


def test_lender_fallback_mapping(monkeypatch):
    from app.core import extraction_rules

    monkeypatch.setitem(extraction_rules.LENDER_FIELD_FALLBACKS, "Muthoot Finance", {"loanAccountNumber": "loanNumber"})
    p = Page()
    p.add("MUTHOOT FINANCE LTD", 100, 50, 500)
    p.add("Loan No: 4512 7788", 30, 100, 300)
    data, _ = run(p.lines)
    assert data.bankName.value == "Muthoot Finance"
    assert data.loanAccountNumber.value == "45127788"
    assert "Mapped from loanNumber" in data.loanAccountNumber.note


def test_unknown_identifier_goes_to_other_identifiers():
    p = Page()
    p.add("Locker No: 4471", 30, 100, 300)
    p.add("Sl. No.", 30, 140, 100)
    data, _ = run(p.lines)
    assert [(o.label, o.value) for o in data.otherIdentifiers] == [("Locker No", "4471")]
    assert data.otherIdentifiers[0].status == FieldStatus.UNCERTAIN
def test_header_word_pledged_is_not_a_pledge_label():
    p = Page()
    p.add("pledged", 219, 392, 285)
    p.add("ingrams", 485, 392, 583)
    data, _ = run(p.lines)
    assert data.pledgeNumber.status == FieldStatus.MISSING


def test_branch_stamp_does_not_override_header_branch():
    p = Page()
    p.add("Green City Branch, Visakhapatnam", 80, 170, 600)
    p.add("Branch", 557, 936, 611)
    p.add("sfe", 552, 945, 580)
    data, _ = run(p.lines)
    assert data.branchName.value == "Green City"
def test_net_far_below_gross_is_uncertain():
    data, warnings = run(slip(necklace_nw="20"))
    item = data.ornamentDetails[0]
    assert item.grossWeight.status == FieldStatus.UNCERTAIN
    assert item.netWeight.status == FieldStatus.UNCERTAIN
    assert any("under 30% of gross weight" in w for w in warnings)