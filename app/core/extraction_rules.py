from dataclasses import dataclass


@dataclass(frozen=True)
class FieldRule:
    name: str
    kind: str
    aliases: tuple[str, ...]
    patterns: tuple[str, ...] = ()
    pattern_confidence: float = 0.6
    known_values: tuple[str, ...] = ()
    max_value_lines: int = 1


KNOWN_LENDERS: tuple[str, ...] = (
    "State Bank of India", "Canara Bank", "Bank of Baroda", "Punjab National Bank",
    "Union Bank of India", "Indian Bank", "Indian Overseas Bank", "Bank of India",
    "Central Bank of India", "UCO Bank", "Bank of Maharashtra", "Punjab & Sind Bank",
    "HDFC Bank", "ICICI Bank", "Axis Bank", "Kotak Mahindra Bank", "Federal Bank",
    "South Indian Bank", "Karur Vysya Bank", "City Union Bank", "Karnataka Bank",
    "IndusInd Bank", "Yes Bank", "IDFC First Bank", "Bandhan Bank", "CSB Bank",
    "Tamilnad Mercantile Bank", "DCB Bank", "RBL Bank", "Equitas Small Finance Bank",
    "AU Small Finance Bank", "Ujjivan Small Finance Bank", "Andhra Pradesh Grameena Vikas Bank",
    "Muthoot Finance", "Muthoot Fincorp", "Manappuram Finance", "IIFL Finance",
    "Shriram Finance", "Bajaj Finance", "Indel Money", "Rupeek",
)

FIELD_RULES: tuple[FieldRule, ...] = (
    FieldRule(
        "customerName",
        "text",
        (
            "name of the borrower", "name of borrower", "borrower name", "borrowers name",
            "customer name", "name of the customer", "name of customer",
            "applicant name", "name of the applicant", "name",
        ),
    ),
    FieldRule("bankName", "text", ("bank name", "name of the bank", "name of bank", "lender name"), known_values=KNOWN_LENDERS),
    FieldRule(
        "loanAccountNumber",
        "id",
        (
            "loan account no", "loan account number", "loan a/c no", "loan ac no", "loan acc no",
            "gold loan account no", "gold loan account number", "gold loan a/c no",
            "account no", "account number", "a/c no", "ac no",
        ),
    ),
    FieldRule(
        "pledgeNumber",
        "id",
        ("pledge no", "pledge number", "pledge id", "pledge ticket no", "pledge ticket number", "pledge card no", "pledge slip no"),
    ),
    FieldRule(
        "loanAmount",
        "amount",
        (
            "amount of loan", "loan amount", "loan amt", "sanctioned amount", "amount sanctioned",
            "disbursed amount", "facility amount", "sanctioned a term loan of rs", "sanctioned a term loan of",
        ),
    ),
    FieldRule(
        "loanDate",
        "date",
        (
            "date of loan", "loan date", "loan start date", "start date", "date of pledge", "pledge date",
            "date of sanction", "sanction date", "date of disbursement", "disbursement date",
        ),
    ),
    FieldRule(
        "dueDate",
        "date",
        (
            "loan due date", "due date", "loan end date", "end date", "date of close", "date of closure",
            "closure date", "maturity date", "date of maturity", "repayment date", "expiry date","repayment dale",
        )
    ),
    FieldRule(
        "branchName",
        "text",
        ("branch name", "name of the branch", "name of branch", "branch"),
        patterns=(r"^\s*(?!name\b)([A-Za-z][A-Za-z .&-]{1,40}?)\s+branch\b",),
        pattern_confidence=0.8,
    ),
    FieldRule(
        "phoneNumber",
        "phone",
        ("mobile no", "mobile number", "mobile", "mob no", "phone no", "phone number", "contact no", "contact number", "cell no"),
    ),
    FieldRule("interestRate", "percent", ("rate of interest", "interest rate", "int rate", "interest", "roi")),
    FieldRule(
        "tenure",
        "tenure",
        ("loan tenure", "tenure of the facility", "tenure", "loan period", "period of loan", "tenor", "period"),
        patterns=(r"\b(\d{1,3}\s*(?:months?|mths?|days?|years?|yrs?))\b",),
    ),
    FieldRule("netWeight", "weight", ("net weight", "net wt", "total net weight")),
    FieldRule(
        "address",
        "text",
        ("address", "residential address", "communication address", "permanent address"),
        max_value_lines=3,
    ),
    FieldRule(
        "loanApplicationNumber",
        "id",
        ("loan application no", "loan application number", "application no", "application number", "application id", "app no", "app id"),
    ),
    FieldRule("loanNumber", "id", ("loan no", "loan number", "loan id", "gold loan no", "gold loan number", "loan ref no")),
    FieldRule("receiptNumber", "id", ("receipt no", "receipt number", "receipt id", "rcpt no")),
    FieldRule(
        "customerId",
        "id",
        ("customer id", "customer no", "customer number", "cust id", "cust no", "cif no", "cif number", "cif id", "cif"),
    ),
    FieldRule(
        "transactionReference",
        "id",
        (
            "transaction id", "transaction no", "transaction reference", "transaction ref no",
            "txn id", "txn no", "reference no", "reference number", "ref no", "utr no", "utr number", "utr",
        ),
    ),
    FieldRule(
        "packetNumber",
        "id",
        ("packet no", "packet number", "gold packet no", "bag no", "bag number", "seal no", "seal number", "pouch no", "pkt no"),
    ),
    FieldRule("branchCode", "id", ("branch code", "br code", "branch id", "branch no")),
    FieldRule("grossWeight", "weight", ("gross weight", "gross wt", "total gross weight")),
    FieldRule(
        "goldPurity",
        "purity",
        ("purity", "fineness", "karat", "carat", "caratage"),
        patterns=(r"\b(\d{2}\s*(?:kt|k|ct|carat|karat))\b",),
    ),
    FieldRule("appraisedValue", "amount", ("appraised value", "market value", "value of gold", "valuation", "gold value")),
)

LENDER_FIELD_FALLBACKS: dict[str, dict[str, str]] = {}

IGNORED_LABEL_ALIASES: tuple[str, ...] = (
    "name of the nominee", "name of nominee", "nominee name", "nominee",
    "name of the father", "name of father", "father name", "fathers name",
    "name of the mother", "name of mother", "mother name", "mothers name",
    "name of the husband", "name of husband", "husband name",
    "name of the spouse", "name of spouse", "spouse name",
    "name of the guardian", "name of guardian", "guardian name",
    "appraiser name", "name of the appraiser", "name of appraiser",
    "s/o", "w/o", "d/o",
    "interest amount", "interest amt", "interest due", "interest paid", "interest payable",
    "interest upto", "interest up to", "penal interest", "penal rate", "overdue interest",
    "signature",
    "interest type", "interest payment frequency", "valuation charges",
)
TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "description": ("description of gold items", "description", "particulars", "ornaments", "ornament", "items"),
    "quantity": ("qty", "quantity", "nos", "no of items", "pcs", "pieces"),
    "grossWeight": ("gw", "gross weight", "gross wt", "gross"),
    "netWeight": ("nw", "net weight", "net wt", "weight net", "net"),
    "goldPurity": ("purity", "karat", "carat", "fineness"),
}

TABLE_IGNORED_COLUMNS: tuple[str, ...] = (
    "m value", "market value", "value per gram", "per gram", "per", "gram rs", "others", "remarks",
)
TABLE_TOTAL_ALIASES: tuple[str, ...] = ("total weight", "total wt", "grand total", "total")
TABLE_END_ALIASES: tuple[str, ...] = ("signature", "acknowledgement", "declaration", "terms and conditions", "remarks")

ITEM_FIELD_KINDS: dict[str, str] = {
    "description": "text",
    "quantity": "quantity",
    "goldPurity": "purity",
    "grossWeight": "weight",
    "netWeight": "weight",
}
VALUE_MINIMUMS: dict[str, float] = {
    "loanAmount": 1000,
    "appraisedValue": 1000,
}

LOAN_PER_GRAM_RANGE: tuple[float, float] = (500, 20000)
MIN_NET_TO_GROSS_RATIO: float = 0.3