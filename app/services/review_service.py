from app.schemas.pledge_slip import ExtractedData, FieldStatus, OrnamentItem, ReviewInfo


def build_review(data: ExtractedData, mandatory: list[str], warnings: list[str]) -> ReviewInfo:
    missing: list[str] = []
    uncertain: list[str] = []
    invalid: list[str] = []
    buckets = {FieldStatus.MISSING: missing, FieldStatus.UNCERTAIN: uncertain, FieldStatus.INVALID: invalid}

    for name in ExtractedData.model_fields:
        if name in ("ornamentDetails", "otherIdentifiers"):
            continue
        f = getattr(data, name)
        if f.status in buckets:
            buckets[f.status].append(name)

    for i, item in enumerate(data.ornamentDetails):
        for name in OrnamentItem.model_fields:
            f = getattr(item, name)
            if f.status == FieldStatus.UNCERTAIN:
                uncertain.append(f"ornamentDetails[{i}].{name}")
            elif f.status == FieldStatus.INVALID:
                invalid.append(f"ornamentDetails[{i}].{name}")

    missing_mandatory = [name for name in mandatory if name in missing]
    mandatory_issue = any(name in missing or name in uncertain or name in invalid for name in mandatory)

    return ReviewInfo(
        required=mandatory_issue or bool(uncertain or invalid or warnings),
        missingMandatoryFields=missing_mandatory,
        missingFields=missing,
        uncertainFields=uncertain,
        invalidFields=invalid,
        warnings=list(warnings),
    )