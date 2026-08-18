"""Centralized label mapping for the DNRTI -> common label space used to
benchmark SecureBERT-NER and CyNER against each other.

Only DNRTI entity types that have a corresponding class in *both* CyNER and
SecureBERT-NER (per the assignment's mapping table) are kept. Everything else
(Time, Area, Purp, Features, and SecureBERT-only indicator sub-types such as
IP/URL/MD5/...) is mapped to UNK_LABEL during preprocessing, so those tokens
can be identified and excluded during evaluation rather than being silently
counted as "O" (true negatives).
"""

# DNRTI entity type -> common label.
# NOTE: DNRTI's "Tool" is mapped to MALWARE (not SYSTEM) per the assignment's
# explicit instruction, even though the table's "System" row also lists
# "Way"/"OffAct" alongside a separate "Malware"/"Tool" row. This ambiguity is
# intentionally not reinterpreted here.
DNRTI_TO_COMMON: dict[str, str] = {
    "HackOrg": "ORG", 
    "SecTeam": "ORG",
    "Idus": "ORG",
    "Org": "ORG",
    "OffAct": "SYSTEM",
    "Way": "SYSTEM",
    "Exp": "VULNERABILITY",
    "Tool": "MALWARE",
    "SamFile": "INDICATOR",
}

SECURE_BERT_TO_COMMON: dict[str, str] = {
    "APT": "ORG", 
    "SECTEAM": "ORG",
    "IDTY": "ORG",
    "ACT": "SYSTEM",
    "OS": "SYSTEM",
    "TOOL": "SYSTEM",
    "VULID": "VULNERABILITY",
    "VULNAME": "VULNERABILITY",
    "MAL": "MALWARE",
    "FILE": "INDICATOR",
}

CYNER_TO_COMMON: dict[str, str] = {
    "Organization": "ORG", 
    "System": "SYSTEM",
    "Vulnerability": "VULNERABILITY",
    "Malware": "MALWARE",
    "Indicator": "INDICATOR",
}

# The canonical label space evaluated in the MVP benchmark.
COMMON_LABELS: set[str] = {
    "ORG",
    "SYSTEM",
    "VULNERABILITY",
    "MALWARE",
    "INDICATOR",
}

# Marker for DNRTI entity types with no corresponding class in COMMON_LABELS
# (e.g. Time, Area, Purp, Features). Kept distinct from "O" so these tokens
# can be excluded during evaluation instead of being treated as true
# negatives.
UNK_LABEL: str = "UNK"


