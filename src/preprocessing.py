"""Preprocessing for the DNRTI dataset (BIO-tagged token/label files).

Turns the raw DNRTI train/valid/test files into a clean, evaluation-ready
representation using the common label space defined in label_mapping.py.
"""

from __future__ import annotations

import re
import warnings
from collections import Counter

try:
    from src.label_mapping import COMMON_LABELS, DNRTI_TO_COMMON, UNK_LABEL
except ImportError:  # running as a plain module from within src/ (e.g. notebooks)
    from label_mapping import COMMON_LABELS, DNRTI_TO_COMMON, UNK_LABEL

# Matches a well-formed BIO label: "O", "B-<Entity>", or "I-<Entity>".
_LABEL_PATTERN = re.compile(r"^(O|[BI]-[A-Za-z]+)$")


def read_dnriti_file(path: str) -> list[dict]:
    """Parse a raw DNRTI BIO file into a list of sentences.

    Each line is expected to be "<token> <label>" (DNRTI uses a single
    space, but some lines in the real files use multiple spaces/tabs, so any
    run of whitespace is treated as a separator). 
    A blank line ends the current sentence. 
    some lines have label without token, I ignore them.

    Args:
        path: Path to a DNRTI-format file (train.txt / valid.txt / test.txt).

    Returns:
        A list of {"tokens": list[str], "labels": list[str]} dicts, one per
        sentence, in file order.
    """
    sentences: list[dict] = []
    tokens: list[str] = []
    labels: list[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            if line == "":
                if tokens:
                    sentences.append({"tokens": tokens, "labels": labels})
                    tokens, labels = [], []
                continue

            parts = line.split()

            if len(parts) == 2:
                token, label = parts
            elif len(parts) > 2:
                print(f"line_num: {line_num} has more than two parts: {parts}")

                # Two tokens merged onto one line (missing newline upstream).
                token, label = " ".join(parts[:-1]), parts[-1]
            elif len(parts) == 1 and _LABEL_PATTERN.match(parts[0]):
                print(f"line_num: {line_num}: line has a label but no token, skipping ({line!r})")
                continue
            else:
                raise ValueError(
                    f"{path}:{line_num}: malformed line, expected "
                    f"'token label': {line!r}"
                )

            tokens.append(token)
            labels.append(label)

    if tokens: 
        sentences.append({"tokens": tokens, "labels": labels})

    return sentences


def map_dnriti_labels(labels: list[str]) -> list[str]:
    """Map DNRTI BIO labels to the common label space.

    Preserves the B-/I- prefix. DNRTI entity types not present in
    DNRTI_TO_COMMON (e.g. Time, Area, Purp, Features) become "B-UNK"/"I-UNK"
    rather than "O", so they stay distinguishable from true negatives and
    can be explicitly ignored during evaluation. "O" stays "O".

    Example:
        ["B-SecTeam", "O", "B-HackOrg", "I-HackOrg", "B-Time"]
        -> ["B-ORG", "O", "B-ORG", "I-ORG", "B-UNK"]
    """
    mapped = []
    for label in labels:
        if label == "O":
            mapped.append("O")
            continue

        prefix, sep, entity = label.partition("-")
        if prefix not in ("B", "I") or not sep or not entity:
            raise ValueError(f"Unrecognized label format: {label!r}")

        common = DNRTI_TO_COMMON.get(entity, UNK_LABEL)
        mapped.append(f"{prefix}-{common}")

    return mapped



def validate_dataset(dataset: list[dict]) -> None:
    """Validate a processed DNRTI dataset, raising on the first problem found.

    Checks:
      * the dataset and each sentence are non-empty
      * tokens and labels are the same length within each sentence
      * every label is "O" or a well-formed "B-*"/"I-*" tag
      * every non-O label's entity type is in COMMON_LABELS or is UNK_LABEL
        (entities DNRTI tags that this benchmark doesn't evaluate)
    """
    if not dataset:
        raise ValueError("Dataset is empty")

    for i, sentence in enumerate(dataset):
        tokens = sentence["tokens"]
        labels = sentence["labels"]

        if not tokens or not labels:
            raise ValueError(f"Sentence {i} is empty")
        if len(tokens) != len(labels):
            raise ValueError(
                f"Sentence {i}: token/label count mismatch "
                f"({len(tokens)} tokens vs {len(labels)} labels)"
            )

        for j, label in enumerate(labels):
            if not _LABEL_PATTERN.match(label):
                raise ValueError(
                    f"Sentence {i}, token {j}: malformed label {label!r}"
                )

            if label != "O":
                entity = label[2:]
                if entity not in COMMON_LABELS and entity != UNK_LABEL:
                    raise ValueError(
                        f"Sentence {i}, token {j}: label {label!r} is not "
                        f"in COMMON_LABELS or {UNK_LABEL!r}"
                    )

def preprocess_dnriti(path: str) -> list[dict]:
    """Read and preprocess a DNRTI file into the common label space.

    Reads the raw file, maps supported DNRTI entity types to the common
    label space, maps unsupported entity types to UNK_LABEL (kept out of
    COMMON_LABELS so they can be ignored during evaluation), normalizes the
    result into valid BIO, and validates it.

    Returns:
        A list of {"tokens": list[str], "original_labels": list[str],
        "labels": list[str]} dicts, where "labels" is the processed common
        label space and "original_labels" preserves the raw DNRTI labels for
        debugging.
    """
    raw_sentences = read_dnriti_file(path)

    processed = []
    for sentence in raw_sentences:
        mapped = map_dnriti_labels(sentence["labels"])
        processed.append(
            {
                "tokens": sentence["tokens"],
                "original_labels": sentence["labels"],
                "labels": mapped,
            }
        )

    validate_dataset(processed)
    return processed


def compute_label_statistics(
    dataset: list[dict],
) -> tuple[dict[str, int], dict[str, int]]:
    """Count how many tokens carry each label across a preprocessed dataset.

    Args:
        dataset: Output of preprocess_dnriti (a list of sentence dicts with
            a "labels" field).

    Returns:
        A tuple of two dicts:
          * per-label counts, keyed by the full BIO tag (e.g. "O", "B-ORG",
            "I-ORG", "B-UNK")
          * per-entity counts, keyed by the label with any "B-"/"I-" prefix
            stripped (e.g. "O", "ORG", "UNK"), i.e. B- and I- tokens of the
            same entity type are combined
    """
    label_counts: Counter = Counter()
    entity_counts: Counter = Counter()
    for sentence in dataset:
        for label in sentence["labels"]:
            label_counts[label] += 1
            entity = label if label == "O" else label[2:]
            entity_counts[entity] += 1
    return dict(label_counts), dict(entity_counts)
