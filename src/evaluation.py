"""Strict CoNLL-style NER evaluation over the canonical BIO label space.

This module only knows about the canonical labels produced by preprocessing
(see label_mapping.py): ORG, SYSTEM, VULNERABILITY, MALWARE, INDICATOR, plus
the UNK marker for gold tokens whose original type isn't tracked by this
benchmark. It has no knowledge of SecureBERT, CyNER, DNRTI's original label
names, tokenizers, model inference, or latency measurement - those live in
other components. Predictions are expected to never contain UNK (a model
adapter maps its own unsupported types straight to "O"); gold UNK tokens are
masked out of scoring entirely by evaluate_model (see _mask_unk).
"""

from __future__ import annotations

import numpy as np
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

try:
    from src.label_mapping import COMMON_LABELS, UNK_LABEL
except ImportError:  # running as a plain module from within src/ (e.g. notebooks)
    from label_mapping import COMMON_LABELS, UNK_LABEL



def validate_predictions(gold_labels: list[list[str]], predicted_labels: list[list[str]]) -> None:
    """Validate gold/predicted label sequences before scoring.

    Checks:
      * the number of sentences matches
      * each sentence has the same number of gold and predicted tokens
      * every label is "O" or a well-formed "B-"/"I-" tag over the canonical
        label space (ORG, SYSTEM, VULNERABILITY, MALWARE, INDICATOR)
      * "I-" tags only follow a "B-" or "I-" tag of the same entity type

    Raises:
        ValueError: with a message identifying the sentence/token at fault.
    """
    if len(gold_labels) != len(predicted_labels):
        raise ValueError(
            f"Sentence count mismatch: {len(gold_labels)} gold sentences "
            f"vs {len(predicted_labels)} predicted sentences"
        )

    for i, (gold_sent, pred_sent) in enumerate(zip(gold_labels, predicted_labels)):
        if len(gold_sent) != len(pred_sent):
            raise ValueError(
                f"Sentence {i}: token count mismatch ({len(gold_sent)} gold "
                f"labels vs {len(pred_sent)} predicted labels)"
            )

def ignore_unk(gold_labels: list[list[str]], predicted_labels: list[list[str]]) -> tuple[list[list[str]], list[list[str]]]:
    clean1 = []
    clean2 = []

    for row1, row2 in zip(gold_labels, predicted_labels):
        keep = list(map(lambda x: "UNK" not in x, row1))
        clean1.append((np.array(row1)[keep]).tolist())
        clean2.append((np.array(row2)[keep]).tolist())
    return clean1, clean2

def evaluate_model(
    gold_labels: list[list[str]],
    predicted_labels: list[list[str]]
) -> dict:
    """Validate and score model predictions against gold labels.

    Args:
        gold_labels: Canonical BIO labels per sentence, from preprocessing.
        predicted_labels: Canonical BIO labels per sentence, from a model adapter.
        include_per_class: Whether to also compute per-entity-type metrics.

    Returns:
        {"overall": {"precision", "recall", "f1"}, "per_class": {...}}
        `per_class` is an empty dict when include_per_class is False.
    """
    validate_predictions(gold_labels, predicted_labels)
    gold_labels, predicted_labels = ignore_unk(gold_labels, predicted_labels)
    return classification_report(
        gold_labels,
        predicted_labels,
        mode="strict",
        scheme=IOB2,
        output_dict=True,
        zero_division=0,
    )
