"""Model adapters: wrap a pretrained NER model behind a common interface.

Each adapter is responsible for its own model loading, inference, and label
mapping so that evaluation.py never has to know about SecureBERT, CyNER,
Hugging Face, or tokenizers - it only ever sees canonical BIO labels
(ORG/SYSTEM/VULNERABILITY/MALWARE/INDICATOR/O), one per input token.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from tqdm import tqdm
from transformers import pipeline

_logger = logging.getLogger(__name__)


def _configure_file_logging(log_path: str) -> None:
    """Attach a file handler to the adapter logger (idempotent per path)."""
    resolved = str(Path(log_path).resolve())
    if any(
        isinstance(h, logging.FileHandler) and h.baseFilename == resolved
        for h in _logger.handlers
    ):
        return

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _logger.addHandler(handler)
    _logger.setLevel(logging.WARNING)


try:
    from src.label_mapping import CYNER_TO_COMMON, SECURE_BERT_TO_COMMON
    from src.preprocessing import preprocess_dnriti
except ImportError:  # running as a plain module from within src/ (e.g. notebooks)
    from label_mapping import CYNER_TO_COMMON, SECURE_BERT_TO_COMMON
    from preprocessing import preprocess_dnriti


def _char_spans(tokens: list[str]) -> list[tuple[int, int]]:
    """Character spans of `tokens` as they appear in " ".join(tokens)."""
    spans = []
    pos = 0
    for token in tokens:
        spans.append((pos, pos + len(token)))
        pos += len(token) + 1  # +1 for the joining space
    return spans


class ModelAdapter(ABC):
    """Common interface for a pretrained NER model used in the benchmark.

    Subclasses adapt one model's tokenizer, label set, and output format to
    the canonical BIO label space. The model is loaded once, at construction
    time.
    """

    def __init__(self) -> None:
        self.model = self.load_model()

    @abstractmethod
    def load_model(self) -> Any:
        """Load and return the underlying model (e.g. a HF pipeline)."""
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, tokens: list[str]) -> Any:
        """Run the model on one pre-tokenized sentence; return its raw output."""
        raise NotImplementedError
        
    @abstractmethod
    def post_process(self, tokens: list[str], raw_output: Any) -> list[str]:
        """Map the model's raw output to canonical BIO labels, one per input token."""
        raise NotImplementedError

    def predict_labels(self, tokens: list[str]) -> list[str]:
        """Convenience: predict() then post_process() in one call."""
        raw_outputs = self.predict(tokens)
        return self.post_process(tokens, raw_outputs), [r['word'] for r in raw_outputs]

    @staticmethod
    def _entities_to_bio(
        tokens: list[str], subword_predictions: list[dict], label_map: dict[str, str]
    ) -> list[str]:
        """Convert raw, per-subword predictions (aggregation_strategy=None) into
        per-token canonical BIO labels.

        The model's own tokenizer may split one input token into several
        subwords (e.g. "supply-chain" -> "supply", "-", "chain"). All
        subwords belonging to one token are expected to agree on the entity
        type; when they do, that type is used directly. When they don't, the
        first subword's label wins for the whole token and the disagreement
        is logged so it can be inspected later. "O" subwords are omitted
        from the pipeline's raw output entirely, so a token with no
        overlapping predictions is left as "O".

        Entity types with no entry in `label_map` (i.e. no canonical-label
        counterpart, e.g. SecureBERT's LOC/TIME/IP/...) are mapped to "O" -
        unlike gold labels, where an unsupported DNRTI type becomes UNK
        (see label_mapping.py), a prediction of a type the benchmark
        doesn't track is just treated as no prediction at all.
        """
        spans = _char_spans(tokens)
        labels = ["O"] * len(tokens)

        for tok_idx, (tok_start, tok_end) in enumerate(spans):
            parts = [
                p for p in subword_predictions if p["start"] < tok_end and p["end"] > tok_start]
            if not parts:
                continue

            part_types = {
                "O" if p["entity"] == "O" else p["entity"].split("-", 1)[1] for p in parts
            }
            if len(part_types) > 1:
                _logger.warning(
                    "Token %r was split into subwords with disagreeing entities "
                    "%s; using the first subword's label %r for the whole token",
                    tokens[tok_idx],
                    [(p["word"], p["entity"]) for p in parts],
                    parts[0]["entity"],
                )

            first_entity = parts[0]["entity"]
            if first_entity == "O":
                continue

            prefix, entity_type = first_entity.split("-", 1)
            common_label = label_map.get(entity_type)
            if common_label is None:
                continue

            labels[tok_idx] = f"{prefix}-{common_label}"

        return labels


class SecureBertAdapter(ModelAdapter):
    """Adapter for CyberPeace-Institute/SecureBERT-NER.

    https://huggingface.co/CyberPeace-Institute/SecureBERT-NER
    """

    MODEL_NAME = "CyberPeace-Institute/SecureBERT-NER"
    def load_model(self) -> Any:
            """Load and return the underlying model (e.g. a HF pipeline)."""
            return pipeline(
                    "token-classification",
                    model=self.MODEL_NAME,
                    tokenizer=self.MODEL_NAME,
                    aggregation_strategy=None,
                )

    def predict(self, tokens: list[str]) -> Any:
            """Run the model on one pre-tokenized sentence; return its raw output."""
            text = " ".join(tokens)
            return self.model(text)
    
    def post_process(self, tokens: list[str], raw_output: Any) -> list[str]:
        return self._entities_to_bio(tokens, raw_output, SECURE_BERT_TO_COMMON)


class CyNERAdapter(ModelAdapter):
    """Adapter for AI4Sec/cyner-xlm-roberta-base.

    https://huggingface.co/AI4Sec/cyner-xlm-roberta-base
    """

    MODEL_NAME = "AI4Sec/cyner-xlm-roberta-base"
    
    def load_model(self) -> Any:
        return pipeline(
            "token-classification",
            model=self.MODEL_NAME,
            tokenizer=self.MODEL_NAME,
            aggregation_strategy=None,
        )
    
    def predict(self, tokens: list[str]) -> Any:
            """Run the model on one pre-tokenized sentence; return its raw output."""
            text = " ".join(tokens)
            return self.model(text)
    
    def post_process(self, tokens: list[str], raw_output: Any) -> list[str]:
        return self._entities_to_bio(tokens, raw_output, CYNER_TO_COMMON)


def run_adapter_on_dataset(
    adapter_cls: type[ModelAdapter],
    dataset: list[dict],
    max_sentences: int | None = None,
    log_file: str | None = "logs/model_adapter.log",
) -> tuple[list[list[str]], list[list[str]], list[float]]:
    """Run a model adapter over every sentence in a preprocessed DNRTI file.

    Args:
        adapter_cls: A ModelAdapter subclass (e.g. SecureBertAdapter). It is
            instantiated here, which loads the model.
        data: A list of dictionaries, each representing a preprocessed sentence.
        max_sentences: If set, only run on the first N sentences - useful for
            a quick, cheap debugging run on a small slice of the data.
        log_file: Where to append subword-disagreement warnings (see
            ModelAdapter._entities_to_bio). Pass None to skip file logging
            and rely on the default stderr output instead.

    Returns:
        (labels, tokens, latencies_s):
          * labels: predicted canonical BIO labels, one list per sentence.
            Pair these with the same file's gold labels (from
            preprocess_dnriti) to call evaluate_model.
          * tokens: the raw subwords the model saw, one list per sentence.
          * latencies_s: wall-clock seconds spent in adapter.predict_labels
            per sentence - tokenization, inference, and post-processing
            only. Excludes model loading (done once before the loop) and
            everything outside the loop (dataset loading, evaluation,
            printing, saving).
    """
    if log_file:
        _configure_file_logging(log_file)

    adapter = adapter_cls()
    if max_sentences is not None:
        dataset = dataset[:max_sentences]
    labels, tokens, latencies = [], [], []
    for sentence in tqdm(dataset):
        start = time.perf_counter()
        s_labels, s_tokens = adapter.predict_labels(sentence["tokens"])
        latencies.append(time.perf_counter() - start)
        labels.append(s_labels)
        tokens.append(s_tokens)
    return labels, tokens, latencies
