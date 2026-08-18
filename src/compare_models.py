"""Evaluate SecureBERT-NER and CyNER on the same DNRTI split and compare them.

Usage:
    python scripts/compare_models.py [--data data/test.txt] [--max-sentences N]

Produces three pandas DataFrames:
    overall_df    - one row per model, precision/recall/f1
    per_class_df  - one row per canonical entity type, columns grouped by
                    model (precision/recall/f1/support)
    latency_df    - one row per model, prediction-call latency stats
                    (tokenization + inference + post-processing only)
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import evaluate_model
from src.label_mapping import COMMON_LABELS
from src.model_adapter import CyNERAdapter, ModelAdapter, SecureBertAdapter, run_adapter_on_dataset
from src.preprocessing import preprocess_dnriti

MODELS: dict[str, type[ModelAdapter]] = {
    "SecureBERT-NER": SecureBertAdapter,
    "CyNER": CyNERAdapter,
}


_AVERAGE_ROWS = {"micro avg", "macro avg", "weighted avg"}




def _extract_prf(metrics: dict) -> dict:
    return {
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1-score"]),
    }


def _extract_class_metrics(metrics: dict) -> dict:
    return {**_extract_prf(metrics), "support": int(metrics["support"])}


def _build_overall_df(results: dict[str, dict]) -> pd.DataFrame:
    """`results[name]` is seqeval's classification_report dict; "macro avg" is
    the standard strict-CoNLL "overall" score (aggregates entity counts
    across classes before computing ratios)."""
    return pd.DataFrame(
        {name: _extract_prf(report["macro avg"]) for name, report in results.items()}
    ).T


def _build_per_class_df(results: dict[str, dict]) -> pd.DataFrame:
    empty_row = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
    per_model = {}
    for name, report in results.items():
        per_class = {
            label: _extract_class_metrics(metrics)
            for label, metrics in report.items()
            if label not in _AVERAGE_ROWS
        }
        df = pd.DataFrame(per_class).T.reindex(sorted(COMMON_LABELS))
        per_model[name] = df.apply(lambda col: col.fillna(empty_row[col.name]))
    return pd.concat(per_model, axis=1)


def _build_latency_df(latencies_by_model: dict[str, list[float]]) -> pd.DataFrame:
    """One row per model: prediction-call latency stats, in milliseconds
    per sentence (except total_time_s, in seconds)."""
    rows = {}
    for name, latencies_s in latencies_by_model.items():
        latencies_ms = [t * 1000 for t in latencies_s]
        rows[name] = {
            "mean_latency_ms": statistics.mean(latencies_ms),
            "median_latency_ms": statistics.median(latencies_ms),
            "total_time_s": sum(latencies_s),
            "sentences": len(latencies_s),
        }
    return pd.DataFrame(rows).T


def compare_models(
    data_path: str,
    max_sentences: int | None = None,
    log_file: str | None = "logs/model_adapter.log",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate every model in MODELS on the same slice of `data_path`.

    Returns:
        (overall_df, per_class_df, latency_df)
    """
    dataset = preprocess_dnriti(data_path)
    if max_sentences is not None:
        dataset = dataset[:max_sentences]
    print(f"Evaluating on {len(dataset)} sentences from {data_path}")

    gold_labels = [sentence["labels"] for sentence in dataset]

    results = {}
    latencies_by_model = {}
    for name, adapter_cls in MODELS.items():
        print(f"Running {name}...")
        predicted_labels, _, latencies = run_adapter_on_dataset(adapter_cls, dataset, log_file=log_file)
        results[name] = evaluate_model(gold_labels, predicted_labels)
        latencies_by_model[name] = latencies

    return _build_overall_df(results), _build_per_class_df(results), _build_latency_df(latencies_by_model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/test.txt", help="DNRTI-format file to evaluate on")
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Only evaluate on the first N sentences (for quick debugging runs)",
    )
    parser.add_argument("--log-file", default="logs/model_adapter.log")
    args = parser.parse_args()

    overall_df, per_class_df, latency_df = compare_models(args.data, args.max_sentences, args.log_file)

    pd.set_option("display.width", 120)
    print("\n=== Overall comparison ===")
    print(overall_df.round(3))
    print("\n=== Per-class comparison ===")
    print(per_class_df.round(3))
    print("\n=== Latency comparison ===")
    print(latency_df.round(3))


if __name__ == "__main__":
    main()
