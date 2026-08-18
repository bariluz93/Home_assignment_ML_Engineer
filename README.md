# DNRTI NER Benchmark

Benchmarks SecureBERT-NER and CyNER against each other on the DNRTI dataset,
on a common label space: preprocessing, strict CoNLL-style evaluation, model
adapters, and a comparison script that reports precision/recall/F1 and
per-sentence latency for both models.

## Structure

```
data/
    train.txt
    valid.txt
    test.txt
src/
    label_mapping.py   # DNRTI/SecureBERT/CyNER -> common label mapping (centralized, easy to change)
    preprocessing.py   # parsing DNRTI files, mapping to the common label space, validation
    evaluation.py       # strict, entity-level CoNLL-style scoring (seqeval), UNK masking
    model_adapter.py    # ModelAdapter interface + SecureBertAdapter/CyNERAdapter, latency timing
    compare_models.py   # CLI: run one or more models on the same data, print comparison tables
app.py                    # Streamlit web UI: model-comparison dashboard
scripts/
    compare_models.sh   # thin wrapper around `python src/compare_models.py`
    run_app.sh            # thin wrapper around `streamlit run app.py`
logs/                    # subword-disagreement warnings written here at runtime (see below)
requirements.txt
```

(The raw files live directly under `data/`, not `data/dnriti/`, matching how
they're actually laid out in this repo.)

## Common label space

Only entity types with a corresponding class in **both** CyNER and
SecureBERT-NER are kept (see `src/label_mapping.py`):

```text
ORG
SYSTEM
VULNERABILITY
MALWARE
INDICATOR
```

| DNRTI                    | Common label    |
| ------------------------- | --------------- |
| HackOrg, SecTeam, Idus, Org | `ORG`          |
| OffAct, Way                | `SYSTEM`       |
| Exp                         | `VULNERABILITY`|
| Tool                        | `MALWARE`      |
| SamFile                     | `INDICATOR`    |

| SecureBERT-NER                  | Common label    |
| -------------------------------- | --------------- |
| APT, SECTEAM, IDTY               | `ORG`          |
| ACT, OS, TOOL                    | `SYSTEM`       |
| VULID, VULNAME                   | `VULNERABILITY`|
| MAL                              | `MALWARE`      |
| FILE                             | `INDICATOR`    |

| CyNER            | Common label    |
| ------------------ | --------------- |
| Organization        | `ORG`          |
| System               | `SYSTEM`       |
| Vulnerability        | `VULNERABILITY`|
| Malware               | `MALWARE`      |
| Indicator              | `INDICATOR`    |

Gold DNRTI entity types with no match above (`Time`, `Area`, `Purp`,
`Features`, ...) are mapped to `UNK` (as `B-UNK`/`I-UNK`), not `O`. This keeps
"no entity here" (`O`) distinguishable from "an entity DNRTI tags but this
benchmark doesn't evaluate" (`UNK`); `evaluate_model` masks `UNK` tokens out
of scoring entirely (see below). Model predictions never contain `UNK` - each
adapter maps its own unsupported label types (e.g. SecureBERT's `LOC`/`TIME`)
straight to `O`, since a prediction of an untracked type is just treated as
no prediction.

## Preprocessing

```python
from src.preprocessing import preprocess_dnriti

dataset = preprocess_dnriti("data/train.txt")
# [{"tokens": [...], "original_labels": [...], "labels": [...]}, ...]
```

## Evaluation (`src/evaluation.py`)

Strict, entity-level CoNLL-style scoring on top of `seqeval`
(`mode="strict"`, `scheme=IOB2`): a predicted entity only counts as correct
if both its span and its type exactly match gold - a partial-span overlap or
a right-span/wrong-type prediction counts as a miss, never partial credit.

```python
from src.evaluation import evaluate_model

result = evaluate_model(gold_labels, predicted_labels)
# seqeval's classification_report as a dict: one entry per entity type that
# appears (precision/recall/f1-score/support), plus "micro avg"/"macro avg"/
# "weighted avg"
```

- `validate_predictions(gold_labels, predicted_labels)` checks sentence and
  per-sentence token counts line up between gold and predictions.
- `evaluate_model` masks out every token where gold is `UNK` (and whatever a
  model predicted at that same position) before scoring, so untracked gold
  entities are neither a false negative nor a false positive for either
  model.

## Model adapters (`src/model_adapter.py`)

`ModelAdapter` is the common interface every model wraps itself in:
`load_model()` (loads the underlying model once, at construction),
`predict(tokens)` (runs inference, returns the model's raw output),
`post_process(tokens, raw_output)` (maps the model's own labels to the
canonical BIO label space). `predict_labels(tokens)` runs both in one call
and returns `(bio_labels, raw_subwords)`.

- `SecureBertAdapter` wraps [`CyberPeace-Institute/SecureBERT-NER`](https://huggingface.co/CyberPeace-Institute/SecureBERT-NER).
- `CyNERAdapter` wraps [`AI4Sec/cyner-xlm-roberta-base`](https://huggingface.co/AI4Sec/cyner-xlm-roberta-base).

Both use a Hugging Face `token-classification` pipeline with
`aggregation_strategy=None` (raw, per-subword predictions) rather than the
pipeline's own word-aggregation heuristic, which doesn't line up with DNRTI's
whitespace tokenization. When the model's tokenizer splits one input token
into several subwords, they're expected to agree on one entity type; when
they don't, the first subword's label wins and the disagreement is logged
(see `run_adapter_on_dataset`'s `log_file` argument, default
`logs/model_adapter.log`).

`run_adapter_on_dataset(adapter_cls, dataset, max_sentences=None, log_file=...)`
runs an adapter over every sentence in a preprocessed dataset and returns
`(labels, tokens, latencies_s)` - predicted BIO labels, the raw subwords the
model saw, and the wall-clock seconds spent in `predict_labels` per sentence
(tokenization + inference + post-processing only; excludes model loading).

## Running the comparison (`src/compare_models.py`)

Runs SecureBERT-NER and CyNER on the same slice of a DNRTI file and prints
three comparison tables: overall precision/recall/F1, per-class
precision/recall/F1/support, and per-sentence prediction latency.

```
python .\src\compare_models.py
```

| Argument | Default | Description |
| --- | --- | --- |
| `--data` | `data/test.txt` | DNRTI-format file to evaluate on. |
| `--max-sentences` | none (full file) | Only evaluate on N sentences chosen at random, with a best-effort balance across entity classes (see `select_balanced_sample` in `src/model_adapter.py`) - useful for a quick, cheap run while debugging. |
| `--log-file` | `logs/model_adapter.log` | Where subword-disagreement warnings (see above) get appended. |

Example, for a quick 20-sentence smoke test:

```
python .\src\compare_models.py --max-sentences 20 --log-file logs\smoke.log
```

`compare_models(data_path, max_sentences, log_file, models=None)` is also
importable directly (e.g. from a notebook) and returns `(overall_df,
per_class_df, latency_df)` as pandas DataFrames instead of printing them.
`models` defaults to all of `MODELS`; pass a subset (e.g.
`{"CyNER": CyNERAdapter}`) to evaluate just one model - the DataFrames come
back the same shape either way, just with fewer rows/columns.

If `data_path` doesn't exist, `compare_models.py`'s CLI (`ensure_dnrti_dataset`)
downloads `DNRTI.rar` from its GitHub repo and extracts it into `data_path`'s
parent directory automatically (via the system `tar` command - see caveats in
the Web UI section below), so a fresh clone doesn't need the dataset committed.

## Web UI (`app.py`)

A single-page Streamlit dashboard around `compare_models()`:

```
streamlit run app.py
```

or, from `scripts/`:

```
./run_app.sh
```

Pick which model(s) to evaluate (a multiselect over `SecureBERT-NER` and
`CyNER` - selecting just one shows that model's own metrics, it's the same
comparison machinery run over a one-model set), the dataset path (default
`data/test.txt`, with a **Download DNRTI dataset** button that fetches and
extracts it if missing), `max_sentences` (0 = full dataset), and `log_file`,
then click **Run Comparison**. A progress bar tracks each selected model's
sentence-by-sentence progress. Results are shown as grouped bar charts
(dataset label statistics from `compute_label_statistics`, overall
precision/recall/F1, a per-class breakdown faceted by metric, and
per-sentence latency) alongside the full DataFrames for exact numbers. A
full-dataset run evaluates every selected model sequentially on CPU and can
take several minutes.

Note on RAR extraction: Python's standard library has no RAR support, so
`ensure_dnrti_dataset` shells out to `tar`. This works out of the box on
Windows 10+ and macOS (both ship a libarchive-based `tar` that reads RAR
archives), but requires a libarchive-based `tar` (not plain GNU tar) on
Linux.

## Docker
to build the docker run:
```
docker compose up --build
```

## Notes
The file `notes_and_decisions_explanation.pdf` contains notes and explanations about some design and algorithmic choices I made in the assignment.