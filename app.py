"""Streamlit web UI for the DNRTI NER benchmark.

Runs one or more NER models (SecureBERT-NER, CyNER) against each other on a
DNRTI-format file and compares them with charts. Selecting a single model
just shows that model's own metrics - it's the same comparison machinery,
run over a one-model set.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import altair as alt
import pandas as pd
import streamlit as st

from src.compare_models import MODELS, compare_models, ensure_dnrti_dataset
from src.preprocessing import compute_label_statistics

st.set_page_config(page_title="Cybersecurity NER", page_icon="🛡️", layout="wide")


def _melt_overall(overall_df: pd.DataFrame) -> pd.DataFrame:
    return (
        overall_df.rename_axis("model")
        .reset_index()
        .melt(id_vars="model", value_vars=["precision", "recall", "f1"], var_name="metric", value_name="value")
    )


def _melt_per_class(per_class_df: pd.DataFrame) -> pd.DataFrame:
    return (
        per_class_df.stack(level=0, future_stack=True)
        .rename_axis(index=["class", "model"])
        .reset_index()
        .melt(id_vars=["class", "model"], value_vars=["precision", "recall", "f1"], var_name="metric", value_name="value")
    )


def _melt_latency(latency_df: pd.DataFrame) -> pd.DataFrame:
    return (
        latency_df.rename_axis("model")
        .reset_index()
        .melt(
            id_vars="model",
            value_vars=["mean_latency_ms", "median_latency_ms"],
            var_name="metric",
            value_name="value",
        )
    )


def _grouped_bar(long_df: pd.DataFrame, y_title: str, y_domain: list[float] | None = None) -> alt.Chart:
    y_scale = alt.Scale(domain=y_domain) if y_domain else alt.Scale()
    return (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X("metric:N", title=None),
            y=alt.Y("value:Q", title=y_title, scale=y_scale),
            xOffset=alt.XOffset("model:N"),
            color=alt.Color("model:N", title="Model"),
            tooltip=["model", "metric", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(height=280)
    )

def _get_inputs() -> tuple[bool, list[str], int, str, list[str], str]:
    st.write(
        "Run one or more NER models on a DNRTI-format file and compare precision, "
        "recall, F1, and latency."
    )

    selected_models = st.multiselect(
        "Models to compare", list(MODELS.keys()), default=list(MODELS.keys())
    )
    # Preserve MODELS' order regardless of selection/click order, for a stable chart legend.
    selected_models = [name for name in MODELS if name in selected_models]

    col1, col2 = st.columns([3, 1])
    with col1:
        data_path = st.text_input("Dataset file (DNRTI format)", value="data/test.txt")
    with col2:
        st.write("")  # vertical spacer to align the button with the text input
        st.write("")
        if st.button("Download DNRTI dataset"):
            try:
                with st.spinner("Downloading and extracting DNRTI dataset..."):
                    ensure_dnrti_dataset(data_path)
                st.success(f"{data_path} is ready.")
            except (RuntimeError, FileNotFoundError) as exc:
                st.error(str(exc))

    col3, col4 = st.columns(2)
    with col3:
        max_sentences_input = st.number_input(
            "Max sentences (0 = full dataset)", min_value=0, value=0, step=10
        )
    with col4:
        log_file = st.text_input("Log file", value="logs/model_adapter.log")

    run_clicked = st.button("Run Comparison", type="primary", disabled=not selected_models)
    if not selected_models:
        st.warning("Select at least one model to run.")
    return run_clicked, selected_models, max_sentences_input, data_path, selected_models, log_file

def get_outputs(data_path: str, max_sentences: int | None, log_file: str, selected_models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]    :
    
        with st.spinner("Preparing dataset..."):
            ensure_dnrti_dataset(data_path)

        progress = st.progress(0.0, text="Running comparison...")

        def on_progress(model_name: str, done: int, total: int) -> None:
            model_idx = selected_models.index(model_name)
            fraction = (model_idx + done / total) / len(selected_models)
            progress.progress(fraction, text=f"Running {model_name}... ({done}/{total} sentences)")

        models = {name: MODELS[name] for name in selected_models}
        overall_df, per_class_df, latency_df, evaluated_dataset = compare_models(
            data_path, max_sentences, log_file, models=models, on_progress=on_progress
        )
        _, entity_counts = compute_label_statistics(evaluated_dataset)
        progress.empty()
        return overall_df, per_class_df, latency_df, entity_counts


def present_results(results: dict) -> None:
    params = results["params"]
    st.caption(
        f"Showing results for {', '.join(params['models'])} on `{params['data_path']}` "
        f"(max_sentences={params['max_sentences'] or 'all'}, log_file=`{params['log_file']}`)."
    )

    overall_df, per_class_df, latency_df = results["overall_df"], results["per_class_df"], results["latency_df"]

    st.subheader("Dataset label statistics")
    stats_df = (
        pd.DataFrame(
            {"class": list(results["entity_counts"].keys()), "token count": list(results["entity_counts"].values())}
        )
        .sort_values("token count", ascending=False)
        .reset_index(drop=True)
    )
    chart_col, table_col = st.columns([2, 1])
    with chart_col:
        stats_chart = (
            alt.Chart(stats_df[stats_df["class"] != "O"])
            .mark_bar()
            .encode(
                x=alt.X("class:N", sort="-y", title=None),
                y=alt.Y("token count:Q"),
                tooltip=["class", "token count"],
            )
            .properties(height=280)
        )
        st.altair_chart(stats_chart, width="stretch")
    with table_col:
        st.dataframe(stats_df, hide_index=True, width="stretch")
    st.caption(
        "Counts collapse B-/I- prefixes (e.g. B-ORG + I-ORG -> ORG); "
        "'O' (non-entity tokens) is excluded from the chart for readability - see the table for the full count. "
        "'UNK' covers DNRTI entity types this benchmark doesn't evaluate (Time, Area, Purp, Features, ...)."
    )

    st.subheader("Overall comparison")
    chart_col, table_col = st.columns([2, 1])
    with chart_col:
        st.altair_chart(_grouped_bar(_melt_overall(overall_df), "score (0-1)", [0, 1]), width="stretch")
    with table_col:
        st.dataframe(overall_df.round(3), width="stretch")

    st.subheader("Per-class comparison")
    base = (
        alt.Chart(_melt_per_class(per_class_df))
        .mark_bar()
        .encode(
            x=alt.X("class:N", title=None),
            y=alt.Y("value:Q", title="score (0-1)", scale=alt.Scale(domain=[0, 1])),
            xOffset=alt.XOffset("model:N"),
            color=alt.Color("model:N", title="Model"),
            tooltip=["class", "model", "metric", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(width=220, height=260)
    )
    st.altair_chart(base.facet(column=alt.Column("metric:N", title=None)), width="content")
    st.dataframe(per_class_df.round(3), width="stretch")
    st.caption("Support (token/entity counts) is identical across models, since all are scored against the same gold labels - see the table above.")

    st.subheader("Latency comparison")
    chart_col, table_col = st.columns([2, 1])
    with chart_col:
        st.altair_chart(_grouped_bar(_melt_latency(latency_df), "milliseconds / sentence"), width="stretch")
    with table_col:
        st.dataframe(latency_df.round(3), width="stretch")
    st.caption(f"Evaluated on {int(latency_df['sentences'].iloc[0])} sentence(s).")

def render_comparison() -> None:
    
    run_clicked, selected_models, max_sentences_input, data_path, selected_models, log_file = _get_inputs()
    if run_clicked and selected_models:
        max_sentences = max_sentences_input or None
        try:
            overall_df, per_class_df, latency_df, entity_counts = get_outputs(data_path, max_sentences, log_file, selected_models)
        except (RuntimeError, FileNotFoundError, ValueError) as exc:
                    st.error(str(exc))
        else:
            st.session_state["comparison_results"] = {
                "overall_df": overall_df,
                "per_class_df": per_class_df,
                "latency_df": latency_df,
                "entity_counts": entity_counts,
                "params": {
                    "data_path": data_path,
                    "max_sentences": max_sentences,
                    "log_file": log_file,
                    "models": selected_models,
                },
            }

    results = st.session_state.get("comparison_results")
    if not results:
        return

    present_results(results)


def main() -> None:
    st.title("🛡️ Cybersecurity NER")
    render_comparison()


if __name__ == "__main__":
    main()
