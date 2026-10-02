"""Multi-page Streamlit data exploration and modeling application."""

from __future__ import annotations

import io
import json
import hashlib
import logging
import os
import secrets
import uuid
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sqlalchemy import select

from app.core.clean import suggest_cleaning
from app.core.ingest import DataIngestor
from app.core.model import AutoMLPipeline, TrainingResult
from app.core.profile import DataProfiler
from app.db.database import SessionLocal, init_db
from app.db.models import Dataset, ModelRun, Profile
from app.utils.auth import configured_api_key

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIRECTORY = PROJECT_ROOT / "data" / "uploads"
MODEL_DIRECTORY = PROJECT_ROOT / "data" / "models"
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}


@st.cache_data(show_spinner=False)
def cached_profile(file_bytes: bytes, file_type: str) -> dict[str, Any]:
    """Parse and profile uploaded file bytes with Streamlit data caching."""
    dataframe = DataIngestor().load_file(io.BytesIO(file_bytes), file_type)
    return DataProfiler(dataframe).profile().to_dict()


@st.cache_resource(show_spinner=False)
def cached_model(model_path: str) -> Any:
    """Cache a loaded model artifact for interactive prediction sessions."""
    return AutoMLPipeline.load(model_path)


def main() -> None:
    """Render navigation and the selected application page."""
    st.set_page_config(
        page_title="Insight Engine",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px;}
        .hero {padding: clamp(1.5rem, 5vw, 4rem); border-radius: 24px;
               background: linear-gradient(125deg, #17152a, #302765 65%, #6C5CE7);
               color: white; margin-bottom: 1.5rem;}
        .hero h1 {font-size: clamp(2rem, 5vw, 3.7rem); margin: 0 0 .5rem 0;}
        .muted {color: #a7a4bb;}
        div[data-testid="stMetric"] {background: #171A23; border: 1px solid #29263d;
          padding: .85rem 1rem; border-radius: 14px;}
        .step-card {background: #171A23; border: 1px solid #302d45; padding: 1rem;
          border-radius: 14px; min-height: 100px;}
        @media (max-width: 700px) {
          .block-container {padding-left: 1rem; padding-right: 1rem;}
          [data-testid="stMetric"] {padding: .5rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if not _authenticate_ui():
        return
    try:
        init_db()
    except Exception as exc:
        logger.exception("Could not initialize local database")
        st.error(f"Could not initialize local database: {exc}")
        return

    pages = [
        "🏠 Home",
        "📤 Upload",
        "📊 Profile",
        "🧹 Clean",
        "🤖 Train",
        "🔮 Predict",
        "🧪 What-If Simulator",
        "📁 History",
    ]
    selected = st.sidebar.radio("Navigate", pages, label_visibility="visible")
    st.sidebar.caption("Insight Engine · Explore your data")

    page_handlers = {
        "🏠 Home": _home_page,
        "📤 Upload": _upload_page,
        "📊 Profile": _profile_page,
        "🧹 Clean": _clean_page,
        "🤖 Train": _train_page,
        "🔮 Predict": _predict_page,
        "🧪 What-If Simulator": _what_if_page,
        "📁 History": _history_page,
    }
    try:
        page_handlers[selected]()
    except Exception as exc:
        logger.exception("Page %s failed", selected)
        st.error(f"Something went wrong on this page: {exc}")


def _authenticate_ui() -> bool:
    """Require the configured shared secret before exposing local app data."""
    expected = configured_api_key()
    if expected is None:
        st.error(
            "Authentication is not configured. Set INSIGHT_API_KEY to a random "
            "secret of at least 32 characters in your environment or .env file."
        )
        return False
    if st.session_state.get("authenticated", False):
        return True
    st.title("Insight Engine sign in")
    candidate = st.text_input("Access key", type="password")
    if st.button("Sign in", type="primary"):
        if secrets.compare_digest(candidate, expected):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Invalid access key.")
    return False


def _home_page() -> None:
    """Render the landing page and quick-start guidance."""
    st.markdown(
        """
        <section class="hero">
          <h1>From raw data to real insight.</h1>
          <p>Upload karo, profile samjho, model banao — all in one workspace.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Your data workflow, in one place")
    first, second, third = st.columns(3)
    first.markdown("### 📥 Ingest\nCSV, Excel, JSON, or Parquet with validation.")
    second.markdown("### 🔎 Understand\nProfiles, quality checks, and interactive charts.")
    third.markdown("### 🧠 Predict\nTrain models, compare metrics, and explore what-if scenarios.")
    st.markdown("### Quick start")
    st.write("1. Open **Upload** and choose a dataset.\n2. Explore **Profile** and review **Clean** suggestions.\n3. Train a model, then predict or experiment in the What-If Simulator.")
    if st.session_state.get("current_dataset") is not None:
        st.success(
            f"Current dataset: {st.session_state.get('current_dataset_name', 'uploaded data')}"
        )


def _upload_page() -> None:
    """Upload, validate, profile, preview, and persist the current dataset."""
    st.title("Upload a dataset")
    st.caption("Choose a data file to preview its shape, columns, sample rows, and quality checks before using it.")
    uploaded_file = st.file_uploader(
        "Choose a tabular data file",
        type=["csv", "xls", "xlsx", "xlsm", "xlsb", "json", "jsonl", "ndjson", "parquet", "pq"],
    )
    if uploaded_file is None:
        st.info("Supported formats: CSV, Excel, JSON, and Parquet.")
        return

    content = uploaded_file.getvalue()
    max_upload_size = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25")) * 1024 * 1024
    if len(content) > max_upload_size:
        st.error(
            f"File is larger than the configured {max_upload_size // (1024 * 1024)} MB upload limit."
        )
        return
    file_type = Path(uploaded_file.name).suffix.lower()
    content_digest = hashlib.sha256(content).hexdigest()
    try:
        with st.spinner("Validating and profiling dataset…"):
            dataframe = DataIngestor().load_file(io.BytesIO(content), file_type)
            profile = cached_profile(content, file_type)
            validation = DataIngestor().validate(dataframe)
    except (ValueError, OSError, ImportError) as exc:
        st.error(f"Could not load this dataset: {exc}")
        return

    if st.session_state.get("last_uploaded_file_digest") != content_digest:
        _set_current_dataset(dataframe, profile, uploaded_file.name, content_digest)
        st.session_state["current_dataset_id"] = None
        st.session_state["dataset_saved_digest"] = None
        st.session_state["last_uploaded_file_digest"] = content_digest
        st.session_state["current_dataset_was_cleaned"] = False
    elif st.session_state.get("current_dataset_digest") != content_digest:
        st.warning(
            "The working dataset has been cleaned or changed. This preview is the original uploaded file."
        )
        if st.button("Restore this uploaded file as my working dataset"):
            _set_current_dataset(dataframe, profile, uploaded_file.name, content_digest)
            st.session_state["current_dataset_id"] = None
            st.session_state["dataset_saved_digest"] = None
            st.session_state["current_dataset_was_cleaned"] = False
            st.rerun()

    st.subheader("Your data at a glance")
    _metric_row(
        {
            "Rows": f"{len(dataframe):,}",
            "Columns": len(dataframe.columns),
            "Duplicate rows": validation.duplicate_rows,
            "Missing cells": sum(validation.missing_values.values()),
        }
    )
    preview_rows = st.number_input(
        "Preview rows",
        min_value=5,
        max_value=max(5, min(len(dataframe), 500)),
        value=min(20, max(5, len(dataframe))),
        step=5,
    )
    st.dataframe(
        dataframe.head(int(preview_rows)),
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("Column names and detected data types", expanded=True):
        st.dataframe(
            pd.DataFrame(
                {
                    "Column": dataframe.columns.astype(str),
                    "Data type": dataframe.dtypes.astype(str).to_numpy(),
                    "Missing values": dataframe.isna().sum().to_numpy(),
                    "Distinct values": dataframe.nunique(dropna=True).to_numpy(),
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    with st.expander("Validation and data-quality details"):
        _render_validation(validation)
        st.metric("Data quality score", f"{profile['data_quality_score']:.1f}/100")

    active_dataset_matches_upload = (
        st.session_state.get("current_dataset_digest") == content_digest
    )
    if not active_dataset_matches_upload:
        st.info("Restore the original upload above before saving it. Your cleaned working data is still active.")
    elif st.session_state.get("dataset_saved_digest") == content_digest:
        st.success("This dataset is saved in History and ready to use.")
    elif st.button("Save dataset to History", type="primary"):
        try:
            dataset_id = _save_dataset(dataframe, uploaded_file.name, content)
            st.session_state["current_dataset_id"] = dataset_id
            st.session_state["dataset_saved_digest"] = content_digest
            st.success("Dataset saved to History. You can now train and save model runs.")
        except Exception as exc:
            logger.exception("Could not save uploaded dataset")
            st.error(f"Could not save dataset: {exc}")


def _profile_page() -> None:
    """Render overview, column, correlation, and quality profile views."""
    st.title("Dataset profile")
    dataframe, profile = _current_dataset_and_profile()
    if dataframe is None or profile is None:
        return
    tabs = st.tabs(["Overview", "Columns", "Correlations", "Quality"])

    with tabs[0]:
        _metric_row(
            {
                "Rows": profile["row_count"],
                "Columns": profile["column_count"],
                "Duplicate rows": profile["duplicate_row_count"],
                "Quality score": f"{profile['data_quality_score']:.1f}/100",
            }
        )
        missing = pd.DataFrame(
            [
                {"column": name, "missing": int(series.isna().sum()), "total": len(series)}
                for name, series in dataframe.items()
            ]
        )
        st.plotly_chart(
            px.bar(
                missing,
                x="column",
                y="missing",
                hover_data=["total"],
                title="Missing values by column",
                color_discrete_sequence=["#6C5CE7"],
            ),
            use_container_width=True,
            config=PLOT_CONFIG,
        )
        _render_distribution(dataframe)

    with tabs[1]:
        summary = pd.DataFrame(
            [
                {
                    "Column": name,
                    "Storage type": stats["dtype"],
                    "Detected meaning": stats.get("semantic_type", "unknown"),
                    "Missing (%)": stats["null_pct"],
                    "Distinct values": stats["unique_count"],
                    "Distinct (%)": stats["unique_pct"],
                }
                for name, stats in profile["columns"].items()
            ]
        )
        st.caption("Every column, its detected meaning, missing-value rate, and distinct-value count.")
        st.dataframe(summary, use_container_width=True, hide_index=True)
        chosen = st.selectbox("Inspect a column", dataframe.columns.astype(str))
        selected = dataframe[chosen]
        details = profile["columns"][chosen]
        st.subheader(f"{chosen} · {details.get('semantic_type', 'column details')}")
        detail_rows = [
            {
                "Statistic": key.replace("_", " ").title(),
                "Value": (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, dict | list)
                    else str(value)
                ),
            }
            for key, value in details.items()
        ]
        st.dataframe(
            pd.DataFrame(detail_rows),
            use_container_width=True,
            hide_index=True,
        )
        if pd.api.types.is_numeric_dtype(selected.dtype):
            chart = make_subplots(
                rows=2, cols=1, shared_xaxes=False, row_heights=[0.7, 0.3],
                subplot_titles=("Histogram", "Box plot"),
            )
            chart.add_trace(
                go.Histogram(x=selected.dropna(), name="Distribution", marker_color="#6C5CE7"),
                row=1, col=1,
            )
            chart.add_trace(
                go.Box(x=selected.dropna(), name="Spread", marker_color="#00CEC9"),
                row=2, col=1,
            )
            st.plotly_chart(chart, use_container_width=True, config=PLOT_CONFIG)
        elif details.get("top_values"):
            st.plotly_chart(
                px.bar(
                    pd.DataFrame(details["top_values"]),
                    x="value",
                    y="count",
                    title=f"Most common values in {chosen}",
                    labels={"value": chosen, "count": "Rows"},
                    color_discrete_sequence=["#6C5CE7"],
                ),
                use_container_width=True,
                config=PLOT_CONFIG,
            )
        st.download_button(
            "Download complete profile as JSON",
            data=json.dumps(profile, indent=2, ensure_ascii=False),
            file_name="insight-engine-profile.json",
            mime="application/json",
        )

    with tabs[2]:
        correlations = dataframe.select_dtypes(include=np.number).corr()
        if correlations.empty:
            st.info("No numeric columns are available for correlation analysis.")
        else:
            st.plotly_chart(
                px.imshow(
                    correlations,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    aspect="auto",
                    title="Numeric correlation heatmap",
                ),
                use_container_width=True,
                config=PLOT_CONFIG,
            )

    with tabs[3]:
        st.metric("Data quality", f"{profile['data_quality_score']:.1f}/100")
        st.write(
            f"Potential targets: {', '.join(profile['potential_target_columns']) or 'None detected'}"
        )
        st.write(
            f"Potential ID columns: {', '.join(profile['potential_id_columns']) or 'None detected'}"
        )
        validation = st.session_state.get("validation_report")
        if validation:
            st.json(validation)
        st.download_button(
            "Download Report (PDF)",
            data=_build_pdf_report(profile, _current_metrics()),
            file_name="insight-engine-report.pdf",
            mime="application/pdf",
            type="primary",
        )


def _clean_page() -> None:
    """Preview selected cleaning operations without mutating the source dataset."""
    st.title("Clean your data")
    st.caption("Choose only the fixes that fit your data. The original stays unchanged until you apply the preview.")
    dataframe = st.session_state.get("current_dataset")
    if dataframe is None:
        st.info("Upload a dataset first.")
        return
    suggestions = suggest_cleaning(dataframe)
    if suggestions:
        with st.expander(f"Suggested fixes ({len(suggestions)})", expanded=True):
            for suggestion in suggestions:
                st.write(f"**{suggestion['column'] or 'Whole dataset'}:** {suggestion['message']}")
    else:
        st.success("No common data-quality issues were detected.")

    duplicate_count = int(dataframe.duplicated().sum())
    constant_columns = [
        str(column)
        for column in dataframe.columns
        if dataframe[column].nunique(dropna=False) <= 1
    ]
    missing_columns = [
        str(column) for column in dataframe.columns if dataframe[column].isna().any()
    ]
    text_columns = [
        str(column)
        for column in dataframe.select_dtypes(include=["object", "string"]).columns
    ]
    dtype_options: list[tuple[str, str, str]] = []
    for column in text_columns:
        values = dataframe[column].dropna()
        if values.empty:
            continue
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.notna().mean() >= 0.8:
            dtype_options.append((f"{column} → number", column, "numeric"))
        parsed_dates = pd.to_datetime(values, errors="coerce", format="mixed")
        if parsed_dates.notna().mean() >= 0.8:
            dtype_options.append((f"{column} → date/time", column, "datetime"))

    st.subheader("Choose changes")
    drop_duplicates = st.checkbox(
        f"Remove duplicate rows ({duplicate_count:,} found)",
        disabled=duplicate_count == 0,
    )
    drop_constant = st.checkbox(
        f"Remove constant columns ({len(constant_columns)} found)",
        disabled=not constant_columns,
    )
    drop_columns = st.multiselect(
        "Remove columns you do not need",
        options=list(dataframe.columns.astype(str)),
        help="This does not remove rows. Avoid dropping columns that contain useful information.",
    )
    fill_columns = st.multiselect(
        "Fill missing values in selected columns",
        options=missing_columns,
        default=missing_columns,
        help="Choose an appropriate fill method for each selected column below.",
    )
    fill_methods: dict[str, tuple[str, Any]] = {}
    for column in fill_columns:
        series = dataframe[column]
        if pd.api.types.is_bool_dtype(series.dtype):
            strategy = st.selectbox(
                f"How should missing values in {column} be filled?",
                ["Most frequent value", "Set all missing values to True", "Set all missing values to False"],
                key=f"clean_method_{column}",
            )
            if strategy == "Most frequent value":
                fill_methods[column] = ("mode", None)
            else:
                fill_methods[column] = (
                    "constant",
                    strategy.endswith("True"),
                )
        elif pd.api.types.is_datetime64_any_dtype(series.dtype):
            strategy = st.selectbox(
                f"How should missing values in {column} be filled?",
                ["Most frequent date", "Use a date"],
                key=f"clean_method_{column}",
            )
            if strategy == "Most frequent date":
                fill_methods[column] = ("mode", None)
            else:
                fill_methods[column] = (
                    "constant",
                    pd.Timestamp(
                        st.date_input(
                            f"Replacement date for {column}",
                            key=f"clean_value_{column}",
                        )
                    ),
                )
        elif pd.api.types.is_numeric_dtype(series.dtype):
            strategy = st.selectbox(
                f"How should missing values in {column} be filled?",
                ["Median", "Mean", "Use a fixed value"],
                key=f"clean_method_{column}",
            )
            if strategy == "Median":
                fill_methods[column] = ("median", None)
            elif strategy == "Mean":
                fill_methods[column] = ("mean", None)
            else:
                fill_methods[column] = (
                    "constant",
                    st.number_input(
                        f"Replacement value for {column}",
                        value=0.0,
                        key=f"clean_value_{column}",
                    ),
                )
        else:
            strategy = st.selectbox(
                f"How should missing values in {column} be filled?",
                ["Most frequent value", "Use a fixed value"],
                key=f"clean_method_{column}",
            )
            if strategy == "Most frequent value":
                fill_methods[column] = ("mode", None)
            else:
                fill_methods[column] = (
                    "constant",
                    st.text_input(
                        f"Replacement value for {column}",
                        key=f"clean_value_{column}",
                    ),
                )

    dtype_labels = [item[0] for item in dtype_options]
    selected_conversions = st.multiselect(
        "Convert columns that look like numbers or dates",
        options=dtype_labels,
        help="Only conversions that successfully parse most existing values are suggested.",
    )
    trim_columns = st.multiselect(
        "Trim extra spaces from text columns",
        options=text_columns,
        help="For example, changes ' Lucknow ' to 'Lucknow'.",
    )

    cleaned = dataframe.copy()
    if drop_duplicates:
        cleaned = cleaned.drop_duplicates()
    for column, (method, replacement) in fill_methods.items():
        if method == "median":
            replacement = cleaned[column].median()
        elif method == "mean":
            replacement = cleaned[column].mean()
        elif method == "mode":
            modes = cleaned[column].mode(dropna=True)
            replacement = modes.iloc[0] if not modes.empty else None
        if replacement is not None and pd.notna(replacement):
            cleaned[column] = cleaned[column].fillna(replacement)
    conversions = {label: (column, kind) for label, column, kind in dtype_options}
    for label in selected_conversions:
        if conversions[label][0] not in cleaned.columns:
            continue
        column, kind = conversions[label]
        if kind == "numeric":
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        else:
            cleaned[column] = pd.to_datetime(
                cleaned[column], errors="coerce", format="mixed"
            )
    for column in trim_columns:
        if column not in cleaned.columns:
            continue
        cleaned[column] = cleaned[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    if drop_constant:
        cleaned = cleaned.drop(columns=constant_columns, errors="ignore")
    if drop_columns:
        cleaned = cleaned.drop(columns=drop_columns, errors="ignore")

    before, after = st.columns(2)
    before.subheader("Before cleaning")
    before.caption(f"{len(dataframe):,} rows · {len(dataframe.columns)} columns")
    before.dataframe(dataframe.head(25), use_container_width=True, hide_index=True)
    after.subheader("Preview after selected changes")
    after.caption(f"{len(cleaned):,} rows · {len(cleaned.columns)} columns")
    after.dataframe(cleaned.head(25), use_container_width=True, hide_index=True)
    before_quality = DataProfiler(dataframe).profile().data_quality_score
    after_quality = DataProfiler(cleaned).profile().data_quality_score
    first, second, third = st.columns(3)
    first.metric("Rows removed", f"{len(dataframe) - len(cleaned):,}")
    second.metric("Columns removed", len(dataframe.columns) - len(cleaned.columns))
    third.metric("Quality score", f"{before_quality:.1f} → {after_quality:.1f}")
    if st.button("Apply these changes to my working data", type="primary"):
        _set_current_dataset(
            cleaned,
            DataProfiler(cleaned).profile().to_dict(),
            f"{Path(st.session_state.get('current_dataset_name', 'dataset')).stem}_cleaned.csv",
            uuid.uuid4().hex,
        )
        st.session_state["current_dataset_id"] = None
        st.session_state["dataset_saved_digest"] = None
        st.session_state["current_dataset_was_cleaned"] = True
        st.success("Cleaned data is now active. Retrain your model to use these changes.")
    if st.session_state.get("current_dataset_was_cleaned"):
        if st.button("Save cleaned data to History"):
            try:
                current = st.session_state["current_dataset"]
                csv_content = current.to_csv(index=False).encode("utf-8")
                dataset_id = _save_dataset(
                    current,
                    st.session_state.get("current_dataset_name", "cleaned-data.csv"),
                    csv_content,
                )
                st.session_state["current_dataset_id"] = dataset_id
                st.session_state["current_dataset_was_cleaned"] = False
                st.success("Cleaned dataset saved to History.")
            except Exception as exc:
                logger.exception("Could not save cleaned dataset")
                st.error(f"Could not save cleaned dataset: {exc}")


def _train_page() -> None:
    """Train a CV-selected model and render its metrics and diagnostics."""
    st.title("Train a model")
    dataframe = st.session_state.get("current_dataset")
    if dataframe is None:
        st.info("Upload a dataset first.")
        return
    st.caption("Choose the result you want to predict. We compare candidate models using cross-validation and show how well they performed.")
    columns = dataframe.columns.astype(str).tolist()
    profile = st.session_state.get("current_profile", {})
    suggested_targets = profile.get("potential_target_columns", [])
    default_target = next(
        (columns.index(name) for name in suggested_targets if name in columns), 0
    )
    target = st.selectbox(
        "What should the model predict?",
        columns,
        index=default_target,
        format_func=lambda name: (
            f"{name} · {dataframe[name].dtype} · "
            f"{dataframe[name].nunique(dropna=True):,} distinct values"
        ),
    )
    task_labels = {
        "Let the app decide": "auto",
        "Classification · choose a category": "classification",
        "Regression · predict a number": "regression",
    }
    task_label = st.selectbox(
        "What kind of result is this?",
        list(task_labels),
        help="Use classification for labels such as hired/not hired; regression for numeric values such as salary or score.",
    )
    algorithm_labels = {
        "Compare models automatically (recommended)": "auto",
        "Logistic regression": "logistic_regression",
        "Ridge regression": "ridge",
        "Random forest": "random_forest",
        "Gradient boosting": "gradient_boosting",
    }
    algorithm_label = st.selectbox(
        "Which model should we try?",
        list(algorithm_labels),
        help="Automatic comparison tests suitable model types and selects the strongest cross-validation score.",
    )
    possible_ids = [
        name for name in profile.get("potential_id_columns", []) if name != target
    ]
    excluded_features = st.multiselect(
        "Exclude columns from model training",
        options=[name for name in columns if name != target],
        default=[name for name in possible_ids if name in columns],
        help="Exclude identifiers or columns that would not be available when making a real prediction. This also helps avoid misleadingly high evaluation scores.",
    )
    if possible_ids:
        st.caption("Possible identifier columns to review: " + ", ".join(possible_ids))
    st.info(f"Selected target: **{target}**. It has {dataframe[target].nunique(dropna=True):,} distinct values.")
    button = st.button("Train and evaluate model", type="primary")
    if button:
        progress = st.progress(0, text="Preparing cross-validation…")

        def update_progress(current: int, total: int, name: str) -> None:
            progress.progress(
                current / total,
                text=f"Evaluating {name} · {current}/{total}",
            )

        try:
            with st.spinner("Training candidate models…"):
                training_data = dataframe.drop(columns=excluded_features)
                result = AutoMLPipeline().train(
                    training_data,
                    target,
                    task_type=task_labels[task_label],
                    algorithm=algorithm_labels[algorithm_label],
                    progress_callback=update_progress,
                )
            st.session_state["current_training_result"] = result
            st.session_state["current_training_dataset_token"] = st.session_state.get(
                "current_dataset_token"
            )
            st.session_state["current_metrics"] = result.metrics
            st.session_state["current_feature_columns"] = result.feature_columns
            _persist_model_run(result)
            progress.progress(1.0, text=f"Selected {result.algorithm}")
            st.success(f"Training complete. Selected **{result.algorithm}**.")
        except (ValueError, OSError) as exc:
            logger.exception("Model training failed")
            st.error(f"Could not train this model: {exc}")
            return

    result = st.session_state.get("current_training_result")
    if result is None:
        st.caption("Choose a target and train the model to see its evaluation here.")
        return
    st.subheader(f"Selected model: {result.algorithm.replace('_', ' ').title()}")
    st.caption(f"Predicting **{result.target_column}** · task: **{result.task.title()}**")
    _render_performance_summary(
        result.task,
        result.metrics,
        result.actual_values,
        result.predicted_values,
        result.class_labels,
        result.confusion_matrix,
    )
    if result.cv_scores:
        st.subheader("Models compared")
        cv_rows = [
            {
                "Model": name.replace("_", " ").title(),
                "Cross-validation score": (
                    f"{score * 100:.1f}%"
                    if result.task == "classification"
                    else f"{score:.4f} R²"
                ),
                "Selected": name == result.algorithm,
            }
            for name, score in result.cv_scores.items()
        ]
        st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)
        st.caption("Scores are averages over held-out folds. They estimate performance; they do not guarantee results on future data.")
    if result.feature_importances:
        importance = pd.DataFrame(
            result.feature_importances.items(), columns=["feature", "importance"]
        ).sort_values("importance", ascending=True).tail(25)
        st.plotly_chart(
            px.bar(
                importance, x="importance", y="feature", orientation="h",
                title="Which inputs influenced the model most", color_discrete_sequence=["#6C5CE7"],
            ),
            use_container_width=True,
            config=PLOT_CONFIG,
        )
    if result.confusion_matrix is not None:
        matrix = np.asarray(result.confusion_matrix)
        labels = result.class_labels or [str(index) for index in range(len(matrix))]
        st.plotly_chart(
            px.imshow(
                matrix,
                text_auto=True,
                x=labels,
                y=labels,
                color_continuous_scale="Purples",
                labels={"x": "Predicted", "y": "Actual", "color": "Count"},
                title="Cross-validated confusion matrix",
            ),
            use_container_width=True,
            config=PLOT_CONFIG,
        )
    if result.roc_curve:
        roc = result.roc_curve
        chart = go.Figure()
        chart.add_trace(
            go.Scatter(
                x=roc["false_positive_rate"],
                y=roc["true_positive_rate"],
                mode="lines",
                name="ROC",
                line={"color": "#6C5CE7", "width": 3},
            )
        )
        chart.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line={"dash": "dash"})
        )
        chart.update_layout(title="ROC curve", xaxis_title="False positive rate",
                            yaxis_title="True positive rate")
        st.plotly_chart(chart, use_container_width=True, config=PLOT_CONFIG)
    if result.actual_values is not None and result.predicted_values is not None:
        comparison = pd.DataFrame(
            {"actual": result.actual_values, "predicted": result.predicted_values}
        )
        chart = px.scatter(
            comparison, x="actual", y="predicted", title="Actual vs predicted",
            color_discrete_sequence=["#6C5CE7"],
        )
        low = float(min(comparison.min()))
        high = float(max(comparison.max()))
        chart.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", name="Ideal"))
        st.plotly_chart(chart, use_container_width=True, config=PLOT_CONFIG)
    if result.residuals:
        st.plotly_chart(
            px.histogram(
                x=result.residuals, nbins=30, title="Cross-validated residuals",
                labels={"x": "Residual"},
                color_discrete_sequence=["#00CEC9"],
            ),
            use_container_width=True,
            config=PLOT_CONFIG,
        )
    profile = st.session_state.get("current_profile")
    if profile:
        st.download_button(
            "Download Report (PDF)",
            data=_build_pdf_report(profile, result.metrics),
            file_name="insight-engine-report.pdf",
            mime="application/pdf",
        )


def _predict_page() -> None:
    """Generate predictions from a session-trained or saved model."""
    st.title("Predict")
    dataframe = st.session_state.get("current_dataset")
    if dataframe is None:
        st.info("Upload the dataset you want to use first. Predictions are matched to the active dataset.")
        return
    active_token = st.session_state.get("current_dataset_token")
    training_result = st.session_state.get("current_training_result")
    if st.session_state.get("current_training_dataset_token") != active_token:
        training_result = None
    dataset_id = st.session_state.get("current_dataset_id")
    runs = [
        run
        for run in _load_model_runs()
        if dataset_id is not None and run.dataset_id == dataset_id
    ]
    if not runs and training_result is None:
        st.info("No model has been trained for this dataset yet. Open **Train**, choose a target, and train one to continue.")
        return
    labels: dict[str, ModelRun | TrainingResult] = {}
    if training_result is not None:
        labels[
            f"Current session · {training_result.algorithm} · "
            f"target {training_result.target_column}"
        ] = training_result
    labels.update(
        {
            f"#{run.id} · {run.algorithm} · target {run.target_column}": run
            for run in runs
        }
    )
    selection = st.selectbox("Trained model", list(labels), key="predict_model")
    selected_model = labels[selection]
    if isinstance(selected_model, TrainingResult):
        feature_columns = selected_model.feature_columns
        model = selected_model.pipeline
        target_column = selected_model.target_column
        task = selected_model.task
        metrics = selected_model.metrics
        cv_scores = selected_model.cv_scores
        actual_values = selected_model.actual_values
        predicted_values = selected_model.predicted_values
        confusion_matrix = selected_model.confusion_matrix
        class_labels = selected_model.class_labels
        st.caption("Using the model trained in this browser session.")
    else:
        run = selected_model
        target_column = run.target_column
        task = run.task_type
        feature_columns = run.metrics_json.get("feature_columns", [])
        model_path = Path(run.model_path)
        if not model_path.is_file():
            st.error("This model file is no longer available. On free hosting, saved files may be removed when the service restarts. Please train it again.")
            return
        model = _load_cached_model(str(model_path))
        metrics = run.metrics_json.get("metrics", {})
        cv_scores = run.metrics_json.get("cv_scores", {})
        actual_values = run.metrics_json.get("actual_values")
        predicted_values = run.metrics_json.get("predicted_values")
        confusion_matrix = run.metrics_json.get("confusion_matrix")
        class_labels = run.metrics_json.get("class_labels")

    st.subheader("How this model performed")
    _render_performance_summary(task, metrics)
    with st.expander("See where the model was right or made mistakes", expanded=True):
        _render_error_analysis(
            task,
            actual_values,
            predicted_values,
            confusion_matrix,
            class_labels,
        )
    if cv_scores:
        with st.expander("Compare cross-validation results"):
            score_rows = [
                {
                    "Model": name.replace("_", " ").title(),
                    "Score": (
                        f"{score * 100:.1f}%"
                        if task == "classification"
                        else f"{score:.4f} R²"
                    ),
                    "Selected model": name == (
                        selected_model.algorithm
                        if isinstance(selected_model, TrainingResult)
                        else selected_model.algorithm
                    ),
                }
                for name, score in cv_scores.items()
            ]
            st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)
    st.caption("These are cross-validation estimates on your dataset, not a guarantee of future predictions.")
    st.subheader("Try a prediction")
    st.write("Enter a new example below. Numeric fields allow values outside the training-data range.")
    values = _feature_form(
        dataframe,
        feature_columns,
        key_prefix="predict",
        allow_outside_training_range=True,
    )
    compare_actual = st.checkbox("I know the real answer and want to compare it")
    actual_value: Any = None
    if compare_actual:
        target_values = dataframe[target_column].dropna()
        if task == "classification":
            actual_options = sorted(target_values.astype(str).unique().tolist())
            if not actual_options:
                st.warning("No actual target values are available for comparison.")
            else:
                actual_value = st.selectbox(
                    f"Real {target_column}",
                    actual_options,
                    key="predict_actual_class",
                )
        elif pd.api.types.is_numeric_dtype(dataframe[target_column].dtype):
            actual_value = st.number_input(
                f"Real {target_column}",
                value=float(target_values.iloc[0]) if not target_values.empty else 0.0,
                key="predict_actual_number",
            )
        else:
            actual_value = st.text_input(f"Real {target_column}", key="predict_actual_text")
    if st.button("Predict this example", type="primary"):
        try:
            result = AutoMLPipeline.predict(model, values)
            st.session_state["last_prediction"] = result
            st.success(f"Predicted **{target_column}**: **{result['prediction']}**")
            if "probabilities" in result:
                probabilities = pd.DataFrame(
                    [
                        {"Possible result": label, "Model confidence": probability}
                        for label, probability in result["probabilities"].items()
                    ]
                ).sort_values("Model confidence", ascending=False)
                st.dataframe(
                    probabilities.style.format({"Model confidence": "{:.1%}"}),
                    use_container_width=True,
                    hide_index=True,
                )
            elif result.get("confidence_interval") and result["confidence_interval"][0] is not None:
                low, high = result["confidence_interval"]
                st.caption(f"Approximate prediction range: {low:.3f} to {high:.3f} (target units).")
            if compare_actual and actual_value is not None:
                predicted_value = result["prediction"]
                if task == "classification":
                    matches = str(predicted_value) == str(actual_value)
                    st.metric("Comparison with actual result", "Correct" if matches else "Incorrect")
                else:
                    error = abs(float(predicted_value) - float(actual_value))
                    st.metric("Absolute prediction error", f"{error:.4f} target units")
        except (ValueError, TypeError, KeyError) as exc:
            st.error(f"Prediction failed: {exc}")


def _what_if_page() -> None:
    """Provide interactive model predictions and one-feature sensitivity."""
    st.title("What-If Simulator")
    st.caption("Explore how your inputs change a prediction. Numeric values are not restricted to the training-data range.")
    dataframe = st.session_state.get("current_dataset")
    if dataframe is None:
        st.info("Upload the dataset you want to simulate first.")
        return
    active_token = st.session_state.get("current_dataset_token")
    training_result = st.session_state.get("current_training_result")
    if st.session_state.get("current_training_dataset_token") != active_token:
        training_result = None
    dataset_id = st.session_state.get("current_dataset_id")
    runs = [
        run
        for run in _load_model_runs()
        if dataset_id is not None and run.dataset_id == dataset_id
    ]
    labels: dict[str, ModelRun | TrainingResult] = {}
    if training_result is not None:
        labels[
            f"Current session · {training_result.algorithm} · "
            f"target {training_result.target_column}"
        ] = training_result
    labels.update(
        {
            f"#{run.id} · {run.algorithm} · {run.target_column}": run
            for run in runs
        }
    )
    if not labels:
        st.info("Train a model on this dataset to unlock the simulator.")
        return
    selection = st.selectbox("Choose a trained model", list(labels), key="whatif_model")
    selected_model = labels[selection]
    if isinstance(selected_model, TrainingResult):
        feature_columns = selected_model.feature_columns
        model = selected_model.pipeline
    else:
        feature_columns = selected_model.metrics_json.get("feature_columns", [])
        if not Path(selected_model.model_path).is_file():
            st.error("This model file is no longer available. Please train it again.")
            return
        model = _load_cached_model(selected_model.model_path)
    baseline = _baseline_features(dataframe, feature_columns)
    values = _feature_form(
        dataframe,
        feature_columns,
        key_prefix=(
            f"whatif_{selected_model.id}"
            if isinstance(selected_model, ModelRun)
            else "whatif_session"
        ),
        defaults=baseline,
        allow_outside_training_range=True,
    )
    try:
        current_result = AutoMLPipeline.predict(model, values)
        baseline_result = AutoMLPipeline.predict(model, baseline)
        current = _numeric_prediction(current_result)
        base = _numeric_prediction(baseline_result)
        delta = current - base
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        first, second, third = st.columns(3)
        first.metric("Prediction", str(current_result["prediction"]))
        second.metric("Baseline", str(baseline_result["prediction"]))
        third.metric("Change vs baseline", f"{arrow} {delta:+.4f}")

        sensitivity_rows: list[dict[str, Any]] = []
        for column in feature_columns:
            changed = dict(baseline)
            changed[column] = values[column]
            changed_result = AutoMLPipeline.predict(model, changed)
            sensitivity_rows.append(
                {
                    "feature": column,
                    "impact": _numeric_prediction(changed_result) - base,
                }
            )
        sensitivity = pd.DataFrame(sensitivity_rows).sort_values(
            "impact", key=lambda values_: values_.abs()
        )
        st.plotly_chart(
            px.bar(
                sensitivity,
                x="impact",
                y="feature",
                orientation="h",
                title="Sensitivity · one feature changed at a time",
                color="impact",
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
            ),
            use_container_width=True,
            config=PLOT_CONFIG,
        )
    except (ValueError, TypeError, KeyError) as exc:
        st.error(f"Could not simulate this input: {exc}")


def _history_page() -> None:
    """List persisted datasets and model runs from SQLite."""
    st.title("History")
    with SessionLocal() as session:
        datasets = session.scalars(
            select(Dataset).order_by(Dataset.uploaded_at.desc())
        ).all()
        model_runs = session.scalars(
            select(ModelRun).order_by(ModelRun.created_at.desc())
        ).all()
        datasets_data = [
            {
                "id": item.id,
                "name": item.name,
                "uploaded": item.uploaded_at,
                "rows": item.row_count,
                "columns": item.column_count,
                "owner": item.owner,
            }
            for item in datasets
        ]
        model_data = [
            {
                "id": item.id,
                "dataset": item.dataset_id,
                "target": item.target_column,
                "task": item.task_type,
                "algorithm": item.algorithm,
                "status": item.status,
                **item.metrics_json.get("metrics", {}),
            }
            for item in model_runs
        ]
    st.subheader("Datasets")
    if datasets_data:
        st.dataframe(pd.DataFrame(datasets_data), use_container_width=True, hide_index=True)
    else:
        st.info("No datasets have been saved.")
    st.subheader("Model runs")
    if model_data:
        st.dataframe(pd.DataFrame(model_data), use_container_width=True, hide_index=True)
    else:
        st.info("No model runs have been saved.")


def _current_dataset_and_profile() -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    """Get current data/profile from session state, computing the profile if needed."""
    dataframe = st.session_state.get("current_dataset")
    if dataframe is None:
        st.info("Upload a dataset first.")
        return None, None
    profile = st.session_state.get("current_profile")
    if profile is None:
        with st.spinner("Profiling data…"):
            profile = DataProfiler(dataframe).profile().to_dict()
        st.session_state["current_profile"] = profile
    return dataframe, profile


def _reset_model_state() -> None:
    """Remove model results that no longer match the active working dataset."""
    for key in (
        "current_training_result",
        "current_training_dataset_token",
        "current_metrics",
        "current_feature_columns",
        "current_model_run_id",
        "current_model_path",
        "last_prediction",
    ):
        st.session_state.pop(key, None)


def _set_current_dataset(
    dataframe: pd.DataFrame,
    profile: dict[str, Any],
    name: str,
    digest: str,
) -> None:
    """Switch all UI pages to one dataset revision and invalidate stale models."""
    report = DataIngestor().validate(dataframe)
    st.session_state["current_dataset"] = dataframe
    st.session_state["current_profile"] = profile
    st.session_state["current_dataset_name"] = name
    st.session_state["current_dataset_digest"] = digest
    st.session_state["current_dataset_token"] = uuid.uuid4().hex
    st.session_state["validation_report"] = {
        "row_count": report.row_count,
        "column_count": report.column_count,
        "missing_values": report.missing_values,
        "duplicate_rows": report.duplicate_rows,
        "dtype_mismatches": report.dtype_mismatches,
        "constant_columns": report.constant_columns,
        "high_cardinality_categorical_columns": (
            report.high_cardinality_categorical_columns
        ),
    }
    st.session_state["current_dataset_was_cleaned"] = False
    _reset_model_state()


def _save_dataset(dataframe: pd.DataFrame, name: str, content: bytes) -> int:
    """Persist a dataset file and its profile, returning its history identifier."""
    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    safe_name = Path(name).name
    destination = UPLOAD_DIRECTORY / f"{uuid.uuid4().hex}_{safe_name}"
    profile = DataProfiler(dataframe).profile().to_dict()
    destination.write_bytes(content)
    try:
        with SessionLocal() as session:
            dataset = Dataset(
                name=safe_name,
                row_count=len(dataframe),
                column_count=len(dataframe.columns),
                file_path=str(destination),
                owner="local",
            )
            dataset.profiles.append(Profile(profile_json=profile))
            session.add(dataset)
            session.commit()
            session.refresh(dataset)
            return dataset.id
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def _render_performance_summary(task: str, metrics: dict[str, Any]) -> None:
    """Explain cross-validated model metrics in user-friendly terms."""
    if task == "classification":
        displayed = {
            "Correct answers on validation data": (
                f"{metrics.get('accuracy', 0.0):.1%}"
            ),
            "Balanced quality across classes (F1)": f"{metrics.get('f1', 0.0):.1%}",
        }
        if "roc_auc" in metrics:
            displayed["Ability to separate classes (ROC-AUC)"] = (
                f"{metrics['roc_auc']:.1%}"
            )
        _metric_row(displayed)
        st.caption("Accuracy is the percentage of validation examples predicted correctly. F1 and ROC-AUC summarize other aspects of classification quality.")
        if metrics.get("accuracy", 0.0) >= 0.99:
            st.warning("This score is exceptionally high. Check that identifier columns and information derived from the target were excluded, then test on separate future data.")
    else:
        r2 = float(metrics.get("r2", 0.0))
        _metric_row(
            {
                "Variation explained (R²)": f"{r2:.1%}",
                "Typical absolute error (MAE)": f"{metrics.get('mae', 0.0):.4g} target units",
                "Error with extra penalty for large misses (RMSE)": (
                    f"{metrics.get('rmse', 0.0):.4g} target units"
                ),
            }
        )
        st.caption("Regression does not have an accuracy percentage. R² describes variation explained; MAE and RMSE show prediction error in the target column's units.")
        if r2 >= 0.99:
            st.warning("This R² is exceptionally high. Check for target leakage or identifiers, and confirm performance on separate future data before relying on it.")


def _render_error_analysis(
    task: str,
    actual_values: list[Any] | None,
    predicted_values: list[Any] | None,
    confusion: list[list[int]] | None,
    class_labels: list[str] | None,
) -> None:
    """Show cross-validated mistakes so users can inspect model failure patterns."""
    if task == "classification":
        if confusion:
            labels = class_labels or [str(index) for index in range(len(confusion))]
            matrix = np.asarray(confusion)
            st.plotly_chart(
                px.imshow(
                    matrix,
                    text_auto=True,
                    x=labels,
                    y=labels,
                    color_continuous_scale="Purples",
                    labels={"x": "Model prediction", "y": "Actual result", "color": "Rows"},
                    title="Correct and incorrect predictions by class",
                ),
                use_container_width=True,
                config=PLOT_CONFIG,
            )
            st.caption("Diagonal cells are correct predictions. Off-diagonal cells show which classes the model confuses.")
        return
    if not actual_values or not predicted_values:
        st.info("No saved validation examples are available to inspect.")
        return
    comparisons = pd.DataFrame(
        {"Actual result": actual_values, "Model prediction": predicted_values}
    )
    comparisons["Absolute error"] = (
        comparisons["Actual result"] - comparisons["Model prediction"]
    ).abs()
    st.caption("Examples with the largest validation errors:")
    st.dataframe(
        comparisons.sort_values("Absolute error", ascending=False).head(10),
        use_container_width=True,
        hide_index=True,
    )
    st.plotly_chart(
        px.histogram(
            comparisons,
            x="Absolute error",
            nbins=30,
            title="How large were the validation errors?",
            color_discrete_sequence=["#6C5CE7"],
        ),
        use_container_width=True,
        config=PLOT_CONFIG,
    )


def _render_validation(report: Any) -> None:
    """Display structured validation findings in a compact layout."""
    _metric_row(
        {
            "Rows": report.row_count,
            "Columns": report.column_count,
            "Duplicate rows": report.duplicate_rows,
        }
    )
    st.write("**Missing values per column**")
    st.json(report.missing_values)
    if report.dtype_mismatches:
        st.warning("Possible dtype mismatches detected.")
        st.json(report.dtype_mismatches)
    st.write("Constant columns:", ", ".join(report.constant_columns) or "None")
    st.write(
        "High-cardinality categorical columns:",
        ", ".join(report.high_cardinality_categorical_columns) or "None",
    )


def _render_distribution(dataframe: pd.DataFrame) -> None:
    """Render a responsive histogram and box plot for a selected numeric column."""
    numeric_columns = dataframe.select_dtypes(include=np.number).columns.tolist()
    if not numeric_columns:
        st.info("No numeric columns are available for distribution plots.")
        return
    column = st.selectbox("Distribution column", numeric_columns, key="profile_dist_column")
    series = dataframe[column].dropna()
    charts = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.68, 0.32],
        subplot_titles=("Histogram", "Box plot"),
    )
    charts.add_trace(
        go.Histogram(x=series, name="Frequency", marker_color="#6C5CE7"),
        row=1,
        col=1,
    )
    charts.add_trace(
        go.Box(x=series, name="Range", marker_color="#00CEC9"),
        row=2,
        col=1,
    )
    charts.update_layout(height=520, showlegend=False)
    st.plotly_chart(charts, use_container_width=True, config=PLOT_CONFIG)


def _metric_row(metrics: dict[str, Any]) -> None:
    """Render metric values as a responsive Streamlit row."""
    columns = st.columns(max(1, min(len(metrics), 4)))
    for index, (name, value) in enumerate(metrics.items()):
        columns[index % len(columns)].metric(
            str(name).replace("_", " ").title(),
            f"{value:.4f}" if isinstance(value, float) else str(value),
        )


def _build_pdf_report(
    profile: dict[str, Any], metrics: dict[str, Any] | None
) -> bytes:
    """Render a profile and optional model metrics as a PDF document."""
    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=letter, title="Insight Engine Report")
    styles = getSampleStyleSheet()
    content: list[Any] = [
        Paragraph("Insight Engine · Dataset Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph(
            f"Rows: {profile['row_count']} &nbsp; | &nbsp; "
            f"Columns: {profile['column_count']} &nbsp; | &nbsp; "
            f"Quality score: {profile['data_quality_score']:.1f}/100",
            styles["BodyText"],
        ),
        Spacer(1, 14),
    ]
    column_rows = [["Column", "Type", "Missing %", "Unique"]]
    detail_rows = [["Column", "Full profile statistics"]]
    for name, item in profile["columns"].items():
        column_rows.append(
            [
                str(name)[:35],
                str(item.get("semantic_type", item.get("dtype", "")))[:20],
                f"{item.get('null_pct', 0):.1f}%",
                str(item.get("unique_count", 0)),
            ]
        )
        detail_json = json.dumps(item, ensure_ascii=False, indent=2)
        detail_rows.append(
            [
                str(name)[:35],
                Paragraph(
                    detail_json.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace("\n", "<br/>"),
                    styles["Code"],
                ),
            ]
        )
    table = Table(column_rows, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6C5CE7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D8D6E8")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F0FA")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    content.extend(
        [
            Paragraph("Column summary", styles["Heading2"]),
            table,
            Spacer(1, 14),
            Paragraph("Full column statistics", styles["Heading2"]),
        ]
    )
    detail_table = Table(detail_rows, colWidths=[110, 400], repeatRows=1)
    detail_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6C5CE7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D8D6E8")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F0FA")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    content.extend([detail_table, Spacer(1, 14)])
    dataset_details = {
        "correlation_matrix": profile.get("correlation_matrix", {}),
        "potential_target_columns": profile.get("potential_target_columns", []),
        "potential_id_columns": profile.get("potential_id_columns", []),
        "data_quality_score": profile.get("data_quality_score"),
        "duplicate_row_count": profile.get("duplicate_row_count"),
    }
    content.extend(
        [
            Paragraph("Dataset-level statistics", styles["Heading2"]),
            Paragraph(
                json.dumps(dataset_details, ensure_ascii=False, indent=2)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br/>"),
                styles["Code"],
            ),
        ]
    )
    if metrics:
        content.extend(
            [
                Spacer(1, 16),
                Paragraph("Model metrics", styles["Heading2"]),
                Paragraph(
                    json.dumps(metrics, ensure_ascii=False, indent=2)
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace("\n", "<br/>"),
                    styles["Code"],
                ),
            ]
        )
    document.build(content)
    return buffer.getvalue()


def _persist_model_run(result: TrainingResult) -> None:
    """Save the fitted model and model-run metadata for history and prediction."""
    dataset_id = st.session_state.get("current_dataset_id")
    if dataset_id is None:
        st.warning("Model trained for this session only; save the dataset to keep history.")
        return
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIRECTORY / f"ui_model_{uuid.uuid4().hex}.joblib"
    AutoMLPipeline.save(result.pipeline, model_path)
    metrics_json = _training_metrics_payload(result)
    try:
        with SessionLocal() as session:
            run = ModelRun(
                dataset_id=dataset_id,
                target_column=result.target_column,
                task_type=result.task,
                algorithm=result.algorithm,
                metrics_json=metrics_json,
                model_path=str(model_path),
                status="completed",
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            st.session_state["current_model_run_id"] = run.id
            st.session_state["current_model_path"] = str(model_path)
    except Exception:
        model_path.unlink(missing_ok=True)
        logger.exception("Could not save model run")
        raise


def _training_metrics_payload(result: TrainingResult) -> dict[str, Any]:
    """Convert model results into JSON-compatible history metadata."""
    return {
        "metrics": result.metrics,
        "feature_importances": result.feature_importances,
        "confusion_matrix": result.confusion_matrix,
        "class_labels": result.class_labels,
        "residuals": result.residuals,
        "actual_values": result.actual_values,
        "predicted_values": result.predicted_values,
        "roc_curve": result.roc_curve,
        "cv_scores": result.cv_scores,
        "feature_columns": result.feature_columns,
    }


def _current_metrics() -> dict[str, Any] | None:
    """Return the current model metric dictionary, if any."""
    result = st.session_state.get("current_training_result")
    if result is not None:
        return result.metrics
    return st.session_state.get("current_metrics")


def _load_model_runs() -> list[ModelRun]:
    """Load completed model metadata without detaching database objects."""
    with SessionLocal() as session:
        runs = session.scalars(
            select(ModelRun)
            .where(ModelRun.status == "completed")
            .order_by(ModelRun.created_at.desc())
        ).all()
        return [
            ModelRun(
                id=run.id,
                dataset_id=run.dataset_id,
                target_column=run.target_column,
                task_type=run.task_type,
                algorithm=run.algorithm,
                metrics_json=run.metrics_json,
                model_path=run.model_path,
                status=run.status,
                created_at=run.created_at,
            )
            for run in runs
        ]


def _dataframe_for_run(run: ModelRun) -> pd.DataFrame | None:
    """Load the source dataset associated with a saved model."""
    with SessionLocal() as session:
        dataset = session.get(Dataset, run.dataset_id)
        path = dataset.file_path if dataset is not None else ""
    if not path or not Path(path).is_file():
        return None
    try:
        return DataIngestor().load_file(path, Path(path).suffix)
    except (ValueError, OSError, ImportError) as exc:
        logger.exception("Could not load source dataset for model run %s", run.id)
        st.error(f"Could not load the model's source data: {exc}")
        return None


def _load_cached_model(path: str) -> Any:
    """Load and cache the model artifact."""
    return cached_model(path)


def _feature_form(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    *,
    key_prefix: str,
    defaults: dict[str, Any] | None = None,
    allow_outside_training_range: bool = False,
) -> dict[str, Any]:
    """Build a form with feature controls based on source column dtypes."""
    defaults = defaults or {}
    values: dict[str, Any] = {}
    columns = st.columns(2)
    for index, name in enumerate(feature_columns):
        if name not in dataframe:
            continue
        series = dataframe[name]
        non_missing = series.dropna()
        if name in defaults:
            default = defaults[name]
        elif len(non_missing) and pd.api.types.is_bool_dtype(series.dtype):
            default = bool(non_missing.mode().iloc[0])
        elif len(non_missing) and pd.api.types.is_numeric_dtype(series.dtype):
            default = non_missing.median()
        else:
            mode = non_missing.mode()
            default = mode.iloc[0] if not mode.empty else 0
        container = columns[index % 2]
        if pd.api.types.is_bool_dtype(series.dtype):
            boolean_options = [False, True]
            values[name] = container.selectbox(
                name,
                boolean_options,
                index=boolean_options.index(bool(default)),
                key=f"{key_prefix}_{name}",
            )
        elif pd.api.types.is_numeric_dtype(series.dtype):
            minimum = float(non_missing.min()) if len(non_missing) else 0.0
            maximum = float(non_missing.max()) if len(non_missing) else 1.0
            numeric_default = float(default)
            step = max(abs(maximum - minimum) / 100, 0.01)
            range_help = (
                f"Training data ranged from {minimum:g} to {maximum:g}. "
                "You may enter a value outside that range."
                if allow_outside_training_range
                else f"Training data ranged from {minimum:g} to {maximum:g}."
            )
            values[name] = container.number_input(
                name,
                value=numeric_default,
                step=step,
                help=range_help,
                key=f"{key_prefix}_{name}",
            )
        elif pd.api.types.is_datetime64_any_dtype(series.dtype):
            default_date = pd.to_datetime(default, errors="coerce")
            if pd.isna(default_date):
                default_date = pd.Timestamp(date.today())
            picked = container.date_input(
                name,
                value=default_date.date(),
                key=f"{key_prefix}_{name}",
            )
            values[name] = pd.Timestamp(picked).isoformat()
        else:
            options = sorted(non_missing.astype(str).unique().tolist())
            if len(options) <= 100 and options:
                selected = str(default) if str(default) in options else options[0]
                values[name] = container.selectbox(
                    name, options, index=options.index(selected),
                    key=f"{key_prefix}_{name}",
                )
            else:
                values[name] = container.text_input(
                    name, value=str(default), key=f"{key_prefix}_{name}"
                )
    return values


def _baseline_features(dataframe: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    """Choose median numeric and mode categorical baseline feature values."""
    baseline: dict[str, Any] = {}
    for column in feature_columns:
        series = dataframe[column]
        values = series.dropna()
        if values.empty:
            baseline[column] = 0
        elif pd.api.types.is_bool_dtype(series.dtype):
            baseline[column] = bool(values.mode().iloc[0])
        elif pd.api.types.is_numeric_dtype(series.dtype):
            baseline[column] = float(values.median())
        elif pd.api.types.is_datetime64_any_dtype(series.dtype):
            baseline[column] = pd.Timestamp(values.median()).isoformat()
        else:
            baseline[column] = values.mode().iloc[0]
    return baseline


def _numeric_prediction(result: dict[str, Any]) -> float:
    """Return a numeric value representing regression or classification output."""
    prediction = result["prediction"]
    if isinstance(prediction, (int, float, np.number)):
        return float(prediction)
    probabilities = result.get("probabilities", {})
    if probabilities:
        return float(max(probabilities.values()))
    return float(str(prediction) in {"True", "true", "yes", "1"})


if __name__ == "__main__":
    main()
