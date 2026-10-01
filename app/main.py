"""Multi-page Streamlit data exploration and modeling application."""

from __future__ import annotations

import io
import json
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
    try:
        with st.spinner("Validating and profiling dataset…"):
            dataframe = DataIngestor().load_file(io.BytesIO(content), file_type)
            profile = cached_profile(content, file_type)
            validation = DataIngestor().validate(dataframe)
    except (ValueError, OSError, ImportError) as exc:
        st.error(f"Could not load this dataset: {exc}")
        return

    st.session_state["current_dataset"] = dataframe
    st.session_state["current_profile"] = profile
    st.session_state["current_dataset_name"] = uploaded_file.name
    st.session_state["validation_report"] = {
        "row_count": validation.row_count,
        "column_count": validation.column_count,
        "missing_values": validation.missing_values,
        "duplicate_rows": validation.duplicate_rows,
        "dtype_mismatches": validation.dtype_mismatches,
        "constant_columns": validation.constant_columns,
        "high_cardinality_categorical_columns": (
            validation.high_cardinality_categorical_columns
        ),
    }

    st.subheader("Preview")
    st.dataframe(dataframe.head(100), use_container_width=True)
    st.subheader("Validation report")
    _render_validation(validation)
    st.caption(f"Data quality score: {profile['data_quality_score']:.1f}/100")

    if st.button("Save dataset to history", type="primary"):
        try:
            UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
            safe_name = Path(uploaded_file.name).name
            destination = UPLOAD_DIRECTORY / f"{uuid.uuid4().hex}_{safe_name}"
            destination.write_bytes(content)
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
                st.session_state["current_dataset_id"] = dataset.id
            st.success(f"Saved {safe_name} to history.")
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
        rows = [{"column": name, **stats} for name, stats in profile["columns"].items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        chosen = st.selectbox("Inspect a column", dataframe.columns.astype(str))
        selected = dataframe[chosen]
        st.json(profile["columns"][chosen])
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
    st.title("Cleaning suggestions")
    dataframe = st.session_state.get("current_dataset")
    if dataframe is None:
        st.info("Upload a dataset first.")
        return
    for suggestion in suggest_cleaning(dataframe):
        st.warning(
            f"{suggestion['column'] or 'Dataset'} — {suggestion['message']}"
        )
    drop_duplicates = st.checkbox("Drop duplicate rows")
    fill_nulls = st.checkbox("Fill nulls (median for numeric, mode for other columns)")
    drop_constants = st.checkbox("Drop constant columns")
    fix_dtypes = st.checkbox("Convert numeric-looking and datetime columns")
    cleaned = dataframe.copy()
    if drop_duplicates:
        cleaned = cleaned.drop_duplicates()
    if fill_nulls:
        for column in cleaned.columns:
            if not cleaned[column].isna().any():
                continue
            if pd.api.types.is_numeric_dtype(cleaned[column].dtype):
                median = cleaned[column].median()
                if pd.notna(median):
                    cleaned[column] = cleaned[column].fillna(median)
            else:
                mode = cleaned[column].mode(dropna=True)
                if not mode.empty:
                    cleaned[column] = cleaned[column].fillna(mode.iloc[0])
    if drop_constants:
        constant_columns = [
            column for column in cleaned.columns if cleaned[column].nunique(dropna=False) <= 1
        ]
        cleaned = cleaned.drop(columns=constant_columns)
    if fix_dtypes:
        for column in cleaned.columns:
            series = cleaned[column]
            if pd.api.types.is_object_dtype(series.dtype):
                numeric = pd.to_numeric(series, errors="coerce")
                if numeric.notna().sum() >= max(1, int(series.notna().sum() * 0.8)):
                    cleaned[column] = numeric
                    continue
                parsed_dates = pd.to_datetime(series, errors="coerce", format="mixed")
                if parsed_dates.notna().sum() >= max(
                    1, int(series.notna().sum() * 0.8)
                ):
                    cleaned[column] = parsed_dates

    before, after = st.columns(2)
    before.subheader(f"Before · {len(dataframe):,} rows")
    before.dataframe(dataframe.head(100), use_container_width=True)
    after.subheader(f"After · {len(cleaned):,} rows")
    after.dataframe(cleaned.head(100), use_container_width=True)
    if st.button("Use cleaned data for this session"):
        st.session_state["current_dataset"] = cleaned
        st.session_state["current_profile"] = DataProfiler(cleaned).profile().to_dict()
        st.success("Cleaned data is now active for this session.")


def _train_page() -> None:
    """Train a CV-selected model and render its metrics and diagnostics."""
    st.title("Train a model")
    dataframe = st.session_state.get("current_dataset")
    if dataframe is None:
        st.info("Upload a dataset first.")
        return
    target = st.selectbox("Target column", dataframe.columns.astype(str))
    task_type = st.selectbox(
        "Task type", ["auto", "classification", "regression"], index=0
    )
    algorithm = st.selectbox(
        "Algorithm", ["auto", "logistic_regression", "ridge", "random_forest", "gradient_boosting"]
    )
    button = st.button("Train with cross-validation", type="primary")
    if button:
        progress = st.progress(0, text="Preparing cross-validation…")

        def update_progress(current: int, total: int, name: str) -> None:
            progress.progress(
                current / total,
                text=f"Evaluating {name} · {current}/{total}",
            )

        try:
            with st.spinner("Training candidate models…"):
                result = AutoMLPipeline().train(
                    dataframe,
                    target,
                    task_type=task_type,
                    algorithm=algorithm,
                    progress_callback=update_progress,
                )
            st.session_state["current_training_result"] = result
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
        st.caption("Trained model results will appear here.")
        return
    st.subheader(f"{result.algorithm.replace('_', ' ').title()} · {result.task}")
    _metric_row(result.metrics)
    if result.cv_scores:
        st.caption("Cross-validation scores: " + " · ".join(
            f"{name}: {score:.3f}" for name, score in result.cv_scores.items()
        ))
    if result.feature_importances:
        importance = pd.DataFrame(
            result.feature_importances.items(), columns=["feature", "importance"]
        ).sort_values("importance", ascending=True).tail(25)
        st.plotly_chart(
            px.bar(
                importance, x="importance", y="feature", orientation="h",
                title="Feature importance", color_discrete_sequence=["#6C5CE7"],
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
    """Generate predictions from a previously saved local model run."""
    st.title("Predict")
    runs = _load_model_runs()
    if not runs:
        st.info("Train a model first.")
        return
    labels = {f"#{run.id} · {run.algorithm} · target {run.target_column}": run for run in runs}
    selection = st.selectbox("Trained model", list(labels))
    run = labels[selection]
    dataframe = _dataframe_for_run(run)
    if dataframe is None:
        st.error("The source dataset for this model is unavailable.")
        return
    feature_columns = run.metrics_json.get("feature_columns", [])
    model = _load_cached_model(run.model_path)
    values = _feature_form(dataframe, feature_columns, key_prefix="predict")
    if st.button("Predict", type="primary"):
        try:
            result = AutoMLPipeline.predict(model, values)
            st.session_state["last_prediction"] = result
            st.json(result)
        except (ValueError, TypeError, KeyError) as exc:
            st.error(f"Prediction failed: {exc}")


def _what_if_page() -> None:
    """Provide interactive model predictions and one-feature sensitivity."""
    st.title("What-If Simulator")
    st.caption("Change a feature and explore how the model's prediction responds.")
    runs = _load_model_runs()
    if not runs:
        st.info("Train a model first to unlock the simulator.")
        return
    labels = {f"#{run.id} · {run.algorithm} · {run.target_column}": run for run in runs}
    selection = st.selectbox("Choose a trained model", list(labels), key="whatif_model")
    run = labels[selection]
    dataframe = _dataframe_for_run(run)
    if dataframe is None:
        st.error("The source dataset for this model is unavailable.")
        return
    feature_columns = run.metrics_json.get("feature_columns", [])
    model = _load_cached_model(run.model_path)
    baseline = _baseline_features(dataframe, feature_columns)
    values = _feature_form(
        dataframe,
        feature_columns,
        key_prefix=f"whatif_{run.id}",
        defaults=baseline,
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
        default = defaults.get(name, non_missing.median() if (
            pd.api.types.is_numeric_dtype(series.dtype) and len(non_missing)
        ) else (non_missing.mode().iloc[0] if len(non_missing) and not non_missing.mode().empty else 0))
        container = columns[index % 2]
        if pd.api.types.is_numeric_dtype(series.dtype):
            minimum = float(non_missing.min()) if len(non_missing) else 0.0
            maximum = float(non_missing.max()) if len(non_missing) else 1.0
            if minimum == maximum:
                maximum = minimum + 1.0
            values[name] = container.slider(
                name,
                min_value=minimum,
                max_value=maximum,
                value=float(np.clip(float(default), minimum, maximum)),
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
