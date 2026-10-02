"""Unit tests for AutoML preprocessing, evaluation, and persistence."""

from __future__ import annotations

import pandas as pd
import pytest

from app.core.model import AutoMLPipeline


def test_detect_task_type_uses_target_dtype_and_cardinality() -> None:
    """Numeric continuous labels are regression and compact labels classification."""
    assert AutoMLPipeline.detect_task_type(pd.Series([1, 2, 3, 4, 5])) == "regression"
    assert AutoMLPipeline.detect_task_type(pd.Series([0, 1, 0, 1])) == "classification"
    assert (
        AutoMLPipeline.detect_task_type(pd.Series([0, 1] * 10))
        == "classification"
    )
    assert (
        AutoMLPipeline.detect_task_type(pd.Series(["a", "b", "a"]))
        == "classification"
    )


def test_build_preprocessor_handles_numeric_categorical_and_datetime() -> None:
    """Preprocessor emits all feature categories including calendar parts."""
    dataframe = pd.DataFrame(
        {
            "amount": [1.0, None, 3.0],
            "kind": ["a", None, "b"],
            "active": [True, False, True],
            "created": pd.to_datetime(["2025-01-01", None, "2025-01-03"]),
            "target": [1, 2, 3],
        }
    )

    result = AutoMLPipeline.build_preprocessor(dataframe, "target").fit_transform(
        dataframe.drop(columns="target")
    )

    assert result.shape[0] == 3
    assert result.shape[1] >= 6


def test_train_classification_provides_metrics_diagnostics_and_progress() -> None:
    """Classification training selects a model and reports CV/diagnostic output."""
    dataframe = pd.DataFrame(
        {
            "value": list(range(20)),
            "group": ["low" if value < 10 else "high" for value in range(20)],
            "target": ["no" if value < 10 else "yes" for value in range(20)],
        }
    )
    progress: list[tuple[int, int, str]] = []

    result = AutoMLPipeline(cv_folds=3).train(
        dataframe,
        "target",
        "classification",
        progress_callback=lambda current, total, name: progress.append(
            (current, total, name)
        ),
    )

    assert set(result.metrics) >= {"accuracy", "f1", "roc_auc"}
    assert result.confusion_matrix is not None
    assert result.roc_curve is not None
    assert result.feature_importances
    assert progress[-1][0] == progress[-1][1]


def test_train_regression_and_save_load_predict(tmp_path) -> None:
    """Regression results persist and a loaded artifact predicts one record."""
    dataframe = pd.DataFrame(
        {
            "size": list(range(20)),
            "category": ["a", "b"] * 10,
            "target": [float(value * 2 + 1) for value in range(20)],
        }
    )
    automl = AutoMLPipeline(cv_folds=3)
    result = automl.train(dataframe, "target", "regression", "ridge")
    artifact = tmp_path / "nested" / "model.joblib"

    automl.save(result.pipeline, artifact)
    loaded = automl.load(artifact)
    prediction = automl.predict(loaded, {"size": 2, "category": "a"})

    assert set(result.metrics) == {"mae", "rmse", "r2"}
    assert isinstance(prediction["prediction"], float)
    assert prediction["confidence_interval"][0] < prediction["prediction"]
    assert prediction["prediction"] < prediction["confidence_interval"][1]


def test_train_rejects_missing_target() -> None:
    """Invalid target selections fail clearly instead of producing a model."""
    with pytest.raises(ValueError, match="not found"):
        AutoMLPipeline().train(pd.DataFrame({"x": [1, 2, 3]}), "missing")
