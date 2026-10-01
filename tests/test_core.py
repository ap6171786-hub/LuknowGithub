"""Tests for the tabular data pipeline."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from app.core.clean import suggest_cleaning
from app.core.ingest import load_dataframe, validate_dataframe
from app.core.model import train_model
from app.core.profile import profile_dataframe


def test_load_dataframe_reads_csv_and_validates_columns() -> None:
    """CSV input is parsed and basic dataset dimensions are retained."""
    dataframe = load_dataframe(io.BytesIO(b"name,score\nAda,10\nLin,9\n"), filename="x.csv")

    assert dataframe.shape == (2, 2)
    assert list(dataframe.columns) == ["name", "score"]


def test_load_dataframe_rejects_unsupported_extensions() -> None:
    """Unsupported file formats produce an actionable validation error."""
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_dataframe(io.BytesIO(b"x"), filename="data.txt")


def test_validate_dataframe_rejects_duplicate_columns() -> None:
    """Duplicate labels are not silently accepted."""
    with pytest.raises(ValueError, match="unique"):
        validate_dataframe(pd.DataFrame([[1, 2]], columns=["value", "value"]))


def test_profile_and_cleaning_suggestions_report_data_quality() -> None:
    """Profile and cleaning output include missing and duplicate observations."""
    dataframe = pd.DataFrame({"score": [1.0, None, 1.0], "name": [" Ada ", "Lin", " Ada "]})

    profile = profile_dataframe(dataframe)
    suggestions = suggest_cleaning(dataframe)

    assert profile["row_count"] == 3
    assert profile["duplicate_row_count"] == 1
    assert profile["columns"][0]["missing_count"] == 1
    assert {suggestion["issue"] for suggestion in suggestions} >= {
        "duplicate_rows",
        "missing_values",
        "surrounding_whitespace",
    }


def test_train_model_returns_holdout_metrics() -> None:
    """A small numeric dataset trains a regression pipeline successfully."""
    dataframe = pd.DataFrame(
        {
            "size": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "category": ["a", "b"] * 5,
            "price": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        }
    )

    result = train_model(dataframe, "price")

    assert result.task == "regression"
    assert "mean_absolute_error" in result.metrics
    assert result.feature_columns == ["size", "category"]
