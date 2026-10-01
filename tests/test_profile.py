"""Tests for the full data profiling report."""

from __future__ import annotations

import json

import pandas as pd

from app.core.profile import DataProfiler, ProfileReport, profile_dataframe


def test_profile_report_includes_numeric_categorical_datetime_and_text_stats() -> None:
    """Profile output contains type-specific statistics and serializes cleanly."""
    dataframe = pd.DataFrame(
        {
            "amount": [1.0, 2.0, 2.0, 100.0],
            "segment": ["a", "b", "a", "b"],
            "event_time": pd.to_datetime(
                ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]
            ),
            "description": [
                "alpha beta",
                "beta gamma",
                "alpha delta",
                "gamma beta",
            ],
        }
    )

    report = DataProfiler(dataframe).profile()
    payload = json.loads(report.to_json())

    assert isinstance(report, ProfileReport)
    assert payload["columns"]["amount"]["outliers"] == 1
    assert payload["columns"]["segment"]["top_values"][0]["value"] == "a"
    assert payload["columns"]["segment"]["entropy"] == 1.0
    assert payload["columns"]["event_time"]["gaps"]["count"] == 3
    assert payload["columns"]["description"]["avg_length"] > 0
    assert payload["columns"]["description"]["most_common_words"][0]["word"] == "beta"
    assert payload["correlation_matrix"]["amount"]["amount"] == 1.0


def test_profile_report_finds_targets_ids_quality_and_html() -> None:
    """Dataset-level findings and report export are available."""
    dataframe = pd.DataFrame(
        {
            "record_id": list(range(20)),
            "label": ["yes", "no"] * 10,
            "value": [float(value) for value in range(20)],
        }
    )
    dataframe.loc[19, "value"] = None

    report = DataProfiler(dataframe).profile()
    compatible = profile_dataframe(dataframe)

    assert "label" in report.potential_target_columns
    assert "record_id" in report.potential_id_columns
    assert 0 <= report.data_quality_score <= 100
    assert "<!doctype html>" in report.to_html().lower()
    assert compatible["row_count"] == 20
