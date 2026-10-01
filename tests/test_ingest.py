"""Unit tests for multi-format data ingestion and schema inference."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from app.core.ingest import DataIngestor, IngestionError, ValidationReport


@pytest.fixture
def ingestor() -> DataIngestor:
    """Provide a fresh data ingestor for each test."""
    return DataIngestor()


@pytest.mark.parametrize(
    ("file_type", "payload"),
    [
        ("csv", b"name,score\nAda,10\nLin,9\n"),
        ("csv", b"name;score\nAda;10\nLin;9\n"),
        ("csv", "name,city\nAndré,Montréal\nAda,Paris\n".encode("cp1252")),
        (
            "csv",
            "name,quote\nAda,\u201ccurious\u201d\nLin,\u201chello\u201d\n".encode("cp1252"),
        ),
        ("json", b'[{"name":"Ada","score":10},{"name":"Lin","score":9}]'),
    ],
)
def test_load_file_reads_csv_and_json(
    ingestor: DataIngestor, file_type: str, payload: bytes
) -> None:
    """Delimited and JSON files load into dataframes."""
    dataframe = ingestor.load_file(io.BytesIO(payload), file_type)

    assert dataframe.shape == (2, 2)


def test_load_file_reads_excel(ingestor: DataIngestor) -> None:
    """Excel buffers are loaded through pandas' Excel reader."""
    buffer = io.BytesIO()
    pd.DataFrame({"name": ["Ada", "Lin"], "score": [10, 9]}).to_excel(
        buffer, index=False
    )

    dataframe = ingestor.load_file(buffer, "xlsx")

    assert dataframe["name"].tolist() == ["Ada", "Lin"]


def test_load_file_reads_parquet(ingestor: DataIngestor) -> None:
    """Parquet buffers are loaded through pandas' Parquet reader."""
    buffer = io.BytesIO()
    expected = pd.DataFrame({"name": ["Ada", "Lin"], "score": [10, 9]})
    expected.to_parquet(buffer, index=False)

    dataframe = ingestor.load_file(buffer, "parquet")

    pd.testing.assert_frame_equal(dataframe, expected)


def test_load_file_rejects_unknown_file_type(ingestor: DataIngestor) -> None:
    """An unsupported format raises the public ingestion exception."""
    with pytest.raises(IngestionError, match="Unsupported file type"):
        ingestor.load_file(io.BytesIO(b"data"), "xlsx-like")


def test_load_file_rejects_empty_csv(ingestor: DataIngestor) -> None:
    """Empty tabular input raises an actionable ingestion error."""
    with pytest.raises(IngestionError, match="at least one row"):
        ingestor.load_file(io.BytesIO(b""), "csv")


def test_validate_returns_all_quality_findings(ingestor: DataIngestor) -> None:
    """Validation reports missing values, duplicates, mismatch, and cardinality."""
    dataframe = pd.DataFrame(
        {
            "numeric_text": ["1", "2", "3", None] + [str(i) for i in range(4, 25)],
            "constant": ["same"] * 25,
            "category": [f"category-{i}" for i in range(25)],
            "other": [*range(24), 23],
        }
    )
    dataframe.loc[24] = dataframe.loc[0]

    report = ingestor.validate(dataframe)

    assert isinstance(report, ValidationReport)
    assert report.row_count == 25
    assert report.column_count == 4
    assert report.missing_values["numeric_text"] == 1
    assert report.duplicate_rows == 1
    assert report.dtype_mismatches["numeric_text"]["expected_dtype"] == "numeric"
    assert "constant" in report.constant_columns
    assert "category" in report.high_cardinality_categorical_columns


def test_validate_rejects_invalid_dataframe(ingestor: DataIngestor) -> None:
    """Validation rejects an empty dataframe with the custom error type."""
    with pytest.raises(IngestionError, match="at least one row"):
        ingestor.validate(pd.DataFrame())


def test_infer_schema_classifies_common_semantic_types(
    ingestor: DataIngestor,
) -> None:
    """Schema inference distinguishes common column types and identifiers."""
    dataframe = pd.DataFrame(
        {
            "amount": [10.5, 20.0, 30.5],
            "segment": ["retail", "enterprise", "retail"],
            "event_time": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03"]
            ),
            "description": ["first event", "second event", "third event"],
            "active": [True, False, True],
            "customer_id": ["a1", "a2", "a3"],
        }
    )

    schema = ingestor.infer_schema(dataframe)

    assert schema == {
        "amount": "numeric",
        "segment": "categorical",
        "event_time": "datetime",
        "description": "text",
        "active": "boolean",
        "customer_id": "id-like",
    }


def test_infer_schema_recognizes_string_boolean_and_datetime(
    ingestor: DataIngestor,
) -> None:
    """Textual true/false and date columns are semantically recognized."""
    dataframe = pd.DataFrame(
        {
            "enabled": ["yes", "no", "yes"],
            "created": ["2025-01-01", "2025-02-01", "2025-03-01"],
        }
    )

    assert ingestor.infer_schema(dataframe) == {
        "enabled": "boolean",
        "created": "datetime",
    }
