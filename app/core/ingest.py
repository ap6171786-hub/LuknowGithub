"""Loading, validating, and inferring schemas for tabular data."""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Literal, TextIO

import numpy as np
import pandas as pd

FileType = Literal["csv", "excel", "json", "parquet"]
DataSource = str | Path | BinaryIO | TextIO
SemanticType = Literal[
    "numeric", "categorical", "datetime", "text", "boolean", "id-like"
]

_SUPPORTED_TYPES: dict[str, FileType] = {
    "csv": "csv",
    "xls": "excel",
    "xlsx": "excel",
    "xlsm": "excel",
    "xlsb": "excel",
    "excel": "excel",
    "json": "json",
    "jsonl": "json",
    "ndjson": "json",
    "parquet": "parquet",
    "pq": "parquet",
}
_ID_NAME_PATTERN = re.compile(r"(^id$|_id$|^id_|uuid|guid|identifier)", re.IGNORECASE)


class IngestionError(ValueError):
    """Raised when a tabular input cannot be loaded or validated."""


@dataclass
class ValidationReport:
    """Summary of structural and data quality findings for a dataframe.

    ``missing_values`` maps each column name to its null count.
    ``dtype_mismatches`` maps columns with numeric-looking string values to
    details containing the detected dtype and number of coercible values.
    """

    row_count: int
    column_count: int
    missing_values: dict[str, int] = field(default_factory=dict)
    duplicate_rows: int = 0
    dtype_mismatches: dict[str, dict[str, str | int]] = field(default_factory=dict)
    constant_columns: list[str] = field(default_factory=list)
    high_cardinality_categorical_columns: list[str] = field(default_factory=list)

    @property
    def duplicate_row_count(self) -> int:
        """Return the duplicate row count using the explicit count naming."""
        return self.duplicate_rows

    @property
    def high_cardinality_columns(self) -> list[str]:
        """Return categorical columns marked as high cardinality."""
        return self.high_cardinality_categorical_columns


class DataIngestor:
    """Load supported tabular files and report inferred structure and quality."""

    def load_file(self, file_path_or_buffer: DataSource, file_type: str) -> pd.DataFrame:
        """Load a CSV, Excel, JSON, or Parquet source into a validated dataframe.

        CSV files are decoded using UTF-8, Latin-1, then CP1252, and their
        delimiters are sniffed from the available text. ``file_type`` accepts
        common extensions, with or without a leading dot, or format names such
        as ``excel``.

        Raises:
            IngestionError: If the format is unsupported or loading or
                validation fails.
        """
        normalized_type = self._normalize_file_type(file_type)
        source = self._prepare_source(file_path_or_buffer)

        try:
            if normalized_type == "csv":
                dataframe = self._load_csv(source)
            elif normalized_type == "excel":
                dataframe = pd.read_excel(source)
            elif normalized_type == "json":
                dataframe = self._load_json(source)
            else:
                dataframe = pd.read_parquet(source)
        except IngestionError:
            raise
        except (OSError, UnicodeError, ValueError, TypeError, ImportError) as exc:
            raise IngestionError(f"Could not load {normalized_type} data: {exc}") from exc

        self.validate(dataframe)
        return dataframe

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """Inspect dataframe structure and common data quality issues.

        Numeric-looking string columns are reported as dtype mismatches when at
        least 80% of their non-empty values can be parsed numerically. Categorical
        columns with at least 20 unique values and at least half as many unique
        values as rows are marked high-cardinality.

        Raises:
            IngestionError: If ``df`` is not a usable dataframe.
        """
        self._validate_structure(df)

        missing_values = {
            str(column): int(count)
            for column, count in df.isna().sum().items()
        }
        dtype_mismatches: dict[str, dict[str, str | int]] = {}
        constant_columns: list[str] = []
        high_cardinality_columns: list[str] = []

        for column in df.columns:
            series = df[column]
            non_missing = series.dropna()
            if series.nunique(dropna=False) <= 1:
                constant_columns.append(str(column))

            if (
                not pd.api.types.is_numeric_dtype(series.dtype)
                and not pd.api.types.is_bool_dtype(series.dtype)
                and not pd.api.types.is_datetime64_any_dtype(series.dtype)
            ):
                numeric_values = pd.to_numeric(non_missing, errors="coerce")
                convertible_count = int(numeric_values.notna().sum())
                if len(non_missing) and convertible_count / len(non_missing) >= 0.8:
                    dtype_mismatches[str(column)] = {
                        "actual_dtype": str(series.dtype),
                        "expected_dtype": "numeric",
                        "numeric_value_count": convertible_count,
                    }

            if self._is_categorical(series) or self._is_high_cardinality_category(
                series, str(column)
            ):
                unique_count = int(series.nunique(dropna=True))
                if unique_count >= 20 and unique_count / max(len(df), 1) >= 0.5:
                    high_cardinality_columns.append(str(column))

        return ValidationReport(
            row_count=int(len(df)),
            column_count=int(len(df.columns)),
            missing_values=missing_values,
            duplicate_rows=int(df.duplicated().sum()),
            dtype_mismatches=dtype_mismatches,
            constant_columns=constant_columns,
            high_cardinality_categorical_columns=high_cardinality_columns,
        )

    def infer_schema(self, df: pd.DataFrame) -> dict[str, SemanticType]:
        """Infer a semantic type for each column in a dataframe.

        Types are assigned in order of specificity: boolean, id-like,
        datetime, numeric, categorical, and text.

        Raises:
            IngestionError: If ``df`` is not a usable dataframe.
        """
        self._validate_structure(df)
        schema: dict[str, SemanticType] = {}
        for column in df.columns:
            series = df[column]
            name = str(column)

            if pd.api.types.is_bool_dtype(series.dtype) or self._looks_boolean(series):
                inferred: SemanticType = "boolean"
            elif _ID_NAME_PATTERN.search(name):
                inferred = "id-like"
            elif pd.api.types.is_datetime64_any_dtype(series.dtype) or self._looks_datetime(
                series
            ):
                inferred = "datetime"
            elif self._looks_id_like(name, series):
                inferred = "id-like"
            elif pd.api.types.is_numeric_dtype(series.dtype):
                inferred = "numeric"
            elif self._is_categorical(series) or self._is_high_cardinality_category(
                series, name
            ):
                inferred = "categorical"
            else:
                inferred = "text"
            schema[name] = inferred
        return schema

    @staticmethod
    def _normalize_file_type(file_type: str) -> FileType:
        """Normalize a format name or file extension."""
        normalized = file_type.lower().strip().lstrip(".")
        if normalized not in _SUPPORTED_TYPES:
            supported = "CSV, Excel, JSON, and Parquet"
            raise IngestionError(f"Unsupported file type {file_type!r}. Use {supported}.")
        return _SUPPORTED_TYPES[normalized]

    @staticmethod
    def _prepare_source(source: DataSource) -> DataSource:
        """Rewind seekable buffers to their beginning before parsing."""
        if not isinstance(source, (str, Path)) and hasattr(source, "seek"):
            try:
                source.seek(0)
            except (OSError, ValueError) as exc:
                raise IngestionError(f"Could not seek input buffer: {exc}") from exc
        return source

    @classmethod
    def _load_csv(cls, source: DataSource) -> pd.DataFrame:
        """Decode CSV content using supported encodings and sniff its delimiter."""
        text = cls._read_text(source)
        if not text.strip():
            raise IngestionError(
                "The dataset must contain at least one row and one column."
            )
        sample = text[:65536]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","
        try:
            return pd.read_csv(io.StringIO(text), sep=delimiter)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
            raise IngestionError(f"Could not parse CSV data: {exc}") from exc

    @staticmethod
    def _read_text(source: DataSource) -> str:
        """Read and decode a CSV source using the supported encoding order."""
        if isinstance(source, (str, Path)):
            try:
                raw_bytes = Path(source).read_bytes()
            except OSError as exc:
                raise IngestionError(f"Could not read CSV file: {exc}") from exc
        else:
            try:
                source.seek(0)
                content = source.read()
            except (OSError, ValueError, AttributeError) as exc:
                raise IngestionError(f"Could not read CSV buffer: {exc}") from exc
            if isinstance(content, str):
                return content
            if not isinstance(content, bytes):
                raise IngestionError("CSV input must provide text or bytes.")
            raw_bytes = content

        latin_fallback: str | None = None
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                decoded = raw_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue
            if encoding == "latin-1" and any(
                "\u0080" <= character <= "\u009f" for character in decoded
            ):
                latin_fallback = decoded
                continue
            return decoded
        if latin_fallback is not None:
            return latin_fallback
        raise IngestionError("CSV encoding is unsupported; tried UTF-8, Latin-1, and CP1252.")

    @staticmethod
    def _load_json(source: DataSource) -> pd.DataFrame:
        """Load standard JSON or newline-delimited JSON into a dataframe."""
        try:
            if isinstance(source, (str, Path)):
                suffix = Path(source).suffix.lower()
                return pd.read_json(source, lines=suffix in {".jsonl", ".ndjson"})
            content = source.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            try:
                json.loads(content)
            except json.JSONDecodeError:
                return pd.read_json(io.StringIO(content), lines=True)
            return pd.read_json(io.StringIO(content))
        except (pd.errors.ParserError, ValueError, TypeError, UnicodeDecodeError) as exc:
            raise IngestionError(f"Could not parse JSON data: {exc}") from exc

    @classmethod
    def _validate_structure(cls, df: pd.DataFrame) -> None:
        """Reject inputs that cannot be profiled as a named, non-empty table."""
        if not isinstance(df, pd.DataFrame):
            raise IngestionError("Input must be a pandas DataFrame.")
        if df.empty or len(df.columns) == 0:
            raise IngestionError("The dataset must contain at least one row and one column.")
        if df.columns.has_duplicates:
            duplicates = df.columns[df.columns.duplicated()].astype(str).tolist()
            raise IngestionError(f"Column names must be unique; duplicates: {duplicates}.")
        if any(not str(column).strip() for column in df.columns):
            raise IngestionError("Column names must not be empty.")

    @staticmethod
    def _is_categorical(series: pd.Series) -> bool:
        """Return whether a column is suitable for categorical interpretation."""
        if (
            pd.api.types.is_bool_dtype(series.dtype)
            or pd.api.types.is_numeric_dtype(series.dtype)
            or pd.api.types.is_datetime64_any_dtype(series.dtype)
        ):
            return False
        values = series.dropna().astype(str)
        if values.empty:
            return False
        if values.map(lambda value: any(character.isspace() for character in value)).any():
            return False
        if values.str.len().mean() >= 64:
            return False
        unique_count = int(series.nunique(dropna=True))
        return unique_count <= max(20, int(np.sqrt(max(len(series), 1))))

    @staticmethod
    def _is_high_cardinality_category(
        series: pd.Series, name: str = ""
    ) -> bool:
        """Identify compact, non-spaced text columns with many distinct labels."""
        if (
            _ID_NAME_PATTERN.search(name)
            or pd.api.types.is_numeric_dtype(series.dtype)
            or pd.api.types.is_bool_dtype(series.dtype)
            or pd.api.types.is_datetime64_any_dtype(series.dtype)
        ):
            return False
        values = series.dropna().astype(str)
        if values.empty or values.map(
            lambda value: any(character.isspace() for character in value)
        ).any():
            return False
        if values.str.len().mean() >= 64:
            return False
        unique_count = int(values.nunique())
        return unique_count >= 20 and unique_count / len(values) >= 0.5

    @staticmethod
    def _looks_boolean(series: pd.Series) -> bool:
        """Recognize common textual encodings of boolean values."""
        values = {
            str(value).strip().lower()
            for value in series.dropna().unique()
        }
        boolean_values = {"true", "false", "yes", "no", "y", "n", "0", "1"}
        return bool(values) and values <= boolean_values

    @staticmethod
    def _looks_id_like(name: str, series: pd.Series) -> bool:
        """Recognize identifier-style names or nearly unique values."""
        if _ID_NAME_PATTERN.search(name):
            return True
        non_missing = series.dropna()
        if len(non_missing) < 2:
            return False
        unique_ratio = non_missing.nunique() / len(non_missing)
        textual = not (
            pd.api.types.is_numeric_dtype(series.dtype)
            or pd.api.types.is_bool_dtype(series.dtype)
            or pd.api.types.is_datetime64_any_dtype(series.dtype)
        )
        values = non_missing.astype(str)
        compact = not values.map(
            lambda value: any(character.isspace() for character in value)
        ).any()
        return textual and compact and unique_ratio >= 0.98

    @staticmethod
    def _looks_datetime(series: pd.Series) -> bool:
        """Recognize string columns where most non-null values parse as dates."""
        if (
            pd.api.types.is_numeric_dtype(series.dtype)
            or pd.api.types.is_bool_dtype(series.dtype)
            or pd.api.types.is_datetime64_any_dtype(series.dtype)
        ):
            return False
        values = series.dropna()
        if values.empty:
            return False
        parsed = pd.to_datetime(values, errors="coerce", format="mixed")
        return bool(parsed.notna().mean() >= 0.8)


def load_dataframe(source: DataSource, *, filename: str | None = None) -> pd.DataFrame:
    """Backward-compatible wrapper for loading CSV and JSON tabular sources."""
    source_name = filename or (str(source) if isinstance(source, (str, Path)) else "")
    suffix = Path(source_name).suffix.lower()
    file_type = "json" if suffix in {".json", ".jsonl", ".ndjson"} else suffix
    return DataIngestor().load_file(source, file_type)


def validate_dataframe(dataframe: pd.DataFrame) -> None:
    """Backward-compatible structural validation helper."""
    DataIngestor()._validate_structure(dataframe)
