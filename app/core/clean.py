"""Non-destructive suggestions for common data quality issues."""

from __future__ import annotations

from typing import TypedDict

import pandas as pd


class CleaningSuggestion(TypedDict):
    """A single actionable observation about dataset quality."""

    column: str
    issue: str
    message: str


def suggest_cleaning(dataframe: pd.DataFrame) -> list[CleaningSuggestion]:
    """Suggest common cleaning operations without modifying the input data."""
    suggestions: list[CleaningSuggestion] = []
    duplicate_rows = int(dataframe.duplicated().sum())
    if duplicate_rows:
        suggestions.append(
            {
                "column": "",
                "issue": "duplicate_rows",
                "message": f"Consider reviewing {duplicate_rows} duplicate row(s).",
            }
        )

    for name in dataframe.columns:
        series = dataframe[name]
        missing_count = int(series.isna().sum())
        if missing_count:
            suggestions.append(
                {
                    "column": str(name),
                    "issue": "missing_values",
                    "message": (
                        f"{missing_count} value(s) are missing; consider imputing "
                        "or removing them."
                    ),
                }
            )

        if series.nunique(dropna=False) <= 1:
            suggestions.append(
                {
                    "column": str(name),
                    "issue": "constant_column",
                    "message": "This column has one value (or is entirely missing).",
                }
            )

        if pd.api.types.is_string_dtype(series.dtype):
            string_values = series.dropna()
            whitespace_count = int(
                string_values.map(lambda value: value != value.strip()).sum()
            )
            if whitespace_count:
                suggestions.append(
                    {
                        "column": str(name),
                        "issue": "surrounding_whitespace",
                        "message": (
                            f"{whitespace_count} text value(s) have surrounding "
                            "whitespace; consider trimming them."
                        ),
                    }
                )

    return suggestions

