"""General-purpose helpers shared by application components."""

from __future__ import annotations

from typing import Any

import pandas as pd


def dataframe_to_records(
    dataframe: pd.DataFrame, *, limit: int = 100
) -> list[dict[str, Any]]:
    """Convert the first ``limit`` dataframe rows to JSON-friendly records."""
    if limit < 0:
        raise ValueError("limit must be non-negative.")
    preview = dataframe.head(limit)
    records = preview.astype(object).where(pd.notna(preview), None)
    return records.to_dict(orient="records")
