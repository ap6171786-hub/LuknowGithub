"""Automatic dataframe profiling and self-contained HTML report rendering."""

from __future__ import annotations

import html
import json
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ProfileReport:
    """JSON-serializable dataset profile and quality summary."""

    row_count: int
    column_count: int
    duplicate_row_count: int
    columns: dict[str, dict[str, Any]]
    correlation_matrix: dict[str, dict[str, float | None]]
    potential_target_columns: list[str]
    potential_id_columns: list[str]
    data_quality_score: float

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-compatible dictionary."""
        return _json_safe(asdict(self))

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize this report to JSON."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_html(self) -> str:
        """Render a clean, standalone HTML summary report."""
        safe = self.to_dict()
        rows = "".join(
            "<tr>"
            f"<th>{html.escape(name)}</th>"
            f"<td>{html.escape(str(item['dtype']))}</td>"
            f"<td>{item['null_pct']:.2f}%</td>"
            f"<td>{item['unique_count']}</td>"
            f"<td>{html.escape(str(item.get('semantic_type', 'unknown')))}</td>"
            "</tr>"
            for name, item in safe["columns"].items()
        )
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Dataset profile</title><style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;color:#202438;background:#f7f7fb}}
h1{{color:#29234a}}.cards{{display:flex;gap:14px;flex-wrap:wrap}}.card{{background:white;border:1px solid #e8e7f0;border-radius:12px;padding:16px 20px;min-width:130px}}
table{{border-collapse:collapse;width:100%;background:white;border-radius:12px;overflow:hidden}}th,td{{text-align:left;padding:11px 14px;border-bottom:1px solid #eee}}thead{{background:#6C5CE7;color:white}}
</style></head><body><h1>Dataset Profile</h1>
<div class="cards"><div class="card"><b>Rows</b><br>{safe['row_count']}</div>
<div class="card"><b>Columns</b><br>{safe['column_count']}</div>
<div class="card"><b>Duplicate rows</b><br>{safe['duplicate_row_count']}</div>
<div class="card"><b>Quality score</b><br>{safe['data_quality_score']:.1f}/100</div></div>
<h2>Columns</h2><table><thead><tr><th>Name</th><th>Dtype</th><th>Missing</th><th>Unique</th><th>Inferred type</th></tr></thead>
<tbody>{rows}</tbody></table>
<h2>Potential targets</h2><p>{html.escape(', '.join(safe['potential_target_columns']) or 'None detected')}</p>
<h2>Potential identifiers</h2><p>{html.escape(', '.join(safe['potential_id_columns']) or 'None detected')}</p>
</body></html>"""


class DataProfiler:
    """Generate descriptive column and dataset statistics for a dataframe."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """Create a profiler for a non-empty pandas dataframe."""
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("DataProfiler requires a pandas DataFrame.")
        self.dataframe = dataframe

    def profile(self) -> ProfileReport:
        """Compute column metrics, dataset statistics, and a quality score."""
        frame = self.dataframe
        columns: dict[str, dict[str, Any]] = {}
        outlier_total = 0
        outlier_eligible = 0

        for column in frame.columns:
            series = frame[column]
            name = str(column)
            non_null = series.dropna()
            unique_count = int(series.nunique(dropna=True))
            null_pct = float(series.isna().mean() * 100) if len(series) else 0.0
            item: dict[str, Any] = {
                "dtype": str(series.dtype),
                "null_pct": null_pct,
                "unique_count": unique_count,
                "unique_pct": (
                    float(unique_count / len(series) * 100) if len(series) else 0.0
                ),
                "semantic_type": self._semantic_type(name, series),
            }
            semantic_type = item["semantic_type"]
            if semantic_type == "numeric":
                numeric = pd.to_numeric(non_null, errors="coerce").dropna()
                item.update(self._numeric_stats(numeric))
                outlier_total += int(item["outliers"])
                outlier_eligible += len(numeric)
            elif semantic_type == "categorical" or semantic_type == "boolean":
                counts = non_null.astype(str).value_counts().head(5)
                frequencies = [
                    {
                        "value": str(value),
                        "count": int(count),
                        "frequency": float(count / max(len(non_null), 1)),
                    }
                    for value, count in counts.items()
                ]
                probabilities = non_null.astype(str).value_counts(normalize=True)
                entropy = (
                    float(-(probabilities * np.log2(probabilities)).sum())
                    if len(probabilities)
                    else 0.0
                )
                item.update({"top_values": frequencies, "entropy": entropy})
            elif semantic_type == "datetime":
                dates = self._as_datetime(series).dropna().sort_values()
                gaps = dates.diff().dropna()
                item.update(
                    {
                        "min": dates.min().isoformat() if len(dates) else None,
                        "max": dates.max().isoformat() if len(dates) else None,
                        "range": (
                            float((dates.max() - dates.min()).total_seconds())
                            if len(dates) > 1
                            else 0.0
                        ),
                        "gaps": {
                            "count": int(len(gaps)),
                            "mean_seconds": (
                                float(gaps.dt.total_seconds().mean()) if len(gaps) else 0.0
                            ),
                            "max_seconds": (
                                float(gaps.dt.total_seconds().max()) if len(gaps) else 0.0
                            ),
                        },
                    }
                )
            elif semantic_type == "text":
                text_values = non_null.astype(str)
                words = re.findall(
                    r"\b[\w'-]+\b", " ".join(text_values.tolist()).lower()
                )
                word_counts = pd.Series(words, dtype="object").value_counts().head(10)
                lengths = text_values.str.len()
                item.update(
                    {
                        "avg_length": float(lengths.mean()) if len(lengths) else 0.0,
                        "max_length": int(lengths.max()) if len(lengths) else 0,
                        "most_common_words": [
                            {"word": str(word), "count": int(count)}
                            for word, count in word_counts.items()
                        ],
                    }
                )
            columns[name] = item

        numeric = frame.select_dtypes(include=np.number)
        correlations = (
            numeric.corr().replace([np.inf, -np.inf], np.nan).to_dict()
            if not numeric.empty
            else {}
        )
        potential_targets = self._potential_targets(frame)
        potential_ids = [
            str(column)
            for column in frame.columns
            if len(frame[column].dropna()) > 0
            and frame[column].nunique(dropna=True) / len(frame[column].dropna()) >= 0.98
        ]
        duplicate_count = int(frame.duplicated().sum())
        null_fraction = (
            float(frame.isna().to_numpy().mean()) if frame.size else 0.0
        )
        duplicate_fraction = duplicate_count / max(len(frame), 1)
        outlier_fraction = outlier_total / max(outlier_eligible, 1)
        quality_score = max(
            0.0,
            min(
                100.0,
                100.0
                - 50.0 * null_fraction
                - 30.0 * duplicate_fraction
                - 20.0 * outlier_fraction,
            ),
        )
        return ProfileReport(
            row_count=int(len(frame)),
            column_count=int(len(frame.columns)),
            duplicate_row_count=duplicate_count,
            columns=columns,
            correlation_matrix=correlations,
            potential_target_columns=potential_targets,
            potential_id_columns=potential_ids,
            data_quality_score=round(quality_score, 2),
        )

    @staticmethod
    def _numeric_stats(values: pd.Series) -> dict[str, Any]:
        """Calculate descriptive statistics and IQR outlier count."""
        if values.empty:
            return {
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "std": None,
                "skew": None,
                "kurtosis": None,
                "outliers": 0,
            }
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
        return {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std()) if len(values) > 1 else 0.0,
            "skew": float(values.skew()) if len(values) > 2 else 0.0,
            "kurtosis": float(values.kurtosis()) if len(values) > 3 else 0.0,
            "outliers": outliers,
        }

    @classmethod
    def _semantic_type(cls, name: str, series: pd.Series) -> str:
        """Infer a practical descriptive type for profile statistics."""
        if pd.api.types.is_bool_dtype(series.dtype):
            return "boolean"
        if pd.api.types.is_numeric_dtype(series.dtype):
            return "numeric"
        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            return "datetime"
        values = series.dropna()
        if values.empty:
            return "categorical"
        if cls._date_values(series):
            return "datetime"
        unique_count = int(series.nunique(dropna=True))
        if name.lower().endswith(("_id", "id")):
            return "id-like"
        is_compact = not values.astype(str).str.contains(r"\s").any()
        if is_compact and unique_count / len(values) >= 0.98:
            return "id-like"
        if cls._categorical_values(series):
            return "categorical"
        return "text"

    @staticmethod
    def _categorical_values(series: pd.Series) -> bool:
        """Identify low-cardinality labels rather than prose."""
        values = series.dropna().astype(str)
        if values.empty:
            return True
        return (
            values.nunique() <= max(20, int(np.sqrt(len(values))))
            and values.str.len().mean() < 48
            and not values.str.contains(r"\s").mean() > 0.5
        )

    @staticmethod
    def _date_values(series: pd.Series) -> bool:
        """Return true when most string values parse as dates."""
        values = series.dropna()
        if values.empty or pd.api.types.is_numeric_dtype(series.dtype):
            return False
        parsed = pd.to_datetime(values, errors="coerce", format="mixed")
        return bool(parsed.notna().mean() >= 0.8)

    @staticmethod
    def _as_datetime(series: pd.Series) -> pd.Series:
        """Convert a series to timestamps while retaining unparseable nulls."""
        return pd.to_datetime(series, errors="coerce", format="mixed")

    @staticmethod
    def _potential_targets(dataframe: pd.DataFrame) -> list[str]:
        """Select low-cardinality columns with non-dominant class balance."""
        candidates: list[str] = []
        for column in dataframe.columns:
            series = dataframe[column].dropna()
            cardinality = int(series.nunique())
            if cardinality < 2 or cardinality > min(20, max(2, len(series) // 2)):
                continue
            proportions = series.value_counts(normalize=True)
            if len(proportions) and float(proportions.max()) <= 0.9:
                candidates.append(str(column))
        return candidates


def _json_safe(value: Any) -> Any:
    """Convert NumPy, pandas, and non-finite values to standard JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def profile_dataframe(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Return a backward-compatible dictionary containing the full profile."""
    report = DataProfiler(dataframe).profile().to_dict()
    column_profiles: list[dict[str, Any]] = []
    for name, item in report["columns"].items():
        item.setdefault(
            "missing_count",
            int(round(item["null_pct"] * report["row_count"] / 100)),
        )
        item.setdefault("missing_percent", item["null_pct"])
        item.setdefault("unique_count", 0)
        column_profiles.append({"name": name, **item})
    report["columns"] = column_profiles
    return report
