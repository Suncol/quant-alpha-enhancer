from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _series_quantiles(series: pd.Series, qs: list[float]) -> dict[str, float | None]:
    quantiles = series.quantile(qs)
    return {str(q): _json_value(float(quantiles.loc[q])) for q in qs}


def profile_return_matrix(input_path: Path, top_n: int = 20) -> dict[str, Any]:
    df = pd.read_pickle(input_path)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(df)!r}")

    rows, cols = df.shape
    index_as_dates = pd.to_datetime(df.index, errors="coerce")
    valid_dates = index_as_dates.notna()

    row_nonnull = df.notna().sum(axis=1)
    col_nonnull = df.notna().sum(axis=0)
    row_coverage = row_nonnull / cols if cols else row_nonnull
    col_coverage = col_nonnull / rows if rows else col_nonnull

    arr = df.to_numpy(dtype=float, copy=False)
    finite_mask = np.isfinite(arr)
    valid_count = int(finite_mask.sum())
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())

    if valid_count:
        valid_values = arr[finite_mask]
        numeric_summary = {
            "min": float(np.nanmin(arr)),
            "p001": float(np.quantile(valid_values, 0.001)),
            "p01": float(np.quantile(valid_values, 0.01)),
            "p05": float(np.quantile(valid_values, 0.05)),
            "median": float(np.quantile(valid_values, 0.50)),
            "mean": float(np.nanmean(arr)),
            "p95": float(np.quantile(valid_values, 0.95)),
            "p99": float(np.quantile(valid_values, 0.99)),
            "p999": float(np.quantile(valid_values, 0.999)),
            "max": float(np.nanmax(arr)),
            "std": float(np.nanstd(arr)),
        }
    else:
        numeric_summary = {}

    threshold_counts = {
        "gt_10pct": int((arr > 0.10).sum()),
        "lt_minus_10pct": int((arr < -0.10).sum()),
        "abs_gt_20pct": int((np.abs(arr) > 0.20).sum()),
        "abs_gt_50pct": int((np.abs(arr) > 0.50).sum()),
        "abs_gt_100pct": int((np.abs(arr) > 1.00).sum()),
        "gt_100pct": int((arr > 1.00).sum()),
        "lt_minus_80pct": int((arr < -0.80).sum()),
        "exact_zero": int((arr == 0.0).sum()),
        "positive": int((arr > 0.0).sum()),
        "negative": int((arr < 0.0).sum()),
    }

    top_extremes: list[dict[str, Any]] = []
    if valid_count:
        abs_arr = np.abs(arr)
        abs_arr = np.where(finite_mask, abs_arr, -1.0)
        flat = abs_arr.ravel()
        n = min(top_n, valid_count)
        idx = np.argpartition(flat, -n)[-n:]
        idx = idx[np.argsort(flat[idx])[::-1]]
        row_idx, col_idx = np.unravel_index(idx, arr.shape)
        top_extremes = [
            {
                "date": str(df.index[i]),
                "stock_code": str(df.columns[j]),
                "value": float(arr[i, j]),
                "abs_value": float(abs_arr[i, j]),
            }
            for i, j in zip(row_idx, col_idx)
        ]

    row_profile = pd.DataFrame(
        {
            "date": index_as_dates,
            "nonnull": row_nonnull.to_numpy(),
            "coverage": row_coverage.to_numpy(),
            "mean": df.mean(axis=1, skipna=True).to_numpy(),
            "std": df.std(axis=1, skipna=True).to_numpy(),
            "absmax": df.abs().max(axis=1, skipna=True).to_numpy(),
        }
    )
    row_profile["year"] = row_profile["date"].dt.year
    annual = (
        row_profile.dropna(subset=["year"])
        .groupby("year")
        .agg(
            rows=("nonnull", "size"),
            avg_nonnull=("nonnull", "mean"),
            min_nonnull=("nonnull", "min"),
            max_nonnull=("nonnull", "max"),
            avg_coverage=("coverage", "mean"),
            min_coverage=("coverage", "min"),
            max_coverage=("coverage", "max"),
            avg_absmax=("absmax", "mean"),
            max_absmax=("absmax", "max"),
        )
        .round(6)
    )

    malformed_codes = [str(c) for c in df.columns if not re.fullmatch(r"\d{6}", str(c))]
    prefix3_counts = pd.Series([str(c)[:3] for c in df.columns]).value_counts().head(20)

    summary: dict[str, Any] = {
        "source": {
            "input_path": input_path,
            "file_size_mb": round(input_path.stat().st_size / 1024 / 1024, 3),
        },
        "shape": {"rows": rows, "columns": cols},
        "grain_assumption": "one row per date, one column per stock_code, values are returns/labels",
        "schema": {
            "object_type": type(df).__name__,
            "index_type": type(df.index).__name__,
            "index_dtype": str(df.index.dtype),
            "index_name": df.index.name,
            "index_unique": bool(df.index.is_unique),
            "index_monotonic_increasing": bool(df.index.is_monotonic_increasing),
            "columns_type": type(df.columns).__name__,
            "columns_dtype": str(df.columns.dtype),
            "columns_name": df.columns.name,
            "columns_unique": bool(df.columns.is_unique),
            "dtype_counts": df.dtypes.astype(str).value_counts().to_dict(),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 3),
        },
        "dates": {
            "parseable": int(valid_dates.sum()),
            "total": rows,
            "min": index_as_dates.min(),
            "max": index_as_dates.max(),
            "weekend_rows": int(index_as_dates[index_as_dates.notna()].dayofweek.isin([5, 6]).sum()),
        },
        "columns": {
            "count": cols,
            "malformed_6_digit_count": len(malformed_codes),
            "malformed_6_digit_sample": malformed_codes[:20],
            "prefix3_top20": prefix3_counts.to_dict(),
            "first_20": [str(c) for c in df.columns[:20]],
            "last_10": [str(c) for c in df.columns[-10:]],
        },
        "completeness": {
            "total_cells": rows * cols,
            "missing_cells": nan_count,
            "missing_pct": round(nan_count / (rows * cols) * 100, 6) if rows * cols else None,
            "inf_cells": inf_count,
            "row_nonnull_min_median_max": [
                int(row_nonnull.min()),
                float(row_nonnull.median()),
                int(row_nonnull.max()),
            ],
            "row_coverage_quantiles": _series_quantiles(row_coverage, [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1]),
            "col_nonnull_min_median_max": [
                int(col_nonnull.min()),
                float(col_nonnull.median()),
                int(col_nonnull.max()),
            ],
            "col_coverage_quantiles": _series_quantiles(col_coverage, [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1]),
            "all_nan_rows_count": int((row_nonnull == 0).sum()),
            "all_nan_rows": [str(x) for x in row_nonnull[row_nonnull == 0].index[:50]],
            "all_nan_columns_count": int((col_nonnull == 0).sum()),
            "all_nan_columns": [str(x) for x in col_nonnull[col_nonnull == 0].index[:50]],
            "one_value_columns_count": int((col_nonnull == 1).sum()),
            "one_value_columns_sample": [str(x) for x in col_nonnull[col_nonnull == 1].index[:50]],
            "columns_lt_20_values": int((col_nonnull < 20).sum()),
            "columns_lt_60_values": int((col_nonnull < 60).sum()),
            "columns_lt_252_values": int((col_nonnull < 252).sum()),
            "columns_lt_500_values": int((col_nonnull < 500).sum()),
            "columns_lt_1000_values": int((col_nonnull < 1000).sum()),
            "lowest_coverage_rows": [
                {"date": str(idx), "nonnull": int(value), "coverage": float(value / cols)}
                for idx, value in row_nonnull.sort_values().head(20).items()
            ],
            "lowest_coverage_columns": [
                {"stock_code": str(idx), "nonnull": int(value), "coverage": float(value / rows)}
                for idx, value in col_nonnull.sort_values().head(20).items()
            ],
        },
        "numeric_validity": {
            "valid_numeric_cells": valid_count,
            "valid_numeric_pct": round(valid_count / (rows * cols) * 100, 6) if rows * cols else None,
            "summary": numeric_summary,
            "threshold_counts": threshold_counts,
            "threshold_rates_of_valid": {
                key: round(value / valid_count * 100, 6) if valid_count else None
                for key, value in threshold_counts.items()
            },
            "top_abs_extremes": top_extremes,
        },
        "temporal_profile": {
            "annual": annual.reset_index().to_dict(orient="records"),
            "last_20_rows": [
                {
                    "date": str(df.index[i]),
                    "nonnull": int(row_nonnull.iloc[i]),
                    "coverage": float(row_coverage.iloc[i]),
                    "mean": _json_value(row_profile["mean"].iloc[i]),
                    "std": _json_value(row_profile["std"].iloc[i]),
                    "absmax": _json_value(row_profile["absmax"].iloc[i]),
                }
                for i in range(max(0, rows - 20), rows)
            ],
        },
    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a date x stock return matrix pickle.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    summary = profile_return_matrix(args.input, top_n=args.top_n)
    text = json.dumps(summary, ensure_ascii=False, indent=2, default=_json_value)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
