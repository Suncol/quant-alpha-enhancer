from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from analysis.evaluate_lgbm_rank_bucket_nav import (
        TRADING_DAYS_PER_YEAR,
        _annualized_sharpe,
        _annualized_vol,
        _finite_values,
        _max_drawdown,
        _path_for_summary,
        _positive_rate,
        _write_csv_with_iso_dates,
    )
except ModuleNotFoundError:  # Allows direct execution via python analysis/script.py.
    from evaluate_lgbm_rank_bucket_nav import (
        TRADING_DAYS_PER_YEAR,
        _annualized_sharpe,
        _annualized_vol,
        _finite_values,
        _max_drawdown,
        _path_for_summary,
        _positive_rate,
        _write_csv_with_iso_dates,
    )


@dataclass(frozen=True)
class PeriodSpec:
    label: str
    start: str | pd.Timestamp
    end: str | pd.Timestamp

    @property
    def start_ts(self) -> pd.Timestamp:
        return _normalize_date(self.start)

    @property
    def end_ts(self) -> pd.Timestamp:
        return _normalize_date(self.end)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize rank-bucket daily returns over explicit date periods without "
            "using fold_id as a grouping key."
        )
    )
    parser.add_argument("--daily-returns", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--periods",
        nargs="+",
        required=True,
        help="Inclusive period specs formatted as label:start_date:end_date.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["test"],
        help="Split values to include before dropping the fold dimension. Defaults to test.",
    )
    args = parser.parse_args(argv)

    daily_returns = pd.read_csv(args.daily_returns)
    summary = write_rank_bucket_period_artifacts(
        daily_returns=daily_returns,
        periods=[parse_period_spec(value) for value in args.periods],
        output_dir=args.output_dir,
        splits=args.splits,
    )
    if argv is None:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_period_spec(value: str) -> PeriodSpec:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError("Period specs must be formatted as label:start_date:end_date.")
    label, start, end = (part.strip() for part in parts)
    if not label:
        raise ValueError("Period label must be non-empty.")
    return PeriodSpec(label=label, start=start, end=end)


def summarize_rank_bucket_periods(
    daily_returns: pd.DataFrame,
    *,
    periods: Sequence[PeriodSpec],
    splits: Sequence[str] | None = ("test",),
) -> dict[str, pd.DataFrame]:
    period_specs = tuple(periods)
    if not period_specs:
        raise ValueError("At least one period must be supplied.")
    _validate_periods(period_specs)

    frame = _prepare_daily_returns(daily_returns, splits=splits)
    assigned = _assign_periods(frame, period_specs)
    _validate_unique_period_date_buckets(assigned)

    nav = _compute_period_nav(assigned)
    summary = _summarize_period_buckets(assigned, nav)
    requested_table = _requested_metric_table(summary)
    return {
        "period_daily_returns": assigned,
        "period_nav": nav,
        "summary": summary,
        "requested_table": requested_table,
    }


def write_rank_bucket_period_artifacts(
    *,
    daily_returns: pd.DataFrame,
    periods: Sequence[PeriodSpec],
    output_dir: Path,
    splits: Sequence[str] | None = ("test",),
) -> dict[str, Any]:
    result = summarize_rank_bucket_periods(
        daily_returns,
        periods=periods,
        splits=splits,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    paths = {
        "period_daily_returns": output_dir / "rank_bucket_period_daily_returns.csv",
        "period_nav": output_dir / "rank_bucket_period_nav.csv",
        "period_summary": output_dir / "rank_bucket_period_summary.csv",
        "requested_table": output_dir / "rank_bucket_period_requested_table.csv",
        "evaluation_summary": output_dir / "rank_bucket_period_evaluation_summary.json",
    }
    _write_csv_with_iso_dates(result["period_daily_returns"], paths["period_daily_returns"])
    _write_csv_with_iso_dates(result["period_nav"], paths["period_nav"])
    _write_csv_with_iso_dates(result["summary"], paths["period_summary"])
    _write_csv_with_iso_dates(result["requested_table"], paths["requested_table"])

    table_paths: dict[str, Path] = {}
    for period_label, group in result["requested_table"].groupby("period_label", sort=True):
        path = tables_dir / f"rank_bucket_period_{_safe_filename(str(period_label))}.csv"
        _write_csv_with_iso_dates(group.drop(columns=["period_label"]), path)
        table_paths[str(period_label)] = path

    summary = _build_period_artifact_summary(
        result,
        periods=periods,
        output_paths=paths,
        table_paths=table_paths,
        splits=splits,
    )
    paths["evaluation_summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _prepare_daily_returns(
    daily_returns: pd.DataFrame,
    *,
    splits: Sequence[str] | None,
) -> pd.DataFrame:
    required = {
        "fold_id",
        "split",
        "score_col",
        "return_col",
        "signal_date",
        "bucket_label",
        "bucket_index",
        "bucket_count",
        "bucket_mode",
        "selected_count",
        "valid_return_count",
        "bucket_return",
    }
    missing = sorted(required.difference(daily_returns.columns))
    if missing:
        raise ValueError(f"Daily returns are missing required columns: {missing}")

    frame = daily_returns[list(required)].copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise").dt.normalize()
    frame["split"] = frame["split"].astype(str)
    frame["score_col"] = frame["score_col"].astype(str)
    frame["return_col"] = frame["return_col"].astype(str)
    frame["bucket_label"] = frame["bucket_label"].astype(str)
    frame["bucket_index"] = pd.to_numeric(frame["bucket_index"], errors="raise").astype(int)
    frame["bucket_count"] = pd.to_numeric(frame["bucket_count"], errors="raise").astype(int)
    frame["selected_count"] = pd.to_numeric(frame["selected_count"], errors="raise").astype(int)
    frame["valid_return_count"] = pd.to_numeric(
        frame["valid_return_count"],
        errors="raise",
    ).astype(int)
    frame["bucket_return"] = pd.to_numeric(frame["bucket_return"], errors="coerce")

    if splits is not None:
        split_set = {str(split) for split in splits}
        frame = frame[frame["split"].isin(split_set)].copy()
    return frame.sort_values(["signal_date", "bucket_index", "bucket_label"]).reset_index(drop=True)


def _assign_periods(frame: pd.DataFrame, periods: Sequence[PeriodSpec]) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for period in periods:
        start = period.start_ts
        end = period.end_ts
        mask = frame["signal_date"].between(start, end, inclusive="both")
        period_frame = frame.loc[mask].copy()
        if period_frame.empty:
            continue
        period_frame["period_label"] = str(period.label)
        period_frame["period_start"] = start
        period_frame["period_end"] = end
        records.append(period_frame)
    if not records:
        return _empty_period_daily_returns()
    assigned = pd.concat(records, ignore_index=True)
    return assigned.sort_values(
        ["period_start", "period_label", "signal_date", "bucket_index", "bucket_label"]
    ).reset_index(drop=True)


def _compute_period_nav(period_daily_returns: pd.DataFrame) -> pd.DataFrame:
    if period_daily_returns.empty:
        return _empty_period_nav()
    records: list[dict[str, Any]] = []
    group_cols = [
        "period_label",
        "period_start",
        "period_end",
        "score_col",
        "return_col",
        "bucket_label",
    ]
    for key, group in period_daily_returns.groupby(group_cols, sort=True):
        period_label, period_start, period_end, score_col, return_col, bucket_label = key
        group = group.sort_values("signal_date")
        running_nav = 1.0
        bucket_index = int(group["bucket_index"].iloc[0])
        bucket_count = int(group["bucket_count"].iloc[0])
        bucket_mode = str(group["bucket_mode"].iloc[0])
        for row in group.itertuples(index=False):
            bucket_return = (
                float(row.bucket_return)
                if row.bucket_return is not None and np.isfinite(row.bucket_return)
                else np.nan
            )
            nav_stale = not np.isfinite(bucket_return)
            applied_return = 0.0 if nav_stale else bucket_return
            running_nav *= 1.0 + applied_return
            records.append(
                {
                    "period_label": str(period_label),
                    "period_start": pd.Timestamp(period_start),
                    "period_end": pd.Timestamp(period_end),
                    "score_col": str(score_col),
                    "return_col": str(return_col),
                    "signal_date": pd.Timestamp(row.signal_date),
                    "bucket_label": str(bucket_label),
                    "bucket_index": bucket_index,
                    "bucket_count": bucket_count,
                    "bucket_mode": bucket_mode,
                    "bucket_return": bucket_return,
                    "applied_return": float(applied_return),
                    "gross_nav": float(running_nav),
                    "nav_base": 1.0,
                    "nav_stale_flag": bool(nav_stale),
                }
            )
    return pd.DataFrame(records).sort_values(
        ["period_start", "period_label", "bucket_index", "bucket_label", "signal_date"]
    ).reset_index(drop=True)


def _summarize_period_buckets(
    period_daily_returns: pd.DataFrame,
    period_nav: pd.DataFrame,
) -> pd.DataFrame:
    if period_daily_returns.empty:
        return _empty_period_summary()
    records: list[dict[str, Any]] = []
    group_cols = [
        "period_label",
        "period_start",
        "period_end",
        "score_col",
        "return_col",
        "bucket_label",
    ]
    nav_lookup = {
        key: group.sort_values("signal_date")
        for key, group in period_nav.groupby(group_cols, sort=True)
    }
    for key, group in period_daily_returns.groupby(group_cols, sort=True):
        period_label, period_start, period_end, score_col, return_col, bucket_label = key
        group = group.sort_values("signal_date")
        nav_group = nav_lookup.get(key, pd.DataFrame())
        date_count = int(len(group))
        finite_returns = _finite_values(group["bucket_return"])
        applied_returns = (
            pd.to_numeric(nav_group["applied_return"], errors="coerce").to_numpy(dtype=float)
            if not nav_group.empty
            else np.array([], dtype=float)
        )
        gross_nav_end = (
            float(nav_group["gross_nav"].iloc[-1])
            if not nav_group.empty and np.isfinite(float(nav_group["gross_nav"].iloc[-1]))
            else np.nan
        )
        annualized_return = (
            float(gross_nav_end ** (TRADING_DAYS_PER_YEAR / date_count) - 1.0)
            if date_count > 0 and np.isfinite(gross_nav_end) and gross_nav_end > 0.0
            else np.nan
        )
        records.append(
            {
                "period_label": str(period_label),
                "period_start": pd.Timestamp(period_start),
                "period_end": pd.Timestamp(period_end),
                "actual_start": pd.Timestamp(group["signal_date"].min()),
                "actual_end": pd.Timestamp(group["signal_date"].max()),
                "score_col": str(score_col),
                "return_col": str(return_col),
                "bucket_label": str(bucket_label),
                "bucket_index": int(group["bucket_index"].iloc[0]),
                "bucket_count": int(group["bucket_count"].iloc[0]),
                "bucket_mode": str(group["bucket_mode"].iloc[0]),
                "date_count": date_count,
                "valid_return_date_count": int(len(finite_returns)),
                "empty_date_count": int(date_count - len(finite_returns)),
                "mean_daily_return": _nanmean(finite_returns),
                "std_daily_return": _nanstd(finite_returns),
                "gross_nav_end": gross_nav_end,
                "annualized_return": annualized_return,
                "annualized_vol": _annualized_vol(applied_returns),
                "sharpe": _annualized_sharpe(applied_returns),
                "max_drawdown": _max_drawdown(nav_group["gross_nav"] if not nav_group.empty else []),
                "longest_underwater_trading_days": _longest_underwater_trading_days(
                    nav_group["gross_nav"] if not nav_group.empty else []
                ),
                "positive_rate": _positive_rate(finite_returns),
                "mean_selected_count": float(group["selected_count"].mean()) if date_count else np.nan,
                "mean_valid_return_count": (
                    float(group["valid_return_count"].mean()) if date_count else np.nan
                ),
                "missing_return_sum": int(
                    (group["selected_count"] - group["valid_return_count"]).sum()
                ),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["period_start", "period_label", "bucket_index", "bucket_label"]
    ).reset_index(drop=True)


def _requested_metric_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return _empty_requested_table()
    bucket_width = max(2, len(str(int(summary["bucket_count"].max()))))
    table = pd.DataFrame(
        {
            "period_label": summary["period_label"],
            "period_start": summary["period_start"],
            "period_end": summary["period_end"],
            "actual_start": summary["actual_start"],
            "actual_end": summary["actual_end"],
            "bucket": summary["bucket_index"].map(lambda value: f"{int(value):0{bucket_width}d}"),
            "mean_daily_return": summary["mean_daily_return"],
            "ann_return": summary["annualized_return"],
            "nav_end": summary["gross_nav_end"],
            "sharpe": summary["sharpe"],
            "pos_rate": summary["positive_rate"],
            "max_drawdown": summary["max_drawdown"],
            "longest_underwater_days": summary["longest_underwater_trading_days"].astype(int),
        }
    )
    return table


def _build_period_artifact_summary(
    result: dict[str, pd.DataFrame],
    *,
    periods: Sequence[PeriodSpec],
    output_paths: dict[str, Path],
    table_paths: dict[str, Path],
    splits: Sequence[str] | None,
) -> dict[str, Any]:
    daily = result["period_daily_returns"]
    summary = result["summary"]
    return {
        "schema_version": "lgbm_rank_bucket_period_summary_v1",
        "metric_contract": {
            "source": "rank_bucket_daily_returns",
            "fold_usage": "fold_id is retained only for duplicate validation and is not a grouping key",
            "period_boundaries": "inclusive signal_date windows",
            "return_alignment": "inherits rank_bucket_daily_returns alignment",
            "nav_rule": "gross NAV starts at 1.0 within each period and bucket; NaN bucket returns apply 0.0 for NAV only",
            "annualized_return_formula": "gross_nav_end ** (252 / date_count) - 1",
            "annualized_vol_formula": "std(applied_daily_returns, ddof=0) * sqrt(252)",
            "sharpe_formula": "mean(applied_daily_returns) / std(applied_daily_returns, ddof=0) * sqrt(252)",
            "positive_rate_formula": "mean(finite bucket_return > 0)",
            "max_drawdown_formula": "min(gross_nav / running_high_water - 1), including base NAV 1.0",
            "longest_underwater_unit": "trading days",
            "longest_underwater_rule": (
                "maximum consecutive signal_date rows where gross_nav is below its prior high-water mark; "
                "a recovery/new-high day is not counted"
            ),
        },
        "periods": [
            {
                "label": str(period.label),
                "start": period.start_ts.date().isoformat(),
                "end": period.end_ts.date().isoformat(),
            }
            for period in periods
        ],
        "splits": list(splits) if splits is not None else None,
        "row_counts": {
            "period_daily_returns": int(len(daily)),
            "period_nav": int(len(result["period_nav"])),
            "period_summary": int(len(summary)),
            "requested_table": int(len(result["requested_table"])),
        },
        "bucket_count": (
            int(summary["bucket_count"].max())
            if not summary.empty and summary["bucket_count"].notna().any()
            else 0
        ),
        "period_count": int(summary["period_label"].nunique()) if not summary.empty else 0,
        "outputs": {
            key: _path_for_summary(path)
            for key, path in output_paths.items()
        }
        | {
            "period_tables": {
                key: _path_for_summary(path)
                for key, path in sorted(table_paths.items())
            }
        },
    }


def _validate_periods(periods: Sequence[PeriodSpec]) -> None:
    normalized: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    labels: set[str] = set()
    for period in periods:
        label = str(period.label)
        if not label:
            raise ValueError("Period label must be non-empty.")
        if label in labels:
            raise ValueError(f"Duplicate period label: {label}")
        labels.add(label)
        start = period.start_ts
        end = period.end_ts
        if end < start:
            raise ValueError(f"Period end is before start for {label}.")
        normalized.append((start, end, label))
    normalized.sort(key=lambda item: (item[0], item[1], item[2]))
    for prev, current in zip(normalized, normalized[1:]):
        prev_start, prev_end, prev_label = prev
        current_start, current_end, current_label = current
        if current_start <= prev_end:
            raise ValueError(
                "Period windows must not overlap: "
                f"{prev_label} {prev_start.date()}:{prev_end.date()} and "
                f"{current_label} {current_start.date()}:{current_end.date()}"
            )


def _validate_unique_period_date_buckets(period_daily_returns: pd.DataFrame) -> None:
    if period_daily_returns.empty:
        return
    duplicate_keys = [
        "period_label",
        "signal_date",
        "score_col",
        "return_col",
        "bucket_label",
        "bucket_index",
    ]
    duplicated = period_daily_returns.duplicated(duplicate_keys, keep=False)
    if duplicated.any():
        sample = period_daily_returns.loc[
            duplicated,
            duplicate_keys + ["fold_id", "split"],
        ].head(10)
        raise ValueError(
            "Period daily returns contain duplicate period/date/bucket rows after dropping "
            f"fold_id as a grouping key: {sample.to_dict('records')}"
        )


def _longest_underwater_trading_days(nav_values: Iterable[Any]) -> int:
    values = _finite_values(nav_values)
    high_water = 1.0
    current = 0
    longest = 0
    for value in values:
        nav = float(value)
        if nav >= high_water:
            high_water = nav
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return int(longest)


def _normalize_date(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value).normalize()
    if pd.isna(timestamp):
        raise ValueError(f"Invalid period date: {value!r}")
    return timestamp


def _safe_filename(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def _nanmean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else np.nan


def _nanstd(values: np.ndarray) -> float:
    return float(np.std(values, ddof=0)) if len(values) else np.nan


def _empty_period_daily_returns() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "fold_id",
            "split",
            "score_col",
            "return_col",
            "signal_date",
            "bucket_label",
            "bucket_index",
            "bucket_count",
            "bucket_mode",
            "selected_count",
            "valid_return_count",
            "bucket_return",
            "period_label",
            "period_start",
            "period_end",
        ]
    )


def _empty_period_nav() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "period_label",
            "period_start",
            "period_end",
            "score_col",
            "return_col",
            "signal_date",
            "bucket_label",
            "bucket_index",
            "bucket_count",
            "bucket_mode",
            "bucket_return",
            "applied_return",
            "gross_nav",
            "nav_base",
            "nav_stale_flag",
        ]
    )


def _empty_period_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "period_label",
            "period_start",
            "period_end",
            "actual_start",
            "actual_end",
            "score_col",
            "return_col",
            "bucket_label",
            "bucket_index",
            "bucket_count",
            "bucket_mode",
            "date_count",
            "valid_return_date_count",
            "empty_date_count",
            "mean_daily_return",
            "std_daily_return",
            "gross_nav_end",
            "annualized_return",
            "annualized_vol",
            "sharpe",
            "max_drawdown",
            "longest_underwater_trading_days",
            "positive_rate",
            "mean_selected_count",
            "mean_valid_return_count",
            "missing_return_sum",
        ]
    )


def _empty_requested_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "period_label",
            "period_start",
            "period_end",
            "actual_start",
            "actual_end",
            "bucket",
            "mean_daily_return",
            "ann_return",
            "nav_end",
            "sharpe",
            "pos_rate",
            "max_drawdown",
            "longest_underwater_days",
        ]
    )


if __name__ == "__main__":
    main()
