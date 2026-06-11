from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


DEFAULT_SCORE_COL = "score_marginal_z"
DEFAULT_SPLITS = ["train", "valid", "test"]
TRADING_DAYS_PER_YEAR = 252.0


@dataclass(frozen=True)
class RankBucket:
    start: int
    end: int
    label: str | None = None

    def __post_init__(self) -> None:
        if self.start < 1:
            raise ValueError("RankBucket.start is 1-based and must be at least 1.")
        if self.end < self.start:
            raise ValueError("RankBucket.end must be greater than or equal to start.")


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest LGBM signal rank buckets as equal-weight gross NAV curves. "
            "The return matrix is assumed to be already aligned to signal dates."
        )
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--return-y", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--score-col", default=DEFAULT_SCORE_COL)
    parser.add_argument("--return-col", default="return_y_hfq_adj")
    parser.add_argument("--rank-bucket-size", default=500, type=int)
    parser.add_argument("--rank-buckets", nargs="*", default=None)
    parser.add_argument("--rank-order", default="descending", choices=["ascending", "descending"])
    parser.add_argument("--splits", nargs="*", default=DEFAULT_SPLITS)
    parser.add_argument("--fold-ids", nargs="*", default=None, type=int)
    parser.add_argument("--min-names", default=1, type=int)
    args = parser.parse_args(argv)

    predictions = _read_predictions(args.predictions, score_col=args.score_col)
    return_y = _read_return_matrix(args.return_y)
    rank_buckets = (
        [_parse_rank_bucket(value) for value in args.rank_buckets]
        if args.rank_buckets
        else None
    )
    summary = write_rank_bucket_nav_artifacts(
        predictions=predictions,
        return_y=return_y,
        output_dir=args.output_dir,
        score_col=args.score_col,
        return_col=args.return_col,
        splits=args.splits,
        fold_ids=args.fold_ids,
        rank_buckets=rank_buckets,
        rank_bucket_size=args.rank_bucket_size,
        rank_order=args.rank_order,
        min_names=args.min_names,
    )
    if argv is None:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def evaluate_rank_bucket_nav(
    *,
    predictions: pd.DataFrame,
    return_y: pd.DataFrame,
    score_col: str = DEFAULT_SCORE_COL,
    return_col: str = "return_y_hfq_adj",
    splits: Sequence[str] | None = DEFAULT_SPLITS,
    fold_ids: Sequence[int] | None = None,
    rank_buckets: Sequence[RankBucket] | None = None,
    rank_bucket_size: int = 500,
    rank_order: str = "descending",
    min_names: int = 1,
) -> dict[str, pd.DataFrame]:
    if rank_order not in {"ascending", "descending"}:
        raise ValueError("rank_order must be 'ascending' or 'descending'.")
    if rank_bucket_size < 1:
        raise ValueError("rank_bucket_size must be positive.")
    if min_names < 1:
        raise ValueError("min_names must be positive.")

    pred = _prepare_predictions(
        predictions,
        score_col=score_col,
        splits=splits,
        fold_ids=fold_ids,
    )
    returns = _prepare_return_matrix(return_y)
    if rank_buckets is None:
        rank_buckets = _default_rank_buckets(pred, score_col=score_col, bucket_size=rank_bucket_size)
    else:
        rank_buckets = tuple(rank_buckets)
    _validate_rank_buckets(rank_buckets)

    daily_returns = _compute_daily_bucket_returns(
        pred,
        returns,
        score_col=score_col,
        return_col=return_col,
        rank_buckets=rank_buckets,
        rank_order=rank_order,
        min_names=min_names,
    )
    nav = _compute_nav(daily_returns)
    bucket_summary = _summarize_buckets(daily_returns, nav)
    return {
        "daily_returns": daily_returns,
        "nav": nav,
        "summary": bucket_summary,
    }


def write_rank_bucket_nav_artifacts(
    *,
    predictions: pd.DataFrame,
    return_y: pd.DataFrame,
    output_dir: Path,
    score_col: str = DEFAULT_SCORE_COL,
    return_col: str = "return_y_hfq_adj",
    splits: Sequence[str] | None = DEFAULT_SPLITS,
    fold_ids: Sequence[int] | None = None,
    rank_buckets: Sequence[RankBucket] | None = None,
    rank_bucket_size: int = 500,
    rank_order: str = "descending",
    min_names: int = 1,
) -> dict[str, Any]:
    result = evaluate_rank_bucket_nav(
        predictions=predictions,
        return_y=return_y,
        score_col=score_col,
        return_col=return_col,
        splits=splits,
        fold_ids=fold_ids,
        rank_buckets=rank_buckets,
        rank_bucket_size=rank_bucket_size,
        rank_order=rank_order,
        min_names=min_names,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"
    paths = {
        "daily_returns": output_dir / "rank_bucket_daily_returns.csv",
        "nav": output_dir / "rank_bucket_nav.csv",
        "bucket_summary": output_dir / "rank_bucket_summary.csv",
        "evaluation_summary": output_dir / "rank_bucket_evaluation_summary.json",
    }
    _write_csv_with_iso_dates(result["daily_returns"], paths["daily_returns"])
    _write_csv_with_iso_dates(result["nav"], paths["nav"])
    _write_csv_with_iso_dates(result["summary"], paths["bucket_summary"])

    chart_paths = write_rank_bucket_nav_charts(result["nav"], charts_dir)
    summary = _build_summary(
        result,
        output_paths=paths,
        chart_paths=chart_paths,
        score_col=score_col,
        return_col=return_col,
        rank_buckets=rank_buckets,
        rank_bucket_size=rank_bucket_size,
        rank_order=rank_order,
        splits=splits,
        fold_ids=fold_ids,
        min_names=min_names,
    )
    paths["evaluation_summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def write_rank_bucket_nav_charts(nav: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _use_static_chart_theme()
    chart_paths: dict[str, Path] = {}
    if nav.empty:
        path = output_dir / "rank_bucket_nav_empty.png"
        _plot_nav_curves(nav, path, title_suffix="empty")
        chart_paths["empty"] = path
        return chart_paths

    for (fold_id, split), group in nav.groupby(["fold_id", "split"], sort=True):
        safe_split = _safe_filename(str(split))
        path = output_dir / f"rank_bucket_nav_fold_{fold_id}_{safe_split}.png"
        _plot_nav_curves(group, path, title_suffix=f"fold {fold_id} {split}")
        chart_paths[f"fold_{fold_id}_{safe_split}"] = path
    return chart_paths


def _compute_daily_bucket_returns(
    pred: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    score_col: str,
    return_col: str,
    rank_buckets: Sequence[RankBucket],
    rank_order: str,
    min_names: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    score_ascending = rank_order == "ascending"
    for (fold_id, split, signal_date), group in pred.groupby(["fold_id", "split", "date"], sort=True):
        scores = pd.to_numeric(group[score_col], errors="coerce")
        finite_signal = np.isfinite(scores.to_numpy(dtype=float))
        ranked = group.loc[finite_signal].copy()
        ranked[score_col] = scores.loc[finite_signal].astype(float)
        ranked = ranked.sort_values(
            [score_col, "stock_code"],
            ascending=[score_ascending, True],
            kind="mergesort",
        ).reset_index(drop=True)
        eligible_count = int(len(ranked))
        return_row = _return_row_for_date(returns, pd.Timestamp(signal_date))

        for bucket in rank_buckets:
            bucket_label = _bucket_label(bucket)
            selected = ranked.iloc[bucket.start - 1 : bucket.end].copy()
            selected_count = int(len(selected))
            selected_codes = selected["stock_code"].astype(str).tolist()
            selected_returns = pd.to_numeric(
                return_row.reindex(selected_codes),
                errors="coerce",
            )
            finite_return = np.isfinite(selected_returns.to_numpy(dtype=float))
            invalid_loss = selected_returns.loc[finite_return & selected_returns.le(-1.0)]
            if not invalid_loss.empty:
                sample = {
                    "date": _date_to_string(pd.Timestamp(signal_date)),
                    "stock_code": str(invalid_loss.index[0]),
                    "return": float(invalid_loss.iloc[0]),
                }
                raise ValueError(f"Return values must be greater than -1. Sample: {sample}")
            valid_return_count = int(finite_return.sum())
            missing_return_count = int(selected_count - valid_return_count)
            if valid_return_count >= min_names:
                bucket_return = float(selected_returns.loc[finite_return].mean())
            else:
                bucket_return = np.nan
            records.append(
                {
                    "fold_id": int(fold_id),
                    "split": str(split),
                    "score_col": score_col,
                    "return_col": return_col,
                    "signal_date": pd.Timestamp(signal_date),
                    "bucket_label": bucket_label,
                    "rank_start": int(bucket.start),
                    "rank_end": int(bucket.end),
                    "rank_bound_mode": "one_based_inclusive",
                    "rank_order": rank_order,
                    "tie_policy": "score_then_stock_code_stable",
                    "eligible_count": eligible_count,
                    "selected_count": selected_count,
                    "valid_return_count": valid_return_count,
                    "missing_return_count": missing_return_count,
                    "min_names": int(min_names),
                    "bucket_return": bucket_return,
                }
            )
    if not records:
        return _empty_daily_returns()
    return pd.DataFrame(records).sort_values(
        ["fold_id", "split", "signal_date", "rank_start"]
    ).reset_index(drop=True)


def _compute_nav(daily_returns: pd.DataFrame) -> pd.DataFrame:
    if daily_returns.empty:
        return _empty_nav()
    records: list[dict[str, Any]] = []
    group_cols = ["fold_id", "split", "score_col", "return_col", "bucket_label"]
    for key, group in daily_returns.groupby(group_cols, sort=True):
        fold_id, split, score_col, return_col, bucket_label = key
        running_nav = 1.0
        group = group.sort_values("signal_date")
        for row in group.itertuples(index=False):
            bucket_return = float(row.bucket_return) if np.isfinite(row.bucket_return) else np.nan
            nav_stale = not np.isfinite(bucket_return)
            applied_return = 0.0 if nav_stale else bucket_return
            running_nav *= 1.0 + applied_return
            records.append(
                {
                    "fold_id": int(fold_id),
                    "split": str(split),
                    "score_col": str(score_col),
                    "return_col": str(return_col),
                    "signal_date": pd.Timestamp(row.signal_date),
                    "bucket_label": str(bucket_label),
                    "rank_start": int(row.rank_start),
                    "rank_end": int(row.rank_end),
                    "bucket_return": bucket_return,
                    "applied_return": float(applied_return),
                    "gross_nav": float(running_nav),
                    "nav_base": 1.0,
                    "nav_stale_flag": bool(nav_stale),
                }
            )
    return pd.DataFrame(records).sort_values(
        ["fold_id", "split", "bucket_label", "signal_date"]
    ).reset_index(drop=True)


def _summarize_buckets(daily_returns: pd.DataFrame, nav: pd.DataFrame) -> pd.DataFrame:
    if daily_returns.empty:
        return _empty_bucket_summary()
    records: list[dict[str, Any]] = []
    group_cols = ["fold_id", "split", "score_col", "return_col", "bucket_label"]
    nav_lookup = {
        key: group.sort_values("signal_date")
        for key, group in nav.groupby(group_cols, sort=True)
    }
    for key, group in daily_returns.groupby(group_cols, sort=True):
        fold_id, split, score_col, return_col, bucket_label = key
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
        annualized_vol = _annualized_vol(applied_returns)
        sharpe = _annualized_sharpe(applied_returns)
        records.append(
            {
                "fold_id": int(fold_id),
                "split": str(split),
                "score_col": str(score_col),
                "return_col": str(return_col),
                "bucket_label": str(bucket_label),
                "rank_start": int(group["rank_start"].iloc[0]),
                "rank_end": int(group["rank_end"].iloc[0]),
                "date_count": date_count,
                "valid_return_date_count": int(len(finite_returns)),
                "empty_date_count": int(date_count - len(finite_returns)),
                "mean_daily_return": _nanmean(finite_returns),
                "std_daily_return": _nanstd(finite_returns),
                "annualized_return": annualized_return,
                "annualized_vol": annualized_vol,
                "sharpe": sharpe,
                "max_drawdown": _max_drawdown(nav_group["gross_nav"] if not nav_group.empty else []),
                "positive_rate": _positive_rate(finite_returns),
                "mean_selected_count": float(group["selected_count"].mean()) if date_count else np.nan,
                "mean_valid_return_count": float(group["valid_return_count"].mean()) if date_count else np.nan,
                "min_selected_count": int(group["selected_count"].min()) if date_count else 0,
                "min_valid_return_count": int(group["valid_return_count"].min()) if date_count else 0,
            }
        )
    return pd.DataFrame(records).sort_values(
        ["fold_id", "split", "rank_start"]
    ).reset_index(drop=True)


def _build_summary(
    result: dict[str, pd.DataFrame],
    *,
    output_paths: dict[str, Path],
    chart_paths: dict[str, Path],
    score_col: str,
    return_col: str,
    rank_buckets: Sequence[RankBucket] | None,
    rank_bucket_size: int,
    rank_order: str,
    splits: Sequence[str] | None,
    fold_ids: Sequence[int] | None,
    min_names: int,
) -> dict[str, Any]:
    daily = result["daily_returns"]
    buckets = (
        list(rank_buckets)
        if rank_buckets is not None
        else [
            RankBucket(
                start=int(row.rank_start),
                end=int(row.rank_end),
                label=str(row.bucket_label),
            )
            for row in daily[["rank_start", "rank_end", "bucket_label"]]
            .drop_duplicates()
            .sort_values(["rank_start", "rank_end"])
            .itertuples(index=False)
        ]
    )
    return {
        "schema_version": "lgbm_rank_bucket_nav_v1",
        "metric_contract": {
            "score_col": score_col,
            "return_col": return_col,
            "return_alignment": "already_aligned_to_signal_date",
            "rank_scope": "daily_cross_section_within_fold_split",
            "rank_order": rank_order,
            "rank_bound_mode": "one_based_inclusive",
            "tie_policy": "score_then_stock_code_stable",
            "bucket_weighting": "equal_weight_valid_returns_within_date_bucket",
            "missing_return_policy": "drop_and_renormalize_within_bucket",
            "empty_bucket_nav_policy": "carry_forward_previous_nav_with_applied_return_zero",
            "cost_model": "gross_no_cost",
            "nav_base": 1.0,
            "nav_formula": "gross_nav_t = gross_nav_t_minus_1 * (1 + applied_return_t)",
            "annualized_return_formula": "gross_nav_end ** (252 / date_count) - 1",
        },
        "filters": {
            "splits": list(splits) if splits is not None else None,
            "fold_ids": list(fold_ids) if fold_ids is not None else None,
        },
        "rank_bucket_size": int(rank_bucket_size),
        "rank_buckets": [
            {
                **asdict(bucket),
                "label": _bucket_label(bucket),
                "bound_mode": "one_based_inclusive",
            }
            for bucket in buckets
        ],
        "min_names": int(min_names),
        "row_counts": {
            "daily_returns": int(len(result["daily_returns"])),
            "nav": int(len(result["nav"])),
            "bucket_summary": int(len(result["summary"])),
        },
        "outputs": {key: _path_for_summary(path) for key, path in output_paths.items()},
        "charts": {key: _path_for_summary(path) for key, path in chart_paths.items()},
        "notes": [
            "Rank buckets are assigned before looking at realized returns.",
            "Only finite signal values enter the daily ranking universe.",
            "Missing realized returns are excluded within the selected bucket and counted in diagnostics.",
            "Each fold_id and split has an independent NAV path; paths are not stitched together.",
            "The module assumes the provided return matrix is already adjusted and aligned to signal dates.",
        ],
    }


def _prepare_predictions(
    predictions: pd.DataFrame,
    *,
    score_col: str,
    splits: Sequence[str] | None,
    fold_ids: Sequence[int] | None,
) -> pd.DataFrame:
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError(f"predictions must be a pandas DataFrame, got {type(predictions)!r}")
    required = {"fold_id", "split", "date", "stock_code", score_col}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing required columns: {missing}")
    frame = predictions[["fold_id", "split", "date", "stock_code", score_col]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["stock_code"] = frame["stock_code"].map(_normalize_stock_code)
    frame["split"] = frame["split"].astype(str)
    frame["score_col"] = score_col
    frame[score_col] = pd.to_numeric(frame[score_col], errors="coerce")
    frame = frame[frame["date"].notna() & frame["stock_code"].ne("")].copy()
    frame["fold_id"] = pd.to_numeric(frame["fold_id"], errors="raise").astype(int)
    if splits is not None:
        split_set = {str(split) for split in splits}
        frame = frame[frame["split"].isin(split_set)].copy()
    if fold_ids is not None:
        fold_set = {int(fold_id) for fold_id in fold_ids}
        frame = frame[frame["fold_id"].isin(fold_set)].copy()
    duplicate_keys = ["fold_id", "split", "date", "stock_code"]
    if frame.duplicated(duplicate_keys).any():
        sample = frame.loc[frame.duplicated(duplicate_keys, keep=False), duplicate_keys].head(5)
        raise ValueError(f"Predictions contain duplicate fold/split/date/stock_code rows: {sample.to_dict('records')}")
    return frame.sort_values(["fold_id", "split", "date", "stock_code"]).reset_index(drop=True)


def _prepare_return_matrix(return_y: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(return_y, pd.DataFrame):
        raise TypeError(f"return_y must be a pandas DataFrame, got {type(return_y)!r}")
    frame = return_y.copy()
    frame.index = pd.DatetimeIndex(pd.to_datetime(list(frame.index), errors="coerce")).normalize()
    frame = frame.loc[~frame.index.isna()].copy()
    if frame.index.duplicated().any():
        duplicates = sorted({pd.Timestamp(date).date().isoformat() for date in frame.index[frame.index.duplicated()]})
        raise ValueError(f"Return matrix contains duplicate normalized dates: {duplicates[:5]}")
    normalized_columns = [_normalize_stock_code(column) for column in frame.columns]
    keep_columns = [bool(column) for column in normalized_columns]
    frame = frame.loc[:, keep_columns].copy()
    normalized_columns = [column for column in normalized_columns if column]
    if len(set(normalized_columns)) != len(normalized_columns):
        duplicates = sorted({column for column in normalized_columns if normalized_columns.count(column) > 1})
        raise ValueError(f"Return matrix contains duplicate normalized stock codes: {duplicates[:5]}")
    frame.columns = normalized_columns
    frame = frame.apply(pd.to_numeric, errors="coerce")
    return frame.sort_index()


def _default_rank_buckets(
    pred: pd.DataFrame,
    *,
    score_col: str,
    bucket_size: int,
) -> tuple[RankBucket, ...]:
    if pred.empty:
        return tuple()
    counts = (
        pred[np.isfinite(pd.to_numeric(pred[score_col], errors="coerce").to_numpy(dtype=float))]
        .groupby(["fold_id", "split", "date"], sort=False)
        .size()
    )
    max_count = int(counts.max()) if not counts.empty else 0
    buckets: list[RankBucket] = []
    start = 1
    while start <= max_count:
        end = min(start + bucket_size - 1, max_count)
        buckets.append(RankBucket(start=start, end=end))
        start = end + 1
    return tuple(buckets)


def _validate_rank_buckets(rank_buckets: Sequence[RankBucket]) -> None:
    previous_end = 0
    seen_labels: set[str] = set()
    for bucket in sorted(rank_buckets, key=lambda item: (item.start, item.end)):
        if bucket.start <= previous_end:
            raise ValueError("Rank buckets must be non-overlapping and sorted by rank range.")
        label = _bucket_label(bucket)
        if label in seen_labels:
            raise ValueError(f"Duplicate rank bucket label: {label!r}")
        seen_labels.add(label)
        previous_end = bucket.end


def _return_row_for_date(returns: pd.DataFrame, signal_date: pd.Timestamp) -> pd.Series:
    date = pd.Timestamp(signal_date).normalize()
    if date not in returns.index:
        return pd.Series(dtype=float)
    row = returns.loc[date]
    if isinstance(row, pd.DataFrame):
        raise ValueError(f"Return matrix contains multiple rows for date {date.date().isoformat()}.")
    return pd.to_numeric(row, errors="coerce")


def _parse_rank_bucket(value: str) -> RankBucket:
    text = str(value).strip()
    match = re.fullmatch(r"(\d+)\s*[:-]\s*(\d+)(?:=(.+))?", text)
    if not match:
        raise ValueError(f"Invalid rank bucket {value!r}; expected START:END or START-END.")
    raw_start = int(match.group(1))
    end = int(match.group(2))
    label = match.group(3).strip() if match.group(3) else None
    start = 1 if raw_start == 0 else raw_start
    return RankBucket(start=start, end=end, label=label)


def _bucket_label(bucket: RankBucket) -> str:
    if bucket.label:
        return bucket.label
    width = max(4, len(str(bucket.end)))
    return f"rank_{bucket.start:0{width}d}_{bucket.end:0{width}d}"


def _normalize_stock_code(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if "." in text:
        prefix = text.split(".", 1)[0]
        if prefix.isdigit():
            text = prefix
    if text.isdigit():
        return text.zfill(6)
    return text


def _read_predictions(path: Path, *, score_col: str) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        frame = pd.read_pickle(path)
    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        usecols = {"fold_id", "split", "date", "stock_code", score_col}
        frame = pd.read_csv(
            path,
            dtype={"stock_code": "string", "split": "string"},
            usecols=lambda column: column in usecols,
        )
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected predictions DataFrame at {path}, got {type(frame)!r}")
    return frame


def _read_return_matrix(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        frame = pd.read_pickle(path)
    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, index_col=0)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected return matrix DataFrame at {path}, got {type(frame)!r}")
    return frame


def _write_csv_with_iso_dates(frame: pd.DataFrame, path: Path) -> None:
    output = frame.copy()
    for column in output.columns:
        if is_datetime64_any_dtype(output[column]):
            output[column] = pd.to_datetime(output[column]).dt.date.astype(str)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, encoding="utf-8-sig")


def _use_static_chart_theme() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "#f7f9fc",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#d7deea",
            "axes.labelcolor": "#283142",
            "axes.titlecolor": "#202938",
            "xtick.color": "#667085",
            "ytick.color": "#667085",
            "grid.color": "#dce5f2",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "#f7f9fc",
            "savefig.dpi": 160,
        }
    )


def _plot_nav_curves(nav: pd.DataFrame, path: Path, *, title_suffix: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.5, 6.0), constrained_layout=True)
    _chart_title(
        fig,
        "LGBM signal rank-bucket gross NAV",
        f"Equal-weight bucket curves, {title_suffix}",
    )
    if nav.empty:
        _plot_empty(ax, "No NAV data available")
        _save_chart(fig, path)
        return
    frame = nav.sort_values(["rank_start", "signal_date"]).copy()
    for bucket_label, group in frame.groupby("bucket_label", sort=False):
        group = group.sort_values("signal_date")
        avg_n = group["gross_nav"].notna().sum()
        ax.plot(
            group["signal_date"],
            group["gross_nav"].astype(float),
            linewidth=1.6,
            label=f"{bucket_label} ({int(avg_n)} days)",
        )
    ax.axhline(1.0, color="#27384c", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Signal date")
    ax.set_ylabel("Gross NAV")
    ax.grid(axis="y")
    ax.legend(frameon=False, fontsize=8.2, ncols=2)
    fig.autofmt_xdate()
    _save_chart(fig, path)


def _chart_title(fig: Any, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.02, ha="left", fontsize=15.0, fontweight="bold")
    fig.text(0.02, 0.925, subtitle, ha="left", va="top", color="#667085", fontsize=9.5)


def _plot_empty(ax: Any, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _save_chart(fig: Any, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def _max_drawdown(nav_values: Iterable[Any]) -> float:
    values = _finite_values(nav_values)
    if len(values) == 0:
        return np.nan
    path = np.concatenate([[1.0], values])
    high_water = np.maximum.accumulate(path)
    drawdowns = path / high_water - 1.0
    return float(np.min(drawdowns))


def _annualized_vol(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return np.nan
    return float(np.std(finite, ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR))


def _annualized_sharpe(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return np.nan
    std = float(np.std(finite, ddof=0))
    if std <= 1e-12:
        return np.nan
    return float(np.mean(finite) / std * math.sqrt(TRADING_DAYS_PER_YEAR))


def _positive_rate(values: np.ndarray) -> float:
    return float(np.mean(values > 0.0)) if len(values) else np.nan


def _nanmean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else np.nan


def _nanstd(values: np.ndarray) -> float:
    return float(np.std(values, ddof=0)) if len(values) else np.nan


def _finite_values(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def _date_to_string(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).date().isoformat()


def _path_for_summary(path: Path) -> str:
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.name


def _empty_daily_returns() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "fold_id",
            "split",
            "score_col",
            "return_col",
            "signal_date",
            "bucket_label",
            "rank_start",
            "rank_end",
            "rank_bound_mode",
            "rank_order",
            "tie_policy",
            "eligible_count",
            "selected_count",
            "valid_return_count",
            "missing_return_count",
            "min_names",
            "bucket_return",
        ]
    )


def _empty_nav() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "fold_id",
            "split",
            "score_col",
            "return_col",
            "signal_date",
            "bucket_label",
            "rank_start",
            "rank_end",
            "bucket_return",
            "applied_return",
            "gross_nav",
            "nav_base",
            "nav_stale_flag",
        ]
    )


def _empty_bucket_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "fold_id",
            "split",
            "score_col",
            "return_col",
            "bucket_label",
            "rank_start",
            "rank_end",
            "date_count",
            "valid_return_date_count",
            "empty_date_count",
            "mean_daily_return",
            "std_daily_return",
            "annualized_return",
            "annualized_vol",
            "sharpe",
            "max_drawdown",
            "positive_rate",
            "mean_selected_count",
            "mean_valid_return_count",
            "min_selected_count",
            "min_valid_return_count",
        ]
    )


if __name__ == "__main__":
    main()
