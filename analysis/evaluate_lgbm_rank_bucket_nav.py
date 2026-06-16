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
COMPOSITION_FEATURE_COLUMNS = (
    "industry",
    "market_cap",
    "log_mcap",
    "log_mcap_z",
    "mcap_rank",
    "size_decile",
)
SIZE_PROFILE_METRICS = ("market_cap", "log_mcap", "log_mcap_z", "mcap_rank", "size_decile")


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
    parser.add_argument("--rank-bucket-count", default=None, type=int)
    parser.add_argument("--rank-buckets", nargs="*", default=None)
    parser.add_argument("--rank-order", default="descending", choices=["ascending", "descending"])
    parser.add_argument("--splits", nargs="*", default=DEFAULT_SPLITS)
    parser.add_argument("--fold-ids", nargs="*", default=None, type=int)
    parser.add_argument("--min-names", default=1, type=int)
    parser.add_argument("--feature-panel", default=None, type=Path)
    parser.add_argument("--composition-target-buckets", nargs="*", default=None, type=int)
    parser.add_argument("--composition-top-n-industries", default=20, type=int)
    args = parser.parse_args(argv)

    predictions = _read_predictions(args.predictions, score_col=args.score_col)
    return_y = _read_return_matrix(args.return_y)
    feature_panel = _read_feature_panel(args.feature_panel) if args.feature_panel else None
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
        rank_bucket_count=args.rank_bucket_count,
        rank_order=args.rank_order,
        min_names=args.min_names,
        feature_panel=feature_panel,
        composition_target_buckets=args.composition_target_buckets,
        composition_top_n_industries=args.composition_top_n_industries,
    )
    if argv is None:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def evaluate_rank_bucket_nav(
    *,
    predictions: pd.DataFrame,
    return_y: pd.DataFrame,
    feature_panel: pd.DataFrame | None = None,
    score_col: str = DEFAULT_SCORE_COL,
    return_col: str = "return_y_hfq_adj",
    splits: Sequence[str] | None = DEFAULT_SPLITS,
    fold_ids: Sequence[int] | None = None,
    rank_buckets: Sequence[RankBucket] | None = None,
    rank_bucket_size: int = 500,
    rank_bucket_count: int | None = None,
    rank_order: str = "descending",
    min_names: int = 1,
    composition_target_buckets: Sequence[int] | None = None,
    composition_top_n_industries: int = 20,
) -> dict[str, pd.DataFrame]:
    if rank_order not in {"ascending", "descending"}:
        raise ValueError("rank_order must be 'ascending' or 'descending'.")
    if rank_bucket_size < 1:
        raise ValueError("rank_bucket_size must be positive.")
    if rank_bucket_count is not None and rank_bucket_count < 1:
        raise ValueError("rank_bucket_count must be positive.")
    if rank_bucket_count is not None and rank_buckets is not None:
        raise ValueError("rank_bucket_count cannot be combined with explicit rank_buckets.")
    if min_names < 1:
        raise ValueError("min_names must be positive.")

    pred = _prepare_predictions(
        predictions,
        score_col=score_col,
        splits=splits,
        fold_ids=fold_ids,
    )
    if feature_panel is not None:
        pred = _attach_feature_panel(
            pred,
            _prepare_feature_panel(feature_panel),
            score_col=score_col,
        )
    returns = _prepare_return_matrix(return_y)
    if rank_buckets is None and rank_bucket_count is None:
        rank_buckets = _default_rank_buckets(pred, score_col=score_col, bucket_size=rank_bucket_size)
    elif rank_buckets is not None:
        rank_buckets = tuple(rank_buckets)
    if rank_buckets is not None:
        _validate_rank_buckets(rank_buckets)

    daily_outputs = _compute_daily_bucket_outputs(
        pred,
        returns,
        score_col=score_col,
        return_col=return_col,
        rank_buckets=rank_buckets,
        rank_bucket_count=rank_bucket_count,
        rank_order=rank_order,
        min_names=min_names,
    )
    daily_returns = daily_outputs["daily_returns"]
    nav = _compute_nav(daily_returns)
    bucket_summary = _summarize_buckets(daily_returns, nav)
    industry_summary = _summarize_industry_composition(daily_outputs["composition_industry_daily"])
    size_summary = _summarize_size_composition(daily_outputs["composition_size_daily"])
    return {
        "daily_returns": daily_returns,
        "nav": nav,
        "summary": bucket_summary,
        "constituents": daily_outputs["constituents"],
        "composition_industry_daily": daily_outputs["composition_industry_daily"],
        "composition_industry_summary": industry_summary,
        "composition_size_daily": daily_outputs["composition_size_daily"],
        "composition_size_summary": size_summary,
    }


def write_rank_bucket_nav_artifacts(
    *,
    predictions: pd.DataFrame,
    return_y: pd.DataFrame,
    feature_panel: pd.DataFrame | None = None,
    output_dir: Path,
    score_col: str = DEFAULT_SCORE_COL,
    return_col: str = "return_y_hfq_adj",
    splits: Sequence[str] | None = DEFAULT_SPLITS,
    fold_ids: Sequence[int] | None = None,
    rank_buckets: Sequence[RankBucket] | None = None,
    rank_bucket_size: int = 500,
    rank_bucket_count: int | None = None,
    rank_order: str = "descending",
    min_names: int = 1,
    composition_target_buckets: Sequence[int] | None = None,
    composition_top_n_industries: int = 20,
) -> dict[str, Any]:
    result = evaluate_rank_bucket_nav(
        predictions=predictions,
        return_y=return_y,
        feature_panel=feature_panel,
        score_col=score_col,
        return_col=return_col,
        splits=splits,
        fold_ids=fold_ids,
        rank_buckets=rank_buckets,
        rank_bucket_size=rank_bucket_size,
        rank_bucket_count=rank_bucket_count,
        rank_order=rank_order,
        min_names=min_names,
        composition_target_buckets=composition_target_buckets,
        composition_top_n_industries=composition_top_n_industries,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"
    paths = {
        "daily_returns": output_dir / "rank_bucket_daily_returns.csv",
        "nav": output_dir / "rank_bucket_nav.csv",
        "bucket_summary": output_dir / "rank_bucket_summary.csv",
        "evaluation_summary": output_dir / "rank_bucket_evaluation_summary.json",
    }
    if feature_panel is not None:
        paths.update(
            {
                "constituents": output_dir / "rank_bucket_constituents.csv",
                "composition_industry_daily": output_dir / "rank_bucket_composition_industry_daily.csv",
                "composition_industry_summary": output_dir / "rank_bucket_composition_industry_summary.csv",
                "composition_size_daily": output_dir / "rank_bucket_composition_size_daily.csv",
                "composition_size_summary": output_dir / "rank_bucket_composition_size_summary.csv",
            }
        )
    _write_csv_with_iso_dates(result["daily_returns"], paths["daily_returns"])
    _write_csv_with_iso_dates(result["nav"], paths["nav"])
    _write_csv_with_iso_dates(result["summary"], paths["bucket_summary"])
    if feature_panel is not None:
        _write_csv_with_iso_dates(result["constituents"], paths["constituents"])
        _write_csv_with_iso_dates(result["composition_industry_daily"], paths["composition_industry_daily"])
        _write_csv_with_iso_dates(result["composition_industry_summary"], paths["composition_industry_summary"])
        _write_csv_with_iso_dates(result["composition_size_daily"], paths["composition_size_daily"])
        _write_csv_with_iso_dates(result["composition_size_summary"], paths["composition_size_summary"])

    chart_paths = write_rank_bucket_nav_charts(result["nav"], charts_dir)
    if feature_panel is not None:
        chart_paths.update(
            write_rank_bucket_composition_charts(
                result,
                charts_dir / "composition",
                target_buckets=composition_target_buckets,
                top_n_industries=composition_top_n_industries,
            )
        )
    summary = _build_summary(
        result,
        output_paths=paths,
        chart_paths=_relative_paths(chart_paths, output_dir),
        score_col=score_col,
        return_col=return_col,
        rank_buckets=rank_buckets,
        rank_bucket_size=rank_bucket_size,
        rank_bucket_count=rank_bucket_count,
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


def write_rank_bucket_composition_charts(
    result: dict[str, pd.DataFrame],
    output_dir: Path,
    *,
    target_buckets: Sequence[int] | None,
    top_n_industries: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _use_static_chart_theme()
    chart_paths: dict[str, Path] = {}
    industry_summary = result["composition_industry_summary"]
    size_summary = result["composition_size_summary"]
    if industry_summary.empty and size_summary.empty:
        return chart_paths

    targets = tuple(int(value) for value in (target_buckets or [1]))
    splits = sorted(
        set(industry_summary["split"].astype(str).tolist())
        | set(size_summary["split"].astype(str).tolist())
    )
    for split in splits:
        safe_split = _safe_filename(str(split))
        split_industry = industry_summary[industry_summary["split"].astype(str).eq(str(split))]
        split_size = size_summary[size_summary["split"].astype(str).eq(str(split))]
        for bucket_index in targets:
            path = output_dir / f"industry_active_weight_bucket_{bucket_index:02d}_{safe_split}.png"
            _plot_industry_active_weight_bucket(
                split_industry,
                path,
                bucket_index=bucket_index,
                split=str(split),
                top_n=top_n_industries,
            )
            chart_paths[f"composition_industry_active_weight_bucket_{bucket_index:02d}_{safe_split}"] = path

        industry_heatmap = output_dir / f"industry_active_weight_heatmap_{safe_split}.png"
        _plot_industry_active_weight_heatmap(
            split_industry,
            industry_heatmap,
            split=str(split),
            top_n=top_n_industries,
        )
        chart_paths[f"composition_industry_active_weight_heatmap_{safe_split}"] = industry_heatmap

        size_decile_heatmap = output_dir / f"size_decile_active_weight_heatmap_{safe_split}.png"
        _plot_size_decile_active_weight_heatmap(split_size, size_decile_heatmap, split=str(split))
        chart_paths[f"composition_size_decile_active_weight_heatmap_{safe_split}"] = size_decile_heatmap

        mcap_rank_path = output_dir / f"mcap_rank_by_bucket_{safe_split}.png"
        _plot_mcap_rank_by_bucket(split_size, mcap_rank_path, split=str(split))
        chart_paths[f"composition_mcap_rank_by_bucket_{safe_split}"] = mcap_rank_path
    return chart_paths


def _compute_daily_bucket_outputs(
    pred: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    score_col: str,
    return_col: str,
    rank_buckets: Sequence[RankBucket] | None,
    rank_bucket_count: int | None,
    rank_order: str,
    min_names: int,
) -> dict[str, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    constituent_records: list[dict[str, Any]] = []
    industry_records: list[dict[str, Any]] = []
    size_records: list[dict[str, Any]] = []
    has_composition_features = _has_composition_features(pred)
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
        ranked["rank_in_date"] = np.arange(1, len(ranked) + 1, dtype=int)
        eligible_count = int(len(ranked))
        return_row = _return_row_for_date(returns, pd.Timestamp(signal_date))
        bucket_specs = _bucket_specs_for_day(
            rank_buckets=rank_buckets,
            rank_bucket_count=rank_bucket_count,
            eligible_count=eligible_count,
        )
        universe_industry_weights = (
            _category_weights(ranked["industry"]) if has_composition_features else pd.Series(dtype=float)
        )
        universe_size_values = (
            _size_metric_means(ranked) if has_composition_features else {}
        )
        universe_size_decile_weights = (
            _category_weights(ranked["size_decile"]) if has_composition_features else pd.Series(dtype=float)
        )

        for bucket in bucket_specs:
            bucket_label = str(bucket["label"])
            start = int(bucket["start"])
            end = int(bucket["end"])
            selected = (
                ranked.iloc[start - 1 : end].copy()
                if end >= start and start >= 1
                else ranked.iloc[0:0].copy()
            )
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
                    "bucket_index": int(bucket["bucket_index"]),
                    "bucket_count": int(bucket["bucket_count"]),
                    "bucket_mode": str(bucket["bucket_mode"]),
                    "rank_start": start,
                    "rank_end": end,
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
            if has_composition_features:
                constituent_records.extend(
                    _constituent_records_for_selection(
                        selected,
                        fold_id=int(fold_id),
                        split=str(split),
                        score_col=score_col,
                        return_col=return_col,
                        signal_date=pd.Timestamp(signal_date),
                        bucket=bucket,
                        selected_count=selected_count,
                    )
                )
                industry_records.extend(
                    _industry_composition_records_for_bucket(
                        selected,
                        universe_industry_weights=universe_industry_weights,
                        fold_id=int(fold_id),
                        split=str(split),
                        score_col=score_col,
                        return_col=return_col,
                        signal_date=pd.Timestamp(signal_date),
                        bucket=bucket,
                    )
                )
                size_records.extend(
                    _size_composition_records_for_bucket(
                        selected,
                        universe_size_values=universe_size_values,
                        universe_size_decile_weights=universe_size_decile_weights,
                        fold_id=int(fold_id),
                        split=str(split),
                        score_col=score_col,
                        return_col=return_col,
                        signal_date=pd.Timestamp(signal_date),
                        bucket=bucket,
                    )
                )
    if not records:
        return {
            "daily_returns": _empty_daily_returns(),
            "constituents": _empty_constituents(),
            "composition_industry_daily": _empty_composition_industry_daily(),
            "composition_size_daily": _empty_composition_size_daily(),
        }
    daily_returns = pd.DataFrame(records).sort_values(
        ["fold_id", "split", "signal_date", "bucket_index", "rank_start"]
    ).reset_index(drop=True)
    constituents = (
        pd.DataFrame(constituent_records).sort_values(
            ["fold_id", "split", "signal_date", "bucket_index", "rank_in_date", "stock_code"]
        ).reset_index(drop=True)
        if constituent_records
        else _empty_constituents()
    )
    industry_daily = (
        pd.DataFrame(industry_records).sort_values(
            ["fold_id", "split", "signal_date", "bucket_index", "industry"]
        ).reset_index(drop=True)
        if industry_records
        else _empty_composition_industry_daily()
    )
    size_daily = (
        pd.DataFrame(size_records).sort_values(
            ["fold_id", "split", "signal_date", "bucket_index", "metric", "segment"]
        ).reset_index(drop=True)
        if size_records
        else _empty_composition_size_daily()
    )
    return {
        "daily_returns": daily_returns,
        "constituents": constituents,
        "composition_industry_daily": industry_daily,
        "composition_size_daily": size_daily,
    }


def _compute_daily_bucket_returns(
    pred: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    score_col: str,
    return_col: str,
    rank_buckets: Sequence[RankBucket] | None,
    rank_bucket_count: int | None,
    rank_order: str,
    min_names: int,
) -> pd.DataFrame:
    return _compute_daily_bucket_outputs(
        pred,
        returns,
        score_col=score_col,
        return_col=return_col,
        rank_buckets=rank_buckets,
        rank_bucket_count=rank_bucket_count,
        rank_order=rank_order,
        min_names=min_names,
    )["daily_returns"]


def _has_composition_features(frame: pd.DataFrame) -> bool:
    return set(COMPOSITION_FEATURE_COLUMNS).issubset(frame.columns)


def _constituent_records_for_selection(
    selected: pd.DataFrame,
    *,
    fold_id: int,
    split: str,
    score_col: str,
    return_col: str,
    signal_date: pd.Timestamp,
    bucket: dict[str, Any],
    selected_count: int,
) -> list[dict[str, Any]]:
    if selected_count == 0:
        return []
    weight = 1.0 / float(selected_count)
    records: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        records.append(
            {
                "fold_id": int(fold_id),
                "split": str(split),
                "score_col": score_col,
                "return_col": return_col,
                "signal_date": pd.Timestamp(signal_date),
                "bucket_label": str(bucket["label"]),
                "bucket_index": int(bucket["bucket_index"]),
                "bucket_count": int(bucket["bucket_count"]),
                "bucket_mode": str(bucket["bucket_mode"]),
                "rank_in_date": int(row["rank_in_date"]),
                "stock_code": str(row["stock_code"]),
                "score": float(row[score_col]),
                "constituent_weight": float(weight),
                "industry": str(row["industry"]),
                "market_cap": float(row["market_cap"]),
                "log_mcap": float(row["log_mcap"]),
                "log_mcap_z": float(row["log_mcap_z"]),
                "mcap_rank": float(row["mcap_rank"]),
                "size_decile": int(row["size_decile"]),
            }
        )
    return records


def _industry_composition_records_for_bucket(
    selected: pd.DataFrame,
    *,
    universe_industry_weights: pd.Series,
    fold_id: int,
    split: str,
    score_col: str,
    return_col: str,
    signal_date: pd.Timestamp,
    bucket: dict[str, Any],
) -> list[dict[str, Any]]:
    bucket_weights = _category_weights(selected["industry"])
    industries = sorted(set(universe_industry_weights.index).union(set(bucket_weights.index)))
    records: list[dict[str, Any]] = []
    for industry in industries:
        bucket_weight = float(bucket_weights.get(industry, 0.0))
        universe_weight = float(universe_industry_weights.get(industry, 0.0))
        records.append(
            {
                "fold_id": int(fold_id),
                "split": str(split),
                "score_col": score_col,
                "return_col": return_col,
                "signal_date": pd.Timestamp(signal_date),
                "bucket_label": str(bucket["label"]),
                "bucket_index": int(bucket["bucket_index"]),
                "bucket_count": int(bucket["bucket_count"]),
                "bucket_mode": str(bucket["bucket_mode"]),
                "industry": str(industry),
                "bucket_weight": bucket_weight,
                "universe_weight": universe_weight,
                "active_weight": float(bucket_weight - universe_weight),
            }
        )
    return records


def _size_composition_records_for_bucket(
    selected: pd.DataFrame,
    *,
    universe_size_values: dict[str, float],
    universe_size_decile_weights: pd.Series,
    fold_id: int,
    split: str,
    score_col: str,
    return_col: str,
    signal_date: pd.Timestamp,
    bucket: dict[str, Any],
) -> list[dict[str, Any]]:
    bucket_size_values = _size_metric_means(selected)
    records: list[dict[str, Any]] = []
    for metric in SIZE_PROFILE_METRICS:
        bucket_value = bucket_size_values.get(metric, np.nan)
        universe_value = universe_size_values.get(metric, np.nan)
        active_value = (
            float(bucket_value - universe_value)
            if np.isfinite(bucket_value) and np.isfinite(universe_value)
            else np.nan
        )
        records.append(
            _size_record(
                fold_id=fold_id,
                split=split,
                score_col=score_col,
                return_col=return_col,
                signal_date=signal_date,
                bucket=bucket,
                metric=metric,
                segment="all",
                bucket_value=bucket_value,
                universe_value=universe_value,
                active_value=active_value,
            )
        )

    bucket_decile_weights = _category_weights(selected["size_decile"])
    deciles = sorted(
        set(universe_size_decile_weights.index).union(set(bucket_decile_weights.index)),
        key=_segment_sort_key,
    )
    for decile in deciles:
        bucket_weight = float(bucket_decile_weights.get(decile, 0.0))
        universe_weight = float(universe_size_decile_weights.get(decile, 0.0))
        records.append(
            _size_record(
                fold_id=fold_id,
                split=split,
                score_col=score_col,
                return_col=return_col,
                signal_date=signal_date,
                bucket=bucket,
                metric="size_decile_weight",
                segment=str(decile),
                bucket_value=bucket_weight,
                universe_value=universe_weight,
                active_value=float(bucket_weight - universe_weight),
            )
        )
    return records


def _size_record(
    *,
    fold_id: int,
    split: str,
    score_col: str,
    return_col: str,
    signal_date: pd.Timestamp,
    bucket: dict[str, Any],
    metric: str,
    segment: str,
    bucket_value: float,
    universe_value: float,
    active_value: float,
) -> dict[str, Any]:
    return {
        "fold_id": int(fold_id),
        "split": str(split),
        "score_col": score_col,
        "return_col": return_col,
        "signal_date": pd.Timestamp(signal_date),
        "bucket_label": str(bucket["label"]),
        "bucket_index": int(bucket["bucket_index"]),
        "bucket_count": int(bucket["bucket_count"]),
        "bucket_mode": str(bucket["bucket_mode"]),
        "metric": str(metric),
        "segment": str(segment),
        "bucket_value": float(bucket_value) if np.isfinite(bucket_value) else np.nan,
        "universe_value": float(universe_value) if np.isfinite(universe_value) else np.nan,
        "active_value": float(active_value) if np.isfinite(active_value) else np.nan,
    }


def _category_weights(values: Iterable[Any]) -> pd.Series:
    series = pd.Series(list(values))
    if series.empty:
        return pd.Series(dtype=float)
    series = series.dropna().map(_category_label)
    if series.empty:
        return pd.Series(dtype=float)
    counts = series.value_counts(sort=False)
    weights = counts.astype(float) / float(counts.sum())
    return weights.sort_index(key=lambda index: index.map(_segment_sort_key))


def _category_label(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(float(value)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _segment_sort_key(value: Any) -> tuple[int, float | str]:
    text = str(value)
    try:
        return (0, float(text))
    except ValueError:
        return (1, text)


def _size_metric_means(frame: pd.DataFrame) -> dict[str, float]:
    means: dict[str, float] = {}
    for metric in SIZE_PROFILE_METRICS:
        values = pd.to_numeric(frame[metric], errors="coerce") if metric in frame else pd.Series(dtype=float)
        finite = _finite_values(values)
        means[metric] = float(np.mean(finite)) if len(finite) else np.nan
    return means


def _summarize_industry_composition(industry_daily: pd.DataFrame) -> pd.DataFrame:
    if industry_daily.empty:
        return _empty_composition_industry_summary()
    records: list[dict[str, Any]] = []
    group_cols = [
        "fold_id",
        "split",
        "score_col",
        "return_col",
        "bucket_label",
        "bucket_index",
        "bucket_count",
        "bucket_mode",
        "industry",
    ]
    for key, group in industry_daily.groupby(group_cols, sort=True):
        (
            fold_id,
            split,
            score_col,
            return_col,
            bucket_label,
            bucket_index,
            bucket_count,
            bucket_mode,
            industry,
        ) = key
        active_values = _finite_values(group["active_weight"])
        records.append(
            {
                "fold_id": int(fold_id),
                "split": str(split),
                "score_col": str(score_col),
                "return_col": str(return_col),
                "bucket_label": str(bucket_label),
                "bucket_index": int(bucket_index),
                "bucket_count": int(bucket_count),
                "bucket_mode": str(bucket_mode),
                "industry": str(industry),
                "date_count": int(len(group)),
                "mean_bucket_weight": _nanmean(_finite_values(group["bucket_weight"])),
                "mean_universe_weight": _nanmean(_finite_values(group["universe_weight"])),
                "mean_active_weight": _nanmean(active_values),
                "active_weight_std": _nanstd(active_values),
                "active_weight_tstat": _tstat(active_values),
                "positive_active_date_rate": _positive_rate(active_values),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["fold_id", "split", "bucket_index", "industry"]
    ).reset_index(drop=True)


def _summarize_size_composition(size_daily: pd.DataFrame) -> pd.DataFrame:
    if size_daily.empty:
        return _empty_composition_size_summary()
    records: list[dict[str, Any]] = []
    group_cols = [
        "fold_id",
        "split",
        "score_col",
        "return_col",
        "bucket_label",
        "bucket_index",
        "bucket_count",
        "bucket_mode",
        "metric",
        "segment",
    ]
    for key, group in size_daily.groupby(group_cols, sort=True):
        (
            fold_id,
            split,
            score_col,
            return_col,
            bucket_label,
            bucket_index,
            bucket_count,
            bucket_mode,
            metric,
            segment,
        ) = key
        active_values = _finite_values(group["active_value"])
        records.append(
            {
                "fold_id": int(fold_id),
                "split": str(split),
                "score_col": str(score_col),
                "return_col": str(return_col),
                "bucket_label": str(bucket_label),
                "bucket_index": int(bucket_index),
                "bucket_count": int(bucket_count),
                "bucket_mode": str(bucket_mode),
                "metric": str(metric),
                "segment": str(segment),
                "date_count": int(len(group)),
                "mean_bucket_value": _nanmean(_finite_values(group["bucket_value"])),
                "mean_universe_value": _nanmean(_finite_values(group["universe_value"])),
                "mean_active_value": _nanmean(active_values),
                "active_value_std": _nanstd(active_values),
                "active_value_tstat": _tstat(active_values),
                "positive_active_date_rate": _positive_rate(active_values),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["fold_id", "split", "bucket_index", "metric", "segment"]
    ).reset_index(drop=True)


def _tstat(values: np.ndarray) -> float:
    if len(values) < 2:
        return np.nan
    std = float(np.std(values, ddof=0))
    if std <= 1e-12:
        return np.nan
    return float(np.mean(values) / (std / math.sqrt(float(len(values)))))


def _compute_nav(daily_returns: pd.DataFrame) -> pd.DataFrame:
    if daily_returns.empty:
        return _empty_nav()
    records: list[dict[str, Any]] = []
    group_cols = ["fold_id", "split", "score_col", "return_col", "bucket_label"]
    for key, group in daily_returns.groupby(group_cols, sort=True):
        fold_id, split, score_col, return_col, bucket_label = key
        running_nav = 1.0
        group = group.sort_values("signal_date")
        bucket_index = int(group["bucket_index"].iloc[0])
        bucket_count = int(group["bucket_count"].iloc[0])
        bucket_mode = str(group["bucket_mode"].iloc[0])
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
                    "bucket_index": bucket_index,
                    "bucket_count": bucket_count,
                    "bucket_mode": bucket_mode,
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
        ["fold_id", "split", "bucket_index", "bucket_label", "signal_date"]
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
                "bucket_index": int(group["bucket_index"].iloc[0]),
                "bucket_count": int(group["bucket_count"].iloc[0]),
                "bucket_mode": str(group["bucket_mode"].iloc[0]),
                "rank_start": int(group["rank_start"].min()),
                "rank_end": int(group["rank_end"].max()),
                "mean_rank_start": float(group["rank_start"].mean()),
                "mean_rank_end": float(group["rank_end"].mean()),
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
        ["fold_id", "split", "bucket_index", "rank_start"]
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
    rank_bucket_count: int | None,
    rank_order: str,
    splits: Sequence[str] | None,
    fold_ids: Sequence[int] | None,
    min_names: int,
) -> dict[str, Any]:
    daily = result["daily_returns"]
    bucket_mode = "daily_equal_count" if rank_bucket_count is not None else "fixed_rank_bounds"
    if rank_bucket_count is not None:
        bucket_records = [
            {
                "bucket_index": int(row.bucket_index),
                "bucket_count": int(row.bucket_count),
                "label": str(row.bucket_label),
                "bound_mode": "daily_equal_count_one_based_inclusive",
            }
            for row in daily[["bucket_index", "bucket_count", "bucket_label"]]
            .drop_duplicates()
            .sort_values(["bucket_index"])
            .itertuples(index=False)
        ]
    else:
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
        bucket_records = [
            {
                **asdict(bucket),
                "bucket_index": int(bucket_index),
                "bucket_count": int(len(buckets)),
                "label": _bucket_label(bucket),
                "bound_mode": "one_based_inclusive",
            }
            for bucket_index, bucket in enumerate(buckets, start=1)
        ]
    return {
        "schema_version": "lgbm_rank_bucket_nav_v1",
        "metric_contract": {
            "score_col": score_col,
            "return_col": return_col,
            "return_alignment": "already_aligned_to_signal_date",
            "rank_scope": "daily_cross_section_within_fold_split",
            "rank_order": rank_order,
            "bucket_mode": bucket_mode,
            "rank_bound_mode": (
                "daily_equal_count_one_based_inclusive"
                if rank_bucket_count is not None
                else "one_based_inclusive"
            ),
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
        "rank_bucket_size": None if rank_bucket_count is not None else int(rank_bucket_size),
        "rank_bucket_count": int(rank_bucket_count) if rank_bucket_count is not None else None,
        "rank_buckets": bucket_records,
        "min_names": int(min_names),
        "row_counts": {
            "daily_returns": int(len(result["daily_returns"])),
            "nav": int(len(result["nav"])),
            "bucket_summary": int(len(result["summary"])),
            "constituents": int(len(result.get("constituents", pd.DataFrame()))),
            "composition_industry_daily": int(len(result.get("composition_industry_daily", pd.DataFrame()))),
            "composition_industry_summary": int(len(result.get("composition_industry_summary", pd.DataFrame()))),
            "composition_size_daily": int(len(result.get("composition_size_daily", pd.DataFrame()))),
            "composition_size_summary": int(len(result.get("composition_size_summary", pd.DataFrame()))),
        },
        "outputs": {key: _path_for_summary(path) for key, path in output_paths.items()},
        "charts": {key: _path_for_summary(path) for key, path in chart_paths.items()},
        "composition_contract": {
            "enabled": not result.get("constituents", pd.DataFrame()).empty,
            "feature_join_key": ["date", "stock_code"],
            "feature_columns": list(COMPOSITION_FEATURE_COLUMNS),
            "feature_panel_duplicate_policy": "fail_on_duplicate_date_stock_code",
            "feature_panel_missing_policy": "fail_for_any_finite_score_prediction_without_required_composition_fields",
            "composition_universe": "finite-score eligible stocks within each fold/split/signal_date ranking universe",
            "constituent_weight": "equal weight within each selected bucket on each date",
            "industry_active_weight_formula": "bucket_industry_weight - eligible_universe_industry_weight",
            "size_active_value_formula": "bucket_mean_metric - eligible_universe_mean_metric",
            "size_decile_active_weight_formula": "bucket_size_decile_weight - eligible_universe_size_decile_weight",
            "mcap_rank_interpretation": "positive active mcap_rank means larger-cap tilt relative to the eligible universe",
        },
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


def _prepare_feature_panel(feature_panel: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(feature_panel, pd.DataFrame):
        raise TypeError(f"feature_panel must be a pandas DataFrame, got {type(feature_panel)!r}")
    required = {"date", "stock_code", *COMPOSITION_FEATURE_COLUMNS}
    missing = sorted(required.difference(feature_panel.columns))
    if missing:
        raise ValueError(f"Feature panel is missing required composition columns: {missing}")

    frame = feature_panel[["date", "stock_code", *COMPOSITION_FEATURE_COLUMNS]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["stock_code"] = frame["stock_code"].map(_normalize_stock_code)
    invalid_keys = frame["date"].isna() | frame["stock_code"].eq("")
    if invalid_keys.any():
        sample = frame.loc[invalid_keys, ["date", "stock_code"]].head(5)
        raise ValueError(f"Feature panel contains invalid date/stock_code keys: {sample.to_dict('records')}")
    duplicate_keys = ["date", "stock_code"]
    if frame.duplicated(duplicate_keys).any():
        sample = frame.loc[frame.duplicated(duplicate_keys, keep=False), duplicate_keys].head(5)
        raise ValueError(f"Feature panel contains duplicate date/stock_code rows: {sample.to_dict('records')}")

    frame["industry"] = frame["industry"].astype("string")
    for column in ["market_cap", "log_mcap", "log_mcap_z", "mcap_rank", "size_decile"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values(["date", "stock_code"]).reset_index(drop=True)


def _attach_feature_panel(
    pred: pd.DataFrame,
    feature_panel: pd.DataFrame,
    *,
    score_col: str,
) -> pd.DataFrame:
    merged = pred.merge(
        feature_panel,
        on=["date", "stock_code"],
        how="left",
        validate="many_to_one",
    )
    finite_signal = np.isfinite(pd.to_numeric(merged[score_col], errors="coerce").to_numpy(dtype=float))
    missing_or_invalid = merged["industry"].isna()
    for column in ["market_cap", "log_mcap", "log_mcap_z", "mcap_rank", "size_decile"]:
        values = pd.to_numeric(merged[column], errors="coerce")
        column_invalid = ~np.isfinite(values.to_numpy(dtype=float))
        if column == "market_cap":
            column_invalid |= values.le(0.0).to_numpy(dtype=bool)
        missing_or_invalid |= column_invalid
    bad = finite_signal & missing_or_invalid.to_numpy(dtype=bool)
    if bad.any():
        sample = merged.loc[bad, ["fold_id", "split", "date", "stock_code"]].head(10).copy()
        sample["date"] = pd.to_datetime(sample["date"]).dt.date.astype(str)
        raise ValueError(
            "Predictions contain finite-score rows with missing feature panel rows "
            f"or invalid composition values: {sample.to_dict('records')}"
        )
    merged["industry"] = merged["industry"].astype(str)
    merged["size_decile"] = pd.to_numeric(merged["size_decile"], errors="raise").astype(int)
    return merged


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


def _bucket_specs_for_day(
    *,
    rank_buckets: Sequence[RankBucket] | None,
    rank_bucket_count: int | None,
    eligible_count: int,
) -> list[dict[str, Any]]:
    if rank_bucket_count is not None:
        width = max(2, len(str(rank_bucket_count)))
        positions = np.array_split(np.arange(max(eligible_count, 0)), rank_bucket_count)
        specs: list[dict[str, Any]] = []
        for bucket_index, bucket_positions in enumerate(positions, start=1):
            if len(bucket_positions):
                start = int(bucket_positions[0]) + 1
                end = int(bucket_positions[-1]) + 1
            else:
                start = int(eligible_count) + 1
                end = int(eligible_count)
            specs.append(
                {
                    "bucket_index": int(bucket_index),
                    "bucket_count": int(rank_bucket_count),
                    "bucket_mode": "daily_equal_count",
                    "start": start,
                    "end": end,
                    "label": f"rank_bucket_{bucket_index:0{width}d}_of_{rank_bucket_count:0{width}d}",
                }
            )
        return specs

    if rank_buckets is None:
        return []
    return [
        {
            "bucket_index": int(bucket_index),
            "bucket_count": int(len(rank_buckets)),
            "bucket_mode": "fixed_rank_bounds",
            "start": int(bucket.start),
            "end": int(bucket.end),
            "label": _bucket_label(bucket),
        }
        for bucket_index, bucket in enumerate(rank_buckets, start=1)
    ]


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


def _read_feature_panel(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        frame = pd.read_pickle(path)
    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, dtype={"stock_code": "string"})
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected feature panel DataFrame at {path}, got {type(frame)!r}")
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
    from matplotlib import font_manager

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "NSimSun",
        "DengXian",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    chart_fonts = [font for font in preferred_fonts if font in available_fonts]
    if not chart_fonts:
        chart_fonts = ["DejaVu Sans"]

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
            "font.family": chart_fonts,
            "font.size": 10.5,
            "axes.unicode_minus": False,
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


def _plot_industry_active_weight_bucket(
    industry_summary: pd.DataFrame,
    path: Path,
    *,
    bucket_index: int,
    split: str,
    top_n: int,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    _chart_title(
        fig,
        f"Industry active weight, bucket {bucket_index:02d}",
        f"Split {split}; active weight = bucket industry weight - daily eligible-universe weight",
    )
    frame = industry_summary[industry_summary["bucket_index"].astype(int).eq(int(bucket_index))].copy()
    if frame.empty:
        _plot_empty(ax, "No industry composition data available")
        _save_chart(fig, path)
        return
    grouped = (
        frame.groupby("industry", sort=True)["mean_active_weight"]
        .mean()
        .sort_values(key=lambda values: values.abs(), ascending=False)
        .head(max(int(top_n), 1))
        .sort_values()
    )
    colors = ["#bf3d3d" if value < 0 else "#2c7a7b" for value in grouped]
    ax.barh(grouped.index.astype(str), grouped.to_numpy(dtype=float), color=colors)
    ax.axvline(0.0, color="#27384c", linewidth=0.9)
    ax.set_xlabel("Mean active weight")
    ax.set_ylabel("Industry")
    ax.grid(axis="x")
    _save_chart(fig, path)


def _plot_industry_active_weight_heatmap(
    industry_summary: pd.DataFrame,
    path: Path,
    *,
    split: str,
    top_n: int,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12.2, 7.0), constrained_layout=True)
    _chart_title(
        fig,
        "Industry active weight by rank bucket",
        f"Split {split}; top industries selected by absolute active-weight magnitude",
    )
    if industry_summary.empty:
        _plot_empty(ax, "No industry composition data available")
        _save_chart(fig, path)
        return
    frame = industry_summary.copy()
    top_industries = (
        frame.groupby("industry", sort=True)["mean_active_weight"]
        .apply(lambda values: float(np.nanmax(np.abs(values.to_numpy(dtype=float)))))
        .sort_values(ascending=False)
        .head(max(int(top_n), 1))
        .index
    )
    frame = frame[frame["industry"].isin(top_industries)].copy()
    pivot = frame.pivot_table(
        index="industry",
        columns="bucket_index",
        values="mean_active_weight",
        aggfunc="mean",
        fill_value=0.0,
    ).sort_index()
    if pivot.empty:
        _plot_empty(ax, "No industry composition data available")
        _save_chart(fig, path)
        return
    max_abs = float(np.nanmax(np.abs(pivot.to_numpy(dtype=float))))
    limit = max(max_abs, 1e-12)
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{int(value):02d}" for value in pivot.columns], rotation=90)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(str))
    ax.set_xlabel("Bucket")
    ax.set_ylabel("Industry")
    fig.colorbar(image, ax=ax, label="Mean active weight")
    _save_chart(fig, path)


def _plot_size_decile_active_weight_heatmap(size_summary: pd.DataFrame, path: Path, *, split: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.6, 5.8), constrained_layout=True)
    _chart_title(
        fig,
        "Size-decile active weight by rank bucket",
        f"Split {split}; active weight = bucket size-decile weight - eligible-universe weight",
    )
    frame = size_summary[size_summary["metric"].eq("size_decile_weight")].copy()
    if frame.empty:
        _plot_empty(ax, "No size-decile composition data available")
        _save_chart(fig, path)
        return
    pivot = frame.pivot_table(
        index="segment",
        columns="bucket_index",
        values="mean_active_value",
        aggfunc="mean",
        fill_value=0.0,
    )
    pivot = pivot.loc[sorted(pivot.index, key=_segment_sort_key)]
    if pivot.empty:
        _plot_empty(ax, "No size-decile composition data available")
        _save_chart(fig, path)
        return
    max_abs = float(np.nanmax(np.abs(pivot.to_numpy(dtype=float))))
    limit = max(max_abs, 1e-12)
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{int(value):02d}" for value in pivot.columns], rotation=90)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(str))
    ax.set_xlabel("Bucket")
    ax.set_ylabel("Size decile")
    fig.colorbar(image, ax=ax, label="Mean active weight")
    _save_chart(fig, path)


def _plot_mcap_rank_by_bucket(size_summary: pd.DataFrame, path: Path, *, split: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.2, 5.8), constrained_layout=True)
    _chart_title(
        fig,
        "Market-cap rank active exposure by bucket",
        f"Split {split}; positive active mcap_rank means larger-cap tilt versus eligible universe",
    )
    frame = size_summary[size_summary["metric"].eq("mcap_rank") & size_summary["segment"].eq("all")].copy()
    if frame.empty:
        _plot_empty(ax, "No mcap_rank composition data available")
        _save_chart(fig, path)
        return
    grouped = (
        frame.groupby("bucket_index", sort=True)["mean_active_value"]
        .mean()
        .sort_index()
    )
    ax.plot(grouped.index.to_numpy(dtype=int), grouped.to_numpy(dtype=float), marker="o", linewidth=1.8)
    ax.axhline(0.0, color="#27384c", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Bucket")
    ax.set_ylabel("Mean active mcap_rank")
    ax.set_xticks(grouped.index.to_numpy(dtype=int))
    ax.grid(axis="y")
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


def _relative_paths(paths: dict[str, Path], base_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for key, path in paths.items():
        try:
            result[key] = path.relative_to(base_dir)
        except ValueError:
            result[key] = path
    return result


def _empty_daily_returns() -> pd.DataFrame:
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
            "bucket_index",
            "bucket_count",
            "bucket_mode",
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
            "bucket_index",
            "bucket_count",
            "bucket_mode",
            "rank_start",
            "rank_end",
            "mean_rank_start",
            "mean_rank_end",
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


def _empty_constituents() -> pd.DataFrame:
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
            "rank_in_date",
            "stock_code",
            "score",
            "constituent_weight",
            "industry",
            "market_cap",
            "log_mcap",
            "log_mcap_z",
            "mcap_rank",
            "size_decile",
        ]
    )


def _empty_composition_industry_daily() -> pd.DataFrame:
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
            "industry",
            "bucket_weight",
            "universe_weight",
            "active_weight",
        ]
    )


def _empty_composition_industry_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "fold_id",
            "split",
            "score_col",
            "return_col",
            "bucket_label",
            "bucket_index",
            "bucket_count",
            "bucket_mode",
            "industry",
            "date_count",
            "mean_bucket_weight",
            "mean_universe_weight",
            "mean_active_weight",
            "active_weight_std",
            "active_weight_tstat",
            "positive_active_date_rate",
        ]
    )


def _empty_composition_size_daily() -> pd.DataFrame:
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
            "metric",
            "segment",
            "bucket_value",
            "universe_value",
            "active_value",
        ]
    )


def _empty_composition_size_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "fold_id",
            "split",
            "score_col",
            "return_col",
            "bucket_label",
            "bucket_index",
            "bucket_count",
            "bucket_mode",
            "metric",
            "segment",
            "date_count",
            "mean_bucket_value",
            "mean_universe_value",
            "mean_active_value",
            "active_value_std",
            "active_value_tstat",
            "positive_active_date_rate",
        ]
    )


if __name__ == "__main__":
    main()
