from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from analysis.evaluate_lgbm_rank_bucket_nav import (
        COMPOSITION_FEATURE_COLUMNS,
        _finite_values,
        _nanmean,
        _nanstd,
        _normalize_stock_code,
        _path_for_summary,
        _positive_rate,
        _prepare_feature_panel,
        _read_return_matrix,
        _tstat,
        _write_csv_with_iso_dates,
        evaluate_rank_bucket_nav,
        write_rank_bucket_composition_charts,
        write_rank_bucket_nav_charts,
    )
    from analysis.summarize_rank_bucket_periods import (
        PeriodSpec,
        parse_period_spec,
        summarize_rank_bucket_periods,
    )
except ModuleNotFoundError:  # Allows direct execution via python analysis/script.py.
    from evaluate_lgbm_rank_bucket_nav import (
        COMPOSITION_FEATURE_COLUMNS,
        _finite_values,
        _nanmean,
        _nanstd,
        _normalize_stock_code,
        _path_for_summary,
        _positive_rate,
        _prepare_feature_panel,
        _read_return_matrix,
        _tstat,
        _write_csv_with_iso_dates,
        evaluate_rank_bucket_nav,
        write_rank_bucket_composition_charts,
        write_rank_bucket_nav_charts,
    )
    from summarize_rank_bucket_periods import (
        PeriodSpec,
        parse_period_spec,
        summarize_rank_bucket_periods,
    )


DEFAULT_ALPHA_NAME = "factor_sss_dx_10"
DEFAULT_SIGNAL_IDS = (
    "raw_alpha_value",
    "neutralized_alpha_value",
    "raw_alpha_rank",
    "neutralized_alpha_rank",
)


@dataclass(frozen=True)
class AlphaBucketSignalSpec:
    signal_id: str
    signal_label: str
    source_column: str
    alpha_name: str
    alpha_input_kind: str
    neutralization: str

    @property
    def score_col(self) -> str:
        return self.signal_id


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description=(
            "Compare raw and neutralized alpha signals as standalone rank-bucket "
            "sorting scores over the current model's fold/split/date-stock sample."
        )
    )
    parser.add_argument("--training-panel", required=True, type=Path)
    parser.add_argument("--template-predictions", required=True, type=Path)
    parser.add_argument("--return-y", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--alpha-name", default=DEFAULT_ALPHA_NAME)
    parser.add_argument("--signals", nargs="*", default=list(DEFAULT_SIGNAL_IDS))
    parser.add_argument("--periods", nargs="+", required=True)
    parser.add_argument("--return-col", default="return_y_hfq_adj")
    parser.add_argument("--rank-bucket-count", default=31, type=int)
    parser.add_argument("--rank-order", default="descending", choices=["ascending", "descending"])
    parser.add_argument("--splits", nargs="*", default=["test"])
    parser.add_argument("--fold-ids", nargs="*", default=None, type=int)
    parser.add_argument("--min-names", default=1, type=int)
    parser.add_argument("--composition-target-buckets", nargs="*", default=[1], type=int)
    parser.add_argument("--composition-top-n-industries", default=20, type=int)
    parser.add_argument("--period-industry-top-n", default=10, type=int)
    args = parser.parse_args(argv)

    training_panel = pd.read_pickle(args.training_panel)
    template_predictions = _read_template_predictions(args.template_predictions)
    return_y = _read_return_matrix(args.return_y)
    summary = write_alpha_signal_rank_bucket_comparison_artifacts(
        training_panel=training_panel,
        template_predictions=template_predictions,
        return_y=return_y,
        output_dir=args.output_dir,
        alpha_name=args.alpha_name,
        signal_ids=args.signals,
        periods=[parse_period_spec(value) for value in args.periods],
        return_col=args.return_col,
        rank_bucket_count=args.rank_bucket_count,
        rank_order=args.rank_order,
        splits=args.splits,
        fold_ids=args.fold_ids,
        min_names=args.min_names,
        composition_target_buckets=args.composition_target_buckets,
        composition_top_n_industries=args.composition_top_n_industries,
        period_industry_top_n=args.period_industry_top_n,
    )
    if argv is None:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def evaluate_alpha_signal_rank_bucket_comparison(
    *,
    training_panel: pd.DataFrame,
    template_predictions: pd.DataFrame,
    return_y: pd.DataFrame,
    alpha_name: str = DEFAULT_ALPHA_NAME,
    signal_ids: Sequence[str] = DEFAULT_SIGNAL_IDS,
    periods: Sequence[PeriodSpec],
    return_col: str = "return_y_hfq_adj",
    rank_bucket_count: int = 31,
    rank_order: str = "descending",
    splits: Sequence[str] | None = ("test",),
    fold_ids: Sequence[int] | None = None,
    min_names: int = 1,
    period_industry_top_n: int = 10,
) -> dict[str, pd.DataFrame]:
    specs = resolve_alpha_bucket_signal_specs(alpha_name=alpha_name, signal_ids=signal_ids)
    score_panel = _prepare_score_panel(training_panel, specs=specs)
    feature_panel = _prepare_feature_panel(training_panel)
    template = _prepare_template_predictions(template_predictions)

    rank_results: list[dict[str, pd.DataFrame]] = []
    period_results: list[dict[str, pd.DataFrame]] = []
    for spec in specs:
        predictions = _build_signal_predictions(template, score_panel, spec)
        rank_result = evaluate_rank_bucket_nav(
            predictions=predictions,
            return_y=return_y,
            feature_panel=feature_panel,
            score_col=spec.score_col,
            return_col=return_col,
            splits=splits,
            fold_ids=fold_ids,
            rank_bucket_count=rank_bucket_count,
            rank_order=rank_order,
            min_names=min_names,
        )
        rank_result = _annotate_rank_result(rank_result, spec)
        rank_results.append(rank_result)

        period_result = summarize_rank_bucket_periods(
            rank_result["daily_returns"],
            periods=periods,
            splits=splits,
        )
        period_result = _annotate_period_result(period_result, spec)
        period_results.append(period_result)

    combined = {
        "signal_manifest": _signal_manifest(specs),
        "daily_returns": _concat_frames(result["daily_returns"] for result in rank_results),
        "nav": _concat_frames(result["nav"] for result in rank_results),
        "bucket_summary": _concat_frames(result["summary"] for result in rank_results),
        "constituents": _concat_frames(result["constituents"] for result in rank_results),
        "composition_industry_daily": _concat_frames(
            result["composition_industry_daily"] for result in rank_results
        ),
        "composition_industry_summary": _concat_frames(
            result["composition_industry_summary"] for result in rank_results
        ),
        "composition_size_daily": _concat_frames(
            result["composition_size_daily"] for result in rank_results
        ),
        "composition_size_summary": _concat_frames(
            result["composition_size_summary"] for result in rank_results
        ),
        "period_daily_returns": _concat_frames(
            result["period_daily_returns"] for result in period_results
        ),
        "period_nav": _concat_frames(result["period_nav"] for result in period_results),
        "period_summary": _concat_frames(result["summary"] for result in period_results),
        "period_requested_table": _concat_frames(
            result["requested_table"] for result in period_results
        ),
    }
    combined["period_industry_summary"] = _annotate_with_signal_metadata(
        summarize_period_industry_composition(
            combined["composition_industry_daily"],
            periods=periods,
            splits=splits,
        ),
        specs,
    )
    combined["period_industry_top"] = top_period_industry_preferences(
        combined["period_industry_summary"],
        target_buckets=[1],
        top_n=period_industry_top_n,
    )
    return combined


def write_alpha_signal_rank_bucket_comparison_artifacts(
    *,
    training_panel: pd.DataFrame,
    template_predictions: pd.DataFrame,
    return_y: pd.DataFrame,
    output_dir: Path,
    alpha_name: str = DEFAULT_ALPHA_NAME,
    signal_ids: Sequence[str] = DEFAULT_SIGNAL_IDS,
    periods: Sequence[PeriodSpec],
    return_col: str = "return_y_hfq_adj",
    rank_bucket_count: int = 31,
    rank_order: str = "descending",
    splits: Sequence[str] | None = ("test",),
    fold_ids: Sequence[int] | None = None,
    min_names: int = 1,
    composition_target_buckets: Sequence[int] | None = (1,),
    composition_top_n_industries: int = 20,
    period_industry_top_n: int = 10,
) -> dict[str, Any]:
    specs = resolve_alpha_bucket_signal_specs(alpha_name=alpha_name, signal_ids=signal_ids)
    result = evaluate_alpha_signal_rank_bucket_comparison(
        training_panel=training_panel,
        template_predictions=template_predictions,
        return_y=return_y,
        alpha_name=alpha_name,
        signal_ids=signal_ids,
        periods=periods,
        return_col=return_col,
        rank_bucket_count=rank_bucket_count,
        rank_order=rank_order,
        splits=splits,
        fold_ids=fold_ids,
        min_names=min_names,
        period_industry_top_n=period_industry_top_n,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "signal_manifest": output_dir / "alpha_signal_manifest.csv",
        "daily_returns": output_dir / "rank_bucket_daily_returns.csv",
        "nav": output_dir / "rank_bucket_nav.csv",
        "bucket_summary": output_dir / "rank_bucket_summary.csv",
        "constituents": output_dir / "rank_bucket_constituents.csv",
        "composition_industry_daily": output_dir / "rank_bucket_composition_industry_daily.csv",
        "composition_industry_summary": output_dir / "rank_bucket_composition_industry_summary.csv",
        "composition_size_daily": output_dir / "rank_bucket_composition_size_daily.csv",
        "composition_size_summary": output_dir / "rank_bucket_composition_size_summary.csv",
        "period_daily_returns": output_dir / "rank_bucket_period_daily_returns.csv",
        "period_nav": output_dir / "rank_bucket_period_nav.csv",
        "period_summary": output_dir / "rank_bucket_period_summary.csv",
        "period_requested_table": output_dir / "rank_bucket_period_requested_table.csv",
        "period_industry_summary": output_dir / "rank_bucket_period_industry_summary.csv",
        "period_industry_top": output_dir / "rank_bucket_period_industry_top.csv",
        "evaluation_summary": output_dir / "alpha_signal_rank_bucket_comparison_summary.json",
    }
    for key, path in paths.items():
        if key == "evaluation_summary":
            continue
        _write_csv_with_iso_dates(result[key], path)

    chart_paths = _write_comparison_charts(
        result,
        output_dir / "charts",
        specs=specs,
        composition_target_buckets=composition_target_buckets,
        composition_top_n_industries=composition_top_n_industries,
    )
    summary = _build_artifact_summary(
        result,
        output_paths=paths,
        chart_paths=chart_paths,
        specs=specs,
        periods=periods,
        alpha_name=alpha_name,
        return_col=return_col,
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


def resolve_alpha_bucket_signal_specs(
    *,
    alpha_name: str,
    signal_ids: Sequence[str],
) -> tuple[AlphaBucketSignalSpec, ...]:
    if not alpha_name:
        raise ValueError("alpha_name must be non-empty.")
    by_id = {
        "raw_alpha_value": AlphaBucketSignalSpec(
            signal_id="raw_alpha_value",
            signal_label="Raw alpha value",
            source_column=f"{alpha_name}_raw",
            alpha_name=alpha_name,
            alpha_input_kind="value",
            neutralization="raw",
        ),
        "neutralized_alpha_value": AlphaBucketSignalSpec(
            signal_id="neutralized_alpha_value",
            signal_label="Neutralized alpha value",
            source_column=f"{alpha_name}_value_neutralized_raw",
            alpha_name=alpha_name,
            alpha_input_kind="value",
            neutralization="neutralized",
        ),
        "raw_alpha_rank": AlphaBucketSignalSpec(
            signal_id="raw_alpha_rank",
            signal_label="Raw alpha rank input",
            source_column=f"{alpha_name}_rank_raw",
            alpha_name=alpha_name,
            alpha_input_kind="rank",
            neutralization="raw",
        ),
        "neutralized_alpha_rank": AlphaBucketSignalSpec(
            signal_id="neutralized_alpha_rank",
            signal_label="Neutralized alpha rank input",
            source_column=f"{alpha_name}_rank_neutralized_raw",
            alpha_name=alpha_name,
            alpha_input_kind="rank",
            neutralization="neutralized",
        ),
    }
    requested = tuple(str(signal_id) for signal_id in signal_ids)
    if not requested:
        raise ValueError("At least one signal_id must be supplied.")
    unknown = sorted(set(requested).difference(by_id))
    if unknown:
        raise ValueError(f"Unknown alpha bucket signal ids: {unknown}")
    return tuple(by_id[signal_id] for signal_id in requested)


def summarize_period_industry_composition(
    industry_daily: pd.DataFrame,
    *,
    periods: Sequence[PeriodSpec],
    splits: Sequence[str] | None = ("test",),
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
        "industry",
        "bucket_weight",
        "universe_weight",
        "active_weight",
    }
    missing = sorted(required.difference(industry_daily.columns))
    if missing:
        raise ValueError(f"Industry daily composition is missing required columns: {missing}")
    if not periods:
        raise ValueError("At least one period must be supplied.")

    frame = industry_daily[list(required)].copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise").dt.normalize()
    frame["split"] = frame["split"].astype(str)
    if splits is not None:
        split_set = {str(split) for split in splits}
        frame = frame[frame["split"].isin(split_set)].copy()

    assigned_frames: list[pd.DataFrame] = []
    for period in periods:
        mask = frame["signal_date"].between(period.start_ts, period.end_ts, inclusive="both")
        period_frame = frame.loc[mask].copy()
        if period_frame.empty:
            continue
        period_frame["period_label"] = str(period.label)
        period_frame["period_start"] = period.start_ts
        period_frame["period_end"] = period.end_ts
        assigned_frames.append(period_frame)
    if not assigned_frames:
        return _empty_period_industry_summary()
    assigned = pd.concat(assigned_frames, ignore_index=True)
    _validate_unique_period_industry_rows(assigned)

    records: list[dict[str, Any]] = []
    group_cols = [
        "period_label",
        "period_start",
        "period_end",
        "score_col",
        "return_col",
        "bucket_label",
        "industry",
    ]
    for key, group in assigned.groupby(group_cols, sort=True):
        (
            period_label,
            period_start,
            period_end,
            score_col,
            return_col,
            bucket_label,
            industry,
        ) = key
        group = group.sort_values("signal_date")
        active_values = _finite_values(group["active_weight"])
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
                "industry": str(industry),
                "date_count": int(len(group)),
                "mean_bucket_weight": _nanmean(group["bucket_weight"]),
                "mean_universe_weight": _nanmean(group["universe_weight"]),
                "mean_active_weight": _nanmean(active_values),
                "active_weight_std": _nanstd(active_values),
                "active_weight_tstat": _tstat(active_values),
                "positive_active_date_rate": _positive_rate(active_values),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["period_start", "period_label", "score_col", "bucket_index", "industry"]
    ).reset_index(drop=True)


def top_period_industry_preferences(
    period_industry_summary: pd.DataFrame,
    *,
    target_buckets: Sequence[int] = (1,),
    top_n: int = 10,
) -> pd.DataFrame:
    if top_n < 1:
        raise ValueError("top_n must be positive.")
    if period_industry_summary.empty:
        return _empty_period_industry_top()
    targets = {int(bucket) for bucket in target_buckets}
    frame = period_industry_summary[
        period_industry_summary["bucket_index"].astype(int).isin(targets)
    ].copy()
    records: list[dict[str, Any]] = []
    group_cols = [
        "period_label",
        "period_start",
        "period_end",
        "score_col",
        "return_col",
        "bucket_label",
    ]
    for key, group in frame.groupby(group_cols, sort=True):
        for side, ordered in [
            ("overweight", group.sort_values("mean_active_weight", ascending=False)),
            ("underweight", group.sort_values("mean_active_weight", ascending=True)),
        ]:
            for rank, row in enumerate(ordered.head(top_n).itertuples(index=False), start=1):
                records.append(
                    {
                        "signal_id": str(getattr(row, "signal_id", row.score_col)),
                        "signal_label": str(getattr(row, "signal_label", row.score_col)),
                        "source_column": str(getattr(row, "source_column", "")),
                        "alpha_input_kind": str(getattr(row, "alpha_input_kind", "")),
                        "neutralization": str(getattr(row, "neutralization", "")),
                        "period_label": str(row.period_label),
                        "period_start": pd.Timestamp(row.period_start),
                        "period_end": pd.Timestamp(row.period_end),
                        "actual_start": pd.Timestamp(row.actual_start),
                        "actual_end": pd.Timestamp(row.actual_end),
                        "score_col": str(row.score_col),
                        "return_col": str(row.return_col),
                        "bucket_label": str(row.bucket_label),
                        "bucket_index": int(row.bucket_index),
                        "industry_preference_side": side,
                        "side_rank": int(rank),
                        "industry": str(row.industry),
                        "mean_active_weight": float(row.mean_active_weight),
                        "mean_bucket_weight": float(row.mean_bucket_weight),
                        "mean_universe_weight": float(row.mean_universe_weight),
                        "active_weight_tstat": float(row.active_weight_tstat)
                        if np.isfinite(row.active_weight_tstat)
                        else np.nan,
                        "positive_active_date_rate": float(row.positive_active_date_rate)
                        if np.isfinite(row.positive_active_date_rate)
                        else np.nan,
                    }
                )
    if not records:
        return _empty_period_industry_top()
    return pd.DataFrame(records).sort_values(
        [
            "period_start",
            "period_label",
            "signal_id",
            "score_col",
            "bucket_index",
            "industry_preference_side",
            "side_rank",
        ]
    ).reset_index(drop=True)


def _read_template_predictions(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        frame = pd.read_pickle(path)
    else:
        frame = pd.read_csv(path, dtype={"stock_code": "string"})
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected template predictions DataFrame at {path}, got {type(frame)!r}")
    return frame


def _prepare_template_predictions(template_predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"fold_id", "split", "date", "stock_code"}
    missing = sorted(required.difference(template_predictions.columns))
    if missing:
        raise ValueError(f"Template predictions are missing required columns: {missing}")
    frame = template_predictions[["fold_id", "split", "date", "stock_code"]].copy()
    frame["fold_id"] = pd.to_numeric(frame["fold_id"], errors="raise").astype(int)
    frame["split"] = frame["split"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["stock_code"] = frame["stock_code"].map(_normalize_stock_code)
    invalid_keys = frame["date"].isna() | frame["stock_code"].eq("")
    if invalid_keys.any():
        sample = frame.loc[invalid_keys, ["fold_id", "split", "date", "stock_code"]].head(5)
        raise ValueError(f"Template predictions contain invalid keys: {sample.to_dict('records')}")
    duplicate_keys = ["fold_id", "split", "date", "stock_code"]
    if frame.duplicated(duplicate_keys).any():
        sample = frame.loc[frame.duplicated(duplicate_keys, keep=False), duplicate_keys].head(5)
        raise ValueError(f"Template predictions contain duplicate rows: {sample.to_dict('records')}")
    return frame.sort_values(duplicate_keys).reset_index(drop=True)


def _prepare_score_panel(
    training_panel: pd.DataFrame,
    *,
    specs: Sequence[AlphaBucketSignalSpec],
) -> pd.DataFrame:
    required = {"date", "stock_code", *(spec.source_column for spec in specs)}
    missing = sorted(required.difference(training_panel.columns))
    if missing:
        raise ValueError(f"Training panel is missing required alpha score columns: {missing}")
    columns = ["date", "stock_code", *(spec.source_column for spec in specs)]
    frame = training_panel[columns].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["stock_code"] = frame["stock_code"].map(_normalize_stock_code)
    invalid_keys = frame["date"].isna() | frame["stock_code"].eq("")
    if invalid_keys.any():
        sample = frame.loc[invalid_keys, ["date", "stock_code"]].head(5)
        raise ValueError(f"Training panel contains invalid date/stock_code keys: {sample.to_dict('records')}")
    duplicate_keys = ["date", "stock_code"]
    if frame.duplicated(duplicate_keys).any():
        sample = frame.loc[frame.duplicated(duplicate_keys, keep=False), duplicate_keys].head(5)
        raise ValueError(f"Training panel contains duplicate date/stock_code rows: {sample.to_dict('records')}")
    for spec in specs:
        frame[spec.source_column] = pd.to_numeric(frame[spec.source_column], errors="coerce")
    return frame.sort_values(duplicate_keys).reset_index(drop=True)


def _build_signal_predictions(
    template: pd.DataFrame,
    score_panel: pd.DataFrame,
    spec: AlphaBucketSignalSpec,
) -> pd.DataFrame:
    merged = template.merge(
        score_panel[["date", "stock_code", spec.source_column]],
        on=["date", "stock_code"],
        how="left",
        indicator=True,
        validate="many_to_one",
    )
    missing_panel = merged["_merge"].ne("both")
    if missing_panel.any():
        sample = merged.loc[
            missing_panel,
            ["fold_id", "split", "date", "stock_code"],
        ].head(10).copy()
        sample["date"] = pd.to_datetime(sample["date"]).dt.date.astype(str)
        raise ValueError(
            "Template predictions contain rows without matching training-panel alpha scores: "
            f"{sample.to_dict('records')}"
        )
    merged = merged.drop(columns=["_merge"]).rename(columns={spec.source_column: spec.score_col})
    return merged[["fold_id", "split", "date", "stock_code", spec.score_col]]


def _annotate_rank_result(
    result: dict[str, pd.DataFrame],
    spec: AlphaBucketSignalSpec,
) -> dict[str, pd.DataFrame]:
    return {
        key: _annotate_frame_with_spec(value, spec)
        for key, value in result.items()
    }


def _annotate_period_result(
    result: dict[str, pd.DataFrame],
    spec: AlphaBucketSignalSpec,
) -> dict[str, pd.DataFrame]:
    return {
        key: _annotate_frame_with_spec(value, spec)
        for key, value in result.items()
    }


def _annotate_frame_with_spec(frame: pd.DataFrame, spec: AlphaBucketSignalSpec) -> pd.DataFrame:
    annotated = frame.copy()
    annotated.insert(0, "signal_id", spec.signal_id)
    annotated.insert(1, "signal_label", spec.signal_label)
    annotated.insert(2, "source_column", spec.source_column)
    annotated.insert(3, "alpha_input_kind", spec.alpha_input_kind)
    annotated.insert(4, "neutralization", spec.neutralization)
    return annotated


def _annotate_with_signal_metadata(
    frame: pd.DataFrame,
    specs: Sequence[AlphaBucketSignalSpec],
) -> pd.DataFrame:
    if frame.empty:
        return frame
    metadata = {spec.signal_id: spec for spec in specs}
    annotated = frame.copy()
    score_values = annotated["score_col"].astype(str)
    annotated.insert(0, "signal_id", score_values)
    annotated.insert(1, "signal_label", score_values.map(lambda value: metadata[value].signal_label))
    annotated.insert(2, "source_column", score_values.map(lambda value: metadata[value].source_column))
    annotated.insert(3, "alpha_input_kind", score_values.map(lambda value: metadata[value].alpha_input_kind))
    annotated.insert(4, "neutralization", score_values.map(lambda value: metadata[value].neutralization))
    return annotated


def _signal_manifest(specs: Sequence[AlphaBucketSignalSpec]) -> pd.DataFrame:
    return pd.DataFrame([{**asdict(spec), "score_col": spec.score_col} for spec in specs])


def _concat_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    materialized = [frame for frame in frames if frame is not None and not frame.empty]
    if not materialized:
        return pd.DataFrame()
    return pd.concat(materialized, ignore_index=True)


def _validate_unique_period_industry_rows(assigned: pd.DataFrame) -> None:
    duplicate_keys = [
        "period_label",
        "signal_date",
        "score_col",
        "return_col",
        "bucket_label",
        "bucket_index",
        "industry",
    ]
    duplicated = assigned.duplicated(duplicate_keys, keep=False)
    if duplicated.any():
        sample = assigned.loc[duplicated, duplicate_keys + ["fold_id", "split"]].head(10)
        raise ValueError(
            "Period industry composition contains duplicate period/date/bucket/industry rows "
            f"after dropping fold_id as a grouping key: {sample.to_dict('records')}"
        )


def _write_comparison_charts(
    result: dict[str, pd.DataFrame],
    output_dir: Path,
    *,
    specs: Sequence[AlphaBucketSignalSpec],
    composition_target_buckets: Sequence[int] | None,
    composition_top_n_industries: int,
) -> dict[str, Path]:
    chart_paths: dict[str, Path] = {}
    for spec in specs:
        signal_dir = output_dir / spec.signal_id
        nav = result["nav"][result["nav"]["signal_id"].eq(spec.signal_id)].copy()
        signal_chart_paths = write_rank_bucket_nav_charts(nav, signal_dir / "nav")
        for key, path in signal_chart_paths.items():
            chart_paths[f"{spec.signal_id}_nav_{key}"] = path

        signal_result = {
            "composition_industry_summary": result["composition_industry_summary"][
                result["composition_industry_summary"]["signal_id"].eq(spec.signal_id)
            ].copy(),
            "composition_size_summary": result["composition_size_summary"][
                result["composition_size_summary"]["signal_id"].eq(spec.signal_id)
            ].copy(),
        }
        composition_chart_paths = write_rank_bucket_composition_charts(
            signal_result,
            signal_dir / "composition",
            target_buckets=composition_target_buckets,
            top_n_industries=composition_top_n_industries,
        )
        for key, path in composition_chart_paths.items():
            chart_paths[f"{spec.signal_id}_{key}"] = path
    return chart_paths


def _build_artifact_summary(
    result: dict[str, pd.DataFrame],
    *,
    output_paths: dict[str, Path],
    chart_paths: dict[str, Path],
    specs: Sequence[AlphaBucketSignalSpec],
    periods: Sequence[PeriodSpec],
    alpha_name: str,
    return_col: str,
    rank_bucket_count: int,
    rank_order: str,
    splits: Sequence[str] | None,
    fold_ids: Sequence[int] | None,
    min_names: int,
) -> dict[str, Any]:
    return {
        "schema_version": "alpha_signal_rank_bucket_comparison_v1",
        "metric_contract": {
            "template_sample_source": "template_predictions fold_id/split/date/stock_code",
            "score_source": "training_panel alpha columns joined by date and stock_code",
            "score_join_key": ["date", "stock_code"],
            "composition_join_key": ["date", "stock_code"],
            "return_col": return_col,
            "return_alignment": "inherits evaluate_lgbm_rank_bucket_nav; return matrix is already aligned to signal dates",
            "rank_scope": "daily_cross_section_within_fold_split",
            "rank_order": rank_order,
            "rank_bucket_count": int(rank_bucket_count),
            "bucket_weighting": "equal_weight_valid_returns_within_date_bucket",
            "period_nav_rule": "gross NAV restarts at 1.0 within each period, signal, and bucket",
            "industry_active_weight_formula": "bucket industry weight - finite-score eligible-universe industry weight",
            "liquidity_feature_policy": (
                "turnover, ADV, amount, volume, and related liquidity fields are not used as bucket scores"
            ),
        },
        "alpha_name": alpha_name,
        "signals": [asdict(spec) | {"score_col": spec.score_col} for spec in specs],
        "periods": [
            {
                "label": str(period.label),
                "start": period.start_ts.date().isoformat(),
                "end": period.end_ts.date().isoformat(),
            }
            for period in periods
        ],
        "filters": {
            "splits": list(splits) if splits is not None else None,
            "fold_ids": list(fold_ids) if fold_ids is not None else None,
        },
        "min_names": int(min_names),
        "row_counts": {
            key: int(len(value))
            for key, value in result.items()
            if isinstance(value, pd.DataFrame)
        },
        "outputs": {
            key: _path_for_summary(path)
            for key, path in output_paths.items()
        },
        "charts": {
            key: _path_for_summary(path)
            for key, path in chart_paths.items()
        },
        "notes": [
            "This analysis does not retrain the model; it only changes the standalone bucket ranking score.",
            "Raw and neutralized alpha signals are evaluated over the same template fold/split/date-stock rows.",
            "Only finite score values enter each daily ranking universe.",
            "The default signal set includes both alpha value and alpha rank inputs because the current model used both.",
        ],
    }


def _empty_period_industry_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "signal_id",
            "signal_label",
            "source_column",
            "alpha_input_kind",
            "neutralization",
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


def _empty_period_industry_top() -> pd.DataFrame:
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
            "industry_preference_side",
            "side_rank",
            "industry",
            "mean_active_weight",
            "mean_bucket_weight",
            "mean_universe_weight",
            "active_weight_tstat",
            "positive_active_date_rate",
        ]
    )


__all__ = [
    "AlphaBucketSignalSpec",
    "PeriodSpec",
    "evaluate_alpha_signal_rank_bucket_comparison",
    "resolve_alpha_bucket_signal_specs",
    "summarize_period_industry_composition",
    "top_period_industry_preferences",
    "write_alpha_signal_rank_bucket_comparison_artifacts",
]


if __name__ == "__main__":
    main()
