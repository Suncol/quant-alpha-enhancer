from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from analysis.build_lgbm_placeholder_training_data import (
        DEFAULT_INDEX_COLUMNS,
        DEFAULT_WALK_FORWARD_FOLDS,
        EPS,
        WalkForwardFold,
        _cross_sectional_robust_z,
        _date_to_string,
        _decile_codes,
        _feature_standardization_diagnostics,
        _index_bucket,
        _normalize_stock_code,
        _path_for_summary,
        _records_with_iso_dates,
        _split_summary,
        _write_csv_with_iso_dates,
        make_embargoed_dates,
        make_walk_forward_fold_assignments,
    )
    from analysis.neutralize_return_y import centered_rank
except ModuleNotFoundError:  # Allows direct execution via python analysis/script.py.
    from build_lgbm_placeholder_training_data import (
        DEFAULT_INDEX_COLUMNS,
        DEFAULT_WALK_FORWARD_FOLDS,
        EPS,
        WalkForwardFold,
        _cross_sectional_robust_z,
        _date_to_string,
        _decile_codes,
        _feature_standardization_diagnostics,
        _index_bucket,
        _normalize_stock_code,
        _path_for_summary,
        _records_with_iso_dates,
        _split_summary,
        _write_csv_with_iso_dates,
        make_embargoed_dates,
        make_walk_forward_fold_assignments,
    )
    from neutralize_return_y import centered_rank


DEFAULT_ALPHA_RAW_COLUMN = "factor_sss_dx_10_raw"
DEFAULT_RAW_RETURN_TARGET_COLUMN = "y_return_hfq_adj_fwd"
DEFAULT_RAW_RETURN_RANK_COLUMN = "y_return_rank_label"
RAW_LIQUIDITY_COLUMNS = ("amount_k", "turnover", "logADV20", "logAmount20", "turnover20")
KLINE_SIGNAL_TRANSFORMS: Mapping[str, str] = {
    "close": "log_positive",
    "amo": "log1p_nonnegative",
    "vol": "log1p_nonnegative",
}
KLINE_RAW_COLUMNS = tuple(f"{name}_raw" for name in KLINE_SIGNAL_TRANSFORMS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a non-neutralized real-alpha training panel with raw forward-return "
            "targets and exposure/context features."
        )
    )
    parser.add_argument("--raw-features", default=None, type=Path)
    parser.add_argument("--exposures", default=None, type=Path)
    parser.add_argument("--alpha-value", default=None, type=Path)
    parser.add_argument("--raw-return", required=True, type=Path)
    parser.add_argument("--output-panel", required=True, type=Path)
    parser.add_argument("--output-fold-assignments", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--diagnostics-dir", required=True, type=Path)
    parser.add_argument("--alpha-raw-col", default=DEFAULT_ALPHA_RAW_COLUMN)
    parser.add_argument("--target-col", default=DEFAULT_RAW_RETURN_TARGET_COLUMN)
    parser.add_argument("--target-rank-col", default=DEFAULT_RAW_RETURN_RANK_COLUMN)
    parser.add_argument(
        "--include-liquidity-context-features",
        action="store_true",
        help="Include same-date amount/turnover/ADV rank-z features as context features.",
    )
    parser.add_argument(
        "--include-kline-signal-features",
        action="store_true",
        help=(
            "Include same-signal-date close/amo/vol raw-value rank-z features as "
            "unconstrained continuous context features."
        ),
    )
    parser.add_argument(
        "--exclude-index-context-features",
        action="store_true",
        help="Exclude index_bucket and is_csi* index-membership context features from the panel and feature roles.",
    )
    parser.add_argument(
        "--fold-train-start-date",
        default=None,
        help="Override the train_start date of all default walk-forward folds without changing valid/test windows.",
    )
    parser.add_argument("--winsor-lower", default=0.01, type=float)
    parser.add_argument("--winsor-upper", default=0.99, type=float)
    parser.add_argument("--z-clip", default=5.0, type=float)
    parser.add_argument("--embargo-trading-days", default=2, type=int)
    args = parser.parse_args()

    if args.raw_features is not None:
        raw_features = _read_frame(args.raw_features)
        raw_feature_source = "raw_features_table"
    elif args.exposures is not None and args.alpha_value is not None:
        raw_features = build_raw_feature_table_from_exposures(
            _read_frame(args.exposures),
            _read_frame(args.alpha_value),
            alpha_raw_column=args.alpha_raw_col,
            include_index_context_features=not args.exclude_index_context_features,
        )
        raw_feature_source = "exposures_plus_alpha_value_matrix"
    else:
        parser.error("Provide either --raw-features or both --exposures and --alpha-value.")

    folds = (
        default_walk_forward_folds_with_train_start(args.fold_train_start_date)
        if args.fold_train_start_date is not None
        else DEFAULT_WALK_FORWARD_FOLDS
    )
    summary = write_non_neutralized_training_data_artifacts(
        raw_features=raw_features,
        raw_return=_read_frame(args.raw_return),
        output_panel=args.output_panel,
        output_fold_assignments=args.output_fold_assignments,
        output_summary=args.output_summary,
        diagnostics_dir=args.diagnostics_dir,
        alpha_raw_column=args.alpha_raw_col,
        raw_feature_source=raw_feature_source,
        target_col=args.target_col,
        target_rank_col=args.target_rank_col,
        include_liquidity_context_features=args.include_liquidity_context_features,
        include_kline_signal_features=args.include_kline_signal_features,
        include_index_context_features=not args.exclude_index_context_features,
        folds=folds,
        winsor_lower=args.winsor_lower,
        winsor_upper=args.winsor_upper,
        z_clip=args.z_clip,
        embargo_trading_days=args.embargo_trading_days,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_non_neutralized_training_panel(
    raw_features: pd.DataFrame,
    raw_return: pd.DataFrame,
    *,
    alpha_raw_column: str = DEFAULT_ALPHA_RAW_COLUMN,
    target_col: str = DEFAULT_RAW_RETURN_TARGET_COLUMN,
    target_rank_col: str = DEFAULT_RAW_RETURN_RANK_COLUMN,
    include_liquidity_context_features: bool = False,
    include_kline_signal_features: bool = False,
    include_index_context_features: bool = True,
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
    z_clip: float = 5.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _validate_winsor_config(winsor_lower, winsor_upper, z_clip)
    _validate_target_names(target_col, target_rank_col)

    feature_frame, input_diagnostics = _build_feature_frame(
        raw_features,
        alpha_raw_column=alpha_raw_column,
        include_liquidity_context_features=include_liquidity_context_features,
        include_kline_signal_features=include_kline_signal_features,
        include_index_context_features=include_index_context_features,
        winsor_lower=winsor_lower,
        winsor_upper=winsor_upper,
        z_clip=z_clip,
    )
    signal_feature_columns = _signal_feature_columns(alpha_raw_column)
    context_feature_columns = _context_feature_columns(
        include_liquidity_context_features=include_liquidity_context_features,
        include_kline_signal_features=include_kline_signal_features,
        include_index_context_features=include_index_context_features,
    )
    feature_eligible_frame = _filter_required_feature_rows(
        feature_frame,
        signal_feature_columns=signal_feature_columns,
        context_feature_columns=context_feature_columns,
    )
    if feature_eligible_frame.empty:
        raise ValueError("No rows remain after filtering same-date raw alpha and context features.")
    feature_date_counts = feature_eligible_frame.groupby("date")["stock_code"].size()

    label_frame, label_diagnostics = _raw_return_labels_to_long_frame(
        raw_return,
        target_col=target_col,
        target_rank_col=target_rank_col,
    )
    feature_row_count = int(len(feature_frame))
    panel = feature_frame.merge(
        label_frame,
        on=["date", "stock_code"],
        how="inner",
        validate="one_to_one",
    )
    dropped_missing_label_rows = int(feature_row_count - len(panel))

    before_feature_filter = int(len(panel))
    panel = _filter_required_feature_rows(
        panel,
        signal_feature_columns=signal_feature_columns,
        context_feature_columns=context_feature_columns,
    )
    dropped_missing_feature_rows = int(before_feature_filter - len(panel))
    if panel.empty:
        raise ValueError("No rows remain after joining raw returns and filtering required features.")

    panel = _add_sample_weights_from_feature_universe(panel, feature_date_counts)
    panel = panel.sort_values(["date", "stock_code"]).reset_index(drop=True)

    feature_roles = _feature_roles(
        alpha_raw_column,
        target_col=target_col,
        target_rank_col=target_rank_col,
        include_liquidity_context_features=include_liquidity_context_features,
        include_kline_signal_features=include_kline_signal_features,
        include_index_context_features=include_index_context_features,
    )
    diagnostics = {
        "summary": {
            "row_count": int(len(panel)),
            "date_count": int(panel["date"].nunique()),
            "stock_count": int(panel["stock_code"].nunique()),
            "date_min": _date_to_string(panel["date"].min()),
            "date_max": _date_to_string(panel["date"].max()),
            "raw_feature_row_count": feature_row_count,
            "feature_eligible_row_count_before_label": int(len(feature_eligible_frame)),
            "label_row_count": int(len(label_frame)),
            "dropped_missing_label_rows": dropped_missing_label_rows,
            "dropped_missing_feature_rows": dropped_missing_feature_rows,
            "feature_transform_universe": "same_date_raw_feature_rows_before_label_join",
            "sample_weight_universe": "same_date_feature_eligible_rows_before_label_join",
            "alpha_raw_column": alpha_raw_column,
            "alpha_source": _alpha_source_from_raw_column(alpha_raw_column),
            "return_neutralization": {"enabled": False},
            "alpha_signal_neutralization": {"enabled": False},
            "target_columns": [target_col],
            "auxiliary_label_columns": [target_rank_col],
            "target_transform": "raw_forward_return_no_winsorization_no_residualization",
            "standardization": {
                "scope": "daily_cross_section",
                "winsor_lower": float(winsor_lower),
                "winsor_upper": float(winsor_upper),
                "z_clip": float(z_clip),
                "rank_convention": "centered_rank_in_roughly_minus_0p5_to_plus_0p5",
                "z_convention": "winsorized_robust_z_with_median_mad_scale",
            },
            "liquidity_context_features_in_model": bool(include_liquidity_context_features),
            "kline_signal_features_in_model": bool(include_kline_signal_features),
            "kline_signal_feature_policy": {
                "enabled": bool(include_kline_signal_features),
                "model_role": "condition_continuous",
                "source_columns": list(KLINE_RAW_COLUMNS) if include_kline_signal_features else [],
                "rank_transform": "daily_centered_rank_of_same_signal_date_raw_values",
                "z_transform": "daily_robust_z_after_domain_transform",
                "raw_transforms": dict(KLINE_SIGNAL_TRANSFORMS) if include_kline_signal_features else {},
            },
            "index_context_features_in_model": bool(include_index_context_features),
        },
        "input_diagnostics": input_diagnostics,
        "label_diagnostics": label_diagnostics,
        "feature_calendar_dates": [
            pd.Timestamp(date).date().isoformat()
            for date in sorted(feature_eligible_frame["date"].unique())
        ],
        "feature_standardization": _feature_standardization_diagnostics(feature_eligible_frame),
        "raw_return_distribution": _raw_return_distribution_by_date(
            panel,
            target_col=target_col,
            target_rank_col=target_rank_col,
        ),
        "feature_roles": feature_roles,
    }
    return panel, diagnostics


def build_raw_feature_table_from_exposures(
    exposures: pd.DataFrame,
    alpha_value: pd.DataFrame,
    *,
    alpha_raw_column: str = DEFAULT_ALPHA_RAW_COLUMN,
    include_index_context_features: bool = True,
) -> pd.DataFrame:
    """Merge raw exposure rows with a raw alpha matrix by signal date and stock code."""

    context = _prepare_exposure_context_for_raw_feature_table(
        exposures,
        include_index_context_features=include_index_context_features,
    )
    alpha_long = _signal_wide_frame_to_long(
        alpha_value,
        signal_name=_alpha_source_from_raw_column(alpha_raw_column),
        raw_column=alpha_raw_column,
        date_index=context["date"].drop_duplicates(),
        stock_codes=context["stock_code"].drop_duplicates(),
    )
    frame = context.merge(alpha_long, on=["date", "stock_code"], how="left", validate="one_to_one")
    frame[alpha_raw_column] = pd.to_numeric(frame[alpha_raw_column], errors="coerce")
    return frame.sort_values(["date", "stock_code"]).reset_index(drop=True)


def default_walk_forward_folds_with_train_start(train_start_date: str) -> tuple[WalkForwardFold, ...]:
    """Return default walk-forward folds with a shared earlier train_start."""

    train_start = pd.Timestamp(train_start_date).normalize()
    folds: list[WalkForwardFold] = []
    for fold in DEFAULT_WALK_FORWARD_FOLDS:
        train_end = pd.Timestamp(fold.train_end).normalize()
        if train_start > train_end:
            raise ValueError(
                f"train_start_date {train_start.date().isoformat()} is after "
                f"fold {fold.fold_id} train_end {train_end.date().isoformat()}."
            )
        folds.append(
            WalkForwardFold(
                fold_id=fold.fold_id,
                train_start=train_start.date().isoformat(),
                train_end=fold.train_end,
                valid_start=fold.valid_start,
                valid_end=fold.valid_end,
                test_start=fold.test_start,
                test_end=fold.test_end,
            )
        )
    return tuple(folds)


def write_non_neutralized_training_data_artifacts(
    *,
    raw_features: pd.DataFrame,
    raw_return: pd.DataFrame,
    output_panel: Path,
    output_fold_assignments: Path,
    output_summary: Path,
    diagnostics_dir: Path,
    alpha_raw_column: str = DEFAULT_ALPHA_RAW_COLUMN,
    raw_feature_source: str = "raw_features_table",
    target_col: str = DEFAULT_RAW_RETURN_TARGET_COLUMN,
    target_rank_col: str = DEFAULT_RAW_RETURN_RANK_COLUMN,
    include_liquidity_context_features: bool = False,
    include_kline_signal_features: bool = False,
    include_index_context_features: bool = True,
    folds: Sequence[WalkForwardFold] = DEFAULT_WALK_FORWARD_FOLDS,
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
    z_clip: float = 5.0,
    embargo_trading_days: int = 2,
) -> dict[str, Any]:
    panel, diagnostics = build_non_neutralized_training_panel(
        raw_features,
        raw_return,
        alpha_raw_column=alpha_raw_column,
        target_col=target_col,
        target_rank_col=target_rank_col,
        include_liquidity_context_features=include_liquidity_context_features,
        include_kline_signal_features=include_kline_signal_features,
        include_index_context_features=include_index_context_features,
        winsor_lower=winsor_lower,
        winsor_upper=winsor_upper,
        z_clip=z_clip,
    )
    feature_dates = pd.to_datetime(diagnostics["feature_calendar_dates"])
    fold_assignments = make_walk_forward_fold_assignments(
        feature_dates,
        folds,
        embargo_trading_days=embargo_trading_days,
    )
    embargoed_dates = make_embargoed_dates(
        feature_dates,
        folds,
        embargo_trading_days=embargo_trading_days,
    )
    split_summary = _split_summary(panel, fold_assignments)

    output_panel.parent.mkdir(parents=True, exist_ok=True)
    output_fold_assignments.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    panel.to_pickle(output_panel)
    _write_csv_with_iso_dates(fold_assignments, output_fold_assignments)
    _write_csv_with_iso_dates(
        diagnostics["feature_standardization"],
        diagnostics_dir / "feature_standardization_diagnostics.csv",
    )
    _write_csv_with_iso_dates(
        diagnostics["raw_return_distribution"],
        diagnostics_dir / "raw_return_distribution_by_date.csv",
    )
    _write_csv_with_iso_dates(split_summary, diagnostics_dir / "split_summary.csv")
    _write_csv_with_iso_dates(embargoed_dates, diagnostics_dir / "embargoed_dates.csv")

    summary = {
        "metadata": {
            "signal_stage": "non_neutralized_real_alpha_raw_return_panel",
            "alpha_source": diagnostics["summary"]["alpha_source"],
            "alpha_is_real": True,
            "production_eligible": False,
            "model_form": "p = g(raw_alpha_rank_z, exposure_context)",
            "optimization_objective": "raw_forward_return",
            "target_col": target_col,
            "fit_target_col": target_col,
            "evaluation_target_col": target_col,
            "auxiliary_rank_label_col": target_rank_col,
            "label_contract": (
                f"{target_col} is a non-neutralized forward return aligned to signal dates; "
                "the builder does not winsorize, residualize, or rank-transform the fit target."
            ),
            "neutralization_policy": {
                "return_y": "not_neutralized",
                "alpha_signal": "not_neutralized",
            },
            "condition_set": (
                _condition_set_name(
                    include_index_context_features=include_index_context_features,
                    include_liquidity_context_features=include_liquidity_context_features,
                    include_kline_signal_features=include_kline_signal_features,
                )
            ),
            "feature_asof": "same_signal_date_raw_feature_inputs",
            "raw_feature_source": raw_feature_source,
        },
        "panel": diagnostics["summary"],
        "input_diagnostics": diagnostics["input_diagnostics"],
        "label_diagnostics": diagnostics["label_diagnostics"],
        "folds": [asdict(fold) for fold in folds],
        "embargo_trading_days": int(embargo_trading_days),
        "embargoed_date_count": int(len(embargoed_dates)),
        "embargoed_dates": _records_with_iso_dates(embargoed_dates),
        "fold_assignment_row_count": int(len(fold_assignments)),
        "feature_roles": diagnostics["feature_roles"],
        "split_summary": _records_with_iso_dates(split_summary),
        "outputs": {
            "panel": _path_for_summary(output_panel),
            "fold_assignments": _path_for_summary(output_fold_assignments),
            "summary": _path_for_summary(output_summary),
            "diagnostics_dir": _path_for_summary(diagnostics_dir),
            "feature_standardization": _path_for_summary(
                diagnostics_dir / "feature_standardization_diagnostics.csv"
            ),
            "raw_return_distribution": _path_for_summary(
                diagnostics_dir / "raw_return_distribution_by_date.csv"
            ),
            "split_summary": _path_for_summary(diagnostics_dir / "split_summary.csv"),
            "embargoed_dates": _path_for_summary(diagnostics_dir / "embargoed_dates.csv"),
        },
        "notes": [
            "No return neutralization is performed; the raw target is copied from the supplied return matrix by date-stock key.",
            "No alpha signal neutralization is performed; alpha rank/z features are daily cross-sectional transforms of the raw alpha column.",
            "Daily rank and robust-z features are computed before joining forward returns.",
            "Sample weights use the same-date feature-eligible universe before label join.",
            "Raw market-cap and alpha columns are retained for traceability and excluded from model feature roles.",
        ],
    }
    if include_kline_signal_features:
        summary["notes"].append(
            "Kline close/amo/vol rank-z features use same-signal-date raw inputs and are included as condition_continuous, not signal_features."
        )
        summary["notes"].append(
            "The builder does not shift Kline features; no future return data is used when computing their daily cross-sectional transforms."
        )
    else:
        summary["notes"].append(
            "Kline close/amo/vol rank-z features are excluded from model feature_roles."
        )
    output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _condition_set_name(
    *,
    include_index_context_features: bool,
    include_liquidity_context_features: bool,
    include_kline_signal_features: bool,
) -> str:
    pieces = ["industry", "board"]
    if include_index_context_features:
        pieces.append("index")
    pieces.append("size")
    if include_liquidity_context_features:
        pieces.extend(["liquidity", "turnover"])
    if include_kline_signal_features:
        pieces.extend(["kline", "close", "amount", "volume"])
    return "_".join(pieces) + "_v1"


def _build_feature_frame(
    raw_features: pd.DataFrame,
    *,
    alpha_raw_column: str,
    include_liquidity_context_features: bool,
    include_kline_signal_features: bool,
    include_index_context_features: bool,
    winsor_lower: float,
    winsor_upper: float,
    z_clip: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, diagnostics = _prepare_raw_features(
        raw_features,
        alpha_raw_column=alpha_raw_column,
        include_liquidity_context_features=include_liquidity_context_features,
        include_kline_signal_features=include_kline_signal_features,
        include_index_context_features=include_index_context_features,
    )
    alpha_feature_name = _alpha_source_from_raw_column(alpha_raw_column)
    frame["alpha_source"] = alpha_feature_name
    frame["log_mcap"] = np.log1p(frame["market_cap"].where(frame["market_cap"] > 0.0))
    frame["log_mcap_z"] = _daily_transform(
        frame,
        "log_mcap",
        lambda values: _cross_sectional_robust_z(
            values,
            winsor_lower=winsor_lower,
            winsor_upper=winsor_upper,
            z_clip=z_clip,
        ),
    )
    frame["mcap_rank"] = _daily_transform(frame, "market_cap", centered_rank)
    frame["size_decile"] = _daily_transform(frame, "market_cap", _decile_codes).astype("int8")
    if include_index_context_features:
        frame["index_bucket"] = _index_bucket_series(frame)

    frame[f"{alpha_feature_name}_rank"] = _daily_transform(frame, alpha_raw_column, centered_rank)
    frame[f"{alpha_feature_name}_z"] = _daily_transform(
        frame,
        alpha_raw_column,
        lambda values: _cross_sectional_robust_z(
            values,
            winsor_lower=winsor_lower,
            winsor_upper=winsor_upper,
            z_clip=z_clip,
        ),
    )

    if include_liquidity_context_features:
        for column in RAW_LIQUIDITY_COLUMNS:
            transformed_column = f"{column}__transform_input"
            frame[transformed_column] = _liquidity_transform_input(frame[column], column)
            frame[f"{column}_rank"] = _daily_transform(frame, transformed_column, centered_rank)
            frame[f"{column}_z"] = _daily_transform(
                frame,
                transformed_column,
                lambda values: _cross_sectional_robust_z(
                    values,
                    winsor_lower=winsor_lower,
                    winsor_upper=winsor_upper,
                    z_clip=z_clip,
                ),
            )

    if include_kline_signal_features:
        for signal_name, raw_transform in KLINE_SIGNAL_TRANSFORMS.items():
            _add_rank_z_columns(
                frame,
                source_column=f"{signal_name}_raw",
                output_name=signal_name,
                raw_transform=raw_transform,
                winsor_lower=winsor_lower,
                winsor_upper=winsor_upper,
                z_clip=z_clip,
            )

    return frame, diagnostics


def _prepare_raw_features(
    raw_features: pd.DataFrame,
    *,
    alpha_raw_column: str,
    include_liquidity_context_features: bool,
    include_kline_signal_features: bool,
    include_index_context_features: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not isinstance(raw_features, pd.DataFrame):
        raise TypeError(f"raw_features must be a pandas DataFrame, got {type(raw_features)!r}")

    required = [
        "date",
        "stock_code",
        "industry",
        "board",
        *(DEFAULT_INDEX_COLUMNS if include_index_context_features else ()),
        "market_cap",
        alpha_raw_column,
    ]
    if include_liquidity_context_features:
        required.extend(RAW_LIQUIDITY_COLUMNS)
    if include_kline_signal_features:
        required.extend(KLINE_RAW_COLUMNS)
    missing = sorted(set(required).difference(raw_features.columns))
    if missing:
        raise ValueError(f"raw_features missing required columns: {missing}")

    columns = list(dict.fromkeys(required))
    frame = raw_features.loc[:, columns].copy()
    frame["date"] = _parse_dates(frame["date"])
    frame["stock_code"] = frame["stock_code"].map(_normalize_stock_code)
    invalid_keys = frame["date"].isna() | frame["stock_code"].eq("")
    invalid_key_count = int(invalid_keys.sum())
    if invalid_key_count:
        frame = frame.loc[~invalid_keys].copy()
    if frame.duplicated(["date", "stock_code"]).any():
        duplicates = frame.loc[
            frame.duplicated(["date", "stock_code"], keep=False),
            ["date", "stock_code"],
        ]
        raise ValueError(f"raw_features contain duplicate date, stock_code keys: {_records_sample(duplicates)}")

    missing_industry_rows = int(frame["industry"].isna().sum())
    missing_board_rows = int(frame["board"].isna().sum())
    frame["industry"] = frame["industry"].fillna("unknown_industry").astype(str)
    frame["board"] = frame["board"].fillna("UNKNOWN_BOARD").astype(str)

    numeric_columns = ["market_cap", alpha_raw_column]
    if include_index_context_features:
        numeric_columns = [*DEFAULT_INDEX_COLUMNS, *numeric_columns]
    if include_liquidity_context_features:
        numeric_columns.extend(RAW_LIQUIDITY_COLUMNS)
    if include_kline_signal_features:
        numeric_columns.extend(KLINE_RAW_COLUMNS)
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if include_index_context_features:
        _validate_index_flags(frame)

    return frame.sort_values(["date", "stock_code"]).reset_index(drop=True), {
        "input_row_count": int(len(raw_features)),
        "invalid_key_rows_dropped": invalid_key_count,
        "prepared_row_count": int(len(frame)),
        "missing_industry_rows_filled": missing_industry_rows,
        "missing_board_rows_filled": missing_board_rows,
        "alpha_raw_column": alpha_raw_column,
        "include_liquidity_context_features": bool(include_liquidity_context_features),
        "include_kline_signal_features": bool(include_kline_signal_features),
        "include_index_context_features": bool(include_index_context_features),
    }


def _prepare_exposure_context_for_raw_feature_table(
    exposures: pd.DataFrame,
    *,
    include_index_context_features: bool = True,
) -> pd.DataFrame:
    if not isinstance(exposures, pd.DataFrame):
        raise TypeError(f"exposures must be a pandas DataFrame, got {type(exposures)!r}")
    required = [
        "date",
        "stock_code",
        "industry",
        "board",
        *(DEFAULT_INDEX_COLUMNS if include_index_context_features else ()),
        "market_cap",
    ]
    missing = sorted(set(required).difference(exposures.columns))
    if missing:
        raise ValueError(f"exposures missing required columns: {missing}")

    optional_liquidity = [column for column in RAW_LIQUIDITY_COLUMNS if column in exposures.columns]
    frame = exposures.loc[:, required + optional_liquidity].copy()
    frame["date"] = _parse_dates(frame["date"])
    frame["stock_code"] = frame["stock_code"].map(_normalize_stock_code)
    frame = frame[frame["date"].notna() & frame["stock_code"].ne("")].copy()
    if frame.duplicated(["date", "stock_code"]).any():
        duplicates = frame.loc[
            frame.duplicated(["date", "stock_code"], keep=False),
            ["date", "stock_code"],
        ]
        raise ValueError(f"exposures contain duplicate date, stock_code keys: {_records_sample(duplicates)}")

    numeric_columns = ["market_cap", *optional_liquidity]
    if include_index_context_features:
        numeric_columns = [*DEFAULT_INDEX_COLUMNS, *numeric_columns]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if include_index_context_features:
        _validate_index_flags(frame)
    return frame.sort_values(["date", "stock_code"]).reset_index(drop=True)


def _validate_index_flags(frame: pd.DataFrame) -> None:
    flags = frame.loc[:, list(DEFAULT_INDEX_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    values = flags.to_numpy(dtype=float, copy=False)
    finite = np.isfinite(values)
    valid_binary = np.isin(values, [0.0, 1.0])
    invalid_binary_rows = (finite & ~valid_binary).any(axis=1)
    if invalid_binary_rows.any():
        sample = frame.loc[invalid_binary_rows, ["date", "stock_code"]].head(5)
        raise ValueError(f"Index membership flags must be finite 0/1 when present. Invalid rows: {_records_sample(sample)}")

    complete = finite.all(axis=1)
    overlapping = complete & (np.nansum(values, axis=1) > 1.0)
    if overlapping.any():
        sample = frame.loc[overlapping, ["date", "stock_code", *DEFAULT_INDEX_COLUMNS]].head(5)
        raise ValueError(f"Rows contain multiple index membership flags: {_records_sample(sample)}")


def _index_bucket_series(frame: pd.DataFrame) -> pd.Series:
    flags = frame.loc[:, list(DEFAULT_INDEX_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    output = pd.Series(pd.NA, index=frame.index, dtype="object")
    complete = np.isfinite(flags.to_numpy(dtype=float, copy=False)).all(axis=1)
    if complete.any():
        output.loc[complete] = [
            _index_bucket(row)
            for row in flags.loc[complete, list(DEFAULT_INDEX_COLUMNS)].itertuples(index=False, name=None)
        ]
    return output


def _raw_return_labels_to_long_frame(
    raw_return: pd.DataFrame,
    *,
    target_col: str,
    target_rank_col: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    returns = _prepare_return_matrix(raw_return)
    rank_label = pd.DataFrame(
        [centered_rank(row).to_numpy(dtype=float) for _, row in returns.iterrows()],
        index=returns.index,
        columns=returns.columns,
    )
    target_long = _wide_frame_to_long(returns, target_col)
    rank_long = _wide_frame_to_long(rank_label, target_rank_col)
    labels = target_long.merge(rank_long, on=["date", "stock_code"], how="inner", validate="one_to_one")
    labels[target_col] = pd.to_numeric(labels[target_col], errors="coerce")
    labels[target_rank_col] = pd.to_numeric(labels[target_rank_col], errors="coerce")
    finite = (
        np.isfinite(labels[target_col].to_numpy(dtype=float, copy=False))
        & np.isfinite(labels[target_rank_col].to_numpy(dtype=float, copy=False))
    )
    labels = labels.loc[finite].sort_values(["date", "stock_code"]).reset_index(drop=True)
    return labels, {
        "raw_return_shape": {"rows": int(returns.shape[0]), "columns": int(returns.shape[1])},
        "raw_return_date_min": returns.index.min().date().isoformat() if len(returns.index) else None,
        "raw_return_date_max": returns.index.max().date().isoformat() if len(returns.index) else None,
        "long_label_row_count": int(len(labels)),
        "target_col": target_col,
        "target_rank_col": target_rank_col,
        "target_source_transform": "raw_matrix_values_copied_without_winsorization_or_residualization",
    }


def _prepare_return_matrix(raw_return: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(raw_return, pd.DataFrame):
        raise TypeError(f"raw_return must be a pandas DataFrame, got {type(raw_return)!r}")
    frame = raw_return.copy()
    frame.index = pd.DatetimeIndex(pd.to_datetime(list(frame.index), errors="coerce")).normalize()
    if frame.index.hasnans:
        raise ValueError("raw_return index contains unparseable dates.")
    if frame.index.duplicated().any():
        duplicates = sorted({pd.Timestamp(date).date().isoformat() for date in frame.index[frame.index.duplicated()]})
        raise ValueError(f"raw_return index contains duplicate normalized dates: {duplicates[:5]}")
    normalized_columns = [_normalize_stock_code(column) for column in frame.columns]
    if len(set(normalized_columns)) != len(normalized_columns):
        duplicates = sorted({code for code in normalized_columns if normalized_columns.count(code) > 1})
        raise ValueError(f"raw_return columns contain duplicate normalized stock codes: {duplicates[:5]}")
    frame.columns = normalized_columns
    frame = frame.apply(pd.to_numeric, errors="coerce")
    return frame.sort_index()


def _wide_frame_to_long(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
    series = frame.rename_axis(index="date", columns="stock_code").stack()
    long = series.rename(value_name).reset_index()
    long["date"] = pd.to_datetime(long["date"], errors="coerce").dt.normalize()
    long["stock_code"] = long["stock_code"].map(_normalize_stock_code)
    return long[long[value_name].notna()].reset_index(drop=True)


def _signal_wide_frame_to_long(
    frame: pd.DataFrame,
    *,
    signal_name: str,
    raw_column: str,
    date_index: Sequence[Any],
    stock_codes: Sequence[Any],
) -> pd.DataFrame:
    matrix = _normalize_signal_matrix(frame, signal_name)
    dates = pd.DatetimeIndex(_parse_dates(pd.Series(list(date_index)))).drop_duplicates()
    normalized_stock_codes = pd.Index([_normalize_stock_code(code) for code in stock_codes])
    normalized_stock_codes = normalized_stock_codes[normalized_stock_codes != ""].drop_duplicates()
    matrix = matrix.reindex(index=dates, columns=normalized_stock_codes)
    long = matrix.copy()
    long.index.name = "date"
    long.columns.name = "stock_code"
    long = long.reset_index().melt(
        id_vars="date",
        var_name="stock_code",
        value_name=raw_column,
    )
    long["date"] = _parse_dates(long["date"])
    long["stock_code"] = long["stock_code"].map(_normalize_stock_code)
    long[raw_column] = pd.to_numeric(long[raw_column], errors="coerce")
    if long.duplicated(["date", "stock_code"]).any():
        duplicates = long.loc[
            long.duplicated(["date", "stock_code"], keep=False),
            ["date", "stock_code"],
        ]
        raise ValueError(f"Signal {signal_name!r} contains duplicate date, stock_code keys: {_records_sample(duplicates)}")
    return long.sort_values(["date", "stock_code"]).reset_index(drop=True)


def _normalize_signal_matrix(frame: pd.DataFrame, signal_name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Signal {signal_name!r} must be a pandas DataFrame, got {type(frame)!r}")
    row_date_ratio = _date_like_ratio(frame.index)
    column_date_ratio = _date_like_ratio(frame.columns)
    if column_date_ratio >= 0.5 and column_date_ratio > row_date_ratio:
        matrix = frame.T.copy()
    elif row_date_ratio >= 0.5 and row_date_ratio > column_date_ratio:
        matrix = frame.copy()
    else:
        raise ValueError(
            f"Cannot infer orientation for signal {signal_name!r}; expected stock x date or date x stock wide frame."
        )

    matrix.index = pd.DatetimeIndex(_parse_dates(pd.Series(list(matrix.index))))
    if matrix.index.hasnans:
        raise ValueError(f"Signal {signal_name!r} has unparseable dates on the date axis.")
    if matrix.index.duplicated().any():
        duplicates = sorted({pd.Timestamp(date).date().isoformat() for date in matrix.index[matrix.index.duplicated()]})
        raise ValueError(f"Signal {signal_name!r} has duplicate normalized dates: {duplicates[:5]}")

    stock_codes = [_normalize_stock_code(column) for column in matrix.columns]
    keep_columns = [bool(code) for code in stock_codes]
    matrix = matrix.loc[:, keep_columns].copy()
    stock_codes = [code for code in stock_codes if code]
    if len(set(stock_codes)) != len(stock_codes):
        duplicates = sorted({code for code in stock_codes if stock_codes.count(code) > 1})
        raise ValueError(f"Signal {signal_name!r} has duplicate normalized stock codes: {duplicates[:5]}")
    matrix.columns = stock_codes
    return matrix.apply(pd.to_numeric, errors="coerce").sort_index()


def _date_like_ratio(values: Sequence[Any] | pd.Index) -> float:
    value_list = list(values)
    if not value_list:
        return 0.0
    candidate_values = [str(value).strip() for value in value_list if _looks_like_calendar_date(str(value).strip())]
    if not candidate_values:
        return 0.0
    parsed = pd.to_datetime(candidate_values, errors="coerce")
    valid_count = 0
    for value in parsed:
        if pd.isna(value):
            continue
        year = pd.Timestamp(value).year
        if 1990 <= year <= 2100:
            valid_count += 1
    return float(valid_count / len(value_list))


def _looks_like_calendar_date(text: str) -> bool:
    if len(text) < 8:
        return False
    year_text = text[:4]
    if not year_text.isdigit():
        return False
    year = int(year_text)
    if not 1990 <= year <= 2100:
        return False
    if len(text) >= 10 and text[4] in {"-", "/"}:
        return text[5:7].replace("-", "").replace("/", "").isdigit()
    return len(text) == 8 and text.isdigit()


def _filter_required_feature_rows(
    frame: pd.DataFrame,
    *,
    signal_feature_columns: Sequence[str],
    context_feature_columns: Sequence[str],
) -> pd.DataFrame:
    valid = np.ones(len(frame), dtype=bool)
    for column in signal_feature_columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float, copy=False)
        valid &= np.isfinite(values)
    for column in context_feature_columns:
        if column in {"industry", "board", "index_bucket"}:
            valid &= frame[column].notna().to_numpy(dtype=bool)
            continue
        if column == "size_decile":
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float, copy=False)
            valid &= np.isfinite(values) & (values >= 0.0)
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float, copy=False)
        valid &= np.isfinite(values)
    return frame.loc[valid].copy()


def _add_sample_weights_from_feature_universe(
    panel: pd.DataFrame,
    feature_date_counts: pd.Series,
) -> pd.DataFrame:
    frame = panel.copy()
    counts = feature_date_counts.copy()
    counts.index = pd.to_datetime(counts.index, errors="coerce").normalize()
    total_rows = float(counts.sum())
    total_dates = float(len(counts))
    if total_rows <= 0.0 or total_dates <= 0.0:
        raise ValueError("Feature universe date counts are empty.")
    date_counts = frame["date"].map(counts).astype(float)
    if date_counts.isna().any():
        missing_dates = sorted(frame.loc[date_counts.isna(), "date"].dt.date.astype(str).unique().tolist())
        raise ValueError(f"Panel contains dates missing from feature universe counts: {missing_dates}")
    frame["sample_weight"] = total_rows / (total_dates * date_counts)
    return frame


def _raw_return_distribution_by_date(
    panel: pd.DataFrame,
    *,
    target_col: str,
    target_rank_col: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for date, group in panel.groupby("date", sort=True):
        target = pd.to_numeric(group[target_col], errors="coerce")
        rank = pd.to_numeric(group[target_rank_col], errors="coerce")
        records.append(
            {
                "date": pd.Timestamp(date),
                "row_count": int(len(group)),
                f"{target_col}_mean": float(target.mean()),
                f"{target_col}_std": float(target.std(ddof=0)),
                f"{target_col}_min": float(target.min()),
                f"{target_col}_max": float(target.max()),
                f"{target_rank_col}_mean": float(rank.mean()),
                f"{target_rank_col}_min": float(rank.min()),
                f"{target_rank_col}_max": float(rank.max()),
            }
        )
    return pd.DataFrame(records)


def _feature_roles(
    alpha_raw_column: str,
    *,
    target_col: str,
    target_rank_col: str,
    include_liquidity_context_features: bool,
    include_kline_signal_features: bool,
    include_index_context_features: bool,
) -> dict[str, list[str]]:
    signal_columns = _signal_feature_columns(alpha_raw_column)
    liquidity_feature_columns = _liquidity_context_feature_columns() if include_liquidity_context_features else []
    liquidity_trace_columns = list(RAW_LIQUIDITY_COLUMNS) if include_liquidity_context_features else []
    liquidity_transform_columns = (
        [f"{column}__transform_input" for column in RAW_LIQUIDITY_COLUMNS]
        if include_liquidity_context_features
        else []
    )
    kline_feature_columns = _kline_context_feature_columns() if include_kline_signal_features else []
    kline_trace_columns = list(KLINE_RAW_COLUMNS) if include_kline_signal_features else []
    kline_transform_columns = (
        _kline_transform_input_columns()
        if include_kline_signal_features
        else []
    )
    index_categorical_columns = ["index_bucket"] if include_index_context_features else []
    index_continuous_columns = list(DEFAULT_INDEX_COLUMNS) if include_index_context_features else []
    roles = {
        "signal_features": signal_columns,
        "alpha_placeholder": signal_columns,
        "condition_categorical": ["industry", "board", *index_categorical_columns, "size_decile"],
        "condition_continuous": [
            *index_continuous_columns,
            "log_mcap_z",
            "mcap_rank",
            *liquidity_feature_columns,
            *kline_feature_columns,
        ],
        "traceability": [
            "date",
            "stock_code",
            "alpha_source",
            "market_cap",
            "log_mcap",
            alpha_raw_column,
            *liquidity_trace_columns,
            *kline_trace_columns,
        ],
        "excluded_from_model": [
            "date",
            "stock_code",
            target_col,
            target_rank_col,
            "sample_weight",
            "alpha_source",
            "market_cap",
            "log_mcap",
            alpha_raw_column,
            *liquidity_trace_columns,
            *liquidity_transform_columns,
            *kline_trace_columns,
            *kline_transform_columns,
        ],
        "targets": [target_col],
    }
    return roles


def _signal_feature_columns(alpha_raw_column: str) -> list[str]:
    alpha_feature_name = _alpha_source_from_raw_column(alpha_raw_column)
    return [f"{alpha_feature_name}_rank", f"{alpha_feature_name}_z"]


def _context_feature_columns(
    *,
    include_liquidity_context_features: bool,
    include_kline_signal_features: bool,
    include_index_context_features: bool,
) -> list[str]:
    return [
        "industry",
        "board",
        *(["index_bucket"] if include_index_context_features else []),
        "size_decile",
        *(list(DEFAULT_INDEX_COLUMNS) if include_index_context_features else []),
        "log_mcap_z",
        "mcap_rank",
        *(_liquidity_context_feature_columns() if include_liquidity_context_features else []),
        *(_kline_context_feature_columns() if include_kline_signal_features else []),
    ]


def _liquidity_context_feature_columns() -> list[str]:
    columns: list[str] = []
    for source_column in RAW_LIQUIDITY_COLUMNS:
        columns.extend([f"{source_column}_rank", f"{source_column}_z"])
    return columns


def _kline_context_feature_columns() -> list[str]:
    columns: list[str] = []
    for signal_name in KLINE_SIGNAL_TRANSFORMS:
        columns.extend([f"{signal_name}_rank", f"{signal_name}_z"])
    return columns


def _kline_transform_input_columns() -> list[str]:
    columns: list[str] = []
    for source_column in KLINE_RAW_COLUMNS:
        columns.extend([f"{source_column}__rank_input", f"{source_column}__z_input"])
    return columns


def _add_rank_z_columns(
    frame: pd.DataFrame,
    *,
    source_column: str,
    output_name: str,
    raw_transform: str,
    winsor_lower: float,
    winsor_upper: float,
    z_clip: float,
) -> None:
    numeric = pd.to_numeric(frame[source_column], errors="coerce").astype(float)
    rank_input, z_input, domain_valid = _signal_inputs_for_transform(
        numeric,
        raw_transform=raw_transform,
    )
    rank_input_column = f"{source_column}__rank_input"
    z_input_column = f"{source_column}__z_input"
    frame[rank_input_column] = rank_input.where(domain_valid)
    frame[z_input_column] = z_input.where(domain_valid)
    frame[f"{output_name}_rank"] = _daily_transform(frame, rank_input_column, centered_rank)
    frame[f"{output_name}_z"] = _daily_transform(
        frame,
        z_input_column,
        lambda values: _cross_sectional_robust_z(
            values,
            winsor_lower=winsor_lower,
            winsor_upper=winsor_upper,
            z_clip=z_clip,
        ),
    )


def _signal_inputs_for_transform(
    values: pd.Series,
    *,
    raw_transform: str,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    finite = pd.Series(np.isfinite(values.to_numpy(dtype=float, copy=False)), index=values.index)
    if raw_transform == "identity":
        return values, values, finite
    if raw_transform == "log1p_nonnegative":
        domain_valid = finite & values.ge(0.0)
        return values, np.log1p(values.where(domain_valid)), domain_valid
    if raw_transform == "log_positive":
        domain_valid = finite & values.gt(0.0)
        return values, np.log(values.where(domain_valid)), domain_valid
    raise ValueError(f"Unsupported raw_transform: {raw_transform!r}")


def _liquidity_transform_input(values: pd.Series, column: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    if column in {"amount_k", "turnover", "turnover20"}:
        return np.log1p(numeric.where(numeric >= 0.0))
    if column in {"logADV20", "logAmount20"}:
        return numeric
    raise ValueError(f"Unsupported liquidity context column: {column!r}")


def _daily_transform(frame: pd.DataFrame, column: str, function: Any) -> pd.Series:
    return frame.groupby("date", group_keys=False)[column].transform(function)


def _alpha_source_from_raw_column(alpha_raw_column: str) -> str:
    if alpha_raw_column.endswith("_raw"):
        return alpha_raw_column[: -len("_raw")]
    return alpha_raw_column


def _validate_target_names(target_col: str, target_rank_col: str) -> None:
    if target_col in {"y_resid_fwd", "y_rank_label"}:
        raise ValueError(f"target_col {target_col!r} would silently reuse residual-label semantics.")
    if target_rank_col in {"y_resid_fwd", "y_rank_label"}:
        raise ValueError(f"target_rank_col {target_rank_col!r} would silently reuse residual-label semantics.")
    if target_col == target_rank_col:
        raise ValueError("target_col and target_rank_col must be distinct.")


def _validate_winsor_config(winsor_lower: float, winsor_upper: float, z_clip: float) -> None:
    if not 0.0 <= winsor_lower < winsor_upper <= 1.0:
        raise ValueError("winsor_lower and winsor_upper must satisfy 0 <= lower < upper <= 1.")
    if z_clip <= EPS:
        raise ValueError("z_clip must be positive.")


def _parse_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", format="mixed").dt.normalize()


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        frame = pd.read_pickle(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path, dtype={"stock_code": "string"})
    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported table suffix: {path.suffix!r}")
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame at {path}, got {type(frame)!r}")
    return frame


def _records_sample(frame: pd.DataFrame, sample_size: int = 5) -> list[dict[str, Any]]:
    output = frame.head(sample_size).copy()
    for column in output.columns:
        if pd.api.types.is_datetime64_any_dtype(output[column]):
            output[column] = pd.to_datetime(output[column]).dt.date.astype(str)
    return output.to_dict("records")


if __name__ == "__main__":
    main()
