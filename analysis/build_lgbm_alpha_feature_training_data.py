from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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
        _label_distribution_by_date,
        _labels_to_long_frame,
        _normalize_stock_code,
        _path_for_summary,
        _records_with_iso_dates,
        _split_summary,
        _wide_frame_to_long,
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
        _label_distribution_by_date,
        _labels_to_long_frame,
        _normalize_stock_code,
        _path_for_summary,
        _records_with_iso_dates,
        _split_summary,
        _wide_frame_to_long,
        _write_csv_with_iso_dates,
        make_embargoed_dates,
        make_walk_forward_fold_assignments,
    )
    from neutralize_return_y import centered_rank


@dataclass(frozen=True)
class SignalTransformSpec:
    name: str
    raw_transform: str


DEFAULT_SIGNAL_SPECS = (
    SignalTransformSpec("factor_sss_dx_10", "identity"),
    SignalTransformSpec("amo", "log1p_nonnegative"),
    SignalTransformSpec("close", "log_positive"),
    SignalTransformSpec("vol", "log1p_nonnegative"),
)

DEFAULT_LIQUIDITY_CONTEXT_SPECS = (
    SignalTransformSpec("amount_k", "log1p_nonnegative"),
    SignalTransformSpec("turnover", "log1p_nonnegative"),
    SignalTransformSpec("logADV20", "identity"),
    SignalTransformSpec("logAmount20", "identity"),
    SignalTransformSpec("turnover20", "log1p_nonnegative"),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a long LGBM training panel with real alpha/kline rank-z features "
            "and context features. All local data paths are passed explicitly."
        )
    )
    parser.add_argument("--exposures", required=True, type=Path)
    parser.add_argument("--y-resid", required=True, type=Path)
    parser.add_argument("--y-rank-label", required=True, type=Path)
    parser.add_argument("--factor-sss-dx-10-value", required=True, type=Path)
    parser.add_argument("--amo-value", required=True, type=Path)
    parser.add_argument("--close-value", required=True, type=Path)
    parser.add_argument("--vol-value", required=True, type=Path)
    parser.add_argument("--output-panel", required=True, type=Path)
    parser.add_argument("--output-fold-assignments", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--diagnostics-dir", required=True, type=Path)
    parser.add_argument("--winsor-lower", default=0.01, type=float)
    parser.add_argument("--winsor-upper", default=0.99, type=float)
    parser.add_argument("--z-clip", default=5.0, type=float)
    parser.add_argument("--embargo-trading-days", default=2, type=int)
    args = parser.parse_args()

    exposures = _read_frame(args.exposures)
    y_resid = _read_frame(args.y_resid)
    y_rank_label = _read_frame(args.y_rank_label)
    signal_value_frames = {
        "factor_sss_dx_10": _read_frame(args.factor_sss_dx_10_value),
        "amo": _read_frame(args.amo_value),
        "close": _read_frame(args.close_value),
        "vol": _read_frame(args.vol_value),
    }

    summary = write_alpha_feature_training_data_artifacts(
        exposures=exposures,
        y_resid=y_resid,
        y_rank_label=y_rank_label,
        signal_value_frames=signal_value_frames,
        output_panel=args.output_panel,
        output_fold_assignments=args.output_fold_assignments,
        output_summary=args.output_summary,
        diagnostics_dir=args.diagnostics_dir,
        winsor_lower=args.winsor_lower,
        winsor_upper=args.winsor_upper,
        z_clip=args.z_clip,
        embargo_trading_days=args.embargo_trading_days,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_alpha_feature_training_panel(
    exposures: pd.DataFrame,
    y_resid: pd.DataFrame,
    y_rank_label: pd.DataFrame,
    *,
    signal_value_frames: Mapping[str, pd.DataFrame],
    signal_specs: Sequence[SignalTransformSpec] = DEFAULT_SIGNAL_SPECS,
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
    z_clip: float = 5.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _validate_winsor_config(winsor_lower, winsor_upper, z_clip)
    specs = tuple(signal_specs)
    _validate_signal_inputs(signal_value_frames, specs)

    context_frame = _prepare_context_exposures(exposures)
    feature_frame, signal_diagnostics = _add_signal_raw_columns(
        context_frame,
        signal_value_frames=signal_value_frames,
        signal_specs=specs,
    )
    feature_frame = _add_alpha_feature_columns(
        feature_frame,
        signal_specs=specs,
        winsor_lower=winsor_lower,
        winsor_upper=winsor_upper,
        z_clip=z_clip,
    )

    signal_feature_columns = _signal_feature_columns(specs)
    context_feature_columns = _context_feature_columns()
    feature_eligible_frame = _filter_required_feature_rows(
        feature_frame,
        signal_feature_columns=signal_feature_columns,
        context_feature_columns=context_feature_columns,
    )
    if feature_eligible_frame.empty:
        raise ValueError("No rows remain after filtering same-date context and signal features.")
    feature_date_counts = feature_eligible_frame.groupby("date")["stock_code"].size()

    label_frame = _labels_to_long_frame(y_resid, y_rank_label)
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
        raise ValueError("No rows remain after joining labels and filtering required features.")

    panel = _add_sample_weights_from_feature_universe(panel, feature_date_counts)
    panel = panel.sort_values(["date", "stock_code"]).reset_index(drop=True)

    diagnostics = {
        "summary": {
            "row_count": int(len(panel)),
            "date_count": int(panel["date"].nunique()),
            "stock_count": int(panel["stock_code"].nunique()),
            "date_min": _date_to_string(panel["date"].min()),
            "date_max": _date_to_string(panel["date"].max()),
            "context_row_count": int(len(context_frame)),
            "feature_row_count_before_label": feature_row_count,
            "feature_eligible_row_count_before_label": int(len(feature_eligible_frame)),
            "label_row_count": int(len(label_frame)),
            "dropped_missing_label_rows": dropped_missing_label_rows,
            "dropped_missing_feature_rows": dropped_missing_feature_rows,
            "feature_transform_universe": "same_date_context_and_signal_rows_before_label_join",
            "sample_weight_universe": "same_date_feature_eligible_rows_before_label_join",
            "signal_sources": [spec.name for spec in specs],
            "liquidity_context_sources": [spec.name for spec in DEFAULT_LIQUIDITY_CONTEXT_SPECS],
            "liquidity_context_transforms": {
                spec.name: spec.raw_transform for spec in DEFAULT_LIQUIDITY_CONTEXT_SPECS
            },
            "standardization": {
                "scope": "daily_cross_section",
                "winsor_lower": float(winsor_lower),
                "winsor_upper": float(winsor_upper),
                "z_clip": float(z_clip),
                "rank_convention": "centered_rank_in_roughly_minus_0p5_to_plus_0p5",
                "z_convention": "winsorized_robust_z_with_median_mad_scale",
            },
            "target_columns": ["y_resid_fwd", "y_rank_label"],
        },
        "signal_inputs": signal_diagnostics,
        "feature_calendar_dates": [
            pd.Timestamp(date).date().isoformat()
            for date in sorted(feature_eligible_frame["date"].unique())
        ],
        "feature_standardization": _feature_standardization_diagnostics(feature_eligible_frame),
        "label_distribution": _label_distribution_by_date(panel),
        "feature_roles": _feature_roles(specs),
    }
    return panel, diagnostics


def write_alpha_feature_training_data_artifacts(
    *,
    exposures: pd.DataFrame,
    y_resid: pd.DataFrame,
    y_rank_label: pd.DataFrame,
    signal_value_frames: Mapping[str, pd.DataFrame],
    output_panel: Path,
    output_fold_assignments: Path,
    output_summary: Path,
    diagnostics_dir: Path,
    signal_specs: Sequence[SignalTransformSpec] = DEFAULT_SIGNAL_SPECS,
    folds: Sequence[WalkForwardFold] = DEFAULT_WALK_FORWARD_FOLDS,
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
    z_clip: float = 5.0,
    embargo_trading_days: int = 2,
) -> dict[str, Any]:
    panel, diagnostics = build_alpha_feature_training_panel(
        exposures,
        y_resid,
        y_rank_label,
        signal_value_frames=signal_value_frames,
        signal_specs=signal_specs,
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
        diagnostics["label_distribution"],
        diagnostics_dir / "label_distribution_by_date.csv",
    )
    _write_csv_with_iso_dates(split_summary, diagnostics_dir / "split_summary.csv")
    _write_csv_with_iso_dates(embargoed_dates, diagnostics_dir / "embargoed_dates.csv")

    summary = {
        "metadata": {
            "signal_stage": "real_alpha_kline_feature_panel",
            "alpha_source": "factor_sss_dx_10",
            "alpha_is_real": True,
            "production_eligible": False,
            "model_form": "p = g(alpha_rank_z_and_kline_rank_z, context_with_liquidity_turnover)",
            "condition_set": "industry_board_index_size_liquidity_turnover_v1",
            "feature_asof": "same_date_eod_for_close_amount_volume_liquidity_turnover_inputs",
            "label_contract": "y_resid_fwd and y_rank_label must already be forward-looking labels aligned to signal dates",
            "liquidity_feature_role": "condition_continuous",
        },
        "panel": diagnostics["summary"],
        "signal_inputs": diagnostics["signal_inputs"],
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
            "embargoed_dates": _path_for_summary(diagnostics_dir / "embargoed_dates.csv"),
        },
        "notes": [
            "Daily rank and robust-z features are computed before joining forward labels.",
            "Liquidity and turnover rank/z features are context features, so context-only predictions keep them fixed.",
            "The legacy feature role name alpha_placeholder is used for all signal columns so existing context-only training zeroes them together.",
            "Industry, board, index membership, size, liquidity, and turnover features remain visible to context-only predictions.",
            "Using same-day close/amount/volume/liquidity/turnover assumes an end-of-day signal; do not use this panel for decisions before those fields are observable.",
            "Rolling liquidity inputs must be trailing windows ending no later than the signal date; use shifted inputs for pre-open or intraday decisions.",
        ],
    }
    output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _prepare_context_exposures(exposures: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "date",
        "stock_code",
        "industry",
        "board",
        *DEFAULT_INDEX_COLUMNS,
        "market_cap",
        *[spec.name for spec in DEFAULT_LIQUIDITY_CONTEXT_SPECS],
    }
    missing = sorted(required_columns.difference(exposures.columns))
    if missing:
        raise ValueError(f"Exposures are missing required context columns: {missing}")

    frame = exposures.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["stock_code"] = frame["stock_code"].map(_normalize_stock_code)
    frame = frame[frame["date"].notna() & frame["stock_code"].ne("")].copy()
    if frame.duplicated(["date", "stock_code"]).any():
        duplicate = frame.loc[frame.duplicated(["date", "stock_code"], keep=False), ["date", "stock_code"]]
        sample = duplicate.head(5).to_dict("records")
        raise ValueError(f"Exposures contain duplicate date, stock_code keys: {sample}")

    frame["industry"] = frame["industry"].fillna("unknown_industry").astype(str)
    frame["board"] = frame["board"].fillna("UNKNOWN_BOARD").astype(str)
    for column in DEFAULT_INDEX_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["market_cap"] = pd.to_numeric(frame["market_cap"], errors="coerce")
    for spec in DEFAULT_LIQUIDITY_CONTEXT_SPECS:
        frame[spec.name] = pd.to_numeric(frame[spec.name], errors="coerce")
    return frame.sort_values(["date", "stock_code"]).reset_index(drop=True)


def _add_signal_raw_columns(
    context_frame: pd.DataFrame,
    *,
    signal_value_frames: Mapping[str, pd.DataFrame],
    signal_specs: Sequence[SignalTransformSpec],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    frame = context_frame.copy()
    diagnostics: dict[str, dict[str, Any]] = {}
    for spec in signal_specs:
        raw_column = _raw_column(spec.name)
        long = _signal_wide_frame_to_long(signal_value_frames[spec.name], spec.name, raw_column)
        before_merge = int(len(frame))
        frame = frame.merge(long, on=["date", "stock_code"], how="left", validate="one_to_one")
        raw_values = pd.to_numeric(frame[raw_column], errors="coerce")
        context_valid = _context_valid_mask(frame)
        _, _, domain_valid = _signal_inputs_for_transform(
            raw_values,
            raw_transform=spec.raw_transform,
        )
        diagnostics[spec.name] = {
            "raw_column": raw_column,
            "raw_transform": spec.raw_transform,
            "input_nonmissing_count": int(long[raw_column].notna().sum()),
            "context_row_count_before_merge": before_merge,
            "matched_nonmissing_count": int(np.isfinite(raw_values.to_numpy(dtype=float, copy=False)).sum()),
            "missing_after_context_merge": int(raw_values.isna().sum()),
            "invalid_domain_count": int((context_valid & raw_values.notna() & ~domain_valid).sum()),
        }
    return frame, diagnostics


def _add_alpha_feature_columns(
    feature_frame: pd.DataFrame,
    *,
    signal_specs: Sequence[SignalTransformSpec],
    winsor_lower: float,
    winsor_upper: float,
    z_clip: float,
) -> pd.DataFrame:
    frame = feature_frame.copy()
    frame["alpha_source"] = "factor_sss_dx_10"

    context_valid = _context_valid_mask(frame)
    market_cap = frame["market_cap"].where(context_valid)
    frame["log_mcap"] = np.log1p(market_cap.to_numpy(dtype=float))
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
    frame["mcap_rank"] = _daily_transform(frame, "market_cap", lambda values: centered_rank(values.where(context_valid.loc[values.index])))
    frame["size_decile"] = _daily_transform(
        frame,
        "market_cap",
        lambda values: _decile_codes(values.where(context_valid.loc[values.index])),
    ).astype("int8")
    frame["index_bucket"] = [
        _index_bucket(row) if is_valid else "UNKNOWN_INDEX"
        for row, is_valid in zip(
            frame[list(DEFAULT_INDEX_COLUMNS)].itertuples(index=False, name=None),
            context_valid.to_numpy(dtype=bool),
            strict=True,
        )
    ]

    for spec in DEFAULT_LIQUIDITY_CONTEXT_SPECS:
        _add_rank_z_columns(
            frame,
            source_column=spec.name,
            output_name=spec.name,
            context_valid=context_valid,
            raw_transform=spec.raw_transform,
            winsor_lower=winsor_lower,
            winsor_upper=winsor_upper,
            z_clip=z_clip,
        )

    for spec in signal_specs:
        raw_column = _raw_column(spec.name)
        _add_rank_z_columns(
            frame,
            source_column=raw_column,
            output_name=spec.name,
            context_valid=context_valid,
            raw_transform=spec.raw_transform,
            winsor_lower=winsor_lower,
            winsor_upper=winsor_upper,
            z_clip=z_clip,
        )
    return frame


def _add_rank_z_columns(
    frame: pd.DataFrame,
    *,
    source_column: str,
    output_name: str,
    context_valid: pd.Series,
    raw_transform: str,
    winsor_lower: float,
    winsor_upper: float,
    z_clip: float,
) -> None:
    numeric = pd.to_numeric(frame[source_column], errors="coerce")
    rank_input, z_input, domain_valid = _signal_inputs_for_transform(
        numeric,
        raw_transform=raw_transform,
    )
    valid_for_feature = context_valid & domain_valid
    rank_input_column = _rank_input_column(source_column)
    z_input_column = _z_input_column(source_column)
    frame[rank_input_column] = rank_input.where(valid_for_feature)
    frame[z_input_column] = z_input.where(valid_for_feature)
    frame[_rank_column(output_name)] = _daily_transform(frame, rank_input_column, centered_rank)
    frame[_z_column(output_name)] = _daily_transform(
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


def _signal_wide_frame_to_long(frame: pd.DataFrame, signal_name: str, raw_column: str) -> pd.DataFrame:
    matrix = _normalize_signal_matrix(frame, signal_name)
    long = _wide_frame_to_long(matrix, raw_column)
    if long.duplicated(["date", "stock_code"]).any():
        duplicates = long.loc[long.duplicated(["date", "stock_code"], keep=False), ["date", "stock_code"]]
        sample = duplicates.head(5).to_dict("records")
        raise ValueError(f"Signal {signal_name!r} contains duplicate date, stock_code keys: {sample}")
    long[raw_column] = pd.to_numeric(long[raw_column], errors="coerce")
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

    matrix.index = pd.DatetimeIndex(pd.to_datetime(list(matrix.index), errors="coerce")).normalize()
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
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    return matrix.sort_index()


def _date_like_ratio(values: Sequence[Any] | pd.Index) -> float:
    value_list = list(values)
    if not value_list:
        return 0.0
    candidate_values: list[str] = []
    for value in value_list:
        text = str(value).strip()
        if _looks_like_calendar_date(text):
            candidate_values.append(text)
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


def _context_valid_mask(frame: pd.DataFrame) -> pd.Series:
    valid = pd.Series(True, index=frame.index)
    market_cap = pd.to_numeric(frame["market_cap"], errors="coerce")
    valid &= np.isfinite(market_cap.to_numpy(dtype=float, copy=False)) & market_cap.gt(0.0)
    for column in DEFAULT_INDEX_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        valid &= np.isfinite(values.to_numpy(dtype=float, copy=False))
    return valid


def _daily_transform(frame: pd.DataFrame, column: str, function: Callable[[pd.Series], pd.Series]) -> pd.Series:
    return frame.groupby("date", group_keys=False)[column].transform(function)


def _feature_roles(signal_specs: Sequence[SignalTransformSpec]) -> dict[str, list[str]]:
    raw_columns = [_raw_column(spec.name) for spec in signal_specs]
    liquidity_columns = [spec.name for spec in DEFAULT_LIQUIDITY_CONTEXT_SPECS]
    input_columns = [*raw_columns, *liquidity_columns]
    signal_feature_columns = _signal_feature_columns(signal_specs)
    return {
        "alpha_placeholder": signal_feature_columns,
        "condition_categorical": ["industry", "board", "index_bucket", "size_decile"],
        "condition_continuous": [
            "is_csi300",
            "is_csi500",
            "is_csi1000",
            "is_csi2000",
            "log_mcap_z",
            "mcap_rank",
            *_liquidity_context_feature_columns(),
        ],
        "traceability": [
            "date",
            "stock_code",
            "alpha_source",
            "market_cap",
            "log_mcap",
            *liquidity_columns,
            *raw_columns,
        ],
        "excluded_from_model": [
            "date",
            "stock_code",
            "y_resid_fwd",
            "y_rank_label",
            "sample_weight",
            "alpha_source",
            "market_cap",
            "log_mcap",
            *liquidity_columns,
            *raw_columns,
            *[_rank_input_column(column) for column in input_columns],
            *[_z_input_column(column) for column in input_columns],
        ],
        "targets": ["y_resid_fwd", "y_rank_label"],
    }


def _signal_feature_columns(signal_specs: Sequence[SignalTransformSpec]) -> list[str]:
    columns: list[str] = []
    for spec in signal_specs:
        columns.extend([_rank_column(spec.name), _z_column(spec.name)])
    return columns


def _context_feature_columns() -> list[str]:
    return [
        "industry",
        "board",
        "index_bucket",
        "size_decile",
        "is_csi300",
        "is_csi500",
        "is_csi1000",
        "is_csi2000",
        "log_mcap_z",
        "mcap_rank",
        *_liquidity_context_feature_columns(),
    ]


def _liquidity_context_feature_columns() -> list[str]:
    columns: list[str] = []
    for spec in DEFAULT_LIQUIDITY_CONTEXT_SPECS:
        columns.extend([_rank_column(spec.name), _z_column(spec.name)])
    return columns


def _raw_column(signal_name: str) -> str:
    return f"{signal_name}_raw"


def _rank_column(signal_name: str) -> str:
    return f"{signal_name}_rank"


def _z_column(signal_name: str) -> str:
    return f"{signal_name}_z"


def _rank_input_column(raw_column: str) -> str:
    return f"{raw_column}__rank_input"


def _z_input_column(raw_column: str) -> str:
    return f"{raw_column}__z_input"


def _validate_signal_inputs(
    signal_value_frames: Mapping[str, pd.DataFrame],
    signal_specs: Sequence[SignalTransformSpec],
) -> None:
    required = {spec.name for spec in signal_specs}
    missing = sorted(required.difference(signal_value_frames.keys()))
    if missing:
        raise ValueError(f"Missing required signal value frames: {missing}")


def _validate_winsor_config(winsor_lower: float, winsor_upper: float, z_clip: float) -> None:
    if not 0.0 <= winsor_lower < winsor_upper <= 1.0:
        raise ValueError("winsor_lower and winsor_upper must satisfy 0 <= lower < upper <= 1.")
    if z_clip <= EPS:
        raise ValueError("z_clip must be positive.")


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


if __name__ == "__main__":
    main()
