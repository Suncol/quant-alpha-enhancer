from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

try:
    from analysis.neutralize_return_y import centered_rank
except ModuleNotFoundError:  # Allows direct execution via python analysis/script.py.
    from neutralize_return_y import centered_rank


DEFAULT_ALPHA_PLACEHOLDER_SOURCE_COLUMNS = ("turnover20", "logADV20")
DEFAULT_INDEX_COLUMNS = ("is_csi300", "is_csi500", "is_csi1000", "is_csi2000")
EPS = 1e-12


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    test_start: str
    test_end: str


DEFAULT_WALK_FORWARD_FOLDS = (
    WalkForwardFold(
        fold_id=1,
        train_start="2023-08-31",
        train_end="2024-09-30",
        valid_start="2024-10-01",
        valid_end="2024-12-31",
        test_start="2025-01-01",
        test_end="2025-06-30",
    ),
    WalkForwardFold(
        fold_id=2,
        train_start="2023-08-31",
        train_end="2025-03-31",
        valid_start="2025-04-01",
        valid_end="2025-06-30",
        test_start="2025-07-01",
        test_end="2025-12-31",
    ),
    WalkForwardFold(
        fold_id=3,
        train_start="2023-08-31",
        train_end="2025-09-30",
        valid_start="2025-10-01",
        valid_end="2025-12-31",
        test_start="2026-01-01",
        test_end="2026-06-01",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a long LGBM training panel for p = g(alpha_placeholder, c). "
            "The script only prepares data; it does not train LightGBM."
        )
    )
    parser.add_argument("--exposures", required=True, type=Path)
    parser.add_argument("--y-resid", required=True, type=Path)
    parser.add_argument("--y-rank-label", required=True, type=Path)
    parser.add_argument("--output-panel", required=True, type=Path)
    parser.add_argument("--output-fold-assignments", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--diagnostics-dir", required=True, type=Path)
    parser.add_argument(
        "--alpha-placeholder-cols",
        nargs="+",
        default=list(DEFAULT_ALPHA_PLACEHOLDER_SOURCE_COLUMNS),
        help="Continuous columns treated as temporary alpha placeholders.",
    )
    parser.add_argument("--winsor-lower", default=0.01, type=float)
    parser.add_argument("--winsor-upper", default=0.99, type=float)
    parser.add_argument("--z-clip", default=5.0, type=float)
    parser.add_argument(
        "--embargo-trading-days",
        default=2,
        type=int,
        help=(
            "Number of tail signal dates to remove from train and valid splits "
            "before the next split."
        ),
    )
    args = parser.parse_args()

    exposures = pd.read_pickle(args.exposures)
    y_resid = pd.read_pickle(args.y_resid)
    y_rank_label = pd.read_pickle(args.y_rank_label)
    if not isinstance(exposures, pd.DataFrame):
        raise TypeError(f"Expected exposures DataFrame, got {type(exposures)!r}")
    if not isinstance(y_resid, pd.DataFrame):
        raise TypeError(f"Expected y_resid DataFrame, got {type(y_resid)!r}")
    if not isinstance(y_rank_label, pd.DataFrame):
        raise TypeError(f"Expected y_rank_label DataFrame, got {type(y_rank_label)!r}")

    summary = write_placeholder_training_data_artifacts(
        exposures=exposures,
        y_resid=y_resid,
        y_rank_label=y_rank_label,
        output_panel=args.output_panel,
        output_fold_assignments=args.output_fold_assignments,
        output_summary=args.output_summary,
        diagnostics_dir=args.diagnostics_dir,
        alpha_placeholder_source_columns=tuple(args.alpha_placeholder_cols),
        winsor_lower=args.winsor_lower,
        winsor_upper=args.winsor_upper,
        z_clip=args.z_clip,
        embargo_trading_days=args.embargo_trading_days,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_placeholder_training_panel(
    exposures: pd.DataFrame,
    y_resid: pd.DataFrame,
    y_rank_label: pd.DataFrame,
    *,
    alpha_placeholder_source_columns: Sequence[str] = DEFAULT_ALPHA_PLACEHOLDER_SOURCE_COLUMNS,
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
    z_clip: float = 5.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    exposure_frame = _prepare_exposures(exposures)
    source_columns = tuple(alpha_placeholder_source_columns)
    _validate_source_columns(exposure_frame, source_columns)

    label_frame = _labels_to_long_frame(y_resid, y_rank_label)
    exposure_row_count = int(len(exposure_frame))
    panel = exposure_frame.merge(
        label_frame,
        on=["date", "stock_code"],
        how="inner",
        validate="one_to_one",
    )
    dropped_missing_label_rows = int(exposure_row_count - len(panel))

    before_feature_filter = int(len(panel))
    panel = _filter_required_feature_rows(panel, source_columns)
    dropped_missing_feature_rows = int(before_feature_filter - len(panel))
    if panel.empty:
        raise ValueError("No rows remain after joining labels and filtering required features.")

    panel = _add_feature_columns(
        panel,
        alpha_placeholder_source_columns=source_columns,
        winsor_lower=winsor_lower,
        winsor_upper=winsor_upper,
        z_clip=z_clip,
    )
    panel = _add_sample_weights(panel)
    panel = panel.sort_values(["date", "stock_code"]).reset_index(drop=True)

    diagnostics = {
        "summary": {
            "row_count": int(len(panel)),
            "date_count": int(panel["date"].nunique()),
            "stock_count": int(panel["stock_code"].nunique()),
            "date_min": _date_to_string(panel["date"].min()),
            "date_max": _date_to_string(panel["date"].max()),
            "exposure_row_count": exposure_row_count,
            "label_row_count": int(len(label_frame)),
            "dropped_missing_label_rows": dropped_missing_label_rows,
            "dropped_missing_feature_rows": dropped_missing_feature_rows,
            "alpha_placeholder_source_columns": list(source_columns),
            "standardization": {
                "scope": "daily_cross_section",
                "winsor_lower": float(winsor_lower),
                "winsor_upper": float(winsor_upper),
                "z_clip": float(z_clip),
                "rank_convention": "centered_rank_in_roughly_minus_0p5_to_plus_0p5",
            },
            "target_columns": ["y_resid_fwd", "y_rank_label"],
        },
        "feature_standardization": _feature_standardization_diagnostics(panel),
        "label_distribution": _label_distribution_by_date(panel),
        "feature_roles": _feature_roles(source_columns),
    }
    return panel, diagnostics


def make_walk_forward_fold_assignments(
    available_dates: Sequence[Any],
    folds: Sequence[WalkForwardFold] = DEFAULT_WALK_FORWARD_FOLDS,
    *,
    embargo_trading_days: int = 0,
) -> pd.DataFrame:
    dates = _normalized_unique_dates(available_dates)
    embargoed_dates = make_embargoed_dates(
        dates,
        folds,
        embargo_trading_days=embargo_trading_days,
    )
    embargoed_keys = {
        (int(row.fold_id), pd.Timestamp(row.date).normalize())
        for row in embargoed_dates.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for fold in folds:
        ranges = [
            ("train", pd.Timestamp(fold.train_start), pd.Timestamp(fold.train_end)),
            ("valid", pd.Timestamp(fold.valid_start), pd.Timestamp(fold.valid_end)),
            ("test", pd.Timestamp(fold.test_start), pd.Timestamp(fold.test_end)),
        ]
        for date in dates:
            matching = [split for split, start, end in ranges if start <= date <= end]
            if len(matching) > 1:
                raise ValueError(
                    f"Fold {fold.fold_id} assigns {date.date().isoformat()} "
                    f"to multiple splits: {matching}"
                )
            if matching:
                if (int(fold.fold_id), date) in embargoed_keys:
                    continue
                rows.append(
                    {
                        "fold_id": int(fold.fold_id),
                        "date": date,
                        "split": matching[0],
                    }
                )
    assignments = pd.DataFrame(rows, columns=["fold_id", "date", "split"])
    if assignments.duplicated(["fold_id", "date"]).any():
        raise ValueError("Fold assignments contain duplicate (fold_id, date) keys.")
    return assignments.sort_values(["fold_id", "date"]).reset_index(drop=True)


def make_embargoed_dates(
    available_dates: Sequence[Any],
    folds: Sequence[WalkForwardFold] = DEFAULT_WALK_FORWARD_FOLDS,
    *,
    embargo_trading_days: int,
) -> pd.DataFrame:
    if embargo_trading_days < 0:
        raise ValueError("embargo_trading_days must be non-negative.")
    columns = ["fold_id", "boundary", "source_split", "next_split", "date"]
    if embargo_trading_days == 0:
        return pd.DataFrame(columns=columns)

    dates = _normalized_unique_dates(available_dates)
    rows: list[dict[str, Any]] = []
    for fold in folds:
        boundaries = [
            (
                "train_to_valid",
                "train",
                "valid",
                pd.Timestamp(fold.train_start),
                pd.Timestamp(fold.train_end),
            ),
            (
                "valid_to_test",
                "valid",
                "test",
                pd.Timestamp(fold.valid_start),
                pd.Timestamp(fold.valid_end),
            ),
        ]
        for boundary, source_split, next_split, start, end in boundaries:
            source_dates = dates[(dates >= start) & (dates <= end)]
            for date in source_dates[-embargo_trading_days:]:
                rows.append(
                    {
                        "fold_id": int(fold.fold_id),
                        "boundary": boundary,
                        "source_split": source_split,
                        "next_split": next_split,
                        "date": date,
                    }
                )
    return pd.DataFrame(rows, columns=columns).sort_values(["fold_id", "date"]).reset_index(drop=True)


def write_placeholder_training_data_artifacts(
    *,
    exposures: pd.DataFrame,
    y_resid: pd.DataFrame,
    y_rank_label: pd.DataFrame,
    output_panel: Path,
    output_fold_assignments: Path,
    output_summary: Path,
    diagnostics_dir: Path,
    alpha_placeholder_source_columns: Sequence[str] = DEFAULT_ALPHA_PLACEHOLDER_SOURCE_COLUMNS,
    folds: Sequence[WalkForwardFold] = DEFAULT_WALK_FORWARD_FOLDS,
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
    z_clip: float = 5.0,
    embargo_trading_days: int = 2,
) -> dict[str, Any]:
    panel, diagnostics = build_placeholder_training_panel(
        exposures,
        y_resid,
        y_rank_label,
        alpha_placeholder_source_columns=alpha_placeholder_source_columns,
        winsor_lower=winsor_lower,
        winsor_upper=winsor_upper,
        z_clip=z_clip,
    )
    fold_assignments = make_walk_forward_fold_assignments(
        panel["date"].unique(),
        folds,
        embargo_trading_days=embargo_trading_days,
    )
    embargoed_dates = make_embargoed_dates(
        panel["date"].unique(),
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
            "signal_stage": "placeholder_alpha_probe",
            "alpha_source": "placeholder_liquidity",
            "alpha_is_real": False,
            "production_eligible": False,
            "model_form": "p = g(alpha_placeholder, c)",
            "condition_set": "industry_board_index_size_v1",
        },
        "panel": diagnostics["summary"],
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
            "This artifact prepares p = g(alpha_placeholder, c) training data only.",
            "The alpha placeholders are liquidity variables for pipeline validation, not real alpha.",
            "Fold assignments remove train and valid tail signal dates according to embargo_trading_days.",
            "The base panel is unchanged by embargo; only fold assignment eligibility changes.",
        ],
    }
    output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _prepare_exposures(exposures: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "date",
        "stock_code",
        "industry",
        "board",
        *DEFAULT_INDEX_COLUMNS,
        "market_cap",
        "amount_k",
        "turnover",
        "logADV20",
        "logAmount20",
        "turnover20",
    }
    missing = sorted(required_columns.difference(exposures.columns))
    if missing:
        raise ValueError(f"Exposures are missing required columns: {missing}")

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
    for column in ["market_cap", "amount_k", "turnover", "logADV20", "logAmount20", "turnover20"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _validate_source_columns(frame: pd.DataFrame, source_columns: Sequence[str]) -> None:
    if not source_columns:
        raise ValueError("At least one alpha placeholder source column is required.")
    missing = sorted(set(source_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Alpha placeholder source columns are missing: {missing}")


def _labels_to_long_frame(y_resid: pd.DataFrame, y_rank_label: pd.DataFrame) -> pd.DataFrame:
    if y_resid.shape != y_rank_label.shape:
        raise ValueError(
            f"Label matrices must have identical shape, got {y_resid.shape} and {y_rank_label.shape}."
        )

    resid = _prepare_label_matrix(y_resid)
    rank = _prepare_label_matrix(y_rank_label)
    resid_long = _wide_frame_to_long(resid, "y_resid_fwd")
    rank_long = _wide_frame_to_long(rank, "y_rank_label")
    labels = resid_long.merge(rank_long, on=["date", "stock_code"], how="inner", validate="one_to_one")
    for column in ["y_resid_fwd", "y_rank_label"]:
        labels[column] = pd.to_numeric(labels[column], errors="coerce")
    labels = labels[
        np.isfinite(labels["y_resid_fwd"].to_numpy(dtype=float))
        & np.isfinite(labels["y_rank_label"].to_numpy(dtype=float))
    ].copy()
    return labels.sort_values(["date", "stock_code"]).reset_index(drop=True)


def _prepare_label_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output.index = pd.DatetimeIndex(pd.to_datetime(output.index, errors="coerce")).normalize()
    output.columns = [_normalize_stock_code(column) for column in output.columns]
    output = output.loc[output.index.notna()]
    if output.columns.duplicated().any():
        duplicates = sorted(set(column for column in output.columns[output.columns.duplicated()]))
        raise ValueError(f"Label matrix contains duplicate normalized stock_code columns: {duplicates}")
    return output


def _wide_frame_to_long(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
    series = frame.rename_axis(index="date", columns="stock_code").stack()
    long = series.rename(value_name).reset_index()
    long["date"] = pd.to_datetime(long["date"], errors="coerce").dt.normalize()
    long["stock_code"] = long["stock_code"].map(_normalize_stock_code)
    return long[long[value_name].notna()].reset_index(drop=True)


def _filter_required_feature_rows(
    panel: pd.DataFrame,
    source_columns: Sequence[str],
) -> pd.DataFrame:
    valid = np.ones(len(panel), dtype=bool)
    market_cap = panel["market_cap"].to_numpy(dtype=float, copy=False)
    valid &= np.isfinite(market_cap) & (market_cap > 0.0)
    for column in DEFAULT_INDEX_COLUMNS:
        values = panel[column].to_numpy(dtype=float, copy=False)
        valid &= np.isfinite(values)
    for column in source_columns:
        values = panel[column].to_numpy(dtype=float, copy=False)
        valid &= np.isfinite(values)
    return panel.loc[valid].copy()


def _add_feature_columns(
    panel: pd.DataFrame,
    *,
    alpha_placeholder_source_columns: Sequence[str],
    winsor_lower: float,
    winsor_upper: float,
    z_clip: float,
) -> pd.DataFrame:
    frame = panel.copy()
    frame["alpha_source"] = "placeholder_liquidity"
    frame["log_mcap"] = np.log1p(frame["market_cap"].to_numpy(dtype=float))
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
    frame["index_bucket"] = [
        _index_bucket(row)
        for row in frame[list(DEFAULT_INDEX_COLUMNS)].itertuples(index=False, name=None)
    ]

    for source_column in alpha_placeholder_source_columns:
        prefix = f"alpha_placeholder_{source_column}"
        frame[f"{prefix}_raw"] = frame[source_column].to_numpy(dtype=float)
        frame[f"{prefix}_z"] = _daily_transform(
            frame,
            source_column,
            lambda values: _cross_sectional_robust_z(
                values,
                winsor_lower=winsor_lower,
                winsor_upper=winsor_upper,
                z_clip=z_clip,
            ),
        )
        frame[f"{prefix}_rank"] = _daily_transform(frame, source_column, centered_rank)

    return frame


def _daily_transform(
    frame: pd.DataFrame,
    column: str,
    function: Any,
) -> pd.Series:
    return frame.groupby("date", group_keys=False)[column].transform(function)


def _cross_sectional_robust_z(
    values: pd.Series,
    *,
    winsor_lower: float,
    winsor_upper: float,
    z_clip: float,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    finite_mask = np.isfinite(numeric.to_numpy(dtype=float, copy=False))
    if not finite_mask.any():
        return result
    finite = numeric.loc[finite_mask]
    lower = float(finite.quantile(winsor_lower))
    upper = float(finite.quantile(winsor_upper))
    clipped = finite.clip(lower=lower, upper=upper)
    median = float(clipped.median())
    mad = float((clipped - median).abs().median())
    scale = 1.4826 * mad
    if scale <= EPS or not np.isfinite(scale):
        std = float(clipped.std(ddof=0))
        scale = std if std > EPS and np.isfinite(std) else 1.0
    z = ((clipped - median) / scale).clip(lower=-z_clip, upper=z_clip)
    result.loc[finite.index] = z.to_numpy(dtype=float)
    return result


def _decile_codes(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(-1, index=values.index, dtype="int8")
    finite = np.isfinite(numeric.to_numpy(dtype=float, copy=False))
    if not finite.any():
        return result
    ranks = numeric.loc[finite].rank(method="first", pct=True)
    decile = np.floor((ranks.to_numpy(dtype=float) - EPS) * 10.0).astype(int)
    decile = np.clip(decile, 0, 9).astype("int8")
    result.loc[finite] = decile
    return result


def _index_bucket(values: tuple[Any, ...]) -> str:
    for column, value in zip(DEFAULT_INDEX_COLUMNS, values, strict=True):
        if float(value) == 1.0:
            return column.replace("is_", "").upper()
    return "NON_INDEX"


def _add_sample_weights(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    n_rows = float(len(frame))
    n_dates = float(frame["date"].nunique())
    date_counts = frame.groupby("date")["stock_code"].transform("size").astype(float)
    frame["sample_weight"] = n_rows / (n_dates * date_counts)
    return frame


def _feature_roles(source_columns: Sequence[str]) -> dict[str, list[str]]:
    alpha_columns: list[str] = []
    for source_column in source_columns:
        prefix = f"alpha_placeholder_{source_column}"
        alpha_columns.extend([f"{prefix}_z", f"{prefix}_rank"])
    return {
        "alpha_placeholder": alpha_columns,
        "condition_categorical": ["industry", "board", "index_bucket", "size_decile"],
        "condition_continuous": [
            "is_csi300",
            "is_csi500",
            "is_csi1000",
            "is_csi2000",
            "log_mcap_z",
            "mcap_rank",
        ],
        "traceability": [
            "date",
            "stock_code",
            "alpha_source",
            "market_cap",
            "amount_k",
            "turnover",
            "logADV20",
            "logAmount20",
            "turnover20",
        ],
        "excluded_from_model": [
            "date",
            "stock_code",
            "y_resid_fwd",
            "y_rank_label",
            "sample_weight",
            "market_cap",
            "amount_k",
            "turnover",
            "logADV20",
            "logAmount20",
            "turnover20",
        ],
        "targets": ["y_resid_fwd", "y_rank_label"],
    }


def _feature_standardization_diagnostics(panel: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        column
        for column in panel.columns
        if column.endswith("_z") or column.endswith("_rank")
    ]
    records: list[dict[str, Any]] = []
    for (date, feature), values in (
        panel.set_index("date")[feature_columns]
        .stack()
        .rename("value")
        .reset_index()
        .rename(columns={"level_1": "feature"})
        .groupby(["date", "feature"])["value"]
    ):
        numeric = pd.to_numeric(values, errors="coerce")
        finite = numeric[np.isfinite(numeric.to_numpy(dtype=float, copy=False))]
        records.append(
            {
                "date": pd.Timestamp(date),
                "feature": str(feature),
                "count": int(len(finite)),
                "missing": int(len(values) - len(finite)),
                "mean": float(finite.mean()) if len(finite) else np.nan,
                "median": float(finite.median()) if len(finite) else np.nan,
                "std": float(finite.std(ddof=0)) if len(finite) else np.nan,
                "min": float(finite.min()) if len(finite) else np.nan,
                "max": float(finite.max()) if len(finite) else np.nan,
            }
        )
    return pd.DataFrame(records)


def _label_distribution_by_date(panel: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for date, group in panel.groupby("date", sort=True):
        y_resid = group["y_resid_fwd"].astype(float)
        y_rank = group["y_rank_label"].astype(float)
        records.append(
            {
                "date": pd.Timestamp(date),
                "row_count": int(len(group)),
                "y_resid_mean": float(y_resid.mean()),
                "y_resid_std": float(y_resid.std(ddof=0)),
                "y_resid_min": float(y_resid.min()),
                "y_resid_max": float(y_resid.max()),
                "y_rank_mean": float(y_rank.mean()),
                "y_rank_min": float(y_rank.min()),
                "y_rank_max": float(y_rank.max()),
            }
        )
    return pd.DataFrame(records)


def _split_summary(panel: pd.DataFrame, fold_assignments: pd.DataFrame) -> pd.DataFrame:
    date_counts = (
        panel.groupby("date")
        .agg(
            row_count=("stock_code", "size"),
            stock_count=("stock_code", "nunique"),
            sample_weight_sum=("sample_weight", "sum"),
        )
        .reset_index()
    )
    joined = fold_assignments.merge(date_counts, on="date", how="left", validate="many_to_one")
    summary = (
        joined.groupby(["fold_id", "split"], sort=True)
        .agg(
            start_date=("date", "min"),
            end_date=("date", "max"),
            date_count=("date", "nunique"),
            row_count=("row_count", "sum"),
            max_daily_stock_count=("stock_count", "max"),
            sample_weight_sum=("sample_weight_sum", "sum"),
        )
        .reset_index()
    )
    for column in ["date_count", "row_count", "max_daily_stock_count"]:
        summary[column] = summary[column].astype(int)
    return summary


def _write_csv_with_iso_dates(frame: pd.DataFrame, path: Path) -> None:
    output = frame.copy()
    for column in output.columns:
        if is_datetime64_any_dtype(output[column]):
            output[column] = pd.to_datetime(output[column]).dt.date.astype(str)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, encoding="utf-8-sig")


def _records_with_iso_dates(frame: pd.DataFrame) -> list[dict[str, Any]]:
    output = frame.copy()
    for column in output.columns:
        if is_datetime64_any_dtype(output[column]):
            output[column] = pd.to_datetime(output[column]).dt.date.astype(str)
    return output.to_dict("records")


def _normalized_unique_dates(values: Sequence[Any]) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(list(values), errors="coerce")).normalize()
    return pd.DatetimeIndex(sorted(dates.dropna().unique()))


def _normalize_stock_code(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if "." in text:
        left, right = text.split(".", 1)
        if left.isdigit() and (right.isalpha() or right.isdigit()):
            text = left
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text.isdigit() and len(text) <= 6:
        return text.zfill(6)
    return text


def _date_to_string(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def _path_for_summary(path: Path) -> str:
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.name


if __name__ == "__main__":
    main()
