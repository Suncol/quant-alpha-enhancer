from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from analysis.build_lgbm_placeholder_training_data import (
        DEFAULT_INDEX_COLUMNS,
        _date_to_string,
        _normalize_stock_code,
        _path_for_summary,
        _write_csv_with_iso_dates,
    )
except ModuleNotFoundError:  # Allows direct execution via python analysis/script.py.
    from build_lgbm_placeholder_training_data import (
        DEFAULT_INDEX_COLUMNS,
        _date_to_string,
        _normalize_stock_code,
        _path_for_summary,
        _write_csv_with_iso_dates,
    )


RAW_LIQUIDITY_COLUMNS = ("amount_k", "turnover", "logADV20", "logAmount20", "turnover20")
INDEX_UNKNOWN_COLUMNS = tuple(f"{column}_unknown" for column in DEFAULT_INDEX_COLUMNS)
INDEX_STATUS_COLUMNS = (
    "index_membership_any_unknown",
    "index_membership_all_known",
    "historical_pit_index_membership",
)
RAW_CONTEXT_COLUMNS = (
    "industry",
    "board",
    *DEFAULT_INDEX_COLUMNS,
    *INDEX_UNKNOWN_COLUMNS,
    *INDEX_STATUS_COLUMNS,
    "market_cap",
    *RAW_LIQUIDITY_COLUMNS,
)
DATE_STOCK_COLUMNS = ("date", "stock_code")
RESERVED_DERIVED_COLUMN_SUFFIXES = ("_z", "_rank")
INDEX_FLAG_TO_NAME = {
    "is_csi300": "CSI300",
    "is_csi500": "CSI500",
    "is_csi1000": "CSI1000",
    "is_csi2000": "CSI2000",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export raw training feature inputs into a directory and zip archive. "
            "The output date column is the signal generation date, equal to buy date minus one."
        )
    )
    parser.add_argument("--exposures", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--archive-path", required=True, type=Path)
    parser.add_argument(
        "--universe-panel",
        default=None,
        type=Path,
        help=(
            "Optional panel/table whose date, stock_code keys define the exported training universe. "
            "Only date and stock_code are read from this file."
        ),
    )
    parser.add_argument(
        "--signal-value",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        default=[],
        help=(
            "Optional raw signal matrix to merge by date and stock_code. "
            "May be repeated, for example: --signal-value close close.pkl."
        ),
    )
    parser.add_argument(
        "--source-note",
        action="append",
        default=[],
        help="Optional provenance note written to manifest.json. May be repeated.",
    )
    args = parser.parse_args()

    exposures = _read_frame(args.exposures)
    universe_frame = _read_frame(args.universe_panel) if args.universe_panel is not None else None
    signal_value_frames = {name: _read_frame(Path(path)) for name, path in args.signal_value}
    summary = export_raw_training_feature_package(
        exposures=exposures,
        output_dir=args.output_dir,
        archive_path=args.archive_path,
        universe_frame=universe_frame,
        signal_value_frames=signal_value_frames,
        source_notes=args.source_note,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def export_raw_training_feature_package(
    *,
    exposures: pd.DataFrame,
    output_dir: Path,
    archive_path: Path,
    universe_frame: pd.DataFrame | None = None,
    signal_value_frames: Mapping[str, pd.DataFrame] | None = None,
    source_notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Write raw feature inputs and a zip archive without feature standardization."""

    context = _prepare_raw_context_exposures(exposures)
    context, universe_summary = _apply_optional_universe(context, universe_frame)
    raw_features, signal_diagnostics = _merge_raw_signals(
        context,
        signal_value_frames=signal_value_frames or {},
    )
    raw_features = raw_features.sort_values(list(DATE_STOCK_COLUMNS)).reset_index(drop=True)
    constituents = _build_index_constituents(raw_features)
    dummy_matrices = _build_dummy_matrices(raw_features)

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    raw_features_csv = output_dir / "raw_training_features.csv"
    raw_features_pickle = output_dir / "raw_training_features.pkl"
    constituents_csv = output_dir / "index_constituents.csv"
    industry_dummy_csv = output_dir / "industry_dummy_matrix.csv"
    board_dummy_csv = output_dir / "board_dummy_matrix.csv"
    index_dummy_csv = output_dir / "index_dummy_matrix.csv"
    manifest_path = output_dir / "manifest.json"
    readme_path = output_dir / "README.md"

    _write_csv_with_iso_dates(raw_features, raw_features_csv)
    raw_features.to_pickle(raw_features_pickle)
    _write_csv_with_iso_dates(constituents, constituents_csv)
    _write_csv_with_iso_dates(dummy_matrices["industry"], industry_dummy_csv)
    _write_csv_with_iso_dates(dummy_matrices["board"], board_dummy_csv)
    _write_csv_with_iso_dates(dummy_matrices["index"], index_dummy_csv)
    readme_path.write_text(_readme_text(), encoding="utf-8")

    outputs = {
        "raw_features_csv": _path_for_summary(raw_features_csv),
        "raw_features_pickle": _path_for_summary(raw_features_pickle),
        "index_constituents_csv": _path_for_summary(constituents_csv),
        "industry_dummy_matrix_csv": _path_for_summary(industry_dummy_csv),
        "board_dummy_matrix_csv": _path_for_summary(board_dummy_csv),
        "index_dummy_matrix_csv": _path_for_summary(index_dummy_csv),
        "manifest": _path_for_summary(manifest_path),
        "readme": _path_for_summary(readme_path),
        "archive": _path_for_summary(archive_path),
    }
    manifest = _build_manifest(
        raw_features,
        constituents,
        signal_diagnostics=signal_diagnostics,
        universe_summary=universe_summary,
        dummy_matrix_summary={
            name: _dummy_matrix_summary(matrix)
            for name, matrix in dummy_matrices.items()
        },
        outputs=outputs,
        source_notes=source_notes,
        artifact_paths=[
            raw_features_csv,
            raw_features_pickle,
            constituents_csv,
            industry_dummy_csv,
            board_dummy_csv,
            index_dummy_csv,
            readme_path,
        ],
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    _write_zip_archive(
        archive_path=archive_path,
        package_dir_name=output_dir.name,
        files=[
            readme_path,
            board_dummy_csv,
            index_dummy_csv,
            constituents_csv,
            industry_dummy_csv,
            manifest_path,
            raw_features_csv,
            raw_features_pickle,
        ],
    )
    return manifest


def _prepare_raw_context_exposures(exposures: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(exposures, pd.DataFrame):
        raise TypeError(f"Expected exposures DataFrame, got {type(exposures)!r}")
    required_columns = {*DATE_STOCK_COLUMNS, *RAW_CONTEXT_COLUMNS}
    missing = sorted(required_columns.difference(exposures.columns))
    if missing:
        raise ValueError(f"Exposures are missing required raw feature columns: {missing}")

    frame = exposures.loc[:, list(DATE_STOCK_COLUMNS) + list(RAW_CONTEXT_COLUMNS)].copy()
    frame["date"] = _parse_date_column(frame["date"])
    invalid_dates = int(frame["date"].isna().sum())
    if invalid_dates:
        raise ValueError(f"Exposures contain {invalid_dates} rows with unparseable date values.")

    frame["stock_code"] = frame["stock_code"].map(_normalize_stock_code)
    invalid_stock_codes = int(frame["stock_code"].eq("").sum())
    if invalid_stock_codes:
        raise ValueError(f"Exposures contain {invalid_stock_codes} rows with empty stock_code values.")

    if frame.duplicated(list(DATE_STOCK_COLUMNS)).any():
        duplicates = frame.loc[
            frame.duplicated(list(DATE_STOCK_COLUMNS), keep=False),
            list(DATE_STOCK_COLUMNS),
        ]
        sample = _records_with_dates(duplicates.head(5))
        raise ValueError(f"Exposures contain duplicate date, stock_code keys: {sample}")

    for column in (*DEFAULT_INDEX_COLUMNS, *INDEX_UNKNOWN_COLUMNS, *INDEX_STATUS_COLUMNS):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        invalid = numeric.isna() | ~numeric.isin([0, 1])
        if invalid.any():
            sample = _records_with_dates(frame.loc[invalid, list(DATE_STOCK_COLUMNS) + [column]].head(5))
            raise ValueError(f"Index membership column {column!r} must contain finite 0/1 values: {sample}")
        frame[column] = numeric.astype("int8")

    for flag_column, unknown_column in zip(DEFAULT_INDEX_COLUMNS, INDEX_UNKNOWN_COLUMNS, strict=True):
        conflict = frame[flag_column].eq(1) & frame[unknown_column].eq(1)
        if conflict.any():
            sample = _records_with_dates(frame.loc[conflict, list(DATE_STOCK_COLUMNS) + [flag_column, unknown_column]].head(5))
            raise ValueError(f"Index membership column {flag_column!r} cannot be true when {unknown_column!r} is true: {sample}")

    expected_any_unknown = frame[list(INDEX_UNKNOWN_COLUMNS)].eq(1).any(axis=1)
    any_unknown_mismatch = frame["index_membership_any_unknown"].astype(bool).ne(expected_any_unknown)
    if any_unknown_mismatch.any():
        sample = _records_with_dates(frame.loc[any_unknown_mismatch, list(DATE_STOCK_COLUMNS)].head(5))
        raise ValueError(f"index_membership_any_unknown is inconsistent with per-index unknown flags: {sample}")
    all_known_mismatch = frame["index_membership_all_known"].astype(bool).ne(~expected_any_unknown)
    if all_known_mismatch.any():
        sample = _records_with_dates(frame.loc[all_known_mismatch, list(DATE_STOCK_COLUMNS)].head(5))
        raise ValueError(f"index_membership_all_known is inconsistent with per-index unknown flags: {sample}")
    historical_invalid = frame["historical_pit_index_membership"].ne(1)
    if historical_invalid.any():
        sample = _records_with_dates(frame.loc[historical_invalid, list(DATE_STOCK_COLUMNS)].head(5))
        raise ValueError(f"historical_pit_index_membership must be 1 for PIT index exposure rows: {sample}")

    for column in ("market_cap", *RAW_LIQUIDITY_COLUMNS):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values(list(DATE_STOCK_COLUMNS)).reset_index(drop=True)


def _apply_optional_universe(
    context: pd.DataFrame,
    universe_frame: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if universe_frame is None:
        return context, {
            "mode": "all_exposure_rows",
            "input_row_count": int(len(context)),
            "exported_row_count": int(len(context)),
            "dropped_exposure_rows_not_in_universe": 0,
        }
    if not isinstance(universe_frame, pd.DataFrame):
        raise TypeError(f"Expected universe_frame DataFrame, got {type(universe_frame)!r}")
    missing = sorted(set(DATE_STOCK_COLUMNS).difference(universe_frame.columns))
    if missing:
        raise ValueError(f"Universe frame is missing required key columns: {missing}")

    universe = universe_frame.loc[:, list(DATE_STOCK_COLUMNS)].copy()
    universe["date"] = _parse_date_column(universe["date"])
    invalid_dates = int(universe["date"].isna().sum())
    if invalid_dates:
        raise ValueError(f"Universe frame contains {invalid_dates} rows with unparseable date values.")
    universe["stock_code"] = universe["stock_code"].map(_normalize_stock_code)
    invalid_stock_codes = int(universe["stock_code"].eq("").sum())
    if invalid_stock_codes:
        raise ValueError(f"Universe frame contains {invalid_stock_codes} rows with empty stock_code values.")
    if universe.duplicated(list(DATE_STOCK_COLUMNS)).any():
        duplicates = universe.loc[
            universe.duplicated(list(DATE_STOCK_COLUMNS), keep=False),
            list(DATE_STOCK_COLUMNS),
        ]
        sample = _records_with_dates(duplicates.head(5))
        raise ValueError(f"Universe frame contains duplicate date, stock_code keys: {sample}")

    matched = universe.merge(context, on=list(DATE_STOCK_COLUMNS), how="left", validate="one_to_one")
    missing_context = matched["market_cap"].isna()
    if missing_context.any():
        sample = _records_with_dates(matched.loc[missing_context, list(DATE_STOCK_COLUMNS)].head(5))
        raise ValueError(f"Universe contains date, stock_code keys missing from exposures: {sample}")

    return matched, {
        "mode": "provided_date_stock_universe",
        "input_row_count": int(len(universe)),
        "exported_row_count": int(len(matched)),
        "dropped_exposure_rows_not_in_universe": int(len(context) - len(matched)),
    }


def _merge_raw_signals(
    context: pd.DataFrame,
    *,
    signal_value_frames: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    frame = context.copy()
    diagnostics: dict[str, dict[str, Any]] = {}
    for signal_name, signal_frame in signal_value_frames.items():
        _validate_signal_name(signal_name)
        raw_column = f"{signal_name}_raw"
        signal_long = _signal_wide_frame_to_long(
            signal_frame,
            signal_name,
            raw_column,
            date_index=context["date"].drop_duplicates(),
            stock_codes=context["stock_code"].drop_duplicates(),
        )
        before = int(len(frame))
        frame = frame.merge(signal_long, on=list(DATE_STOCK_COLUMNS), how="left", validate="one_to_one")
        values = pd.to_numeric(frame[raw_column], errors="coerce")
        frame[raw_column] = values
        diagnostics[signal_name] = {
            "raw_column": raw_column,
            "input_nonmissing_count": int(signal_long[raw_column].notna().sum()),
            "context_row_count_before_merge": before,
            "matched_nonmissing_count": int(np.isfinite(values.to_numpy(dtype=float, copy=False)).sum()),
            "missing_after_context_merge": int(values.isna().sum()),
            "transform_policy": "raw_numeric_values_no_domain_transform",
        }
    return frame, diagnostics


def _build_index_constituents(raw_features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in DEFAULT_INDEX_COLUMNS:
        values = pd.to_numeric(raw_features[column], errors="coerce").fillna(0.0)
        members = raw_features.loc[values.eq(1.0), ["date", "stock_code"]]
        for row in members.itertuples(index=False):
            rows.append(
                {
                    "date": row.date,
                    "stock_code": row.stock_code,
                    "index_name": INDEX_FLAG_TO_NAME[column],
                    "membership_flag_column": column,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["date", "stock_code", "index_name", "membership_flag_column"])
    return pd.DataFrame(rows).sort_values(["date", "stock_code", "index_name"]).reset_index(drop=True)


def _build_dummy_matrices(raw_features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "industry": _categorical_dummy_matrix(
            raw_features,
            source_column="industry",
            prefix="industry",
            missing_category="unknown_industry",
        ),
        "board": _categorical_dummy_matrix(
            raw_features,
            source_column="board",
            prefix="board",
            missing_category="UNKNOWN_BOARD",
        ),
        "index": _index_dummy_matrix(raw_features),
    }


def _categorical_dummy_matrix(
    raw_features: pd.DataFrame,
    *,
    source_column: str,
    prefix: str,
    missing_category: str,
) -> pd.DataFrame:
    keys = raw_features.loc[:, list(DATE_STOCK_COLUMNS)].copy()
    values = raw_features[source_column].copy()
    values = values.where(values.notna(), missing_category).astype(str).str.strip()
    values = values.where(values.ne(""), missing_category)
    dummies = pd.get_dummies(values, prefix=prefix, prefix_sep="__", dtype="int8")
    dummies = dummies.reindex(sorted(dummies.columns), axis=1)
    matrix = pd.concat([keys, dummies], axis=1)
    _validate_one_hot_matrix(matrix, matrix_name=f"{source_column}_dummy_matrix")
    return matrix


def _index_dummy_matrix(raw_features: pd.DataFrame) -> pd.DataFrame:
    keys = raw_features.loc[:, list(DATE_STOCK_COLUMNS)].copy()
    flags = raw_features.loc[:, list(DEFAULT_INDEX_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    unknown = raw_features.loc[:, list(INDEX_UNKNOWN_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    combined = pd.concat([flags, unknown], axis=1)
    finite = np.isfinite(combined.to_numpy(dtype=float, copy=False))
    zero_or_one = np.isin(combined.to_numpy(dtype=float, copy=False), [0.0, 1.0])
    if not (finite & zero_or_one).all():
        invalid_mask = pd.Series(~(finite & zero_or_one).all(axis=1), index=raw_features.index)
        sample = _records_with_dates(raw_features.loc[invalid_mask, list(DATE_STOCK_COLUMNS)].head(5))
        raise ValueError(f"Index membership and unknown flags must be finite 0/1 values. Invalid rows: {sample}")

    for flag_column, unknown_column in zip(DEFAULT_INDEX_COLUMNS, INDEX_UNKNOWN_COLUMNS, strict=True):
        conflict = flags[flag_column].eq(1.0) & unknown[unknown_column].eq(1.0)
        if conflict.any():
            sample = _records_with_dates(raw_features.loc[conflict, list(DATE_STOCK_COLUMNS)].head(5))
            raise ValueError(f"Rows contain conflicting index membership and unknown flags: {sample}")

    row_sums = flags.sum(axis=1)
    overlapping = row_sums.gt(1.0)
    if overlapping.any():
        sample = _records_with_dates(raw_features.loc[overlapping, list(DATE_STOCK_COLUMNS)].head(5))
        raise ValueError(f"Rows contain multiple index membership flags and cannot be one-hot encoded: {sample}")

    any_unknown = unknown.sum(axis=1).gt(0.0)
    matrix = keys.copy()
    for column in DEFAULT_INDEX_COLUMNS:
        matrix[f"index__{INDEX_FLAG_TO_NAME[column]}"] = flags[column].astype("int8")
    matrix["index__UNKNOWN_INDEX"] = (row_sums.eq(0.0) & any_unknown).astype("int8")
    matrix["index__NON_INDEX"] = (row_sums.eq(0.0) & ~any_unknown).astype("int8")
    _validate_one_hot_matrix(matrix, matrix_name="index_dummy_matrix")
    return matrix

def _validate_one_hot_matrix(matrix: pd.DataFrame, *, matrix_name: str) -> None:
    values = matrix.drop(columns=list(DATE_STOCK_COLUMNS))
    if values.empty:
        raise ValueError(f"{matrix_name} has no dummy columns.")
    numeric = values.apply(pd.to_numeric, errors="coerce")
    raw_values = numeric.to_numpy(dtype=float, copy=False)
    if not np.isin(raw_values, [0.0, 1.0]).all():
        raise ValueError(f"{matrix_name} contains non-binary dummy values.")
    row_sums = numeric.sum(axis=1)
    if not row_sums.eq(1).all():
        bad = matrix.loc[~row_sums.eq(1), list(DATE_STOCK_COLUMNS)].head(5)
        sample = _records_with_dates(bad)
        raise ValueError(f"{matrix_name} is not one-hot; row sums must equal 1. Bad rows: {sample}")


def _dummy_matrix_summary(matrix: pd.DataFrame) -> dict[str, Any]:
    values = matrix.drop(columns=list(DATE_STOCK_COLUMNS)).apply(pd.to_numeric, errors="coerce")
    row_sums = values.sum(axis=1)
    return {
        "row_count": int(len(matrix)),
        "category_count": int(values.shape[1]),
        "columns": list(values.columns),
        "one_hot_row_sum_min": int(row_sums.min()) if len(row_sums) else None,
        "one_hot_row_sum_max": int(row_sums.max()) if len(row_sums) else None,
    }


def _build_manifest(
    raw_features: pd.DataFrame,
    constituents: pd.DataFrame,
    *,
    signal_diagnostics: Mapping[str, Mapping[str, Any]],
    universe_summary: Mapping[str, Any],
    dummy_matrix_summary: Mapping[str, Mapping[str, Any]],
    outputs: Mapping[str, str],
    source_notes: Sequence[str],
    artifact_paths: Sequence[Path],
) -> dict[str, Any]:
    date_count = int(raw_features["date"].nunique())
    stock_count = int(raw_features["stock_code"].nunique())
    return {
        "metadata": {
            "artifact_version": "raw_training_features_v1",
            "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "date_index_semantics": "signal_generation_date_equals_buy_date_minus_one",
            "date_column": "date",
            "stock_column": "stock_code",
            "transform_policy": "raw_inputs_only_no_zscore_no_rank_no_winsorization",
            "dummy_matrix_policy": "industry_board_index_are_exported_as_one_hot_matrices",
            "feature_universe": "date_stock_rows_used_for_training_when_universe_frame_is_provided",
            "timestamped_feature_alignment": {
                "market_cap": "date is signal_generation_date, equal to buy_date_minus_one",
                "amount_k": "date is signal_generation_date, equal to buy_date_minus_one",
                "turnover": "date is signal_generation_date, equal to buy_date_minus_one",
                "logADV20": "date is signal_generation_date, equal to buy_date_minus_one",
                "logAmount20": "date is signal_generation_date, equal to buy_date_minus_one",
                "turnover20": "date is signal_generation_date, equal to buy_date_minus_one",
            },
        },
        "row_count": int(len(raw_features)),
        "date_count": date_count,
        "stock_count": stock_count,
        "date_min": _date_to_string(raw_features["date"].min()) if len(raw_features) else None,
        "date_max": _date_to_string(raw_features["date"].max()) if len(raw_features) else None,
        "raw_feature_columns": list(raw_features.columns),
        "raw_context_columns": list(RAW_CONTEXT_COLUMNS),
        "raw_liquidity_columns": list(RAW_LIQUIDITY_COLUMNS),
        "index_membership_flag_columns": list(DEFAULT_INDEX_COLUMNS),
        "index_membership_unknown_columns": list(INDEX_UNKNOWN_COLUMNS),
        "index_membership_status_columns": list(INDEX_STATUS_COLUMNS),
        "index_constituent_row_count": int(len(constituents)),
        "dummy_matrices": _json_safe(dummy_matrix_summary),
        "signal_inputs": _json_safe(signal_diagnostics),
        "universe": dict(universe_summary),
        "outputs": dict(outputs),
        "file_checksums": {
            path.name: {
                "sha256": _sha256(path),
                "bytes": int(path.stat().st_size),
            }
            for path in artifact_paths
        },
        "source_notes": list(source_notes),
        "notes": [
            "No cross-sectional z-score, rank feature, winsorization, neutralization, label, or sample_weight column is exported.",
            "Index constituents are derived directly from raw index membership flag columns.",
            "Index dummy matrices include UNKNOWN_INDEX for rows with no confirmed CSI membership and at least one unknown index-date source.",
            "Industry, board, and index dummy matrices are one-hot: every exported date-stock row has exactly one active category per matrix.",
            "The package stores CSV for portability and pickle for dtype-preserving Python reloads.",
        ],
    }


def _write_zip_archive(
    *,
    archive_path: Path,
    package_dir_name: str,
    files: Sequence[Path],
) -> None:
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=f"{package_dir_name}/{path.name}")


def _signal_wide_frame_to_long(
    frame: pd.DataFrame,
    signal_name: str,
    raw_column: str,
    *,
    date_index: Sequence[Any] | pd.Series | None = None,
    stock_codes: Sequence[Any] | pd.Series | None = None,
) -> pd.DataFrame:
    matrix = _normalize_signal_matrix(frame, signal_name)
    if date_index is not None:
        dates = pd.DatetimeIndex(_parse_date_column(pd.Series(list(date_index)))).drop_duplicates()
        matrix = matrix.reindex(dates)
    if stock_codes is not None:
        normalized_stock_codes = pd.Index([_normalize_stock_code(code) for code in stock_codes])
        normalized_stock_codes = normalized_stock_codes[normalized_stock_codes != ""].drop_duplicates()
        matrix = matrix.reindex(columns=normalized_stock_codes)
    long = matrix.copy()
    long.index.name = "date"
    long.columns.name = "stock_code"
    long = long.reset_index().melt(
        id_vars="date",
        var_name="stock_code",
        value_name=raw_column,
    )
    long["date"] = _parse_date_column(long["date"])
    long["stock_code"] = long["stock_code"].map(_normalize_stock_code)
    long = long[long["date"].notna() & long["stock_code"].ne("")].copy()
    if long.duplicated(list(DATE_STOCK_COLUMNS)).any():
        duplicates = long.loc[
            long.duplicated(list(DATE_STOCK_COLUMNS), keep=False),
            list(DATE_STOCK_COLUMNS),
        ]
        sample = _records_with_dates(duplicates.head(5))
        raise ValueError(f"Signal {signal_name!r} contains duplicate date, stock_code keys: {sample}")
    long[raw_column] = pd.to_numeric(long[raw_column], errors="coerce")
    return long.sort_values(list(DATE_STOCK_COLUMNS)).reset_index(drop=True)


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

    matrix.index = pd.DatetimeIndex(_parse_date_column(pd.Series(list(matrix.index))))
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


def _validate_signal_name(signal_name: str) -> None:
    if not signal_name or not signal_name.replace("_", "").isalnum():
        raise ValueError(f"Signal name must be non-empty and use only letters, numbers, and underscores: {signal_name!r}")
    if signal_name.endswith(RESERVED_DERIVED_COLUMN_SUFFIXES):
        raise ValueError(
            f"Signal name {signal_name!r} ends with a reserved derived-feature suffix; "
            "pass the original raw input under a name that does not look like a rank/z feature."
        )


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


def _parse_date_column(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", format="mixed").dt.normalize()


def _readme_text() -> str:
    return (
        "# Raw Training Feature Package\n\n"
        "This directory contains raw input features for the training date-stock universe.\n\n"
        "- `raw_training_features.csv` and `raw_training_features.pkl` contain raw feature values only.\n"
        "- `index_constituents.csv` expands index membership flags into long constituent rows.\n"
        "- `industry_dummy_matrix.csv`, `board_dummy_matrix.csv`, and `index_dummy_matrix.csv` are one-hot matrices.\n"
        "- `manifest.json` records schema, date semantics, counts, and file checksums.\n\n"
        "`date` is the signal generation date. For timestamped fields such as market cap, ADV, "
        "and turnover, this date equals the buy date minus one trading day. No z-score, rank, "
        "winsorization, neutralization, target label, or sample weight is included in the raw export.\n"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records_with_dates(frame: pd.DataFrame) -> list[dict[str, Any]]:
    output = frame.copy()
    for column in output.columns:
        if pd.api.types.is_datetime64_any_dtype(output[column]):
            output[column] = pd.to_datetime(output[column]).dt.date.astype(str)
    return output.to_dict("records")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return value


if __name__ == "__main__":
    main()




