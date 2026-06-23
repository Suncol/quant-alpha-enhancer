from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


DATE_STOCK_COLUMNS = ("date", "stock_code")
REQUESTED_DAILY_COLUMNS = (
    "change",
    "pct_change",
    "volume",
    "amount",
    "turnover_rate",
    "free_float_turnover_rate",
    "volume_ratio",
    "pe",
    "pe_ttm",
    "pb",
    "ps",
    "ps_ttm",
    "dividend_yield",
    "dividend_yield_ttm",
    "total_share_capital",
    "float_share_capital",
    "free_float_share_capital",
    "total_market_cap",
    "float_market_cap",
)
SW_COLUMNS = (
    "sw_industry_standard",
    "sw_l1_code",
    "sw_l1_name",
    "sw_l2_code",
    "sw_l2_name",
    "sw_l3_code",
    "sw_l3_name",
    "sw_classification_snapshot_date",
    "sw_classification_mode",
    "historical_pit_industry",
)
LISTING_BOARD_SEGMENTS = ("CHINEXT", "STAR", "BSE", "SSE_MAIN", "SZSE_MAIN")
MAIN_BOARD_SEGMENTS = frozenset({"SSE_MAIN", "SZSE_MAIN"})
LISTING_BOARD_COLUMNS = (
    "listing_board_standard",
    "listing_board_code",
    "listing_board_name",
    "listing_board_segment",
    "exchange_code",
    "exchange_suffix",
    "listing_board_reference_mode",
    "historical_pit_listing_board",
    "is_chinext",
    "is_star",
    "is_bse",
    "is_sse_main",
    "is_szse_main",
    "is_main_board",
)
INDEX_FLAG_COLUMNS = (
    "is_csi300",
    "is_csi500",
    "is_csi1000",
    "is_csi2000",
)
INDEX_UNKNOWN_COLUMNS = (
    "is_csi300_unknown",
    "is_csi500_unknown",
    "is_csi1000_unknown",
    "is_csi2000_unknown",
)
INDEX_MEMBERSHIP_COLUMNS = (
    *INDEX_FLAG_COLUMNS,
    *INDEX_UNKNOWN_COLUMNS,
    "index_membership_any_unknown",
    "index_membership_all_known",
    "historical_pit_index_membership",
)
OUTPUT_COLUMNS = (
    "date",
    "stock_code",
    "symbol",
    *REQUESTED_DAILY_COLUMNS,
    *SW_COLUMNS,
    *LISTING_BOARD_COLUMNS,
    *INDEX_MEMBERSHIP_COLUMNS,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a date-stock dataset from generated A-share daily metrics, a static "
            "SW2021 current industry reference, a static current listing board reference, "
            "and a PIT index membership panel. When --universe-panel is supplied, date "
            "keeps the same semantics as that universe, e.g. signal generation date."
        )
    )
    parser.add_argument("--daily-metrics-dir", required=True, type=Path)
    parser.add_argument("--sw-current-snapshot", required=True, type=Path)
    parser.add_argument("--listing-board-current-snapshot", required=True, type=Path)
    parser.add_argument("--index-membership-panel", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--archive-path", default=None, type=Path)
    parser.add_argument("--universe-panel", default=None, type=Path)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--allow-missing-static-industry",
        action="store_true",
        help="Keep rows whose symbol is absent from the static SW snapshot instead of failing.",
    )
    parser.add_argument(
        "--allow-missing-static-listing-board",
        action="store_true",
        help="Keep rows whose symbol is absent from the static listing board snapshot instead of failing.",
    )
    parser.add_argument(
        "--allow-missing-pit-index-membership",
        action="store_true",
        help="Keep rows whose date-stock key is absent from the PIT index membership panel instead of failing.",
    )
    args = parser.parse_args()

    universe_frame = _read_frame(args.universe_panel) if args.universe_panel is not None else None
    index_membership_panel = _read_frame(args.index_membership_panel)
    panel, summary = build_stock_daily_metrics_sw_static_panel(
        daily_metrics_dir=args.daily_metrics_dir,
        sw_current_snapshot=args.sw_current_snapshot,
        listing_board_current_snapshot=args.listing_board_current_snapshot,
        index_membership_panel=index_membership_panel,
        index_membership_panel_source=args.index_membership_panel,
        universe_frame=universe_frame,
        start=args.start_date,
        end=args.end_date,
        require_industry_match=not args.allow_missing_static_industry,
        require_listing_board_match=not args.allow_missing_static_listing_board,
        require_index_membership_match=not args.allow_missing_pit_index_membership,
    )
    manifest = write_stock_daily_metrics_sw_static_dataset(
        panel=panel,
        summary=summary,
        output_dir=args.output_dir,
        archive_path=args.archive_path,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_stock_daily_metrics_sw_static_panel(
    *,
    daily_metrics_dir: Path,
    sw_current_snapshot: Path,
    listing_board_current_snapshot: Path,
    index_membership_panel: pd.DataFrame,
    index_membership_panel_source: Path | None = None,
    universe_frame: pd.DataFrame | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    require_industry_match: bool = True,
    require_listing_board_match: bool = True,
    require_index_membership_match: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return daily metrics joined to static SW, static board, and PIT index membership."""

    daily_root = Path(daily_metrics_dir)
    daily_index = _read_daily_index(daily_root, start=start, end=end)
    universe, universe_summary = _prepare_universe(universe_frame)
    selected_index = _select_daily_index(daily_index, universe)
    daily_panel = _load_daily_metric_rows(daily_root, selected_index, universe)

    sw_reference, sw_summary = _read_static_sw_reference(sw_current_snapshot)
    panel = daily_panel.merge(sw_reference, on="symbol", how="left", validate="many_to_one")
    missing_industry = panel["sw_l1_code"].isna()
    missing_industry_count = int(missing_industry.sum())
    if require_industry_match and missing_industry_count:
        sample = panel.loc[missing_industry, ["date", "stock_code", "symbol"]].head(10)
        raise ValueError(f"Rows have missing static SW industry rows: {_records_with_iso_dates(sample)}")

    board_reference, board_summary = _read_static_listing_board_reference(listing_board_current_snapshot)
    panel = panel.merge(board_reference, on="symbol", how="left", validate="many_to_one")
    missing_board = panel["listing_board_segment"].isna()
    missing_board_count = int(missing_board.sum())
    if require_listing_board_match and missing_board_count:
        sample = panel.loc[missing_board, ["date", "stock_code", "symbol"]].head(10)
        raise ValueError(f"Rows have missing static listing board rows: {_records_with_iso_dates(sample)}")

    index_reference, index_summary = _prepare_pit_index_membership_panel(index_membership_panel)
    panel = panel.merge(index_reference, on=list(DATE_STOCK_COLUMNS), how="left", validate="one_to_one")
    missing_index = panel["historical_pit_index_membership"].isna()
    missing_index_count = int(missing_index.sum())
    if require_index_membership_match and missing_index_count:
        sample = panel.loc[missing_index, ["date", "stock_code", "symbol"]].head(10)
        raise ValueError(f"Rows have missing PIT index membership rows: {_records_with_iso_dates(sample)}")

    panel["historical_pit_industry"] = False
    panel = panel.loc[:, list(OUTPUT_COLUMNS)].sort_values(list(DATE_STOCK_COLUMNS)).reset_index(drop=True)
    _validate_output_panel(panel)

    summary = _build_summary(
        panel,
        daily_metrics_dir=daily_root,
        sw_current_snapshot=Path(sw_current_snapshot),
        listing_board_current_snapshot=Path(listing_board_current_snapshot),
        index_membership_panel_source=index_membership_panel_source,
        universe_summary=universe_summary,
        sw_summary=sw_summary,
        listing_board_summary=board_summary,
        index_membership_summary=index_summary,
        missing_industry_count=missing_industry_count,
        missing_listing_board_count=missing_board_count,
        missing_index_membership_count=missing_index_count,
    )
    return panel, summary


def write_stock_daily_metrics_sw_static_dataset(
    *,
    panel: pd.DataFrame,
    summary: Mapping[str, Any],
    output_dir: Path,
    archive_path: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(panel, pd.DataFrame):
        raise TypeError(f"panel must be a pandas DataFrame, got {type(panel)!r}")
    missing = sorted(set(OUTPUT_COLUMNS).difference(panel.columns))
    if missing:
        raise ValueError(f"Panel is missing output columns: {missing}")
    _validate_output_panel(panel)

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    partition_dir = target / "daily_features"
    if partition_dir.exists():
        shutil.rmtree(partition_dir)
    partition_dir.mkdir(parents=True)

    daily_index_rows: list[dict[str, Any]] = []
    partition_paths: list[Path] = []
    for date, group in panel.groupby("date", sort=True):
        timestamp = pd.Timestamp(date)
        month_dir = partition_dir / timestamp.strftime("%Y-%m")
        month_dir.mkdir(parents=True, exist_ok=True)
        output_path = month_dir / f"{timestamp.strftime('%Y-%m-%d')}.csv"
        _write_csv_with_iso_dates(group, output_path)
        partition_paths.append(output_path)
        daily_index_rows.append(
            {
                "date": timestamp,
                "file_path": output_path.relative_to(target).as_posix(),
                "rows": int(len(group)),
                "unique_stocks": int(group["stock_code"].nunique()),
                "missing_static_sw_industry_rows": int(group["sw_l1_code"].isna().sum()),
                "missing_static_listing_board_rows": int(group["listing_board_segment"].isna().sum()),
                "missing_pit_index_membership_rows": int(group["historical_pit_index_membership"].isna().sum()),
                "index_membership_any_unknown_rows": int(group["index_membership_any_unknown"].sum()),
            }
        )

    daily_index = pd.DataFrame(daily_index_rows)
    daily_index_path = target / "daily_features.csv"
    panel_pickle_path = target / "stock_daily_metrics_sw_static.pkl"
    manifest_path = target / "manifest.json"
    readme_path = target / "README.md"

    _write_csv_with_iso_dates(daily_index, daily_index_path)
    panel.to_pickle(panel_pickle_path)
    readme_path.write_text(_readme_text(), encoding="utf-8")

    outputs = {
        "daily_features_index": _path_for_summary(daily_index_path),
        "daily_features_dir": _path_for_summary(partition_dir),
        "panel_pickle": _path_for_summary(panel_pickle_path),
        "manifest": _path_for_summary(manifest_path),
        "readme": _path_for_summary(readme_path),
    }
    if archive_path is not None:
        outputs["archive"] = _path_for_summary(Path(archive_path))

    manifest = dict(summary)
    manifest["metadata"] = {
        **dict(summary.get("metadata", {})),
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    manifest["outputs"] = outputs
    manifest["daily_partitions"] = {
        "partitioning": "daily_features/YYYY-MM/YYYY-MM-DD.csv",
        "date_count": int(len(daily_index)),
        "row_count": int(len(panel)),
    }
    artifact_paths = [daily_index_path, panel_pickle_path, readme_path, *partition_paths]
    manifest["file_checksums"] = {
        path.relative_to(target).as_posix(): {
            "sha256": _sha256(path),
            "bytes": int(path.stat().st_size),
        }
        for path in artifact_paths
    }
    manifest_path.write_text(json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if archive_path is not None:
        _write_zip_archive(
            archive_path=Path(archive_path),
            package_dir_name=target.name,
            base_dir=target,
            files=[daily_index_path, panel_pickle_path, manifest_path, readme_path, *partition_paths],
        )
    return manifest


def _read_daily_index(
    daily_root: Path,
    *,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
) -> pd.DataFrame:
    index_path = daily_root / "daily_metrics.csv"
    if not index_path.is_file():
        raise FileNotFoundError(f"daily_metrics.csv not found: {index_path}")
    index = pd.read_csv(index_path, dtype={"file_path": "string"})
    missing = sorted({"date", "file_path"}.difference(index.columns))
    if missing:
        raise ValueError(f"daily_metrics.csv is missing required columns: {missing}")
    index = index.loc[:, ["date", "file_path"]].copy()
    index["date"] = _parse_dates(index["date"])
    if index["date"].isna().any():
        raise ValueError("daily_metrics.csv contains unparseable date values.")
    if index.duplicated("date").any():
        duplicates = index.loc[index.duplicated("date", keep=False), ["date"]].head(10)
        raise ValueError(f"daily_metrics.csv contains duplicate dates: {_records_with_iso_dates(duplicates)}")
    if start is not None:
        index = index[index["date"] >= pd.Timestamp(start).normalize()]
    if end is not None:
        index = index[index["date"] <= pd.Timestamp(end).normalize()]
    return index.sort_values("date").reset_index(drop=True)


def _prepare_universe(universe_frame: pd.DataFrame | None) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    if universe_frame is None:
        return None, {"mode": "all_daily_metric_rows"}
    if not isinstance(universe_frame, pd.DataFrame):
        raise TypeError(f"universe_frame must be a pandas DataFrame, got {type(universe_frame)!r}")
    missing = sorted(set(DATE_STOCK_COLUMNS).difference(universe_frame.columns))
    if missing:
        raise ValueError(f"Universe frame is missing required key columns: {missing}")
    universe = universe_frame.loc[:, list(DATE_STOCK_COLUMNS)].copy()
    universe["date"] = _parse_dates(universe["date"])
    invalid_dates = int(universe["date"].isna().sum())
    if invalid_dates:
        raise ValueError(f"Universe frame contains {invalid_dates} unparseable date values.")
    universe["stock_code"] = universe["stock_code"].map(_normalize_stock_code)
    invalid_stock_codes = int(universe["stock_code"].eq("").sum())
    if invalid_stock_codes:
        raise ValueError(f"Universe frame contains {invalid_stock_codes} empty stock_code values.")
    if universe.duplicated(list(DATE_STOCK_COLUMNS)).any():
        duplicates = universe.loc[
            universe.duplicated(list(DATE_STOCK_COLUMNS), keep=False),
            list(DATE_STOCK_COLUMNS),
        ].head(10)
        raise ValueError(f"Universe frame contains duplicate date, stock_code keys: {_records_with_iso_dates(duplicates)}")
    return (
        universe.sort_values(list(DATE_STOCK_COLUMNS)).reset_index(drop=True),
        {
            "mode": "provided_date_stock_universe",
            "input_row_count": int(len(universe)),
        },
    )


def _select_daily_index(daily_index: pd.DataFrame, universe: pd.DataFrame | None) -> pd.DataFrame:
    if universe is None:
        return daily_index
    required_dates = pd.DatetimeIndex(sorted(universe["date"].drop_duplicates()))
    available_dates = set(pd.DatetimeIndex(daily_index["date"]))
    missing_dates = [date for date in required_dates if date not in available_dates]
    if missing_dates:
        date_text = [pd.Timestamp(date).date().isoformat() for date in missing_dates[:10]]
        raise ValueError(f"Universe contains dates with missing daily metrics partitions: {date_text}")
    selected = daily_index[daily_index["date"].isin(required_dates)].copy()
    return selected.sort_values("date").reset_index(drop=True)


def _load_daily_metric_rows(
    daily_root: Path,
    daily_index: pd.DataFrame,
    universe: pd.DataFrame | None,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    universe_by_date = {
        pd.Timestamp(date): group.loc[:, list(DATE_STOCK_COLUMNS)].copy()
        for date, group in (universe.groupby("date") if universe is not None else [])
    }
    for row in daily_index.itertuples(index=False):
        date = pd.Timestamp(row.date).normalize()
        path = daily_root / str(row.file_path)
        daily = _read_daily_metrics_partition(path, expected_date=date)
        if universe is not None:
            wanted = universe_by_date.get(date)
            if wanted is None:
                continue
            daily = wanted.merge(daily, on=list(DATE_STOCK_COLUMNS), how="left", validate="one_to_one")
            missing_daily = daily["symbol"].isna()
            if missing_daily.any():
                sample = daily.loc[missing_daily, list(DATE_STOCK_COLUMNS)].head(10)
                raise ValueError(f"Universe keys are missing from daily metrics rows: {_records_with_iso_dates(sample)}")
        chunks.append(daily)
    if not chunks:
        raise ValueError("No daily metrics rows selected for the requested date range or universe.")
    panel = pd.concat(chunks, ignore_index=True)
    if panel.duplicated(list(DATE_STOCK_COLUMNS)).any():
        duplicates = panel.loc[
            panel.duplicated(list(DATE_STOCK_COLUMNS), keep=False),
            list(DATE_STOCK_COLUMNS),
        ].head(10)
        raise ValueError(f"Daily metrics contain duplicate date, stock_code keys: {_records_with_iso_dates(duplicates)}")
    return panel.sort_values(list(DATE_STOCK_COLUMNS)).reset_index(drop=True)


def _read_daily_metrics_partition(path: Path, *, expected_date: pd.Timestamp) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Daily metrics partition not found: {path}")
    required = {"symbol", "trade_date", *REQUESTED_DAILY_COLUMNS}
    frame = pd.read_csv(path, dtype={"symbol": "string"})
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Daily metrics partition {path} is missing required columns: {missing}")
    frame = frame.loc[:, ["symbol", "trade_date", *REQUESTED_DAILY_COLUMNS]].copy()
    frame["date"] = _parse_dates(frame["trade_date"])
    frame = frame[frame["date"].eq(expected_date)].copy()
    if frame.empty:
        raise ValueError(f"Daily metrics partition {path} has no rows for {expected_date.date().isoformat()}.")
    frame["symbol"] = frame["symbol"].astype("string")
    frame["stock_code"] = frame["symbol"].map(_normalize_stock_code)
    empty_codes = int(frame["stock_code"].eq("").sum())
    if empty_codes:
        raise ValueError(f"Daily metrics partition {path} contains {empty_codes} empty normalized stock codes.")
    for column in REQUESTED_DAILY_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.duplicated(list(DATE_STOCK_COLUMNS)).any():
        duplicates = frame.loc[
            frame.duplicated(list(DATE_STOCK_COLUMNS), keep=False),
            list(DATE_STOCK_COLUMNS),
        ].head(10)
        raise ValueError(f"Daily metrics partition {path} contains duplicate date, stock_code rows: {_records_with_iso_dates(duplicates)}")
    return frame.loc[:, ["date", "stock_code", "symbol", *REQUESTED_DAILY_COLUMNS]]


def _read_static_sw_reference(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Static SW current snapshot not found: {path}")
    frame = pd.read_csv(path, dtype={"stock_code": "string"})
    required = {
        "stock_code",
        "industry_standard",
        "industry_level1_code",
        "industry_level1",
        "industry_level2_code",
        "industry_level2",
        "industry_level3_code",
        "industry_level3",
        "classification_snapshot_date",
        "classification_mode",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Static SW current snapshot is missing required columns: {missing}")
    reference = pd.DataFrame(
        {
            "symbol": frame["stock_code"].astype("string"),
            "sw_industry_standard": frame["industry_standard"].astype("string"),
            "sw_l1_code": frame["industry_level1_code"].astype("string"),
            "sw_l1_name": frame["industry_level1"].astype("string"),
            "sw_l2_code": frame["industry_level2_code"].astype("string"),
            "sw_l2_name": frame["industry_level2"].astype("string"),
            "sw_l3_code": frame["industry_level3_code"].astype("string"),
            "sw_l3_name": frame["industry_level3"].astype("string"),
            "sw_classification_snapshot_date": _parse_dates(frame["classification_snapshot_date"]).dt.date.astype(str),
            "sw_classification_mode": frame["classification_mode"].astype("string"),
        }
    )
    reference = reference[reference["symbol"].notna() & reference["symbol"].ne("")]
    if reference.duplicated("symbol").any():
        duplicates = reference.loc[reference.duplicated("symbol", keep=False), ["symbol"]].head(10)
        raise ValueError(f"Static SW current snapshot contains duplicate symbols: {duplicates.to_dict('records')}")
    return reference.sort_values("symbol").reset_index(drop=True), {
        "path": _path_for_summary(Path(path)),
        "row_count": int(len(reference)),
        "industry_standard_values": sorted(reference["sw_industry_standard"].dropna().astype(str).unique().tolist()),
        "classification_snapshot_dates": sorted(reference["sw_classification_snapshot_date"].dropna().astype(str).unique().tolist()),
        "classification_modes": sorted(reference["sw_classification_mode"].dropna().astype(str).unique().tolist()),
        "historical_pit_industry": False,
    }


def _read_static_listing_board_reference(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Static listing board snapshot not found: {path}")
    frame = pd.read_csv(path, dtype={"stock_code": "string"})
    required = {
        "stock_code",
        "dimension_standard",
        "listing_board_code",
        "listing_board",
        "exchange_code",
        "exchange_suffix",
        "reference_mode",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Static listing board snapshot is missing required columns: {missing}")

    reference = pd.DataFrame(
        {
            "symbol": frame["stock_code"].astype("string"),
            "listing_board_standard": frame["dimension_standard"].astype("string"),
            "listing_board_code": frame["listing_board_code"].astype("string"),
            "listing_board_name": frame["listing_board"].astype("string"),
            "exchange_code": frame["exchange_code"].astype("string"),
            "exchange_suffix": frame["exchange_suffix"].astype("string"),
            "listing_board_reference_mode": frame["reference_mode"].astype("string"),
        }
    )
    reference = reference[reference["symbol"].notna() & reference["symbol"].ne("")].copy()
    reference["listing_board_segment"] = [
        _listing_board_segment(code, exchange)
        for code, exchange in zip(reference["listing_board_code"], reference["exchange_code"], strict=True)
    ]
    invalid_segment = reference["listing_board_segment"].eq("")
    if invalid_segment.any():
        sample = reference.loc[invalid_segment, ["symbol", "listing_board_code", "exchange_code"]].head(10)
        raise ValueError(f"Static listing board snapshot contains invalid board rows: {sample.to_dict('records')}")
    if reference.duplicated("symbol").any():
        duplicates = reference.loc[reference.duplicated("symbol", keep=False), ["symbol"]].head(10)
        raise ValueError(f"Static listing board snapshot contains duplicate symbols: {duplicates.to_dict('records')}")

    for segment in LISTING_BOARD_SEGMENTS:
        reference[f"is_{segment.lower()}"] = reference["listing_board_segment"].eq(segment)
    reference["is_main_board"] = reference["listing_board_segment"].isin(MAIN_BOARD_SEGMENTS)
    reference["historical_pit_listing_board"] = False
    reference = reference.loc[:, ["symbol", *LISTING_BOARD_COLUMNS]]
    return reference.sort_values("symbol").reset_index(drop=True), {
        "path": _path_for_summary(Path(path)),
        "row_count": int(len(reference)),
        "dimension_standard_values": sorted(reference["listing_board_standard"].dropna().astype(str).unique().tolist()),
        "reference_modes": sorted(reference["listing_board_reference_mode"].dropna().astype(str).unique().tolist()),
        "listing_board_segments": sorted(reference["listing_board_segment"].dropna().astype(str).unique().tolist()),
        "historical_pit_listing_board": False,
        "membership_mode": "static_current_reference",
    }


def _listing_board_segment(listing_board_code: Any, exchange_code: Any) -> str:
    code = "" if pd.isna(listing_board_code) else str(listing_board_code).strip().upper()
    exchange = "" if pd.isna(exchange_code) else str(exchange_code).strip().upper()
    if code == "CHINEXT":
        return "CHINEXT"
    if code == "STAR":
        return "STAR"
    if code in {"BSE", "BJSE"}:
        return "BSE"
    if code == "MAIN":
        if exchange in {"SSE", "SHSE", "SH"}:
            return "SSE_MAIN"
        if exchange in {"SZSE", "SZ"}:
            return "SZSE_MAIN"
    if code in LISTING_BOARD_SEGMENTS:
        return code
    return ""


def _prepare_pit_index_membership_panel(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"index_membership_panel must be a pandas DataFrame, got {type(frame)!r}")
    required = {"date", "stock_code", *INDEX_MEMBERSHIP_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"PIT index membership panel is missing required columns: {missing}")

    output = frame.loc[:, ["date", "stock_code", *INDEX_MEMBERSHIP_COLUMNS]].copy()
    output["date"] = _parse_dates(output["date"])
    invalid_dates = int(output["date"].isna().sum())
    if invalid_dates:
        raise ValueError(f"PIT index membership panel contains {invalid_dates} unparseable date values.")
    output["stock_code"] = output["stock_code"].map(_normalize_stock_code)
    empty_codes = int(output["stock_code"].eq("").sum())
    if empty_codes:
        raise ValueError(f"PIT index membership panel contains {empty_codes} empty stock_code values.")
    if output.duplicated(list(DATE_STOCK_COLUMNS)).any():
        duplicates = output.loc[
            output.duplicated(list(DATE_STOCK_COLUMNS), keep=False),
            list(DATE_STOCK_COLUMNS),
        ].head(10)
        raise ValueError(f"PIT index membership panel contains duplicate date, stock_code keys: {_records_with_iso_dates(duplicates)}")

    for column in INDEX_MEMBERSHIP_COLUMNS:
        numeric = pd.to_numeric(output[column], errors="coerce")
        invalid = numeric.isna() | ~numeric.isin([0, 1])
        if invalid.any():
            sample = output.loc[invalid, list(DATE_STOCK_COLUMNS) + [column]].head(10)
            raise ValueError(f"PIT index membership column {column!r} must contain only 0/1 values: {_records_with_iso_dates(sample)}")
        output[column] = numeric.astype(bool)

    for flag_column, unknown_column in zip(INDEX_FLAG_COLUMNS, INDEX_UNKNOWN_COLUMNS, strict=True):
        conflict = output[flag_column] & output[unknown_column]
        if conflict.any():
            sample = output.loc[conflict, list(DATE_STOCK_COLUMNS) + [flag_column, unknown_column]].head(10)
            raise ValueError(f"PIT index membership column {flag_column!r} cannot be true when {unknown_column!r} is true: {_records_with_iso_dates(sample)}")

    expected_any_unknown = output[list(INDEX_UNKNOWN_COLUMNS)].any(axis=1)
    any_unknown_mismatch = output["index_membership_any_unknown"].ne(expected_any_unknown)
    if any_unknown_mismatch.any():
        sample = output.loc[any_unknown_mismatch, list(DATE_STOCK_COLUMNS) + [*INDEX_UNKNOWN_COLUMNS, "index_membership_any_unknown"]].head(10)
        raise ValueError(f"PIT index membership any-unknown flag is inconsistent with per-index unknown flags: {_records_with_iso_dates(sample)}")
    all_known_mismatch = output["index_membership_all_known"].ne(~expected_any_unknown)
    if all_known_mismatch.any():
        sample = output.loc[all_known_mismatch, list(DATE_STOCK_COLUMNS) + [*INDEX_UNKNOWN_COLUMNS, "index_membership_all_known"]].head(10)
        raise ValueError(f"PIT index membership all-known flag is inconsistent with per-index unknown flags: {_records_with_iso_dates(sample)}")
    historical_invalid = ~output["historical_pit_index_membership"]
    if historical_invalid.any():
        sample = output.loc[historical_invalid, list(DATE_STOCK_COLUMNS) + ["historical_pit_index_membership"]].head(10)
        raise ValueError(f"PIT index membership panel must mark historical_pit_index_membership=true: {_records_with_iso_dates(sample)}")

    return output.sort_values(list(DATE_STOCK_COLUMNS)).reset_index(drop=True), {
        "row_count": int(len(output)),
        "date_count": int(output["date"].nunique()),
        "stock_count": int(output["stock_code"].nunique()),
        "date_min": _date_to_string(output["date"].min()),
        "date_max": _date_to_string(output["date"].max()),
        "historical_pit_index_membership": True,
        "membership_mode": "historical_pit_date_stock_panel",
        "unknown_flags_present": True,
        "columns": list(INDEX_MEMBERSHIP_COLUMNS),
        "membership_flag_columns": list(INDEX_FLAG_COLUMNS),
        "unknown_flag_columns": list(INDEX_UNKNOWN_COLUMNS),
        "unknown_row_counts_by_index": {column: int(output[f"{column}_unknown"].sum()) for column in INDEX_FLAG_COLUMNS},
        "member_row_counts_by_index": {column: int(output[column].sum()) for column in INDEX_FLAG_COLUMNS},
    }

def _validate_output_panel(panel: pd.DataFrame) -> None:
    if panel.empty:
        raise ValueError("Output panel is empty.")
    if panel.duplicated(list(DATE_STOCK_COLUMNS)).any():
        duplicates = panel.loc[
            panel.duplicated(list(DATE_STOCK_COLUMNS), keep=False),
            list(DATE_STOCK_COLUMNS),
        ].head(10)
        raise ValueError(f"Output panel contains duplicate date, stock_code keys: {_records_with_iso_dates(duplicates)}")
    for column in REQUESTED_DAILY_COLUMNS:
        values = pd.to_numeric(panel[column], errors="coerce")
        invalid = values.notna() & ~np.isfinite(values.to_numpy(dtype=float, copy=False))
        if invalid.any():
            sample = panel.loc[invalid, list(DATE_STOCK_COLUMNS)].head(10)
            raise ValueError(f"Output column {column!r} contains non-finite values: {_records_with_iso_dates(sample)}")
    boolean_columns = [
        column
        for column in (*LISTING_BOARD_COLUMNS, *INDEX_MEMBERSHIP_COLUMNS)
        if column.startswith("is_") or column.startswith("historical_pit_") or column.startswith("index_membership_")
    ]
    for column in boolean_columns:
        values = panel[column]
        invalid = values.notna() & ~values.isin([True, False])
        if invalid.any():
            sample = panel.loc[invalid, list(DATE_STOCK_COLUMNS)].head(10)
            raise ValueError(f"Output column {column!r} contains non-boolean values: {_records_with_iso_dates(sample)}")


def _build_summary(
    panel: pd.DataFrame,
    *,
    daily_metrics_dir: Path,
    sw_current_snapshot: Path,
    listing_board_current_snapshot: Path,
    index_membership_panel_source: Path | None,
    universe_summary: Mapping[str, Any],
    sw_summary: Mapping[str, Any],
    listing_board_summary: Mapping[str, Any],
    index_membership_summary: Mapping[str, Any],
    missing_industry_count: int,
    missing_listing_board_count: int,
    missing_index_membership_count: int,
) -> dict[str, Any]:
    universe = dict(universe_summary)
    universe["exported_row_count"] = int(len(panel))
    index_membership_source = (
        _path_for_summary(Path(index_membership_panel_source))
        if index_membership_panel_source is not None
        else "provided_dataframe"
    )
    return {
        "metadata": {
            "artifact_version": "stock_daily_metrics_sw_static_v1",
            "description": "Requested raw A-share daily metrics joined to static current SW2021 industry, static current listing board, and PIT index membership fields.",
            "date_column": "date",
            "stock_column": "stock_code",
            "symbol_column": "symbol",
            "date_index_semantics": (
                "same_as_input_universe_date_signal_generation_date_when_universe_is_provided"
                if universe_summary["mode"] == "provided_date_stock_universe"
                else "trade_date_from_ashare_daily_metrics"
            ),
            "transform_policy": "raw_daily_metric_values_copied_no_zscore_no_rank_no_winsorization",
            "industry_membership": "static_current_reference",
            "historical_pit_industry": False,
            "classification_mode": "static_current_reference",
            "listing_board_membership": "static_current_reference",
            "historical_pit_listing_board": False,
            "index_membership": "historical_pit_date_stock_panel",
            "historical_pit_index_membership": True,
            "index_membership_unknown_flags": True,
            "index_membership_unknown_semantics": "is_csi*=false and is_csi*_unknown=true means membership is unknown, not confirmed non-member",
        },
        "row_count": int(len(panel)),
        "date_count": int(panel["date"].nunique()),
        "stock_count": int(panel["stock_code"].nunique()),
        "date_min": _date_to_string(panel["date"].min()),
        "date_max": _date_to_string(panel["date"].max()),
        "daily_columns": list(REQUESTED_DAILY_COLUMNS),
        "sw_columns": list(SW_COLUMNS),
        "listing_board_columns": list(LISTING_BOARD_COLUMNS),
        "index_membership_columns": list(INDEX_MEMBERSHIP_COLUMNS),
        "index_membership_flag_columns": list(INDEX_FLAG_COLUMNS),
        "index_membership_unknown_columns": list(INDEX_UNKNOWN_COLUMNS),
        "index_membership_unknown_row_counts": {column: int(panel[f"{column}_unknown"].sum()) for column in INDEX_FLAG_COLUMNS},
        "index_membership_member_row_counts": {column: int(panel[column].sum()) for column in INDEX_FLAG_COLUMNS},
        "output_columns": list(OUTPUT_COLUMNS),
        "universe": universe,
        "source_policy": {
            "daily_metrics": {
                "path": _path_for_summary(daily_metrics_dir),
                "dataset": "finfact_io.generated_data.ashare_daily_metrics",
                "source_columns": list(REQUESTED_DAILY_COLUMNS),
            },
            "sw_current_snapshot": {
                "path": _path_for_summary(sw_current_snapshot),
                "dataset": "finfact_io.generated_data.industry_sw_current_reference.current_snapshot",
                "historical_pit_industry": False,
                "classification_mode": "static_current_reference",
            },
            "listing_board_current_snapshot": {
                "path": _path_for_summary(listing_board_current_snapshot),
                "dataset": "finfact_io.generated_data.listing_board_current_reference.current_snapshot",
                "historical_pit_listing_board": False,
                "reference_mode": "static_current_reference",
            },
            "index_membership_panel": {
                "path": index_membership_source,
                "historical_pit_index_membership": True,
                "membership_mode": "historical_pit_date_stock_panel",
                "unknown_flags_present": True,
                "source_columns": list(INDEX_MEMBERSHIP_COLUMNS),
            },
        },
        "static_sw_reference": dict(sw_summary),
        "static_listing_board_reference": dict(listing_board_summary),
        "pit_index_membership_panel": dict(index_membership_summary),
        "missing_static_sw_industry_rows": int(missing_industry_count),
        "missing_static_listing_board_rows": int(missing_listing_board_count),
        "missing_pit_index_membership_rows": int(missing_index_membership_count),
        "notes": [
            "The SW industry fields intentionally use the user-approved static current reference policy.",
            "The dataset does not claim point-in-time historical SW industry membership.",
            "The listing board fields intentionally use the static current listing board reference policy.",
            "The dataset does not claim point-in-time historical listing board membership.",
            "The index membership fields are joined from a provided date-stock PIT membership panel.",
            "Index unknown flags distinguish missing or unreliable index-date constituent sources from confirmed non-membership.",
            "Rows missing from the static SW snapshot, static listing board snapshot, or PIT index membership panel fail by default unless the corresponding allow-missing flag is explicitly requested.",
            "Numeric daily metric columns are copied from the standardized finfact_io daily metrics partitions without imputation.",
        ],
    }

def _read_frame(path: Path | None) -> pd.DataFrame:
    if path is None:
        raise ValueError("path must not be None")
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


def _parse_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", format="mixed").dt.normalize()


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


def _write_csv_with_iso_dates(frame: pd.DataFrame, path: Path) -> None:
    output = frame.copy()
    for column in output.columns:
        if is_datetime64_any_dtype(output[column]):
            output[column] = pd.to_datetime(output[column]).dt.date.astype(str)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, encoding="utf-8-sig")


def _readme_text() -> str:
    return (
        "# Stock Daily Metrics With Static SW Industry, Static Board, And PIT Index Membership\n\n"
        "This dataset contains requested raw daily stock metrics, SW2021 L1/L2/L3 industry fields, "
        "static current listing board fields, and PIT index membership fields.\n\n"
        "`historical_pit_industry=false`: SW industry columns use the static current reference snapshot, "
        "not historical point-in-time industry membership.\n\n"
        "`historical_pit_listing_board=false`: listing board columns use the static current listing board "
        "reference snapshot.\n\n"
        "`historical_pit_index_membership=true`: CSI300/CSI500/CSI1000/CSI2000 membership columns are "
        "joined from a date-stock PIT membership panel.\n\n"
        "`is_csi*_unknown=true`: the corresponding index-date constituent source is missing or unreliable; "
        "`is_csi*=false` must not be interpreted as confirmed non-membership for that index.\n\n"
        "Files:\n\n"
        "- `daily_features.csv`: date-level partition index.\n"
        "- `daily_features/YYYY-MM/YYYY-MM-DD.csv`: on-demand date partitions.\n"
        "- `stock_daily_metrics_sw_static.pkl`: dtype-preserving full panel.\n"
        "- `manifest.json`: schema, source, static-industry policy, static-board policy, PIT-index policy, row counts, and checksums.\n"
    )

def _write_zip_archive(*, archive_path: Path, package_dir_name: str, base_dir: Path, files: Sequence[Path]) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=f"{package_dir_name}/{path.relative_to(base_dir).as_posix()}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records_with_iso_dates(frame: pd.DataFrame) -> list[dict[str, Any]]:
    output = frame.copy()
    for column in output.columns:
        if is_datetime64_any_dtype(output[column]):
            output[column] = pd.to_datetime(output[column]).dt.date.astype(str)
    return output.to_dict("records")


def _date_to_string(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def _path_for_summary(path: Path) -> str:
    path = Path(path)
    if not path.is_absolute():
        return path.as_posix()
    cwd = Path.cwd().resolve()
    resolved = path.resolve(strict=False)
    try:
        return resolved.relative_to(cwd, walk_up=True).as_posix()
    except ValueError:
        return f"<external>/{resolved.name}"


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


















