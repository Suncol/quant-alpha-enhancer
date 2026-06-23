from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


DATE_STOCK_COLUMNS = ("date", "stock_code")
DEFAULT_WEIGHT_SUM_TOLERANCE = 0.005


@dataclass(frozen=True)
class IndexSpec:
    directory: str
    index_code: str
    index_name: str
    expected_member_count: int


INDEX_SPECS: dict[str, IndexSpec] = {
    "is_csi300": IndexSpec("csi300", "000300.SH", "CSI300", 300),
    "is_csi500": IndexSpec("csi500", "000905.SH", "CSI500", 500),
    "is_csi1000": IndexSpec("csi1000", "000852.SH", "CSI1000", 1000),
    "is_csi2000": IndexSpec("csi2000", "932000.CSI", "CSI2000", 2000),
}

DEFAULT_MEMBER_COUNT_TOLERANCES: dict[str, tuple[int, int]] = {
    "is_csi300": (297, 303),
    "is_csi500": (495, 505),
    "is_csi1000": (990, 1010),
    "is_csi2000": (1980, 2020),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a date-stock PIT CSI membership panel with explicit unknown flags "
            "when an index-date constituent source is missing or unreliable."
        )
    )
    parser.add_argument("--daily-metrics-dir", required=True, type=Path)
    parser.add_argument("--index-data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--archive-path", default=None, type=Path)
    parser.add_argument("--start-date", default="2017-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--weight-sum-tolerance", default=DEFAULT_WEIGHT_SUM_TOLERANCE, type=float)
    args = parser.parse_args()

    panel, quality, summary = build_pit_index_membership_panel(
        daily_metrics_dir=args.daily_metrics_dir,
        index_data_root=args.index_data_root,
        start=args.start_date,
        end=args.end_date,
        weight_sum_tolerance=args.weight_sum_tolerance,
    )
    manifest = write_pit_index_membership_panel(
        panel=panel,
        quality=quality,
        summary=summary,
        output_dir=args.output_dir,
        archive_path=args.archive_path,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_pit_index_membership_panel(
    *,
    daily_metrics_dir: Path,
    index_data_root: Path,
    start: str | pd.Timestamp = "2017-01-01",
    end: str | pd.Timestamp | None = None,
    index_specs: Mapping[str, IndexSpec] = INDEX_SPECS,
    member_count_tolerances: Mapping[str, tuple[int, int]] = DEFAULT_MEMBER_COUNT_TOLERANCES,
    weight_sum_tolerance: float = DEFAULT_WEIGHT_SUM_TOLERANCE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if weight_sum_tolerance < 0.0:
        raise ValueError("weight_sum_tolerance must be non-negative.")
    if not index_specs:
        raise ValueError("index_specs must not be empty.")

    universe = _read_daily_stock_universe(Path(daily_metrics_dir), start=start, end=end)
    dates = pd.DatetimeIndex(sorted(universe["date"].drop_duplicates()))
    frame = universe.copy()
    groups = frame.groupby("date", sort=False).groups

    asof_indexes = {
        column: _read_asof_index(Path(index_data_root) / spec.directory, column, spec)
        for column, spec in index_specs.items()
    }
    quality_records: list[dict[str, Any]] = []

    for column, spec in index_specs.items():
        unknown_column = f"{column}_unknown"
        frame[column] = False
        frame[unknown_column] = False
        asof_index = asof_indexes[column]
        min_members, max_members = member_count_tolerances.get(
            column,
            (
                int(np.floor(spec.expected_member_count * 0.99)),
                int(np.ceil(spec.expected_member_count * 1.01)),
            ),
        )

        for date in dates:
            date = pd.Timestamp(date).normalize()
            row_index = groups[date]
            asof_row = asof_index.get(date)
            if asof_row is None:
                frame.loc[row_index, unknown_column] = True
                quality_records.append(
                    _missing_quality_record(
                        date=date,
                        column=column,
                        spec=spec,
                        min_members=min_members,
                        max_members=max_members,
                        weight_sum_tolerance=weight_sum_tolerance,
                    )
                )
                continue

            members, record = _read_and_validate_asof_members(
                asof_row=asof_row,
                column=column,
                spec=spec,
                index_dir=Path(index_data_root) / spec.directory,
                min_members=min_members,
                max_members=max_members,
                weight_sum_tolerance=weight_sum_tolerance,
            )
            quality_records.append(record)
            if not record["reliable"]:
                frame.loc[row_index, unknown_column] = True
                continue

            frame.loc[row_index, column] = frame.loc[row_index, "stock_code"].isin(members).to_numpy()

    flag_columns = list(index_specs)
    unknown_columns = [f"{column}_unknown" for column in flag_columns]
    row_flag_sum = frame[flag_columns].astype("int8").sum(axis=1)
    overlapping = row_flag_sum.gt(1)
    if overlapping.any():
        sample = _records_with_iso_dates(frame.loc[overlapping, ["date", "stock_code", *flag_columns]].head(10))
        raise ValueError(f"Rows contain overlapping reliable index membership flags: {sample}")

    frame["index_membership_any_unknown"] = frame[unknown_columns].any(axis=1)
    frame["index_membership_all_known"] = ~frame["index_membership_any_unknown"]
    frame["historical_pit_index_membership"] = True

    output_columns = [
        "date",
        "stock_code",
        *flag_columns,
        *unknown_columns,
        "index_membership_any_unknown",
        "index_membership_all_known",
        "historical_pit_index_membership",
    ]
    frame = frame.loc[:, output_columns].sort_values(list(DATE_STOCK_COLUMNS)).reset_index(drop=True)
    quality = pd.DataFrame(quality_records).sort_values(["date", "membership_flag_column"]).reset_index(drop=True)
    _validate_pit_panel(frame, flag_columns=flag_columns, unknown_columns=unknown_columns)
    summary = _build_summary(
        frame,
        quality,
        daily_metrics_dir=Path(daily_metrics_dir),
        index_data_root=Path(index_data_root),
        index_specs=index_specs,
        member_count_tolerances=member_count_tolerances,
        weight_sum_tolerance=weight_sum_tolerance,
    )
    return frame, quality, summary


def write_pit_index_membership_panel(
    *,
    panel: pd.DataFrame,
    quality: pd.DataFrame,
    summary: Mapping[str, Any],
    output_dir: Path,
    archive_path: Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel_csv = output_dir / "pit_index_membership.csv"
    panel_pickle = output_dir / "pit_index_membership.pkl"
    quality_csv = output_dir / "index_membership_quality_by_date.csv"
    manifest_path = output_dir / "manifest.json"
    readme_path = output_dir / "README.md"

    _write_csv_with_iso_dates(panel, panel_csv)
    panel.to_pickle(panel_pickle)
    _write_csv_with_iso_dates(quality, quality_csv)
    readme_path.write_text(_readme_text(), encoding="utf-8")

    manifest = dict(summary)
    manifest["metadata"] = {
        **dict(summary.get("metadata", {})),
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    manifest["outputs"] = {
        "pit_index_membership_csv": _path_for_summary(panel_csv),
        "pit_index_membership_pickle": _path_for_summary(panel_pickle),
        "index_membership_quality_by_date_csv": _path_for_summary(quality_csv),
        "manifest": _path_for_summary(manifest_path),
        "readme": _path_for_summary(readme_path),
    }
    if archive_path is not None:
        manifest["outputs"]["archive"] = _path_for_summary(Path(archive_path))
    manifest["file_checksums"] = {
        path.relative_to(output_dir).as_posix(): {
            "sha256": _sha256(path),
            "bytes": int(path.stat().st_size),
        }
        for path in [panel_csv, panel_pickle, quality_csv, readme_path]
    }
    manifest_path.write_text(json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if archive_path is not None:
        _write_zip_archive(
            archive_path=Path(archive_path),
            package_dir_name=output_dir.name,
            files=[readme_path, quality_csv, manifest_path, panel_csv, panel_pickle],
        )
    return manifest


def _read_daily_stock_universe(
    daily_metrics_dir: Path,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None,
) -> pd.DataFrame:
    index_path = daily_metrics_dir / "daily_metrics.csv"
    if not index_path.is_file():
        raise FileNotFoundError(f"daily_metrics.csv not found: {index_path}")
    index = pd.read_csv(index_path, dtype={"file_path": "string"})
    required = {"date", "file_path"}
    missing = sorted(required.difference(index.columns))
    if missing:
        raise ValueError(f"daily_metrics.csv is missing required columns: {missing}")
    index = index.loc[:, ["date", "file_path"]].copy()
    index["date"] = _parse_dates(index["date"])
    if index["date"].isna().any():
        raise ValueError("daily_metrics.csv contains unparseable date values.")
    index = index[index["date"].ge(pd.Timestamp(start).normalize())]
    if end is not None:
        index = index[index["date"].le(pd.Timestamp(end).normalize())]
    if index.empty:
        raise ValueError("No daily metric dates remain after applying the requested date range.")

    chunks: list[pd.DataFrame] = []
    for row in index.sort_values("date").itertuples(index=False):
        date = pd.Timestamp(row.date).normalize()
        path = daily_metrics_dir / str(row.file_path)
        daily = pd.read_csv(path, usecols=["symbol", "trade_date"], dtype={"symbol": "string"})
        daily["date"] = _parse_dates(daily["trade_date"])
        daily = daily[daily["date"].eq(date)].copy()
        if daily.empty:
            raise ValueError(f"Daily metrics partition {path} has no rows for {date.date().isoformat()}.")
        daily["stock_code"] = daily["symbol"].map(_normalize_stock_code)
        empty_codes = int(daily["stock_code"].eq("").sum())
        if empty_codes:
            raise ValueError(f"Daily metrics partition {path} contains {empty_codes} empty normalized stock codes.")
        duplicate_keys = daily.duplicated(list(DATE_STOCK_COLUMNS))
        if duplicate_keys.any():
            sample = _records_with_iso_dates(daily.loc[duplicate_keys, list(DATE_STOCK_COLUMNS)].head(10))
            raise ValueError(f"Daily metrics partition {path} contains duplicate date-stock rows: {sample}")
        chunks.append(daily.loc[:, list(DATE_STOCK_COLUMNS)])

    frame = pd.concat(chunks, ignore_index=True).sort_values(list(DATE_STOCK_COLUMNS)).reset_index(drop=True)
    if frame.duplicated(list(DATE_STOCK_COLUMNS)).any():
        sample = _records_with_iso_dates(
            frame.loc[frame.duplicated(list(DATE_STOCK_COLUMNS), keep=False), list(DATE_STOCK_COLUMNS)].head(10)
        )
        raise ValueError(f"Daily metrics universe contains duplicate date-stock keys: {sample}")
    return frame


def _read_asof_index(index_dir: Path, column: str, spec: IndexSpec) -> dict[pd.Timestamp, dict[str, Any]]:
    path = index_dir / "constituent_weights_daily_asof.csv"
    if not index_dir.is_dir():
        raise FileNotFoundError(f"Index directory not found for {column}: {index_dir}")
    if not path.is_file():
        raise FileNotFoundError(f"Daily as-of index not found for {column}: {path}")
    frame = pd.read_csv(path, dtype={"file_path": "string", "index_code": "string", "quality_status": "string"})
    required = {
        "date",
        "file_path",
        "rows",
        "index_code",
        "weight_snapshot_date",
        "effective_date",
        "days_since_snapshot",
        "quality_status",
        "weight_sum",
        "duplicate_member_rows",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Daily as-of index {path} is missing required columns: {missing}")
    frame["date"] = _parse_dates(frame["date"])
    if frame["date"].isna().any():
        raise ValueError(f"Daily as-of index {path} contains unparseable date values.")
    if frame.duplicated("date").any():
        sample = _records_with_iso_dates(frame.loc[frame.duplicated("date", keep=False), ["date"]].head(10))
        raise ValueError(f"Daily as-of index {path} contains duplicate dates: {sample}")
    return {
        pd.Timestamp(row.date).normalize(): row._asdict()
        for row in frame.sort_values("date").itertuples(index=False)
    }


def _read_and_validate_asof_members(
    *,
    asof_row: Mapping[str, Any],
    column: str,
    spec: IndexSpec,
    index_dir: Path,
    min_members: int,
    max_members: int,
    weight_sum_tolerance: float,
) -> tuple[set[str], dict[str, Any]]:
    date = pd.Timestamp(asof_row["date"]).normalize()
    partition_path = index_dir / str(asof_row["file_path"])
    if not partition_path.is_file():
        return set(), _quality_record(
            date=date,
            column=column,
            spec=spec,
            asof_row=asof_row,
            min_members=min_members,
            max_members=max_members,
            weight_sum_tolerance=weight_sum_tolerance,
            observed_member_count=0,
            weight_sum_percent=np.nan,
            duplicate_member_rows=np.nan,
            reliable=False,
            unreliable_reasons=["missing_daily_asof_partition"],
            source_file_path=partition_path,
        )

    frame = pd.read_csv(
        partition_path,
        dtype={"member_symbol": "string", "index_code": "string", "quality_status": "string", "weight_unit": "string"},
    )
    required = {"date", "index_code", "member_symbol", "weight", "weight_unit", "quality_status"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Daily as-of partition {partition_path} is missing required columns: {missing}")
    frame["date"] = _parse_dates(frame["date"])
    wrong_date = frame["date"].ne(date)
    if wrong_date.any():
        sample = _records_with_iso_dates(frame.loc[wrong_date, ["date", "member_symbol"]].head(10))
        raise ValueError(f"Daily as-of partition {partition_path} contains rows for unexpected dates: {sample}")

    members = frame["member_symbol"].map(_normalize_stock_code)
    valid_members = members[members.ne("")]
    duplicate_member_rows = int(valid_members.duplicated().sum())
    observed_member_count = int(valid_members.nunique())
    numeric_weight = pd.to_numeric(frame["weight"], errors="coerce")
    weight_sum_percent = float(numeric_weight.sum(skipna=True))
    weight_sum_fraction = weight_sum_percent / 100.0
    source_quality = _clean_string(asof_row.get("quality_status"))
    partition_quality_values = sorted(frame["quality_status"].dropna().astype(str).str.strip().unique().tolist())

    reasons: list[str] = []
    if source_quality != "complete":
        reasons.append(f"source_quality_status={source_quality or '<missing>'}")
    if partition_quality_values != ["complete"]:
        reasons.append(f"partition_quality_status={partition_quality_values}")
    if duplicate_member_rows != 0:
        reasons.append(f"duplicate_member_rows={duplicate_member_rows}")
    if observed_member_count < min_members or observed_member_count > max_members:
        reasons.append(f"observed_member_count={observed_member_count}, expected_range=[{min_members},{max_members}]")
    if not np.isfinite(weight_sum_fraction) or abs(weight_sum_fraction - 1.0) > weight_sum_tolerance:
        reasons.append(f"weight_sum_fraction={weight_sum_fraction:.12g}")

    return set(valid_members), _quality_record(
        date=date,
        column=column,
        spec=spec,
        asof_row=asof_row,
        min_members=min_members,
        max_members=max_members,
        weight_sum_tolerance=weight_sum_tolerance,
        observed_member_count=observed_member_count,
        weight_sum_percent=weight_sum_percent,
        duplicate_member_rows=duplicate_member_rows,
        reliable=not reasons,
        unreliable_reasons=reasons,
        source_file_path=partition_path,
    )


def _missing_quality_record(
    *,
    date: pd.Timestamp,
    column: str,
    spec: IndexSpec,
    min_members: int,
    max_members: int,
    weight_sum_tolerance: float,
) -> dict[str, Any]:
    return _quality_record(
        date=date,
        column=column,
        spec=spec,
        asof_row={
            "index_code": spec.index_code,
            "quality_status": pd.NA,
            "weight_snapshot_date": pd.NaT,
            "effective_date": pd.NaT,
            "days_since_snapshot": pd.NA,
        },
        min_members=min_members,
        max_members=max_members,
        weight_sum_tolerance=weight_sum_tolerance,
        observed_member_count=0,
        weight_sum_percent=np.nan,
        duplicate_member_rows=np.nan,
        reliable=False,
        unreliable_reasons=["missing_daily_asof_index_row"],
        source_file_path=None,
    )


def _quality_record(
    *,
    date: pd.Timestamp,
    column: str,
    spec: IndexSpec,
    asof_row: Mapping[str, Any],
    min_members: int,
    max_members: int,
    weight_sum_tolerance: float,
    observed_member_count: int,
    weight_sum_percent: float,
    duplicate_member_rows: int | float,
    reliable: bool,
    unreliable_reasons: Sequence[str],
    source_file_path: Path | None,
) -> dict[str, Any]:
    return {
        "date": date,
        "index_name": spec.index_name,
        "index_code": _clean_string(asof_row.get("index_code")) or spec.index_code,
        "membership_flag_column": column,
        "unknown_flag_column": f"{column}_unknown",
        "expected_member_count": spec.expected_member_count,
        "member_count_min": min_members,
        "member_count_max": max_members,
        "observed_member_count": int(observed_member_count),
        "weight_sum_percent": float(weight_sum_percent) if np.isfinite(weight_sum_percent) else np.nan,
        "weight_sum_fraction": float(weight_sum_percent) / 100.0 if np.isfinite(weight_sum_percent) else np.nan,
        "weight_sum_tolerance": float(weight_sum_tolerance),
        "duplicate_member_rows": duplicate_member_rows,
        "source_quality_status": _clean_string(asof_row.get("quality_status")),
        "weight_snapshot_date": _date_or_none(asof_row.get("weight_snapshot_date")),
        "effective_date": _date_or_none(asof_row.get("effective_date")),
        "days_since_snapshot": _int_or_none(asof_row.get("days_since_snapshot")),
        "reliable": bool(reliable),
        "unreliable_reason": ";".join(unreliable_reasons),
        "source_file_path": _path_for_summary(source_file_path) if source_file_path is not None else "",
    }


def _validate_pit_panel(
    panel: pd.DataFrame,
    *,
    flag_columns: Sequence[str],
    unknown_columns: Sequence[str],
) -> None:
    if panel.empty:
        raise ValueError("PIT index membership panel is empty.")
    if panel.duplicated(list(DATE_STOCK_COLUMNS)).any():
        sample = _records_with_iso_dates(
            panel.loc[panel.duplicated(list(DATE_STOCK_COLUMNS), keep=False), list(DATE_STOCK_COLUMNS)].head(10)
        )
        raise ValueError(f"PIT index membership panel contains duplicate date-stock keys: {sample}")
    for column in [*flag_columns, *unknown_columns, "index_membership_any_unknown", "index_membership_all_known", "historical_pit_index_membership"]:
        values = panel[column]
        invalid = values.notna() & ~values.isin([True, False])
        if invalid.any():
            sample = _records_with_iso_dates(panel.loc[invalid, list(DATE_STOCK_COLUMNS)].head(10))
            raise ValueError(f"PIT index membership column {column!r} contains non-boolean values: {sample}")
    for flag_column, unknown_column in zip(flag_columns, unknown_columns, strict=True):
        invalid = panel[flag_column] & panel[unknown_column]
        if invalid.any():
            sample = _records_with_iso_dates(panel.loc[invalid, list(DATE_STOCK_COLUMNS)].head(10))
            raise ValueError(f"{flag_column} cannot be true when {unknown_column} is true: {sample}")


def _build_summary(
    panel: pd.DataFrame,
    quality: pd.DataFrame,
    *,
    daily_metrics_dir: Path,
    index_data_root: Path,
    index_specs: Mapping[str, IndexSpec],
    member_count_tolerances: Mapping[str, tuple[int, int]],
    weight_sum_tolerance: float,
) -> dict[str, Any]:
    flag_columns = list(index_specs)
    unknown_columns = [f"{column}_unknown" for column in flag_columns]
    unknown_row_counts = {column: int(panel[f"{column}_unknown"].sum()) for column in flag_columns}
    member_row_counts = {column: int(panel[column].sum()) for column in flag_columns}
    reliable_dates = quality.groupby("membership_flag_column")["reliable"].sum().astype(int).to_dict()
    total_dates = quality.groupby("membership_flag_column")["date"].nunique().astype(int).to_dict()
    unreliable_dates = {
        column: int(total_dates.get(column, 0) - reliable_dates.get(column, 0))
        for column in flag_columns
    }
    return {
        "metadata": {
            "artifact_version": "pit_index_membership_2017_v1",
            "description": "Date-stock PIT CSI membership panel with explicit unknown flags for missing or unreliable index-date sources.",
            "historical_pit_index_membership": True,
            "index_membership_source_mode": "historical_pit_daily_asof_with_unknown_flags",
            "unknown_flags_present": True,
            "unknown_semantics": "is_csi*=false and is_csi*_unknown=true means membership is unknown, not confirmed non-member",
        },
        "row_count": int(len(panel)),
        "date_count": int(panel["date"].nunique()),
        "stock_count": int(panel["stock_code"].nunique()),
        "date_min": _date_to_string(panel["date"].min()),
        "date_max": _date_to_string(panel["date"].max()),
        "index_membership_columns": [*flag_columns, *unknown_columns, "index_membership_any_unknown", "index_membership_all_known", "historical_pit_index_membership"],
        "membership_flag_columns": flag_columns,
        "unknown_flag_columns": unknown_columns,
        "member_row_counts_by_index": member_row_counts,
        "unknown_row_counts_by_index": unknown_row_counts,
        "reliable_date_counts_by_index": {column: int(reliable_dates.get(column, 0)) for column in flag_columns},
        "unreliable_date_counts_by_index": unreliable_dates,
        "quality_row_count": int(len(quality)),
        "weight_sum_tolerance": float(weight_sum_tolerance),
        "member_count_tolerances": {
            column: list(member_count_tolerances.get(column, (spec.expected_member_count, spec.expected_member_count)))
            for column, spec in index_specs.items()
        },
        "source_policy": {
            "daily_metrics": {
                "path": _path_for_summary(daily_metrics_dir),
                "dataset": "finfact_io.generated_data.ashare_daily_metrics",
            },
            "index_data_root": {
                "path": _path_for_summary(index_data_root),
                "datasets": {
                    column: {
                        "directory": spec.directory,
                        "index_code": spec.index_code,
                        "index_name": spec.index_name,
                        "expected_member_count": spec.expected_member_count,
                    }
                    for column, spec in index_specs.items()
                },
                "asof_source": "constituent_weights_daily_asof.csv and daily partition files",
            },
        },
        "notes": [
            "Reliable index-date membership requires complete source quality, no duplicate members, member count inside tolerance, and weight sum close to 1 after converting percent to fraction.",
            "Missing or unreliable index-date sources set the corresponding is_csi*_unknown flag to true for every stock on that date.",
            "Membership flags remain boolean; unknown is represented only by separate unknown flag columns.",
        ],
    }


def _parse_dates(values: Any) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.normalize()


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


def _clean_string(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _date_or_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp).date().isoformat()


def _int_or_none(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _date_to_string(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def _write_csv_with_iso_dates(frame: pd.DataFrame, path: Path) -> None:
    output = frame.copy()
    for column in output.columns:
        if is_datetime64_any_dtype(output[column]):
            output[column] = pd.to_datetime(output[column]).dt.date.astype(str)
    output.to_csv(path, index=False, encoding="utf-8-sig")


def _records_with_iso_dates(frame: pd.DataFrame) -> list[dict[str, Any]]:
    output = frame.copy()
    for column in output.columns:
        if is_datetime64_any_dtype(output[column]):
            output[column] = pd.to_datetime(output[column]).dt.date.astype(str)
    return output.to_dict("records")


def _path_for_summary(path: Path | None) -> str:
    if path is None:
        return ""
    path = Path(path)
    if not path.is_absolute():
        return path.as_posix()
    cwd = Path.cwd().resolve()
    resolved = path.resolve(strict=False)
    try:
        return resolved.relative_to(cwd, walk_up=True).as_posix()
    except ValueError:
        return f"<external>/{resolved.name}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _write_zip_archive(*, archive_path: Path, package_dir_name: str, files: Sequence[Path]) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=f"{package_dir_name}/{path.name}")


def _readme_text() -> str:
    return """# PIT Index Membership Panel

This package contains date-stock CSI membership flags generated from daily as-of constituent weight files.

`is_csi*` columns are boolean membership flags only when the corresponding index-date source is reliable.
`is_csi*_unknown=true` means that index membership is unknown for that index-date and must not be interpreted as non-membership.
"""


if __name__ == "__main__":
    main()
