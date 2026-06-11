from __future__ import annotations

import argparse
import json
import re
import zipfile
from bisect import bisect_right
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


COL_TS_CODE = "TS\u4ee3\u7801"
COL_STOCK_CODE = "\u80a1\u7968\u4ee3\u7801"
COL_STOCK_INDUSTRY = "\u6240\u5c5e\u884c\u4e1a"
COL_MARKET_TYPE = "\u5e02\u573a\u7c7b\u578b"
COL_EXCHANGE = "\u4ea4\u6613\u6240\u4ee3\u7801"
COL_TRADE_DATE = "\u4ea4\u6613\u65e5\u671f"
COL_MARKET_CAP = "\u603b\u5e02\u503c(\u4e07\u5143)"
COL_AMOUNT = "\u6210\u4ea4\u989d(\u5343\u5143)"
COL_VOLUME = "\u6210\u4ea4\u91cf(\u624b)"
COL_TURNOVER = "\u6362\u624b\u7387"
COL_INDEX_COMPONENT = "\u6210\u5206\u80a1\u7968\u4ee3\u7801"

SSE_INDEX_DIR = "\u4e0a\u4ea4\u6240\u6307\u6570\u6210\u5206"
CSI_INDEX_DIR = "\u4e2d\u8bc1\u6307\u6570\u6210\u5206"

INDEX_SPECS = {
    "is_csi300": (SSE_INDEX_DIR, "000300.SH.csv"),
    "is_csi500": (SSE_INDEX_DIR, "000905.SH.csv"),
    "is_csi1000": (SSE_INDEX_DIR, "000852.SH.csv"),
    "is_csi2000": (CSI_INDEX_DIR, "932000.CSI.csv"),
}

UNKNOWN_BOARD = "UNKNOWN_BOARD"
CANONICAL_BOARD_CODES = {
    "CHINEXT",
    "STAR",
    "BSE",
    "SSE_MAIN",
    "SZSE_MAIN",
    UNKNOWN_BOARD,
}

BOARD_CODE_ALIASES = {
    "CHINEXT": "CHINEXT",
    "CHI_NEXT": "CHINEXT",
    "CHINEXT_BOARD": "CHINEXT",
    "创业板": "CHINEXT",
    "STAR": "STAR",
    "STAR_BOARD": "STAR",
    "科创板": "STAR",
    "BSE": "BSE",
    "BJSE": "BSE",
    "北交所": "BSE",
    "SSE_MAIN": "SSE_MAIN",
    "SSE_MAIN_BOARD": "SSE_MAIN",
    "SZSE_MAIN": "SZSE_MAIN",
    "SZSE_MAIN_BOARD": "SZSE_MAIN",
}

EXCHANGE_MARKET_BOARD_MAP = {
    ("SZSE", "创业板"): "CHINEXT",
    ("SSE", "科创板"): "STAR",
    ("BSE", "北交所"): "BSE",
    ("BJSE", "北交所"): "BSE",
    ("SSE", "主板"): "SSE_MAIN",
    ("SZSE", "主板"): "SZSE_MAIN",
    ("SZSE", "中小板"): "SZSE_MAIN",
}

PREFIX_BOARD_MAP = (
    (("688", "689"), "STAR"),
    (("300", "301", "302"), "CHINEXT"),
    (("920", "830", "831", "832", "833", "834", "835", "836", "837", "838", "839"), "BSE"),
    (("600", "601", "603", "605"), "SSE_MAIN"),
    (("000", "001", "002", "003"), "SZSE_MAIN"),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build daily exposure rows for return_y neutralization."
    )
    parser.add_argument("--return-y", required=True, type=Path)
    parser.add_argument("--daily-data-root", required=True, type=Path)
    parser.add_argument("--index-data-root", required=True, type=Path)
    parser.add_argument("--industry-board", required=True, type=Path)
    parser.add_argument(
        "--listing-board-reference",
        type=Path,
        default=None,
        help=(
            "Optional canonical listing board snapshot. When supplied, "
            "listing_board_code is used before local fallback inference."
        ),
    )
    parser.add_argument("--output-exposures", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--start-date", default="2023-08-11")
    parser.add_argument("--min-rolling-observations", default=5, type=int)
    args = parser.parse_args()

    return_y = pd.read_pickle(args.return_y)
    if not isinstance(return_y, pd.DataFrame):
        raise TypeError(f"Expected return_y DataFrame, got {type(return_y)!r}")

    requested_start = pd.Timestamp(args.start_date).normalize()
    return_dates = pd.DatetimeIndex(pd.to_datetime(return_y.index)).normalize()
    stock_codes = sorted({_normalize_stock_code(column) for column in return_y.columns})
    stock_codes = [code for code in stock_codes if code]

    index_snapshots = {
        column: _read_index_snapshots(args.index_data_root / directory, member_name)
        for column, (directory, member_name) in INDEX_SPECS.items()
    }
    first_complete_snapshot = max(
        min(snapshots) for snapshots in index_snapshots.values() if snapshots
    )
    effective_start = max(requested_start, first_complete_snapshot)
    required_dates = pd.DatetimeIndex(
        sorted(date for date in return_dates.unique() if date >= effective_start)
    )
    if required_dates.empty:
        raise ValueError("No return_y dates remain after applying start date and index snapshots.")

    metadata = _build_static_metadata(
        args.daily_data_root / "\u80a1\u7968\u5217\u8868.csv",
        args.industry_board,
        stock_codes,
        listing_board_reference_path=args.listing_board_reference,
    )
    daily = _read_daily_indicator_rows(
        args.daily_data_root / "\u6bcf\u65e5\u6307\u6807.zip",
        stock_codes,
        required_dates,
        min_rolling_observations=args.min_rolling_observations,
    )
    if daily.empty:
        raise ValueError("No daily indicator rows matched return_y dates and stock universe.")

    exposures = daily.merge(metadata, on="stock_code", how="left", validate="many_to_one")
    exposures["industry"] = exposures["industry"].fillna("unknown_industry")
    exposures["board"] = exposures["board"].fillna(UNKNOWN_BOARD)
    exposures["board_source"] = exposures["board_source"].fillna("unknown")
    exposures = _attach_index_flags(exposures, index_snapshots)

    output_columns = [
        "date",
        "stock_code",
        "industry",
        "board",
        "is_csi300",
        "is_csi500",
        "is_csi1000",
        "is_csi2000",
        "market_cap",
        "amount_k",
        "turnover",
        "logADV20",
        "logAmount20",
        "turnover20",
    ]
    exposures = exposures.sort_values(["date", "stock_code"]).reset_index(drop=True)

    args.output_exposures.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    exposures[output_columns].to_pickle(args.output_exposures)

    latest_board_counts = _latest_category_counts(exposures, "board")
    singleton_board_summary = _singleton_category_summary(exposures, "board")
    unexpected_board_values = sorted(
        str(value)
        for value in exposures["board"].dropna().unique()
        if str(value) not in CANONICAL_BOARD_CODES
    )
    unknown_board = exposures["board"].eq(UNKNOWN_BOARD)
    summary = {
        "return_y_shape": {"rows": int(return_y.shape[0]), "columns": int(return_y.shape[1])},
        "requested_start_date": requested_start.date().isoformat(),
        "effective_start_date": effective_start.date().isoformat(),
        "date_min": required_dates.min().date().isoformat(),
        "date_max": required_dates.max().date().isoformat(),
        "date_count": int(len(required_dates)),
        "stock_count_requested": int(len(stock_codes)),
        "stock_count_with_daily_rows": int(exposures["stock_code"].nunique()),
        "exposure_rows": int(len(exposures)),
        "missing_industry_rows": int(exposures["industry"].eq("unknown_industry").sum()),
        "missing_board_rows": int(unknown_board.sum()),
        "unknown_board_rows": int(unknown_board.sum()),
        "unknown_board_symbols": sorted(exposures.loc[unknown_board, "stock_code"].unique().tolist()),
        "listing_board_reference_used": args.listing_board_reference is not None,
        "board_source_counts": {
            str(source): int(count)
            for source, count in exposures["board_source"].value_counts(dropna=False).items()
        },
        "board_category_counts_latest": latest_board_counts,
        "singleton_board_categories_latest": [
            category for category, count in latest_board_counts.items() if count == 1
        ],
        "singleton_board_category_date_count": singleton_board_summary["date_count"],
        "singleton_board_categories_by_date_sample": singleton_board_summary["sample"],
        "unexpected_board_values": unexpected_board_values,
        "index_snapshot_ranges": {
            column: {
                "snapshot_count": int(len(snapshots)),
                "date_min": min(snapshots).date().isoformat(),
                "date_max": max(snapshots).date().isoformat(),
            }
            for column, snapshots in index_snapshots.items()
        },
        "index_member_row_counts": {
            column: int(exposures[column].sum()) for column in INDEX_SPECS
        },
    }
    args.output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


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


def _build_static_metadata(
    stock_list_path: Path,
    industry_board_path: Path,
    stock_codes: list[str],
    *,
    listing_board_reference_path: Path | None = None,
) -> pd.DataFrame:
    base = pd.DataFrame({"stock_code": stock_codes})

    stock_list = pd.read_csv(stock_list_path, encoding="utf-8-sig", dtype=str)
    stock_list["stock_code"] = stock_list[COL_TS_CODE].map(_normalize_stock_code)
    stock_list = stock_list.drop_duplicates("stock_code", keep="last")
    stock_list = stock_list[
        ["stock_code", COL_STOCK_INDUSTRY, COL_MARKET_TYPE, COL_EXCHANGE]
    ].rename(columns={COL_STOCK_INDUSTRY: "stock_list_industry"})

    industry = pd.read_csv(
        industry_board_path,
        sep="\t",
        header=None,
        names=["industry_code", "industry", "stock_code", "stock_name"],
        dtype=str,
        encoding="gbk",
    )
    industry["stock_code"] = industry["stock_code"].map(_normalize_stock_code)
    industry = industry.drop_duplicates("stock_code", keep="last")[["stock_code", "industry"]]

    meta = base.merge(stock_list, on="stock_code", how="left", validate="one_to_one")
    meta = meta.merge(industry, on="stock_code", how="left", validate="one_to_one")
    meta["industry"] = meta["industry"].fillna(meta["stock_list_industry"])

    listing_board_reference = _read_listing_board_reference(listing_board_reference_path)
    if not listing_board_reference.empty:
        meta = meta.merge(
            listing_board_reference,
            on="stock_code",
            how="left",
            validate="one_to_one",
        )
    else:
        meta["reference_board"] = pd.NA

    inferred = [
        _infer_board_with_source(code, market_type, exchange)
        for code, market_type, exchange in zip(
            meta["stock_code"],
            meta[COL_MARKET_TYPE],
            meta[COL_EXCHANGE],
            strict=True,
        )
    ]
    meta["inferred_board"] = [board for board, _source in inferred]
    meta["inferred_board_source"] = [source for _board, source in inferred]
    has_reference = meta["reference_board"].notna() & meta["reference_board"].ne("")
    meta["board"] = meta["reference_board"].where(has_reference, meta["inferred_board"])
    meta["board_source"] = np.where(
        has_reference,
        "listing_board_reference",
        meta["inferred_board_source"],
    )
    return meta[["stock_code", "industry", "board", "board_source"]]


def _infer_board(code: str, market_type: Any, exchange: Any) -> str:
    board, _source = _infer_board_with_source(code, market_type, exchange)
    return board


def _infer_board_with_source(code: str, market_type: Any, exchange: Any) -> tuple[str, str]:
    stock_code = _normalize_stock_code(code)
    market_text = "" if pd.isna(market_type) else str(market_type).strip()
    exchange_text = _normalize_exchange_code(exchange)

    mapped = EXCHANGE_MARKET_BOARD_MAP.get((exchange_text, market_text))
    if mapped is not None:
        return mapped, "exchange_market_mapping"

    alias = _normalize_board_code(market_text)
    if alias and alias != UNKNOWN_BOARD:
        return alias, "market_type_alias"

    for prefixes, board in PREFIX_BOARD_MAP:
        if stock_code.startswith(prefixes):
            return board, "prefix_fallback"

    return UNKNOWN_BOARD, "unknown"


def _read_listing_board_reference(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["stock_code", "reference_board"])

    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    stock_column = _first_existing_column(
        frame,
        ["stock_code", "ts_code", "TS\u4ee3\u7801", COL_TS_CODE, COL_STOCK_CODE],
    )
    if stock_column is None:
        raise ValueError(
            "Listing board reference must contain stock_code, ts_code, TS\u4ee3\u7801, "
            "or \u80a1\u7968\u4ee3\u7801."
        )

    board_code_column = _first_existing_column(
        frame,
        ["listing_board_code", "board_code"],
    )
    board_name_column = _first_existing_column(
        frame,
        ["listing_board", "board", COL_MARKET_TYPE],
    )
    exchange_column = _first_existing_column(frame, ["exchange", "exchange_code", COL_EXCHANGE])
    if board_code_column is None and board_name_column is None:
        raise ValueError(
            "Listing board reference must contain listing_board_code or listing_board."
        )

    output = pd.DataFrame({"stock_code": frame[stock_column].map(_normalize_stock_code)})
    boards: list[str] = []
    for row_index, stock_code in output["stock_code"].items():
        exchange_value = frame.at[row_index, exchange_column] if exchange_column is not None else None
        board = ""
        if board_code_column is not None:
            board = _normalize_reference_board_code(frame.at[row_index, board_code_column], exchange_value)
        if board == "" and board_name_column is not None:
            board_name = frame.at[row_index, board_name_column]
            board = _normalize_reference_board_code(board_name, exchange_value)
            if board == "" and exchange_column is not None:
                board = _infer_board(stock_code, board_name, exchange_value)
        boards.append(board)
    output["reference_board"] = boards

    output = output[output["stock_code"].ne("")]
    output = output[output["reference_board"].ne("")]
    output = output.drop_duplicates("stock_code", keep="last")
    invalid = sorted(
        value
        for value in output["reference_board"].dropna().unique()
        if value not in CANONICAL_BOARD_CODES
    )
    if invalid:
        raise ValueError(f"Unexpected listing board codes in reference: {invalid}")
    return output


def _normalize_board_code(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    alias = BOARD_CODE_ALIASES.get(text)
    if alias is not None:
        return alias
    normalized = re.sub(r"[\s\-]+", "_", text.upper())
    return BOARD_CODE_ALIASES.get(normalized, normalized)


def _normalize_reference_board_code(value: Any, exchange: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.upper() == "MAIN" or text == "\u4e3b\u677f":
        return _main_board_for_exchange(exchange)
    board = _normalize_board_code(text)
    return board if board in CANONICAL_BOARD_CODES else ""


def _main_board_for_exchange(exchange: Any) -> str:
    exchange_code = _normalize_exchange_code(exchange)
    if exchange_code == "SSE":
        return "SSE_MAIN"
    if exchange_code == "SZSE":
        return "SZSE_MAIN"
    return ""


def _normalize_exchange_code(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().upper()
    if text in {"SH", "SHSE"}:
        return "SSE"
    if text in {"SZ"}:
        return "SZSE"
    if text in {"BJ", "BJSE"}:
        return "BSE"
    return text


def _first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized_columns = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        match = normalized_columns.get(str(candidate).strip().lower())
        if match is not None:
            return str(match)
    return None


def _latest_category_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty:
        return {}
    latest_date = frame["date"].max()
    counts = frame.loc[frame["date"].eq(latest_date), column].value_counts(dropna=False)
    return {str(category): int(count) for category, count in counts.sort_index().items()}


def _singleton_category_summary(
    frame: pd.DataFrame,
    column: str,
    *,
    sample_size: int = 20,
) -> dict[str, Any]:
    if frame.empty:
        return {"date_count": 0, "sample": []}

    counts = frame.groupby(["date", column], dropna=False).size()
    singleton_counts = counts[counts.eq(1)]
    if singleton_counts.empty:
        return {"date_count": 0, "sample": []}

    singleton_frame = singleton_counts.reset_index(name="count").sort_values(["date", column])
    sample = [
        {
            "date": pd.Timestamp(row["date"]).date().isoformat(),
            "category": str(row[column]),
        }
        for row in singleton_frame.head(sample_size).to_dict("records")
    ]
    return {
        "date_count": int(singleton_frame["date"].nunique()),
        "sample": sample,
    }


def _read_daily_indicator_rows(
    daily_zip_path: Path,
    stock_codes: list[str],
    required_dates: pd.DatetimeIndex,
    *,
    min_rolling_observations: int,
) -> pd.DataFrame:
    required_date_ints = {int(date.strftime("%Y%m%d")) for date in required_dates}
    pre_start = required_dates.min() - pd.Timedelta(days=90)
    max_date_int = int(required_dates.max().strftime("%Y%m%d"))
    pre_start_int = int(pre_start.strftime("%Y%m%d"))
    chunks: list[pd.DataFrame] = []
    missing_entries: list[str] = []
    usecols = [
        COL_STOCK_CODE,
        COL_TRADE_DATE,
        COL_MARKET_CAP,
        COL_AMOUNT,
        COL_VOLUME,
        COL_TURNOVER,
    ]

    with zipfile.ZipFile(daily_zip_path) as zf:
        entry_by_code = {
            _normalize_stock_code(Path(info.filename).stem): info.filename
            for info in zf.infolist()
            if info.filename.lower().endswith(".csv")
        }
        for offset, code in enumerate(stock_codes, start=1):
            entry = entry_by_code.get(code)
            if entry is None:
                missing_entries.append(code)
                continue
            with zf.open(entry) as file:
                frame = pd.read_csv(
                    file,
                    encoding="utf-8-sig",
                    usecols=usecols,
                    dtype={COL_STOCK_CODE: str},
                )
            frame[COL_TRADE_DATE] = pd.to_numeric(frame[COL_TRADE_DATE], errors="coerce")
            frame = frame[
                (frame[COL_TRADE_DATE] >= pre_start_int)
                & (frame[COL_TRADE_DATE] <= max_date_int)
            ].copy()
            if frame.empty:
                continue
            frame["stock_code"] = code
            frame["date"] = pd.to_datetime(
                frame[COL_TRADE_DATE].astype("Int64").astype(str),
                format="%Y%m%d",
                errors="coerce",
            ).dt.normalize()
            frame = frame.sort_values("date")
            for source, target in [
                (COL_MARKET_CAP, "market_cap"),
                (COL_AMOUNT, "amount_k"),
                (COL_VOLUME, "volume_hand"),
                (COL_TURNOVER, "turnover"),
            ]:
                frame[target] = pd.to_numeric(frame[source], errors="coerce")

            frame["amount20"] = (
                frame["amount_k"].where(frame["amount_k"] > 0.0)
                .rolling(20, min_periods=min_rolling_observations)
                .mean()
            )
            frame["adv20"] = (
                frame["volume_hand"].where(frame["volume_hand"] > 0.0)
                .rolling(20, min_periods=min_rolling_observations)
                .mean()
            )
            frame["turnover20"] = (
                frame["turnover"].where(frame["turnover"] >= 0.0)
                .rolling(20, min_periods=min_rolling_observations)
                .mean()
            )
            frame["logAmount20"] = np.log1p(frame["amount20"])
            frame["logADV20"] = np.log1p(frame["adv20"])
            frame = frame[frame[COL_TRADE_DATE].isin(required_date_ints)]
            if not frame.empty:
                chunks.append(
                    frame[
                        [
                            "date",
                            "stock_code",
                            "market_cap",
                            "amount_k",
                            "turnover",
                            "logADV20",
                            "logAmount20",
                            "turnover20",
                        ]
                    ]
                )
            if offset % 500 == 0:
                print(f"read daily indicators for {offset}/{len(stock_codes)} stocks")

    if missing_entries:
        print(f"missing daily indicator files for {len(missing_entries)} stocks")
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _read_index_snapshots(index_dir: Path, member_name: str) -> dict[pd.Timestamp, set[str]]:
    snapshots: dict[pd.Timestamp, set[str]] = {}
    for zip_path in sorted(index_dir.glob("*.zip")):
        match = re.search(r"(\d{8})", zip_path.name)
        if match is None:
            continue
        snapshot_date = pd.Timestamp(match.group(1)).normalize()
        with zipfile.ZipFile(zip_path) as zf:
            if member_name not in zf.namelist():
                continue
            with zf.open(member_name) as file:
                frame = pd.read_csv(
                    file,
                    encoding="utf-8-sig",
                    usecols=[COL_INDEX_COMPONENT],
                    dtype={COL_INDEX_COMPONENT: str},
                )
        members = {
            code
            for code in frame[COL_INDEX_COMPONENT].map(_normalize_stock_code)
            if code
        }
        snapshots[snapshot_date] = members
    if not snapshots:
        raise ValueError(f"No index snapshots found for {member_name} under {index_dir}")
    return snapshots


def _attach_index_flags(
    exposures: pd.DataFrame,
    index_snapshots: dict[str, dict[pd.Timestamp, set[str]]],
) -> pd.DataFrame:
    frame = exposures.copy()
    dates = sorted(pd.Timestamp(date).normalize() for date in frame["date"].unique())
    groups = frame.groupby("date", sort=False).groups
    for column, snapshots in index_snapshots.items():
        snapshot_dates = sorted(snapshots)
        frame[column] = np.int8(0)
        for date in dates:
            position = bisect_right(snapshot_dates, date) - 1
            if position < 0:
                continue
            members = snapshots[snapshot_dates[position]]
            row_index = groups[date]
            frame.loc[row_index, column] = (
                frame.loc[row_index, "stock_code"].isin(members).astype(np.int8).to_numpy()
            )
    return frame


if __name__ == "__main__":
    main()
