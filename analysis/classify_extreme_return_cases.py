from __future__ import annotations

import argparse
import json
import math
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def read_stock_metadata(root: Path) -> pd.DataFrame:
    frames = []
    for filename in ["股票列表.csv", "退市股票列表.csv"]:
        path = root / filename
        if path.exists():
            frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
            frame["metadata_source"] = filename
            frames.append(frame)
    meta = pd.concat(frames, ignore_index=True)
    meta = meta.drop_duplicates(subset=["股票代码"], keep="first")
    for col in ["上市日期", "退市日期"]:
        meta[col] = pd.to_datetime(meta[col], format="%Y%m%d", errors="coerce").dt.date
    return meta.set_index("股票代码", drop=False)


def read_trading_calendar(root: Path) -> pd.DatetimeIndex:
    path = root / "交易日历.csv"
    cal = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    date_col = "日期"
    trade_col = "是否交易"
    trading = cal[cal[trade_col].astype(str).isin(["1", "True", "true", "是", "交易"])]
    return pd.DatetimeIndex(pd.to_datetime(trading[date_col], errors="coerce").dropna()).sort_values()


def trading_days_between(calendar: pd.DatetimeIndex, start: date | None, end: date | None) -> int | None:
    if start is None or end is None:
        return None
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if end_ts <= start_ts:
        return 0
    return int(((calendar > start_ts) & (calendar < end_ts)).sum())


def build_member_map(zip_path: Path) -> dict[str, str]:
    with zipfile.ZipFile(zip_path) as zf:
        return {name[:6]: name for name in zf.namelist() if name.endswith(".csv") and len(name) >= 10}


def read_daily_history(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    df = pd.read_csv(zf.open(member), encoding="utf-8-sig")
    df["交易日期"] = pd.to_datetime(df["交易日期"].astype(str), format="%Y%m%d", errors="coerce").dt.date
    df = df.sort_values("交易日期").reset_index(drop=True)
    df["raw_close_ret"] = df["收盘价"].pct_change()
    return df


def first_non_null(values: list[Any]) -> Any:
    for value in values:
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            return value
    return None


def classify_row(row: dict[str, Any]) -> tuple[str, str, str]:
    flags: list[str] = []

    if not row["has_local_daily_history"]:
        return "missing_local_daily_history", "High", "No local daily history member matched this stock code."

    if not row["has_base_or_future_daily_record"]:
        return (
            "pre_listing_or_no_trading_record_on_label_date",
            "High",
            "Label date has no local trading record; event likely shifted before first trade or generated from a different calendar.",
        )

    days_to_listing = row.get("calendar_days_to_listing")
    trading_days_since_first = row.get("trading_days_since_first_record")
    event_gap_before = row.get("event_gap_before_market_days")
    event_raw_pct = row.get("event_raw_pct_change")
    event_is_first_record = row.get("event_is_first_record")
    event_date = row.get("event_date")
    label_date = row.get("label_date")
    one_day_captures = row.get("event_raw_pct_captures_label")
    delist_date = row.get("delist_date")

    if event_is_first_record or (trading_days_since_first is not None and trading_days_since_first <= 5):
        flags.append("new_listing_first_days")
    elif trading_days_since_first is not None and trading_days_since_first <= 60:
        flags.append("recent_listing_first_60_trading_days")

    if days_to_listing is not None and days_to_listing > 0 and days_to_listing <= 10:
        flags.append("label_shifted_before_listing_event")

    if delist_date is not None and not pd.isna(delist_date) and event_date is not None:
        days_to_delist = (delist_date - event_date).days
        if -5 <= days_to_delist <= 90:
            flags.append("delisting_period_or_pre_delisting_trade")

    if event_gap_before is not None and event_gap_before >= 20:
        flags.append("resumption_after_long_suspension")
    elif event_gap_before is not None and event_gap_before >= 5:
        flags.append("resumption_after_short_suspension")

    raw_close_ret = row.get("event_raw_close_ret")
    raw_cum = row.get("raw_cum_from_base_or_next_to_event")
    local_raw_jump = False
    if raw_close_ret is not None and abs(raw_close_ret) >= 0.5:
        local_raw_jump = True
    if raw_cum is not None and abs(raw_cum) >= 0.5:
        local_raw_jump = True

    official_pct_abs = abs(event_raw_pct) if event_raw_pct is not None else None
    label_like_raw_discontinuity = False
    if raw_close_ret is not None and abs(raw_close_ret) >= 0.4:
        label_like_raw_discontinuity = True
    if raw_cum is not None and abs(raw_cum) >= 0.4:
        label_like_raw_discontinuity = True

    if (local_raw_jump or label_like_raw_discontinuity) and official_pct_abs is not None and official_pct_abs < 20:
        flags.append("unadjusted_price_discontinuity_ex_rights_or_corporate_action")
    elif event_raw_pct is not None and abs(event_raw_pct) >= 50:
        flags.append("single_day_raw_price_jump_over_50pct")
    elif event_raw_pct is not None and abs(event_raw_pct) >= 20:
        flags.append("single_day_raw_price_jump_20_to_50pct")

    if one_day_captures:
        flags.append("label_matches_local_raw_move")

    if not flags:
        return (
            "unexplained_by_local_daily_metadata",
            "High",
            "Extreme label is not explained by local listing date, suspension gap, or nearby raw one-day price move.",
        )

    if "delisting_period_or_pre_delisting_trade" in flags and (
        "single_day_raw_price_jump_over_50pct" in flags
        or "resumption_after_long_suspension" in flags
        or "resumption_after_short_suspension" in flags
    ):
        primary = "delisting_or_resumption_extreme_trade"
        severity = "High"
    elif "new_listing_first_days" in flags and "label_shifted_before_listing_event" in flags:
        primary = "new_listing_label_shift"
        severity = "Medium"
    elif "new_listing_first_days" in flags:
        primary = "new_listing_first_days"
        severity = "Medium"
    elif "unadjusted_price_discontinuity_ex_rights_or_corporate_action" in flags:
        primary = "corporate_action_unadjusted_price_discontinuity"
        severity = "High"
    elif "resumption_after_long_suspension" in flags:
        primary = "resumption_after_long_suspension"
        severity = "Medium"
    elif "resumption_after_short_suspension" in flags:
        primary = "resumption_after_short_suspension"
        severity = "Medium"
    elif "single_day_raw_price_jump_over_50pct" in flags:
        primary = "large_raw_price_move"
        severity = "Medium"
    else:
        primary = flags[0]
        severity = "Medium"

    explanation = "; ".join(flags)
    if event_date and label_date and event_date != label_date:
        explanation += f"; nearest/event trading date {event_date} differs from label date {label_date}"
    return primary, severity, explanation


def classify_extremes(
    return_y_path: Path,
    data_root: Path,
    threshold: float,
    event_window: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    y = pd.read_pickle(return_y_path)
    values = y.to_numpy(dtype=float, copy=False)
    mask = np.isfinite(values) & (np.abs(values) > threshold)
    row_idx, col_idx = np.where(mask)

    meta = read_stock_metadata(data_root)
    calendar = read_trading_calendar(data_root)
    daily_zip = data_root / "每日指标.zip"
    member_map = build_member_map(daily_zip)

    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(daily_zip) as zf:
        history_cache: dict[str, pd.DataFrame | None] = {}

        for i, j in zip(row_idx, col_idx):
            label_dt = y.index[i]
            if isinstance(label_dt, pd.Timestamp):
                label_dt = label_dt.date()
            stock = str(y.columns[j])
            value = float(values[i, j])
            member = member_map.get(stock)

            meta_row = meta.loc[stock] if stock in meta.index else None
            ts_code = str(meta_row["TS代码"]) if meta_row is not None else None
            stock_name = str(meta_row["股票名称"]) if meta_row is not None else None
            listing_date = meta_row["上市日期"] if meta_row is not None else None
            delist_date = meta_row["退市日期"] if meta_row is not None else None
            market_type = str(meta_row["市场类型"]) if meta_row is not None else None
            exchange = str(meta_row["交易所代码"]) if meta_row is not None else None
            status = str(meta_row["上市状态"]) if meta_row is not None else None

            if member and stock not in history_cache:
                try:
                    history_cache[stock] = read_daily_history(zf, member)
                except Exception:
                    history_cache[stock] = None
            hist = history_cache.get(stock)

            base_or_future = None
            event = None
            prev_event = None
            next_event = None
            event_window_rows = None
            base_pos = None

            if hist is not None and not hist.empty:
                dates = list(hist["交易日期"])
                pos_exact = {d: k for k, d in enumerate(dates)}.get(label_dt)
                if pos_exact is not None:
                    base_pos = pos_exact
                    start_pos = pos_exact
                else:
                    future_positions = [k for k, d in enumerate(dates) if d >= label_dt]
                    start_pos = future_positions[0] if future_positions else None

                if start_pos is not None:
                    base_or_future = hist.iloc[start_pos]
                    end_pos = min(len(hist), start_pos + event_window + 1)
                    event_window_rows = hist.iloc[start_pos:end_pos].copy()
                    if not event_window_rows.empty:
                        event_idx = event_window_rows["raw_close_ret"].abs().idxmax()
                        if pd.isna(event_window_rows.loc[event_idx, "raw_close_ret"]):
                            event_idx = event_window_rows.index[0]
                        event = hist.loc[event_idx]
                        event_pos = int(event_idx)
                        if event_pos > 0:
                            prev_event = hist.iloc[event_pos - 1]
                        if event_pos + 1 < len(hist):
                            next_event = hist.iloc[event_pos + 1]

            def row_value(source: Any, col: str) -> Any:
                if source is None:
                    return None
                value = source[col]
                if pd.isna(value):
                    return None
                return value

            event_date = row_value(event, "交易日期")
            event_raw_pct = row_value(event, "涨跌幅")
            event_raw_close_ret = row_value(event, "raw_close_ret")
            event_abs_raw_pct = abs(float(event_raw_pct)) if event_raw_pct is not None else None

            first_record_date = row_value(hist.iloc[0], "交易日期") if hist is not None and not hist.empty else None
            trading_days_since_first = None
            if hist is not None and event_date is not None:
                event_positions = hist.index[hist["交易日期"] == event_date].tolist()
                if event_positions:
                    trading_days_since_first = int(event_positions[0])

            event_gap_before_market_days = trading_days_between(
                calendar,
                row_value(prev_event, "交易日期"),
                event_date,
            )
            event_gap_after_market_days = trading_days_between(
                calendar,
                event_date,
                row_value(next_event, "交易日期"),
            )

            raw_cum_from_start_to_event = None
            if base_or_future is not None and event is not None:
                base_close = row_value(base_or_future, "收盘价")
                event_close = row_value(event, "收盘价")
                if base_close not in [None, 0] and event_close is not None:
                    raw_cum_from_start_to_event = float(event_close) / float(base_close) - 1.0

            previous_actual_close = row_value(prev_event, "收盘价")
            event_reference_prev_close = row_value(event, "昨收价")
            reference_vs_previous_actual_ret = None
            if previous_actual_close not in [None, 0] and event_reference_prev_close is not None:
                reference_vs_previous_actual_ret = float(event_reference_prev_close) / float(previous_actual_close) - 1.0

            event_raw_pct_captures_label = False
            if event_raw_pct is not None:
                event_raw_pct_captures_label = abs(float(event_raw_pct) / 100.0 - value) < 0.02
            if raw_cum_from_start_to_event is not None:
                event_raw_pct_captures_label = event_raw_pct_captures_label or abs(raw_cum_from_start_to_event - value) < 0.02

            rec: dict[str, Any] = {
                "label_date": label_dt,
                "stock_code": stock,
                "ts_code": ts_code or (member[:-4] if member else None),
                "stock_name": stock_name,
                "return_y": value,
                "abs_return_y": abs(value),
                "listing_date": listing_date,
                "delist_date": delist_date,
                "calendar_days_since_listing": (label_dt - listing_date).days if listing_date and label_dt >= listing_date else None,
                "calendar_days_to_listing": (listing_date - label_dt).days if listing_date and label_dt < listing_date else None,
                "market_type": market_type,
                "exchange": exchange,
                "listing_status": status,
                "daily_member": member,
                "has_local_daily_history": hist is not None and not hist.empty,
                "has_base_or_future_daily_record": base_or_future is not None,
                "first_record_date": first_record_date,
                "base_or_next_record_date": row_value(base_or_future, "交易日期"),
                "base_or_next_close": row_value(base_or_future, "收盘价"),
                "base_or_next_pct_change": row_value(base_or_future, "涨跌幅"),
                "event_date": event_date,
                "event_stock_code": row_value(event, "股票代码"),
                "event_open": row_value(event, "开盘价"),
                "event_high": row_value(event, "最高价"),
                "event_low": row_value(event, "最低价"),
                "event_close": row_value(event, "收盘价"),
                "event_prev_close": row_value(event, "昨收价"),
                "previous_actual_close": previous_actual_close,
                "reference_vs_previous_actual_ret": reference_vs_previous_actual_ret,
                "event_raw_pct_change": event_raw_pct,
                "event_raw_close_ret": event_raw_close_ret,
                "event_abs_raw_pct_change": event_abs_raw_pct,
                "event_volume": row_value(event, "成交量(手)"),
                "event_amount": row_value(event, "成交额(千元)"),
                "event_turnover": row_value(event, "换手率"),
                "prev_event_date": row_value(prev_event, "交易日期"),
                "next_event_date": row_value(next_event, "交易日期"),
                "event_gap_before_market_days": event_gap_before_market_days,
                "event_gap_after_market_days": event_gap_after_market_days,
                "trading_days_since_first_record": trading_days_since_first,
                "event_is_first_record": bool(trading_days_since_first == 0) if trading_days_since_first is not None else False,
                "raw_cum_from_base_or_next_to_event": raw_cum_from_start_to_event,
                "event_raw_pct_captures_label": event_raw_pct_captures_label,
                "event_window": event_window,
            }
            primary, severity, explanation = classify_row(rec)
            rec["primary_cause"] = primary
            rec["severity"] = severity
            rec["cause_explanation"] = explanation
            rows.append(rec)

    cases = pd.DataFrame(rows)
    if not cases.empty:
        cases = cases.sort_values(["abs_return_y", "label_date", "stock_code"], ascending=[False, True, True])

    summary = {
        "threshold_abs_return_y": threshold,
        "event_window_trading_days": event_window,
        "case_count": int(len(cases)),
        "cause_counts": cases["primary_cause"].value_counts(dropna=False).to_dict() if not cases.empty else {},
        "cause_abs_return_stats": (
            cases.groupby("primary_cause")["abs_return_y"].agg(["count", "mean", "median", "max"]).reset_index().to_dict(orient="records")
            if not cases.empty
            else []
        ),
        "severity_counts": cases["severity"].value_counts(dropna=False).to_dict() if not cases.empty else {},
        "top_20_abs_return": cases.head(20).to_dict(orient="records") if not cases.empty else [],
    }
    return cases, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify extreme return_y cases with local A-share metadata.")
    parser.add_argument("--return-y", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--event-window", type=int, default=10)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    cases, summary = classify_extremes(args.return_y, args.data_root, args.threshold, args.event_window)
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        cases.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    text = json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
