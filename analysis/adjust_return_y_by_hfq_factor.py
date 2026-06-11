from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd


SuspiciousPolicy = Literal["keep_raw", "nan"]


@dataclass(frozen=True)
class AdjustmentConfig:
    suspicious_policy: SuspiciousPolicy = "keep_raw"
    no_change_atol: float = 1e-12
    factor_jump_threshold: float = 0.02
    large_factor_jump_threshold: float = 0.20
    reversal_window: int = 10
    reversal_tolerance: float = 0.02
    suspicious_reversal_min_jump: float = 0.20
    include_unchanged_diagnostics: bool = False


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.date().isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _sanitize_local_paths(text: str) -> str:
    return re.sub(r"[A-Za-z]:\\[^\"'\r\n]+", "<local-path>", text)


def _normalize_datetime_index(values: Sequence[Any] | pd.Index) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(list(values), errors="coerce")).normalize()


def _normalize_factor_series(values: pd.Series) -> pd.Series:
    factors = values.copy()
    factors.index = _normalize_datetime_index(factors.index)
    factors = pd.to_numeric(factors, errors="coerce")
    factors = factors[~factors.index.isna()]
    factors = factors[~factors.index.duplicated(keep="last")]
    return factors.sort_index()


def _date_string(value: pd.Timestamp | pd.NaT | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def _relative_close(values: pd.Series, target: float, tolerance: float) -> bool:
    if not np.isfinite(target) or target == 0:
        return False
    arr = values.to_numpy(dtype=float, copy=False)
    arr = arr[np.isfinite(arr) & (arr != 0)]
    if arr.size == 0:
        return False
    return bool(np.any(np.abs(arr / target - 1.0) <= tolerance))


def build_trade_date_map(
    signal_dates: Sequence[Any] | pd.Index,
    trading_days: Sequence[Any] | pd.Index,
) -> pd.DataFrame:
    normalized_signals = _normalize_datetime_index(signal_dates)
    normalized_trading_days = _normalize_datetime_index(trading_days).dropna().unique().sort_values()
    position_by_day = {day: pos for pos, day in enumerate(normalized_trading_days)}

    rows: list[dict[str, pd.Timestamp | pd.NaT]] = []
    for signal_date in normalized_signals:
        pos = position_by_day.get(signal_date)
        buy_date: pd.Timestamp | pd.NaT = pd.NaT
        sell_date: pd.Timestamp | pd.NaT = pd.NaT
        if pos is not None:
            if pos + 1 < len(normalized_trading_days):
                buy_date = normalized_trading_days[pos + 1]
            if pos + 2 < len(normalized_trading_days):
                sell_date = normalized_trading_days[pos + 2]
        rows.append({"buy_date": buy_date, "sell_date": sell_date})

    return pd.DataFrame(rows, index=normalized_signals)


def is_suspicious_factor_reversal(
    factors: pd.Series,
    buy_date: pd.Timestamp,
    sell_date: pd.Timestamp,
    *,
    factor_buy: float,
    factor_sell: float,
    config: AdjustmentConfig,
) -> bool:
    if not (np.isfinite(factor_buy) and np.isfinite(factor_sell)):
        return False
    if factor_buy <= 0 or factor_sell <= 0:
        return False
    ratio = factor_sell / factor_buy
    if abs(ratio - 1.0) <= config.suspicious_reversal_min_jump:
        return False

    normalized = _normalize_factor_series(factors)
    buy = pd.Timestamp(buy_date).normalize()
    sell = pd.Timestamp(sell_date).normalize()
    if buy not in normalized.index or sell not in normalized.index:
        return False

    buy_pos = int(normalized.index.get_loc(buy))
    sell_pos = int(normalized.index.get_loc(sell))
    before_buy = normalized.iloc[max(0, buy_pos - config.reversal_window) : buy_pos]
    after_sell = normalized.iloc[sell_pos + 1 : sell_pos + 1 + config.reversal_window]

    pre_returns_to_sell_level = _relative_close(
        before_buy,
        factor_sell,
        config.reversal_tolerance,
    )
    post_returns_to_buy_level = _relative_close(
        after_sell,
        factor_buy,
        config.reversal_tolerance,
    )
    return pre_returns_to_sell_level or post_returns_to_buy_level


def adjust_stock_returns(
    symbol: str,
    raw_returns: pd.Series,
    adjustment_factors: pd.Series,
    date_map: pd.DataFrame,
    config: AdjustmentConfig,
    official_pct_change: pd.Series | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    raw = raw_returns.astype(float).copy()
    raw.index = _normalize_datetime_index(raw.index)
    factors = _normalize_factor_series(adjustment_factors)

    pct_change: pd.Series | None = None
    if official_pct_change is not None:
        pct_change = official_pct_change.copy()
        pct_change.index = _normalize_datetime_index(pct_change.index)
        pct_change = pd.to_numeric(pct_change, errors="coerce")
        pct_change = pct_change[~pct_change.index.duplicated(keep="last")].sort_index()

    normalized_date_map = date_map.copy()
    normalized_date_map.index = _normalize_datetime_index(normalized_date_map.index)

    aligned_map = normalized_date_map.reindex(raw.index)
    buy_dates = pd.DatetimeIndex(aligned_map["buy_date"])
    sell_dates = pd.DatetimeIndex(aligned_map["sell_date"])

    factor_buy_series = factors.reindex(buy_dates)
    factor_sell_series = factors.reindex(sell_dates)
    factor_buy_series.index = raw.index
    factor_sell_series.index = raw.index

    if pct_change is not None:
        official_pct_series = pct_change.reindex(sell_dates)
        official_pct_series.index = raw.index
    else:
        official_pct_series = pd.Series(np.nan, index=raw.index, dtype=float)

    raw_values = raw.to_numpy(dtype=float, copy=False)
    factor_buy_values = factor_buy_series.to_numpy(dtype=float, copy=False)
    factor_sell_values = factor_sell_series.to_numpy(dtype=float, copy=False)

    finite_raw = np.isfinite(raw_values)
    missing_trade_date = finite_raw & (pd.isna(buy_dates) | pd.isna(sell_dates))
    missing_factor = (
        finite_raw
        & ~missing_trade_date
        & (~np.isfinite(factor_buy_values) | ~np.isfinite(factor_sell_values))
    )
    invalid_factor = (
        finite_raw
        & ~missing_trade_date
        & ~missing_factor
        & ((factor_buy_values <= 0.0) | (factor_sell_values <= 0.0))
    )
    valid_factor = finite_raw & ~missing_trade_date & ~missing_factor & ~invalid_factor

    adjusted_values = np.full(len(raw), np.nan, dtype=float)
    factor_ratio_values = np.full(len(raw), np.nan, dtype=float)
    candidate_values = np.full(len(raw), np.nan, dtype=float)

    factor_ratio_values[valid_factor] = factor_sell_values[valid_factor] / factor_buy_values[valid_factor]
    candidate_values[valid_factor] = (1.0 + raw_values[valid_factor]) * factor_ratio_values[valid_factor] - 1.0

    no_change = valid_factor & (np.abs(factor_ratio_values - 1.0) <= config.no_change_atol)
    changed = valid_factor & ~no_change
    adjusted_values[no_change] = raw_values[no_change]
    adjusted_values[changed] = candidate_values[changed]

    actions = np.full(len(raw), "", dtype=object)
    reasons = np.full(len(raw), "", dtype=object)
    actions[missing_trade_date] = "nan_missing_trade_date"
    reasons[missing_trade_date] = "buy_date_or_sell_date_missing_from_trading_calendar"
    actions[missing_factor] = "nan_missing_factor"
    reasons[missing_factor] = "factor_buy_or_factor_sell_missing"
    actions[invalid_factor] = "nan_invalid_factor"
    reasons[invalid_factor] = "factor_buy_or_factor_sell_is_non_positive_or_non_finite"
    actions[no_change] = "unchanged_no_factor_change"
    reasons[no_change] = "factor_ratio_is_one"
    actions[changed] = "adjusted_by_factor_ratio"
    reasons[changed] = "applied_factor_sell_over_factor_buy"

    factor_jump = valid_factor & (np.abs(factor_ratio_values - 1.0) > config.factor_jump_threshold)
    large_factor_jump = valid_factor & (
        np.abs(factor_ratio_values - 1.0) > config.large_factor_jump_threshold
    )
    suspicious = np.full(len(raw), False, dtype=bool)
    for pos in np.flatnonzero(large_factor_jump):
        suspicious[pos] = is_suspicious_factor_reversal(
            factors,
            buy_dates[pos],
            sell_dates[pos],
            factor_buy=float(factor_buy_values[pos]),
            factor_sell=float(factor_sell_values[pos]),
            config=config,
        )

    suspicious_changed = suspicious & changed
    if config.suspicious_policy == "keep_raw":
        adjusted_values[suspicious_changed] = raw_values[suspicious_changed]
        actions[suspicious_changed] = "kept_raw_suspicious_factor_reversal"
    else:
        adjusted_values[suspicious_changed] = np.nan
        actions[suspicious_changed] = "nan_suspicious_factor_reversal"
    reasons[suspicious_changed] = "large_factor_jump_reverses_within_nearby_window"

    adjusted = pd.Series(adjusted_values, index=raw.index, name=raw.name)

    diagnostic_mask = finite_raw & (
        (actions != "unchanged_no_factor_change") | config.include_unchanged_diagnostics
    )
    diagnostic_positions = np.flatnonzero(diagnostic_mask)
    diagnostics: list[dict[str, Any]] = []
    for pos in diagnostic_positions:
        raw_value = float(raw_values[pos])
        final_value = float(adjusted_values[pos]) if np.isfinite(adjusted_values[pos]) else np.nan
        candidate = float(candidate_values[pos]) if np.isfinite(candidate_values[pos]) else np.nan
        official_value = official_pct_series.iloc[pos]
        official_sell_pct_change = (
            float(official_value)
            if pd.notna(official_value) and np.isfinite(float(official_value))
            else np.nan
        )
        diagnostics.append(
            {
                "signal_date": _date_string(raw.index[pos]),
                "stock_code": symbol.split(".")[0],
                "ts_code": symbol,
                "buy_date": _date_string(buy_dates[pos]),
                "sell_date": _date_string(sell_dates[pos]),
                "return_y_raw": raw_value,
                "factor_buy": float(factor_buy_values[pos])
                if np.isfinite(factor_buy_values[pos])
                else np.nan,
                "factor_sell": float(factor_sell_values[pos])
                if np.isfinite(factor_sell_values[pos])
                else np.nan,
                "factor_ratio": float(factor_ratio_values[pos])
                if np.isfinite(factor_ratio_values[pos])
                else np.nan,
                "return_y_adj_candidate": candidate,
                "return_y_adj_final": final_value,
                "delta": final_value - raw_value if np.isfinite(final_value) else np.nan,
                "official_sell_pct_change": official_sell_pct_change,
                "candidate_minus_official_sell_pct_change": (
                    candidate - official_sell_pct_change
                    if np.isfinite(candidate) and np.isfinite(official_sell_pct_change)
                    else np.nan
                ),
                "factor_jump_flag": bool(factor_jump[pos]),
                "large_factor_jump_flag": bool(large_factor_jump[pos]),
                "suspicious_factor_reversal_flag": bool(suspicious[pos]),
                "adjustment_action": str(actions[pos]),
                "reason": str(reasons[pos]),
            }
        )

    return adjusted, pd.DataFrame(diagnostics)


def _read_return_y(path: Path) -> pd.DataFrame:
    frame = pd.read_pickle(path)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected return_y to be a pandas DataFrame, got {type(frame)!r}")
    frame = frame.copy()
    frame.index = _normalize_datetime_index(frame.index)
    frame.columns = frame.columns.astype(str)
    return frame


def _load_finfact_store(ashare_daily_root: Path):
    try:
        from finfact_io import FinfactStore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "finfact_io is required. Install it or set PYTHONPATH to the finfact_io src directory."
        ) from exc
    return FinfactStore(ashare_daily_dir=ashare_daily_root)


def _load_trading_days(ashare: Any, exchange: str) -> pd.DatetimeIndex:
    calendar = ashare.trading_calendar(exchange=exchange, is_open=True, columns="standard")
    return _normalize_datetime_index(calendar["date"]).dropna().unique().sort_values()


def _build_symbol_map(ashare: Any) -> dict[str, str]:
    stocks = ashare.stocks(include_delisted=True, columns="standard")
    if "stock_code" not in stocks.columns or "symbol" not in stocks.columns:
        return {}
    mapping = (
        stocks[["stock_code", "symbol"]]
        .dropna()
        .astype(str)
        .drop_duplicates(subset=["stock_code"], keep="first")
    )
    return dict(zip(mapping["stock_code"], mapping["symbol"]))


def _infer_symbol(stock_code: str) -> str | None:
    code = str(stock_code)
    if "." in code:
        return code
    if code.startswith("6"):
        return f"{code}.SH"
    if code.startswith(("0", "2", "3")):
        return f"{code}.SZ"
    if code.startswith(("8", "9")):
        return f"{code}.BJ"
    return None


def _resolve_symbol(stock_code: str, symbol_map: dict[str, str]) -> str | None:
    code = str(stock_code)
    return symbol_map.get(code) or _infer_symbol(code)


def _load_factor_inputs(
    ashare: Any,
    symbol: str,
    *,
    adjustment: str,
) -> tuple[pd.Series, pd.Series]:
    factors = ashare.technical_factors(symbol, adjustment=adjustment, columns="standard")
    required = {"trade_date", "adjustment_factor", "pct_change"}
    missing = sorted(required - set(factors.columns))
    if missing:
        raise ValueError(f"technical_factors({symbol}) missing standard columns: {missing}")
    factors = factors.copy()
    factors["trade_date"] = _normalize_datetime_index(factors["trade_date"])
    factors = factors.drop_duplicates(subset=["trade_date"], keep="last").set_index("trade_date")
    adjustment_factors = pd.to_numeric(factors["adjustment_factor"], errors="coerce").sort_index()
    official_pct_change = (pd.to_numeric(factors["pct_change"], errors="coerce") / 100.0).sort_index()
    return adjustment_factors, official_pct_change


def _parse_symbols(values: Sequence[str] | None) -> set[str] | None:
    if not values:
        return None
    parsed: set[str] = set()
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                parsed.add(item.split(".")[0])
    return parsed or None


def adjust_return_y_matrix(
    return_y: pd.DataFrame,
    ashare: Any,
    trading_days: pd.DatetimeIndex,
    *,
    config: AdjustmentConfig,
    symbols: set[str] | None = None,
    technical_adjustment: str = "hfq",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    symbol_map = _build_symbol_map(ashare)
    date_map = build_trade_date_map(return_y.index, trading_days)
    adjusted = return_y.astype(float).copy()

    diagnostics: list[pd.DataFrame] = []
    per_symbol: list[dict[str, Any]] = []
    processed_stock_codes: list[str] = []
    processed = 0
    skipped = 0

    for stock_code in return_y.columns.astype(str):
        if symbols is not None and stock_code not in symbols:
            skipped += 1
            continue
        symbol = _resolve_symbol(stock_code, symbol_map)
        if symbol is None:
            adjusted[stock_code] = np.nan
            per_symbol.append(
                {
                    "stock_code": stock_code,
                    "ts_code": None,
                    "status": "missing_symbol_mapping",
                    "raw_points": int(return_y[stock_code].notna().sum()),
                    "testable_points": 0,
                }
            )
            continue

        try:
            factors, official_pct_change = _load_factor_inputs(
                ashare,
                symbol,
                adjustment=technical_adjustment,
            )
        except Exception as exc:
            adjusted[stock_code] = np.nan
            per_symbol.append(
                {
                    "stock_code": stock_code,
                    "ts_code": symbol,
                    "status": "factor_load_error",
                    "error": _sanitize_local_paths(f"{type(exc).__name__}: {exc}"),
                    "raw_points": int(return_y[stock_code].notna().sum()),
                    "testable_points": 0,
                }
            )
            continue

        adjusted_series, diag = adjust_stock_returns(
            symbol,
            return_y[stock_code],
            factors,
            date_map,
            config,
            official_pct_change,
        )
        adjusted[stock_code] = adjusted_series.to_numpy(dtype=float)
        if not diag.empty:
            diagnostics.append(diag)

        finite = pd.DataFrame({"raw": return_y[stock_code], "adj": adjusted_series}).dropna()
        processed_stock_codes.append(stock_code)
        processed += 1
        per_symbol.append(
            {
                "stock_code": stock_code,
                "ts_code": symbol,
                "status": "processed",
                "raw_points": int(return_y[stock_code].notna().sum()),
                "testable_points": int(len(finite)),
                "raw_abs_gt_50pct_count": int((finite["raw"].abs() > 0.5).sum()),
                "adj_abs_gt_50pct_count": int((finite["adj"].abs() > 0.5).sum()),
            }
        )

    diagnostics_frame = pd.concat(diagnostics, ignore_index=True) if diagnostics else pd.DataFrame()
    action_counts = (
        diagnostics_frame["adjustment_action"].value_counts(dropna=False).to_dict()
        if not diagnostics_frame.empty and "adjustment_action" in diagnostics_frame.columns
        else {}
    )
    before_after = pd.DataFrame(
        {
            "raw": return_y.to_numpy(dtype=float, copy=False).ravel(),
            "adj": adjusted.to_numpy(dtype=float, copy=False).ravel(),
        }
    )
    before_after = before_after[np.isfinite(before_after["raw"]) | np.isfinite(before_after["adj"])]
    if processed_stock_codes:
        processed_before_after = pd.DataFrame(
            {
                "raw": return_y.loc[:, processed_stock_codes].to_numpy(dtype=float, copy=False).ravel(),
                "adj": adjusted.loc[:, processed_stock_codes].to_numpy(dtype=float, copy=False).ravel(),
            }
        )
        processed_before_after = processed_before_after[
            np.isfinite(processed_before_after["raw"]) | np.isfinite(processed_before_after["adj"])
        ]
    else:
        processed_before_after = pd.DataFrame({"raw": [], "adj": []})
    summary = {
        "config": asdict(config),
        "technical_adjustment": technical_adjustment,
        "shape": {"rows": int(return_y.shape[0]), "columns": int(return_y.shape[1])},
        "symbols": {
            "processed": processed,
            "skipped_by_filter": skipped,
            "requested_filter_count": None if symbols is None else len(symbols),
        },
        "diagnostic_row_count": int(len(diagnostics_frame)),
        "diagnostic_action_counts": action_counts,
        "raw_abs_gt_50pct_count": int((before_after["raw"].abs() > 0.5).sum()),
        "adjusted_abs_gt_50pct_count": int((before_after["adj"].abs() > 0.5).sum()),
        "processed_symbols_raw_abs_gt_50pct_count": int(
            (processed_before_after["raw"].abs() > 0.5).sum()
        ),
        "processed_symbols_adjusted_abs_gt_50pct_count": int(
            (processed_before_after["adj"].abs() > 0.5).sum()
        ),
        "per_symbol": per_symbol,
    }
    return adjusted, diagnostics_frame, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adjust return_y labels with post-adjustment factors and factor-jump diagnostics."
    )
    parser.add_argument("--return-y", required=True, type=Path)
    parser.add_argument("--ashare-daily-root", required=True, type=Path)
    parser.add_argument("--output-return-y", required=True, type=Path)
    parser.add_argument("--output-diagnostics", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--symbols", nargs="*", help="Optional stock codes or TS codes to process.")
    parser.add_argument("--calendar-exchange", default="SSE")
    parser.add_argument("--technical-adjustment", default="hfq", choices=["hfq", "qfq"])
    parser.add_argument("--suspicious-policy", default="keep_raw", choices=["keep_raw", "nan"])
    parser.add_argument("--factor-jump-threshold", default=0.02, type=float)
    parser.add_argument("--large-factor-jump-threshold", default=0.20, type=float)
    parser.add_argument("--reversal-window", default=10, type=int)
    parser.add_argument("--reversal-tolerance", default=0.02, type=float)
    parser.add_argument("--suspicious-reversal-min-jump", default=0.20, type=float)
    args = parser.parse_args()

    config = AdjustmentConfig(
        suspicious_policy=args.suspicious_policy,
        factor_jump_threshold=args.factor_jump_threshold,
        large_factor_jump_threshold=args.large_factor_jump_threshold,
        reversal_window=args.reversal_window,
        reversal_tolerance=args.reversal_tolerance,
        suspicious_reversal_min_jump=args.suspicious_reversal_min_jump,
    )

    store = _load_finfact_store(args.ashare_daily_root)
    ashare = store.ashare
    trading_days = _load_trading_days(ashare, args.calendar_exchange)
    return_y = _read_return_y(args.return_y)
    symbols = _parse_symbols(args.symbols)

    adjusted, diagnostics, summary = adjust_return_y_matrix(
        return_y,
        ashare,
        trading_days,
        config=config,
        symbols=symbols,
        technical_adjustment=args.technical_adjustment,
    )

    args.output_return_y.parent.mkdir(parents=True, exist_ok=True)
    args.output_diagnostics.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)

    adjusted.to_pickle(args.output_return_y)
    diagnostics.to_csv(args.output_diagnostics, index=False, encoding="utf-8-sig")
    args.output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
