from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def build_member_map(zip_path: Path) -> dict[str, str]:
    with zipfile.ZipFile(zip_path) as zf:
        return {name[:6]: name for name in zf.namelist() if name.endswith(".csv")}


def read_stock_history(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    df = pd.read_csv(zf.open(member), encoding="utf-8-sig")
    df["交易日期"] = pd.to_datetime(df["交易日期"].astype(str), format="%Y%m%d").dt.date
    df = df.sort_values("交易日期").reset_index(drop=True)
    return df[["交易日期", "收盘价"]]


def infer_horizon(return_y_path: Path, daily_zip_path: Path, sample_n: int = 400, max_horizon: int = 30) -> dict[str, Any]:
    y = pd.read_pickle(return_y_path)
    finite = np.isfinite(y.to_numpy(dtype=float, copy=False))
    row_idx, col_idx = np.where(finite)

    rng = np.random.default_rng(0)
    if len(row_idx) > sample_n:
        take = rng.choice(len(row_idx), sample_n, replace=False)
        row_idx = row_idx[take]
        col_idx = col_idx[take]

    by_stock: dict[str, list[tuple[int, date, float]]] = defaultdict(list)
    for i, j in zip(row_idx, col_idx):
        stock = str(y.columns[j])
        dt = y.index[i]
        if isinstance(dt, pd.Timestamp):
            dt = dt.date()
        by_stock[stock].append((i, dt, float(y.iat[i, j])))

    member_map = build_member_map(daily_zip_path)
    errors = {h: [] for h in range(1, max_horizon + 1)}
    matched_points = 0

    with zipfile.ZipFile(daily_zip_path) as zf:
        for stock, points in by_stock.items():
            member = member_map.get(stock)
            if member is None:
                continue
            hist = read_stock_history(zf, member)
            pos_by_date = {d: k for k, d in enumerate(hist["交易日期"])}
            closes = hist["收盘价"].to_numpy(dtype=float)
            for _, dt, y_value in points:
                pos = pos_by_date.get(dt)
                if pos is None:
                    continue
                matched_points += 1
                base = closes[pos]
                if not np.isfinite(base) or base == 0:
                    continue
                for h in range(1, max_horizon + 1):
                    if pos + h >= len(closes):
                        continue
                    ret = closes[pos + h] / base - 1.0
                    if np.isfinite(ret):
                        errors[h].append(abs(ret - y_value))

    rows = []
    for h, vals in errors.items():
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        rows.append(
            {
                "horizon": h,
                "n": int(arr.size),
                "median_abs_error": float(np.median(arr)),
                "p90_abs_error": float(np.quantile(arr, 0.90)),
                "p99_abs_error": float(np.quantile(arr, 0.99)),
                "exactish_count_lt_1bp": int((arr < 0.0001).sum()),
                "exactish_rate_lt_1bp": float((arr < 0.0001).mean()),
            }
        )

    rows = sorted(rows, key=lambda r: (r["median_abs_error"], r["p90_abs_error"]))
    return {
        "sample_requested": sample_n,
        "matched_points": matched_points,
        "max_horizon": max_horizon,
        "results_sorted": rows,
        "best": rows[0] if rows else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--return-y", required=True, type=Path)
    parser.add_argument("--daily-zip", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    result = infer_horizon(args.return_y, args.daily_zip)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
