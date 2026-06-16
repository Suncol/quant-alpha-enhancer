from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.summarize_rank_bucket_periods import (
    PeriodSpec,
    summarize_rank_bucket_periods,
)


def _make_daily_returns() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    returns_by_date = {
        "2024-01-02": {1: 0.10, 2: -0.10},
        "2024-01-03": {1: -0.20, 2: 0.00},
        "2024-01-04": {1: 0.05, 2: 0.02},
        "2024-01-05": {1: 0.05, 2: np.nan},
    }
    for date, bucket_returns in returns_by_date.items():
        fold_id = 1 if date <= "2024-01-03" else 2
        for bucket_index, bucket_return in bucket_returns.items():
            records.append(
                {
                    "fold_id": fold_id,
                    "split": "test",
                    "score_col": "score_marginal_z",
                    "return_col": "return_y",
                    "signal_date": date,
                    "bucket_label": f"rank_bucket_{bucket_index:02d}_of_02",
                    "bucket_index": bucket_index,
                    "bucket_count": 2,
                    "bucket_mode": "daily_equal_count",
                    "selected_count": 10,
                    "valid_return_count": 10 if np.isfinite(bucket_return) else 0,
                    "bucket_return": bucket_return,
                }
            )
    return pd.DataFrame(records)


def test_summarize_rank_bucket_periods_restarts_nav_and_ignores_fold_dimension() -> None:
    result = summarize_rank_bucket_periods(
        _make_daily_returns(),
        periods=[
            PeriodSpec(label="p1", start="2024-01-02", end="2024-01-03"),
            PeriodSpec(label="p2", start="2024-01-04", end="2024-01-05"),
        ],
        splits=["test"],
    )

    summary = result["summary"].sort_values(["period_label", "bucket_index"])
    assert "fold_id" not in summary.columns
    assert summary["period_label"].tolist() == ["p1", "p1", "p2", "p2"]
    assert summary["date_count"].tolist() == [2, 2, 2, 2]

    p1_top = summary[
        summary["period_label"].eq("p1") & summary["bucket_index"].eq(1)
    ].iloc[0]
    assert np.isclose(p1_top["gross_nav_end"], 0.88)
    assert np.isclose(p1_top["max_drawdown"], -0.20)
    assert p1_top["longest_underwater_trading_days"] == 1

    p2_top = summary[
        summary["period_label"].eq("p2") & summary["bucket_index"].eq(1)
    ].iloc[0]
    assert np.isclose(p2_top["gross_nav_end"], 1.1025)
    assert np.isclose(p2_top["mean_daily_return"], 0.05)
    assert p2_top["longest_underwater_trading_days"] == 0

    p2_bottom = summary[
        summary["period_label"].eq("p2") & summary["bucket_index"].eq(2)
    ].iloc[0]
    assert np.isclose(p2_bottom["gross_nav_end"], 1.02)
    assert p2_bottom["valid_return_date_count"] == 1
    assert p2_bottom["empty_date_count"] == 1


def test_summarize_rank_bucket_periods_rejects_duplicate_dates_after_dropping_fold() -> None:
    daily = _make_daily_returns()
    duplicate = daily[daily["signal_date"].eq("2024-01-02")].copy()
    duplicate["fold_id"] = 99
    duplicated = pd.concat([daily, duplicate], ignore_index=True)

    try:
        summarize_rank_bucket_periods(
            duplicated,
            periods=[PeriodSpec(label="p1", start="2024-01-02", end="2024-01-03")],
            splits=["test"],
        )
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("Expected duplicate period/date/bucket rows to raise ValueError.")
