from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.analyze_signal_rank_alignment import (
    CandidateSpec,
    PeriodSpec,
    compare_signal_rank_alignment,
)


def _reference_panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date in ["2024-01-02", "2024-01-03"]:
        for stock_idx, reference_score in enumerate([4.0, 3.0, 2.0, 1.0], start=1):
            rows.append(
                {
                    "date": date,
                    "stock_code": f"{stock_idx:06d}",
                    "neutralized_alpha": reference_score,
                }
            )
    return pd.DataFrame(rows)


def _candidate_predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date, scores in {
        "2024-01-02": [4.0, 3.0, 1.0, 2.0],
        "2024-01-03": [1.0, 2.0, 3.0, 4.0],
    }.items():
        for stock_idx, score in enumerate(scores, start=1):
            rows.append(
                {
                    "fold_id": 1,
                    "split": "test",
                    "date": date,
                    "stock_code": f"{stock_idx:06d}",
                    "candidate_score": score,
                }
            )
    return pd.DataFrame(rows)


def test_compare_signal_rank_alignment_summarizes_rank_corr_and_top_tail_overlap() -> None:
    result = compare_signal_rank_alignment(
        reference_panel=_reference_panel(),
        candidates=[
            CandidateSpec(
                label="candidate",
                predictions=_candidate_predictions(),
                score_col="candidate_score",
            )
        ],
        reference_col="neutralized_alpha",
        periods=[PeriodSpec(label="sample", start="2024-01-02", end="2024-01-03")],
        rank_bucket_count=2,
        splits=["test"],
    )

    daily = result["daily_alignment"].sort_values("signal_date")
    assert daily["eligible_count"].tolist() == [4, 4]
    assert daily["top_count"].tolist() == [2, 2]
    assert daily["top_overlap_count"].tolist() == [2, 0]
    assert daily["top_overlap_rate"].tolist() == [1.0, 0.0]
    assert np.isclose(daily["spearman_rank_corr"].iloc[0], 0.8)
    assert np.isclose(daily["spearman_rank_corr"].iloc[1], -1.0)

    summary = result["period_summary"].iloc[0]
    assert summary["date_count"] == 2
    assert np.isclose(summary["mean_spearman_rank_corr"], -0.1)
    assert np.isclose(summary["mean_top_overlap_rate"], 0.5)
