from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis.return_y_neutralization_analysis import (
    build_neutralization_analysis,
    write_neutralization_analysis_artifacts,
)


def test_build_neutralization_analysis_recomputes_distribution_metrics() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    columns = ["000001", "000002", "000003"]
    raw = pd.DataFrame(
        [
            [0.03, -0.02, 0.01],
            [0.06, -0.04, 0.02],
            [np.nan, -0.03, 0.00],
        ],
        index=dates,
        columns=columns,
    )
    residual = pd.DataFrame(
        [
            [0.01, -0.01, 0.00],
            [0.02, -0.02, 0.00],
            [np.nan, -0.01, 0.01],
        ],
        index=dates,
        columns=columns,
    )
    rank_label = pd.DataFrame(
        [
            [0.25, -0.25, 0.00],
            [0.30, -0.30, 0.00],
            [np.nan, -0.20, 0.20],
        ],
        index=dates,
        columns=columns,
    )
    diagnostics = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "skipped": [False, False, False],
            "r2": [0.70, 0.85, 0.95],
            "max_abs_industry_mean_after": [1e-8, 2e-8, 3e-8],
            "max_abs_board_mean_after": [4e-8, 5e-8, 6e-8],
            "max_abs_continuous_exposure_after": [7e-10, 8e-10, 9e-10],
            "n_used": [3, 3, 2],
        }
    )

    analysis = build_neutralization_analysis(
        raw,
        residual,
        rank_label,
        diagnostics,
        example_symbols=["000001", "000003"],
    )

    raw_values = raw.where(residual.notna()).to_numpy(dtype=float).ravel()
    raw_values = raw_values[np.isfinite(raw_values)]
    residual_values = residual.to_numpy(dtype=float).ravel()
    residual_values = residual_values[np.isfinite(residual_values)]

    assert analysis["headline_metrics"]["raw_std"] == np.std(raw_values)
    assert analysis["headline_metrics"]["residual_std"] == np.std(residual_values)
    assert analysis["headline_metrics"]["residual_mean"] == np.mean(residual_values)
    assert analysis["headline_metrics"]["max_industry_residual_mean"] == 3e-8
    assert analysis["headline_metrics"]["max_board_residual_mean"] == 6e-8
    assert analysis["headline_metrics"]["max_continuous_exposure"] == 9e-10
    assert analysis["headline_metrics"]["median_r2"] == 0.85
    assert analysis["headline_metrics"]["rank_label_min"] == -0.30
    assert analysis["headline_metrics"]["rank_label_max"] == 0.30
    assert [item["symbol"] for item in analysis["example_symbols"]] == ["000001", "000003"]


def test_write_neutralization_analysis_artifacts_creates_report_files(tmp_path) -> None:
    dates = pd.date_range("2024-01-02", periods=40, freq="B")
    columns = ["000001", "000002", "000003", "000004"]
    base = np.linspace(-0.03, 0.03, len(dates))
    raw = pd.DataFrame(
        {
            "000001": base + 0.01,
            "000002": -base,
            "000003": base * 0.5,
            "000004": np.sin(np.arange(len(dates))) * 0.02,
        },
        index=dates,
    )
    residual = raw * 0.4
    rank_label = residual.rank(axis=1, pct=True) - 0.5
    diagnostics = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "skipped": False,
            "r2": np.linspace(0.2, 0.8, len(dates)),
            "max_abs_industry_mean_after": np.linspace(1e-7, 1e-8, len(dates)),
            "max_abs_board_mean_after": np.linspace(2e-7, 2e-8, len(dates)),
            "max_abs_continuous_exposure_after": np.linspace(3e-9, 3e-10, len(dates)),
            "n_used": 4,
        }
    )

    artifacts = write_neutralization_analysis_artifacts(
        raw,
        residual,
        rank_label,
        diagnostics,
        output_dir=tmp_path,
        example_symbols=["000001", "000002"],
    )

    assert artifacts["summary_json"].exists()
    assert artifacts["report_html"].exists()
    for chart_path in artifacts["charts"].values():
        assert chart_path.exists()
        assert chart_path.stat().st_size > 0

    summary = json.loads(artifacts["summary_json"].read_text(encoding="utf-8"))
    assert summary["headline_metrics"]["raw_std"] > summary["headline_metrics"]["residual_std"]
    assert "000001" in artifacts["report_html"].read_text(encoding="utf-8")
