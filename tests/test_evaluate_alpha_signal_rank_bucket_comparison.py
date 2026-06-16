from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.evaluate_alpha_signal_rank_bucket_comparison import (
    PeriodSpec,
    evaluate_alpha_signal_rank_bucket_comparison,
)


def _make_training_panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    industries = {
        "000001": "bank",
        "000002": "bank",
        "000003": "tech",
        "000004": "tech",
    }
    market_caps = {
        "000001": 400.0,
        "000002": 300.0,
        "000003": 200.0,
        "000004": 100.0,
    }
    raw_scores = {
        "000001": 4.0,
        "000002": 3.0,
        "000003": 2.0,
        "000004": 1.0,
    }
    neutralized_scores = {
        "000001": 1.0,
        "000002": 2.0,
        "000003": 3.0,
        "000004": 4.0,
    }
    for date in ["2024-01-02", "2024-01-03"]:
        for stock_code in ["000001", "000002", "000003", "000004"]:
            market_cap = market_caps[stock_code]
            rows.append(
                {
                    "date": date,
                    "stock_code": stock_code,
                    "industry": industries[stock_code],
                    "market_cap": market_cap,
                    "log_mcap": float(np.log1p(market_cap)),
                    "log_mcap_z": float((market_cap - 250.0) / 125.0),
                    "mcap_rank": float((market_cap - 250.0) / 400.0),
                    "size_decile": int(8 if market_cap >= 300 else 2),
                    "factor_sss_dx_10_raw": raw_scores[stock_code],
                    "factor_sss_dx_10_value_neutralized_raw": neutralized_scores[stock_code],
                }
            )
    return pd.DataFrame(rows)


def _make_template_predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date in ["2024-01-02", "2024-01-03"]:
        for stock_code in ["000001", "000002", "000003", "000004"]:
            rows.append(
                {
                    "fold_id": 1,
                    "split": "test",
                    "date": date,
                    "stock_code": stock_code,
                }
            )
    return pd.DataFrame(rows)


def _make_return_y() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [0.02, 0.04, 0.08, 0.10],
            [0.02, 0.04, 0.08, 0.10],
        ],
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        columns=["000001", "000002", "000003", "000004"],
    )


def test_alpha_signal_comparison_uses_raw_and_neutralized_alpha_scores_for_bucket_nav() -> None:
    result = evaluate_alpha_signal_rank_bucket_comparison(
        training_panel=_make_training_panel(),
        template_predictions=_make_template_predictions(),
        return_y=_make_return_y(),
        alpha_name="factor_sss_dx_10",
        signal_ids=["raw_alpha_value", "neutralized_alpha_value"],
        periods=[PeriodSpec(label="sample", start="2024-01-02", end="2024-01-03")],
        rank_bucket_count=2,
        splits=["test"],
    )

    manifest = result["signal_manifest"].sort_values("signal_id").reset_index(drop=True)
    assert manifest["signal_id"].tolist() == ["neutralized_alpha_value", "raw_alpha_value"]
    assert manifest["source_column"].tolist() == [
        "factor_sss_dx_10_value_neutralized_raw",
        "factor_sss_dx_10_raw",
    ]

    requested = result["period_requested_table"].sort_values(["signal_id", "bucket"])
    raw_top = requested[
        requested["signal_id"].eq("raw_alpha_value") & requested["bucket"].eq("01")
    ].iloc[0]
    neutralized_top = requested[
        requested["signal_id"].eq("neutralized_alpha_value") & requested["bucket"].eq("01")
    ].iloc[0]

    assert np.isclose(raw_top["mean_daily_return"], 0.03)
    assert np.isclose(raw_top["nav_end"], 1.0609)
    assert np.isclose(neutralized_top["mean_daily_return"], 0.09)
    assert np.isclose(neutralized_top["nav_end"], 1.1881)

    industry = result["period_industry_summary"]
    raw_bank = industry[
        industry["signal_id"].eq("raw_alpha_value")
        & industry["period_label"].eq("sample")
        & industry["bucket_index"].eq(1)
        & industry["industry"].eq("bank")
    ].iloc[0]
    neutralized_tech = industry[
        industry["signal_id"].eq("neutralized_alpha_value")
        & industry["period_label"].eq("sample")
        & industry["bucket_index"].eq(1)
        & industry["industry"].eq("tech")
    ].iloc[0]

    assert np.isclose(raw_bank["mean_active_weight"], 0.5)
    assert np.isclose(neutralized_tech["mean_active_weight"], 0.5)

    industry_top = result["period_industry_top"]
    assert "signal_id" in industry_top.columns
    assert set(industry_top["signal_id"]) == {"raw_alpha_value", "neutralized_alpha_value"}


def test_alpha_signal_comparison_rejects_missing_requested_signal_column() -> None:
    panel = _make_training_panel().drop(columns=["factor_sss_dx_10_value_neutralized_raw"])

    try:
        evaluate_alpha_signal_rank_bucket_comparison(
            training_panel=panel,
            template_predictions=_make_template_predictions(),
            return_y=_make_return_y(),
            alpha_name="factor_sss_dx_10",
            signal_ids=["neutralized_alpha_value"],
            periods=[PeriodSpec(label="sample", start="2024-01-02", end="2024-01-03")],
            rank_bucket_count=2,
            splits=["test"],
        )
    except ValueError as exc:
        assert "factor_sss_dx_10_value_neutralized_raw" in str(exc)
    else:
        raise AssertionError("Expected missing neutralized alpha score column to raise ValueError.")


def test_alpha_signal_comparison_cli_writes_artifacts(tmp_path: Path) -> None:
    training_panel_path = tmp_path / "training_panel.pkl"
    template_predictions_path = tmp_path / "template_predictions.csv"
    return_y_path = tmp_path / "return_y.pkl"
    output_dir = tmp_path / "alpha_signal_comparison"

    _make_training_panel().to_pickle(training_panel_path)
    _make_template_predictions().to_csv(template_predictions_path, index=False, encoding="utf-8-sig")
    _make_return_y().to_pickle(return_y_path)

    completed = subprocess.run(
        [
            sys.executable,
            "analysis/evaluate_alpha_signal_rank_bucket_comparison.py",
            "--training-panel",
            str(training_panel_path),
            "--template-predictions",
            str(template_predictions_path),
            "--return-y",
            str(return_y_path),
            "--output-dir",
            str(output_dir),
            "--alpha-name",
            "factor_sss_dx_10",
            "--signals",
            "raw_alpha_value",
            "neutralized_alpha_value",
            "--periods",
            "sample:2024-01-02:2024-01-03",
            "--rank-bucket-count",
            "2",
            "--splits",
            "test",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "alpha_signal_rank_bucket_comparison_summary.json").exists()
    assert (output_dir / "rank_bucket_period_requested_table.csv").exists()
