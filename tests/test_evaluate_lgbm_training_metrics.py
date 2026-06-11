from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.evaluate_lgbm_training_metrics import (
    build_feature_gain_artifacts,
    evaluate_model_predictions,
    write_model_evaluation_artifacts,
)


def test_overall_ic_is_mean_of_daily_cross_sections_not_pooled() -> None:
    rows: list[dict[str, object]] = []
    panel_rows: list[dict[str, object]] = []
    for stock_idx, stock_code in enumerate(["000001", "000002"], start=1):
        rows.append(
            {
                "fold_id": 1,
                "split": "test",
                "date": "2024-01-02",
                "stock_code": stock_idx,
                "y_true": float(stock_idx),
                "score": float(stock_idx),
                "sample_weight": 1.0,
            }
        )
        panel_rows.append(_panel_row("2024-01-02", stock_code, stock_idx))
    for stock_idx in range(1, 21):
        stock_code = f"{stock_idx + 100:06d}"
        rows.append(
            {
                "fold_id": 1,
                "split": "test",
                "date": "2024-01-03",
                "stock_code": int(stock_code),
                "y_true": float(21 - stock_idx),
                "score": float(stock_idx),
                "sample_weight": 1.0,
            }
        )
        panel_rows.append(_panel_row("2024-01-03", stock_code, stock_idx))

    predictions = pd.DataFrame(rows)
    pooled_ic = float(np.corrcoef(predictions["score"], predictions["y_true"])[0, 1])

    result = evaluate_model_predictions(
        predictions=predictions,
        training_panel=pd.DataFrame(panel_rows),
        score_cols=["score"],
        target_col="y_true",
        spread_target_col="y_true",
        top_bottom_quantiles=2,
        condition_bucket_count=3,
    )

    metric = result["overall_metrics"].iloc[0]
    assert metric["date_count"] == 2
    assert np.isclose(metric["mean_ic"], 0.0)
    assert np.isclose(metric["mean_rankic"], 0.0)
    assert not np.isclose(pooled_ic, metric["mean_ic"])


def test_evaluate_model_predictions_computes_top_bottom_and_group_ic() -> None:
    predictions, panel = _balanced_predictions_and_panel()

    result = evaluate_model_predictions(
        predictions=predictions,
        training_panel=panel,
        score_cols=["score"],
        target_col="y_true",
        spread_target_col="y_true",
        top_bottom_quantiles=3,
        condition_bucket_count=3,
    )

    spread_by_date = result["top_bottom_spread_by_date"].sort_values("date")
    assert spread_by_date["spread"].round(10).tolist() == [4.0, -4.0]
    spread_summary = result["top_bottom_spread_summary"].iloc[0]
    assert spread_summary["date_count"] == 2
    assert np.isclose(spread_summary["mean_spread"], 0.0)
    assert np.isclose(spread_summary["mean_top_count"], 2.0)
    assert np.isclose(spread_summary["mean_bottom_count"], 2.0)

    group_summary = result["group_ic_summary"]
    assert {"industry", "board", "index_bucket", "size_decile", "adv_tercile", "turnover_tercile"}.issubset(
        set(group_summary["group_dimension"])
    )
    industry_a = group_summary[
        group_summary["group_dimension"].eq("industry") & group_summary["group_value"].eq("A")
    ].iloc[0]
    assert industry_a["date_count"] == 2
    assert np.isclose(industry_a["mean_ic"], 0.0)
    assert np.isclose(industry_a["mean_rankic"], 0.0)


def test_write_model_evaluation_artifacts_creates_reproducible_outputs(tmp_path: Path) -> None:
    predictions, panel = _balanced_predictions_and_panel()
    feature_gain = pd.DataFrame(
        [
            {"fold_id": 1, "feature": "score_alpha", "importance_gain": 3.0, "importance_split": 2.0},
            {"fold_id": 1, "feature": "industry", "importance_gain": 1.0, "importance_split": 1.0},
        ]
    )

    summary = write_model_evaluation_artifacts(
        predictions=predictions,
        training_panel=panel,
        output_dir=tmp_path,
        score_cols=["score"],
        target_col="y_true",
        spread_target_col="y_true",
        top_bottom_quantiles=3,
        condition_bucket_count=3,
        feature_gain=feature_gain,
        feature_roles={
            "alpha_placeholder": ["score_alpha"],
            "condition_categorical": ["industry"],
        },
        feature_columns=["score_alpha", "industry"],
    )

    expected_files = {
        "overall_metrics_by_fold_split_score.csv",
        "daily_ic_by_fold_split_score.csv",
        "top_bottom_spread_by_date.csv",
        "top_bottom_spread_summary.csv",
        "group_ic_by_date.csv",
        "group_ic_summary.csv",
        "feature_gain_by_fold.csv",
        "feature_gain_summary.csv",
        "feature_gain_role_summary.csv",
        "feature_gain_diagnostics.json",
        "evaluation_summary.json",
    }
    assert expected_files.issubset({path.name for path in tmp_path.iterdir()})
    loaded_summary = json.loads((tmp_path / "evaluation_summary.json").read_text(encoding="utf-8"))
    assert loaded_summary["metric_contract"]["ic_aggregation"] == "mean_of_daily_cross_section_ic"
    assert loaded_summary["metric_contract"]["top_bottom_quantiles"] == 3
    assert "feature_gain_summary" in loaded_summary["outputs"]
    for output_path in loaded_summary["outputs"].values():
        assert not Path(output_path).is_absolute()
    assert summary["row_counts"]["overall_metrics"] == 1


def test_build_feature_gain_artifacts_computes_shares_roles_and_zero_gain() -> None:
    feature_gain = pd.DataFrame(
        [
            {"fold_id": 1, "feature": "alpha_a", "importance_gain": 3.0, "importance_split": 2.0},
            {"fold_id": 1, "feature": "context_b", "importance_gain": 1.0, "importance_split": 2.0},
            {"fold_id": 2, "feature": "alpha_a", "importance_gain": 0.0, "importance_split": 0.0},
            {"fold_id": 2, "feature": "context_b", "importance_gain": 0.0, "importance_split": 0.0},
        ]
    )

    artifacts = build_feature_gain_artifacts(
        feature_gain=feature_gain,
        feature_roles={
            "alpha_placeholder": ["alpha_a"],
            "condition_continuous": ["context_b"],
        },
        feature_columns=["alpha_a", "context_b"],
    )

    by_fold = artifacts["feature_gain_by_fold"]
    fold1 = by_fold[by_fold["fold_id"].eq(1)].sort_values("feature")
    fold2 = by_fold[by_fold["fold_id"].eq(2)].sort_values("feature")
    assert np.isclose(fold1["gain_share_in_fold"].sum(), 1.0)
    assert np.isclose(fold1["split_share_in_fold"].sum(), 1.0)
    assert np.isclose(fold2["gain_share_in_fold"].sum(), 0.0)
    assert np.isclose(fold2["split_share_in_fold"].sum(), 0.0)
    assert set(by_fold["feature_role"]) == {"alpha_placeholder", "condition_continuous"}

    summary = artifacts["feature_gain_summary"]
    alpha_summary = summary[summary["feature"].eq("alpha_a")].iloc[0]
    assert np.isclose(alpha_summary["total_gain"], 3.0)
    assert np.isclose(alpha_summary["mean_gain_share"], 0.375)
    role_summary = artifacts["feature_gain_role_summary"]
    assert np.isclose(role_summary["gain_share"].sum(), 1.0)
    diagnostics = artifacts["feature_gain_diagnostics"]
    assert diagnostics["zero_total_gain_folds"] == [2]


def test_build_feature_gain_artifacts_rejects_ambiguous_feature_roles() -> None:
    feature_gain = pd.DataFrame(
        [{"fold_id": 1, "feature": "dup", "importance_gain": 1.0, "importance_split": 1.0}]
    )

    try:
        build_feature_gain_artifacts(
            feature_gain=feature_gain,
            feature_roles={
                "alpha_placeholder": ["dup"],
                "condition_continuous": ["dup"],
            },
            feature_columns=["dup"],
        )
    except ValueError as exc:
        assert "multiple feature roles" in str(exc)
    else:
        raise AssertionError("Expected ambiguous feature role mapping to raise ValueError.")


def test_top_bottom_spread_skips_constant_scores() -> None:
    predictions, panel = _balanced_predictions_and_panel()
    predictions["score"] = 1.0

    result = evaluate_model_predictions(
        predictions=predictions,
        training_panel=panel,
        score_cols=["score"],
        target_col="y_true",
        spread_target_col="y_true",
        top_bottom_quantiles=3,
        condition_bucket_count=3,
    )

    assert result["top_bottom_spread_by_date"].empty
    assert result["top_bottom_spread_summary"].empty


def test_enrichment_rejects_inconsistent_panel_targets() -> None:
    predictions, panel = _balanced_predictions_and_panel()
    panel.loc[0, "y_true"] = panel.loc[0, "y_true"] + 1.0

    try:
        evaluate_model_predictions(
            predictions=predictions,
            training_panel=panel,
            score_cols=["score"],
            target_col="y_true",
            spread_target_col="y_true",
            top_bottom_quantiles=3,
            condition_bucket_count=3,
        )
    except ValueError as exc:
        assert "inconsistent with training panel" in str(exc)
    else:
        raise AssertionError("Expected mismatched targets to raise ValueError.")


def _balanced_predictions_and_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = ["2024-01-02", "2024-01-03"]
    predictions: list[dict[str, object]] = []
    panel_rows: list[dict[str, object]] = []
    industries = ["A", "A", "B", "B", "C", "C"]
    size_deciles = [0, 0, 1, 1, 2, 2]
    for date_idx, date in enumerate(dates):
        y_values = [1, 2, 3, 4, 5, 6] if date_idx == 0 else [6, 5, 4, 3, 2, 1]
        for stock_idx, y_value in enumerate(y_values, start=1):
            stock_code = f"{stock_idx:06d}"
            predictions.append(
                {
                    "fold_id": 1,
                    "split": "test",
                    "date": date,
                    "stock_code": stock_idx,
                    "y_true": float(y_value),
                    "score": float(stock_idx),
                    "sample_weight": 1.0,
                }
            )
            panel_rows.append(
                {
                    "date": date,
                    "stock_code": stock_code,
                    "industry": industries[stock_idx - 1],
                    "board": "MAIN" if stock_idx <= 3 else "CHINEXT",
                    "index_bucket": "CSI300" if stock_idx % 2 else "CSI1000",
                    "size_decile": size_deciles[stock_idx - 1],
                    "mcap_rank": float(size_deciles[stock_idx - 1] - 1),
                    "logADV20": float(stock_idx),
                    "turnover20": float(stock_idx),
                    "y_true": float(y_value),
                    "sample_weight": 1.0,
                }
            )
    return pd.DataFrame(predictions), pd.DataFrame(panel_rows)


def _panel_row(date: str, stock_code: str, stock_idx: int) -> dict[str, object]:
    return {
        "date": date,
        "stock_code": stock_code,
        "industry": "A" if stock_idx % 2 else "B",
        "board": "MAIN" if stock_idx % 2 else "CHINEXT",
        "index_bucket": "CSI300" if stock_idx % 2 else "CSI1000",
        "size_decile": stock_idx % 3,
        "mcap_rank": float(stock_idx),
        "logADV20": float(stock_idx),
        "turnover20": float(stock_idx),
        "sample_weight": 1.0,
    }
