from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.evaluate_lgbm_rank_bucket_nav import (
    RankBucket,
    evaluate_rank_bucket_nav,
    main,
    write_rank_bucket_nav_artifacts,
)


def _make_predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in ["train", "test"]:
        for date_idx, date in enumerate(["2024-01-02", "2024-01-03"]):
            scores = [4.0, 3.0, 2.0, 1.0] if date_idx == 0 else [1.0, 2.0, 3.0, 4.0]
            for stock_idx, score in enumerate(scores, start=1):
                rows.append(
                    {
                        "fold_id": 1,
                        "split": split,
                        "date": date,
                        "stock_code": stock_idx,
                        "score_marginal_z": score,
                    }
                )
    return pd.DataFrame(rows)


def _make_return_y() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [0.10, 0.06, -0.02, -0.04],
            [-0.01, 0.01, 0.03, 0.05],
        ],
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        columns=["000001", "000002", "000003", "000004"],
    )


def _make_feature_panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    market_caps_by_date = {
        "2024-01-02": [400.0, 300.0, 200.0, 100.0],
        "2024-01-03": [100.0, 200.0, 300.0, 400.0],
    }
    industries_by_date = {
        "2024-01-02": ["bank", "bank", "tech", "tech"],
        "2024-01-03": ["bank", "tech", "tech", "bank"],
    }
    mcap_ranks_by_date = {
        "2024-01-02": [0.375, 0.125, -0.125, -0.375],
        "2024-01-03": [-0.375, -0.125, 0.125, 0.375],
    }
    size_deciles_by_date = {
        "2024-01-02": [10, 8, 3, 1],
        "2024-01-03": [1, 3, 8, 10],
    }
    for date in ["2024-01-02", "2024-01-03"]:
        for idx, stock_code in enumerate(["000001", "000002", "000003", "000004"]):
            market_cap = market_caps_by_date[date][idx]
            rows.append(
                {
                    "date": date,
                    "stock_code": stock_code,
                    "industry": industries_by_date[date][idx],
                    "market_cap": market_cap,
                    "log_mcap": float(np.log1p(market_cap)),
                    "log_mcap_z": mcap_ranks_by_date[date][idx],
                    "mcap_rank": mcap_ranks_by_date[date][idx],
                    "size_decile": size_deciles_by_date[date][idx],
                }
            )
    return pd.DataFrame(rows)


def test_evaluate_rank_bucket_nav_assigns_fixed_rank_buckets_and_compounds_by_group() -> None:
    result = evaluate_rank_bucket_nav(
        predictions=_make_predictions(),
        return_y=_make_return_y(),
        rank_bucket_size=2,
    )

    daily = result["daily_returns"]
    nav = result["nav"]
    top_train = daily[
        daily["split"].eq("train") & daily["bucket_label"].eq("rank_0001_0002")
    ].sort_values("signal_date")

    assert top_train["selected_count"].tolist() == [2, 2]
    assert top_train["valid_return_count"].tolist() == [2, 2]
    assert top_train["bucket_return"].round(10).tolist() == [0.08, 0.04]

    bottom_train = daily[
        daily["split"].eq("train") & daily["bucket_label"].eq("rank_0003_0004")
    ].sort_values("signal_date")
    assert bottom_train["bucket_return"].round(10).tolist() == [-0.03, 0.0]

    top_nav = nav[
        nav["split"].eq("train") & nav["bucket_label"].eq("rank_0001_0002")
    ].sort_values("signal_date")
    assert top_nav["nav_base"].tolist() == [1.0, 1.0]
    assert top_nav["gross_nav"].round(10).tolist() == [1.08, 1.1232]

    test_top_nav = nav[
        nav["split"].eq("test") & nav["bucket_label"].eq("rank_0001_0002")
    ].sort_values("signal_date")
    assert test_top_nav["gross_nav"].round(10).tolist() == [1.08, 1.1232]


def test_evaluate_rank_bucket_nav_drops_nan_signal_and_renormalizes_missing_returns() -> None:
    predictions = pd.DataFrame(
        [
            {"fold_id": 1, "split": "test", "date": "2024-01-02", "stock_code": "000001", "score_marginal_z": 4.0},
            {"fold_id": 1, "split": "test", "date": "2024-01-02", "stock_code": "000002", "score_marginal_z": np.nan},
            {"fold_id": 1, "split": "test", "date": "2024-01-02", "stock_code": "000003", "score_marginal_z": 2.0},
            {"fold_id": 1, "split": "test", "date": "2024-01-02", "stock_code": "000004", "score_marginal_z": 1.0},
        ]
    )
    return_y = pd.DataFrame(
        [[0.02, 0.03, np.nan, -0.01]],
        index=pd.to_datetime(["2024-01-02"]),
        columns=["000001", "000002", "000003", "000004"],
    )

    result = evaluate_rank_bucket_nav(
        predictions=predictions,
        return_y=return_y,
        rank_buckets=[RankBucket(start=1, end=2), RankBucket(start=3, end=4)],
    )

    daily = result["daily_returns"].sort_values("bucket_label")
    top = daily[daily["bucket_label"].eq("rank_0001_0002")].iloc[0]
    bottom = daily[daily["bucket_label"].eq("rank_0003_0004")].iloc[0]

    assert top["eligible_count"] == 3
    assert top["selected_count"] == 2
    assert top["valid_return_count"] == 1
    assert top["missing_return_count"] == 1
    assert np.isclose(top["bucket_return"], 0.02)
    assert bottom["selected_count"] == 1
    assert np.isclose(bottom["bucket_return"], -0.01)


def test_evaluate_rank_bucket_nav_supports_daily_equal_count_bucket_count() -> None:
    rows: list[dict[str, object]] = []
    for date, scores in {
        "2024-01-02": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        "2024-01-03": [10, 9, 8, 7, 6, 5, 4, 3, np.nan, np.nan],
    }.items():
        for stock_idx, score in enumerate(scores, start=1):
            rows.append(
                {
                    "fold_id": 1,
                    "split": "test",
                    "date": date,
                    "stock_code": f"{stock_idx:06d}",
                    "score_marginal_z": score,
                }
            )
    predictions = pd.DataFrame(rows)
    return_y = pd.DataFrame(
        np.full((2, 10), 0.01),
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        columns=[f"{idx:06d}" for idx in range(1, 11)],
    )

    result = evaluate_rank_bucket_nav(
        predictions=predictions,
        return_y=return_y,
        rank_bucket_count=3,
        splits=["test"],
    )

    daily = result["daily_returns"].sort_values(["signal_date", "bucket_index"])
    assert daily["bucket_label"].unique().tolist() == [
        "rank_bucket_01_of_03",
        "rank_bucket_02_of_03",
        "rank_bucket_03_of_03",
    ]
    per_date = daily.groupby("signal_date", sort=True)
    assert per_date["bucket_label"].nunique().tolist() == [3, 3]
    assert per_date["selected_count"].sum().tolist() == [10, 8]
    assert daily["selected_count"].tolist() == [4, 3, 3, 3, 3, 2]
    assert daily["valid_return_count"].tolist() == [4, 3, 3, 3, 3, 2]
    assert daily["rank_start"].tolist() == [1, 5, 8, 1, 4, 7]
    assert daily["rank_end"].tolist() == [4, 7, 10, 3, 6, 8]


def test_evaluate_rank_bucket_nav_builds_constituents_and_composition_active_weights() -> None:
    result = evaluate_rank_bucket_nav(
        predictions=_make_predictions(),
        return_y=_make_return_y(),
        feature_panel=_make_feature_panel(),
        rank_bucket_size=2,
        splits=["test"],
    )

    constituents = result["constituents"].sort_values(
        ["signal_date", "bucket_index", "rank_in_date"]
    )
    assert constituents["constituent_weight"].round(10).tolist() == [0.5] * 8
    assert constituents[
        constituents["bucket_label"].eq("rank_0001_0002")
        & constituents["signal_date"].eq(pd.Timestamp("2024-01-02"))
    ]["stock_code"].tolist() == ["000001", "000002"]
    assert {
        "industry",
        "market_cap",
        "log_mcap",
        "log_mcap_z",
        "mcap_rank",
        "size_decile",
    }.issubset(constituents.columns)

    industry_summary = result["composition_industry_summary"]
    top_bank = industry_summary[
        industry_summary["bucket_index"].eq(1)
        & industry_summary["industry"].eq("bank")
    ].iloc[0]
    assert np.isclose(top_bank["mean_bucket_weight"], 0.75)
    assert np.isclose(top_bank["mean_universe_weight"], 0.50)
    assert np.isclose(top_bank["mean_active_weight"], 0.25)
    assert np.isclose(top_bank["positive_active_date_rate"], 0.50)

    size_summary = result["composition_size_summary"]
    top_mcap_rank = size_summary[
        size_summary["bucket_index"].eq(1)
        & size_summary["metric"].eq("mcap_rank")
    ].iloc[0]
    assert np.isclose(top_mcap_rank["mean_bucket_value"], 0.25)
    assert np.isclose(top_mcap_rank["mean_universe_value"], 0.0)
    assert np.isclose(top_mcap_rank["mean_active_value"], 0.25)


def test_evaluate_rank_bucket_nav_rejects_duplicate_feature_panel_keys() -> None:
    feature_panel = pd.concat(
        [_make_feature_panel(), _make_feature_panel().iloc[[0]]],
        ignore_index=True,
    )

    try:
        evaluate_rank_bucket_nav(
            predictions=_make_predictions(),
            return_y=_make_return_y(),
            feature_panel=feature_panel,
            rank_bucket_size=2,
            splits=["test"],
        )
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("Expected duplicate feature-panel keys to raise ValueError.")


def test_evaluate_rank_bucket_nav_rejects_missing_feature_panel_rows() -> None:
    feature_panel = _make_feature_panel()
    feature_panel = feature_panel[
        ~(
            feature_panel["date"].eq("2024-01-02")
            & feature_panel["stock_code"].eq("000001")
        )
    ].copy()

    try:
        evaluate_rank_bucket_nav(
            predictions=_make_predictions(),
            return_y=_make_return_y(),
            feature_panel=feature_panel,
            rank_bucket_size=2,
            splits=["test"],
        )
    except ValueError as exc:
        assert "missing feature panel rows" in str(exc)
    else:
        raise AssertionError("Expected missing feature rows to raise ValueError.")


def test_evaluate_rank_bucket_nav_rejects_duplicate_prediction_keys() -> None:
    predictions = _make_predictions()
    duplicated = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)

    try:
        evaluate_rank_bucket_nav(
            predictions=duplicated,
            return_y=_make_return_y(),
            rank_bucket_size=2,
        )
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("Expected duplicate prediction keys to raise ValueError.")


def test_evaluate_rank_bucket_nav_keeps_nav_stale_when_bucket_has_no_valid_return() -> None:
    predictions = _make_predictions()
    return_y = _make_return_y()
    return_y.loc[pd.Timestamp("2024-01-03"), ["000003", "000004"]] = np.nan

    result = evaluate_rank_bucket_nav(
        predictions=predictions[predictions["split"].eq("test")],
        return_y=return_y,
        rank_bucket_size=2,
        splits=["test"],
    )

    top_nav = result["nav"][
        result["nav"]["bucket_label"].eq("rank_0001_0002")
    ].sort_values("signal_date")

    assert top_nav["bucket_return"].iloc[0] == 0.08
    assert pd.isna(top_nav["bucket_return"].iloc[1])
    assert top_nav["applied_return"].tolist() == [0.08, 0.0]
    assert top_nav["nav_stale_flag"].tolist() == [False, True]
    assert top_nav["gross_nav"].round(10).tolist() == [1.08, 1.08]


def test_rank_bucket_cli_writes_outputs_with_relative_summary_paths(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.csv"
    return_y_path = tmp_path / "return_y.pkl"
    output_dir = tmp_path / "rank_bucket_outputs"
    _make_predictions().to_csv(predictions_path, index=False, encoding="utf-8-sig")
    _make_return_y().to_pickle(return_y_path)

    summary = main(
        [
            "--predictions",
            str(predictions_path),
            "--return-y",
            str(return_y_path),
            "--output-dir",
            str(output_dir),
            "--rank-bucket-size",
            "2",
            "--splits",
            "train",
            "test",
        ]
    )

    expected_files = {
        "rank_bucket_daily_returns.csv",
        "rank_bucket_nav.csv",
        "rank_bucket_summary.csv",
        "rank_bucket_evaluation_summary.json",
    }
    assert expected_files.issubset({path.name for path in output_dir.iterdir()})
    assert (output_dir / "charts" / "rank_bucket_nav_fold_1_train.png").exists()
    assert (output_dir / "charts" / "rank_bucket_nav_fold_1_test.png").exists()

    loaded = json.loads(
        (output_dir / "rank_bucket_evaluation_summary.json").read_text(encoding="utf-8")
    )
    assert loaded["schema_version"] == "lgbm_rank_bucket_nav_v1"
    assert loaded["metric_contract"]["score_col"] == "score_marginal_z"
    assert loaded["metric_contract"]["return_alignment"] == "already_aligned_to_signal_date"
    assert summary["outputs"] == loaded["outputs"]
    for output_path in loaded["outputs"].values():
        if isinstance(output_path, str):
            assert not Path(output_path).is_absolute()


def test_rank_bucket_artifacts_write_composition_outputs_and_charts(tmp_path: Path) -> None:
    output_dir = tmp_path / "rank_bucket_outputs"

    summary = write_rank_bucket_nav_artifacts(
        predictions=_make_predictions(),
        return_y=_make_return_y(),
        feature_panel=_make_feature_panel(),
        output_dir=output_dir,
        rank_bucket_size=2,
        splits=["test"],
    )

    expected_outputs = {
        "constituents",
        "composition_industry_daily",
        "composition_industry_summary",
        "composition_size_daily",
        "composition_size_summary",
    }
    assert expected_outputs.issubset(summary["outputs"])
    for key in expected_outputs:
        assert (output_dir / summary["outputs"][key]).exists()

    expected_charts = {
        "composition_industry_active_weight_bucket_01_test",
        "composition_industry_active_weight_heatmap_test",
        "composition_size_decile_active_weight_heatmap_test",
        "composition_mcap_rank_by_bucket_test",
    }
    assert expected_charts.issubset(summary["charts"])
    for key in expected_charts:
        assert (output_dir / summary["charts"][key]).exists()
