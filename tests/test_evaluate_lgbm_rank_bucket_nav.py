from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.evaluate_lgbm_rank_bucket_nav import (
    RankBucket,
    evaluate_rank_bucket_nav,
    main,
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
