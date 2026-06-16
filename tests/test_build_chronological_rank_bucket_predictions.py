from __future__ import annotations

import csv
import json
from pathlib import Path

from analysis.build_chronological_rank_bucket_predictions import build_chronological_predictions


def _write_training_summary(path: Path) -> None:
    summary = {
        "folds": [
            {
                "fold_id": 1,
                "date_ranges": {
                    "train": {"start": "2024-01-02", "end": "2024-01-02", "date_count": 1},
                    "valid": {"start": "2024-01-03", "end": "2024-01-03", "date_count": 1},
                    "test": {"start": "2024-01-04", "end": "2024-01-04", "date_count": 1},
                },
            },
            {
                "fold_id": 2,
                "date_ranges": {
                    "train": {"start": "2024-01-02", "end": "2024-01-04", "date_count": 3},
                    "valid": {"start": "2024-01-04", "end": "2024-01-04", "date_count": 1},
                    "test": {"start": "2024-01-05", "end": "2024-01-05", "date_count": 1},
                },
            },
        ]
    }
    path.write_text(json.dumps(summary), encoding="utf-8")


def _write_predictions(path: Path, *, include_duplicate: bool = False) -> None:
    rows = [
        {"fold_id": 1, "split": "train", "date": "2024-01-02", "stock_code": "1", "pred_direct": "0.4"},
        {"fold_id": 1, "split": "valid", "date": "2024-01-03", "stock_code": "1", "pred_direct": "0.5"},
        {"fold_id": 1, "split": "test", "date": "2024-01-04", "stock_code": "1", "pred_direct": "0.6"},
        {"fold_id": 2, "split": "train", "date": "2024-01-02", "stock_code": "1", "pred_direct": "9.9"},
        {"fold_id": 2, "split": "valid", "date": "2024-01-04", "stock_code": "1", "pred_direct": "8.8"},
        {"fold_id": 2, "split": "test", "date": "2024-01-05", "stock_code": "1", "pred_direct": "0.7"},
    ]
    if include_duplicate:
        rows.append(
            {"fold_id": 1, "split": "train", "date": "2024-01-02", "stock_code": "000001", "pred_direct": "0.1"}
        )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["fold_id", "split", "date", "stock_code", "pred_direct"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_build_chronological_predictions_selects_non_overlapping_segments(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.csv"
    training_summary = tmp_path / "training_summary.json"
    output_dir = tmp_path / "chronological"
    _write_predictions(predictions)
    _write_training_summary(training_summary)

    manifest = build_chronological_predictions(
        predictions_path=predictions,
        training_summary_path=training_summary,
        output_dir=output_dir,
    )

    with (output_dir / "predictions.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["date"] for row in rows] == [
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
    ]
    assert {row["fold_id"] for row in rows} == {"0"}
    assert {row["split"] for row in rows} == {"full_chronological"}
    assert [row["source_fold_id"] for row in rows] == ["1", "1", "1", "2"]
    assert [row["source_split"] for row in rows] == ["train", "valid", "test", "test"]
    assert manifest["row_counts"]["selected_prediction_rows"] == 4
    assert manifest["date_range"] == {"start": "2024-01-02", "end": "2024-01-05"}


def test_build_chronological_predictions_rejects_duplicate_selected_date_stock(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.csv"
    training_summary = tmp_path / "training_summary.json"
    _write_predictions(predictions, include_duplicate=True)
    _write_training_summary(training_summary)

    try:
        build_chronological_predictions(
            predictions_path=predictions,
            training_summary_path=training_summary,
            output_dir=tmp_path / "chronological",
        )
    except ValueError as exc:
        assert "duplicate date/stock_code" in str(exc)
    else:
        raise AssertionError("Expected duplicate selected date/stock rows to raise ValueError.")
