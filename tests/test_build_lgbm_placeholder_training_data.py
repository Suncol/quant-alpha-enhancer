from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis.build_lgbm_placeholder_training_data import (
    WalkForwardFold,
    build_placeholder_training_panel,
    make_walk_forward_fold_assignments,
    write_placeholder_training_data_artifacts,
)


def _make_exposures() -> pd.DataFrame:
    dates = pd.to_datetime(
        [
            "2024-09-30",
            "2024-10-01",
            "2024-12-31",
            "2025-01-01",
        ]
    )
    stock_codes = ["000001", "000002", "000003", "000004"]
    rows: list[dict[str, object]] = []
    for date_idx, date in enumerate(dates):
        for stock_idx, stock_code in enumerate(stock_codes):
            rows.append(
                {
                    "date": date,
                    "stock_code": stock_code,
                    "industry": ["bank", "bank", "tech", "tech"][stock_idx],
                    "board": ["SZSE_MAIN", "SZSE_MAIN", "CHINEXT", "STAR"][stock_idx],
                    "is_csi300": 1 if stock_idx == 0 else 0,
                    "is_csi500": 1 if stock_idx == 1 else 0,
                    "is_csi1000": 1 if stock_idx == 2 else 0,
                    "is_csi2000": 1 if stock_idx == 3 else 0,
                    "market_cap": float(100 + 10 * stock_idx + date_idx),
                    "amount_k": float(1000 + 100 * stock_idx + date_idx),
                    "turnover": float(1 + stock_idx + date_idx),
                    "logADV20": float(10 + 0.5 * stock_idx + 0.1 * date_idx),
                    "logAmount20": float(11 + 0.4 * stock_idx + 0.1 * date_idx),
                    "turnover20": float(2 + stock_idx + 0.2 * date_idx),
                }
            )
    return pd.DataFrame(rows)


def _make_labels() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(
        [
            "2024-09-30",
            "2024-10-01",
            "2024-12-31",
            "2025-01-01",
        ]
    )
    columns = ["000001", "000002", "000003", "000004"]
    y_resid = pd.DataFrame(
        [
            [0.01, 0.02, -0.01, -0.02],
            [0.02, np.nan, -0.01, -0.03],
            [0.04, 0.01, -0.02, -0.01],
            [0.03, 0.02, -0.02, -0.04],
        ],
        index=dates,
        columns=columns,
    )
    y_rank = pd.DataFrame(
        [
            [0.125, 0.375, -0.125, -0.375],
            [0.333333, np.nan, 0.0, -0.333333],
            [0.375, 0.125, -0.375, -0.125],
            [0.375, 0.125, -0.125, -0.375],
        ],
        index=dates,
        columns=columns,
    )
    return y_resid, y_rank


def test_build_placeholder_training_panel_aligns_labels_and_standardizes_by_date() -> None:
    panel, diagnostics = build_placeholder_training_panel(
        _make_exposures(),
        *_make_labels(),
    )

    assert set(panel["alpha_source"]) == {"placeholder_liquidity"}
    assert panel["date"].min() == pd.Timestamp("2024-09-30")
    assert panel["date"].max() == pd.Timestamp("2025-01-01")
    assert len(panel) == 15
    assert not panel.duplicated(["date", "stock_code"]).any()

    missing_label_row = panel[
        panel["date"].eq(pd.Timestamp("2024-10-01")) & panel["stock_code"].eq("000002")
    ]
    assert missing_label_row.empty

    expected_columns = {
        "y_resid_fwd",
        "y_rank_label",
        "alpha_placeholder_turnover20_raw",
        "alpha_placeholder_turnover20_z",
        "alpha_placeholder_turnover20_rank",
        "alpha_placeholder_logADV20_raw",
        "alpha_placeholder_logADV20_z",
        "alpha_placeholder_logADV20_rank",
        "log_mcap",
        "log_mcap_z",
        "mcap_rank",
        "size_decile",
        "index_bucket",
        "sample_weight",
    }
    assert expected_columns.issubset(panel.columns)

    for date, group in panel.groupby("date"):
        assert abs(float(group["mcap_rank"].mean())) < 1e-12
        assert abs(float(group["alpha_placeholder_turnover20_rank"].mean())) < 1e-12
        assert abs(float(group["alpha_placeholder_logADV20_rank"].mean())) < 1e-12
        assert abs(float(group["sample_weight"].sum()) - len(panel) / panel["date"].nunique()) < 1e-12
        for column in [
            "alpha_placeholder_turnover20_z",
            "alpha_placeholder_logADV20_z",
            "log_mcap_z",
        ]:
            assert abs(float(group[column].median())) < 1e-12, (date, column)

    assert diagnostics["summary"]["row_count"] == 15
    assert diagnostics["summary"]["dropped_missing_label_rows"] == 1
    assert diagnostics["summary"]["alpha_placeholder_source_columns"] == [
        "turnover20",
        "logADV20",
    ]


def test_make_walk_forward_fold_assignments_is_inclusive_and_per_fold_unique() -> None:
    available_dates = pd.to_datetime(
        ["2024-09-30", "2024-10-01", "2024-12-31", "2025-01-01", "2025-06-30"]
    )
    folds = [
        WalkForwardFold(
            fold_id=1,
            train_start="2024-09-30",
            train_end="2024-09-30",
            valid_start="2024-10-01",
            valid_end="2024-12-31",
            test_start="2025-01-01",
            test_end="2025-06-30",
        )
    ]

    assignments = make_walk_forward_fold_assignments(
        available_dates,
        folds,
        embargo_trading_days=0,
    )

    assert not assignments.duplicated(["fold_id", "date"]).any()
    split_by_date = assignments.set_index("date")["split"].to_dict()
    assert split_by_date[pd.Timestamp("2024-09-30")] == "train"
    assert split_by_date[pd.Timestamp("2024-10-01")] == "valid"
    assert split_by_date[pd.Timestamp("2024-12-31")] == "valid"
    assert split_by_date[pd.Timestamp("2025-01-01")] == "test"
    assert split_by_date[pd.Timestamp("2025-06-30")] == "test"


def test_make_walk_forward_fold_assignments_drops_tail_signal_dates_with_embargo() -> None:
    available_dates = pd.to_datetime(
        [
            "2024-09-26",
            "2024-09-27",
            "2024-09-30",
            "2024-10-08",
            "2024-12-30",
            "2024-12-31",
            "2025-01-02",
        ]
    )
    folds = [
        WalkForwardFold(
            fold_id=1,
            train_start="2024-09-01",
            train_end="2024-09-30",
            valid_start="2024-10-01",
            valid_end="2024-12-31",
            test_start="2025-01-01",
            test_end="2025-06-30",
        )
    ]

    assignments = make_walk_forward_fold_assignments(
        available_dates,
        folds,
        embargo_trading_days=2,
    )

    assert not assignments.duplicated(["fold_id", "date"]).any()
    split_by_date = assignments.set_index("date")["split"].to_dict()
    assert split_by_date[pd.Timestamp("2024-09-26")] == "train"
    assert pd.Timestamp("2024-09-27") not in split_by_date
    assert pd.Timestamp("2024-09-30") not in split_by_date
    assert split_by_date[pd.Timestamp("2024-10-08")] == "valid"
    assert pd.Timestamp("2024-12-30") not in split_by_date
    assert pd.Timestamp("2024-12-31") not in split_by_date
    assert split_by_date[pd.Timestamp("2025-01-02")] == "test"


def test_build_placeholder_training_panel_rejects_duplicate_exposure_keys() -> None:
    exposures = pd.concat([_make_exposures(), _make_exposures().head(1)], ignore_index=True)

    try:
        build_placeholder_training_panel(
            exposures,
            *_make_labels(),
        )
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()
        assert "date" in str(exc)
        assert "stock_code" in str(exc)
    else:
        raise AssertionError("Expected duplicate exposure keys to raise ValueError")


def test_write_placeholder_training_data_artifacts(tmp_path) -> None:
    output_panel = tmp_path / "panel.pkl"
    output_folds = tmp_path / "folds.csv"
    output_summary = tmp_path / "summary.json"
    diagnostics_dir = tmp_path / "diagnostics"

    write_placeholder_training_data_artifacts(
        exposures=_make_exposures(),
        y_resid=_make_labels()[0],
        y_rank_label=_make_labels()[1],
        output_panel=output_panel,
        output_fold_assignments=output_folds,
        output_summary=output_summary,
        diagnostics_dir=diagnostics_dir,
        embargo_trading_days=0,
    )

    panel = pd.read_pickle(output_panel)
    folds = pd.read_csv(output_folds)
    summary = json.loads(output_summary.read_text(encoding="utf-8"))

    assert len(panel) == 15
    assert {"fold_id", "date", "split"}.issubset(folds.columns)
    assert summary["metadata"] == {
        "signal_stage": "placeholder_alpha_probe",
        "alpha_source": "placeholder_liquidity",
        "alpha_is_real": False,
        "production_eligible": False,
        "model_form": "p = g(alpha_placeholder, c)",
        "condition_set": "industry_board_index_size_v1",
    }
    assert summary["panel"]["row_count"] == 15
    assert summary["feature_roles"]["alpha_placeholder"] == [
        "alpha_placeholder_turnover20_z",
        "alpha_placeholder_turnover20_rank",
        "alpha_placeholder_logADV20_z",
        "alpha_placeholder_logADV20_rank",
    ]
    assert (diagnostics_dir / "feature_standardization_diagnostics.csv").exists()
    assert (diagnostics_dir / "split_summary.csv").exists()


def test_write_placeholder_training_data_artifacts_records_embargoed_dates(tmp_path) -> None:
    output_panel = tmp_path / "panel.pkl"
    output_folds = tmp_path / "folds.csv"
    output_summary = tmp_path / "summary.json"
    diagnostics_dir = tmp_path / "diagnostics"
    folds = [
        WalkForwardFold(
            fold_id=1,
            train_start="2024-09-30",
            train_end="2024-09-30",
            valid_start="2024-10-01",
            valid_end="2024-12-31",
            test_start="2025-01-01",
            test_end="2025-01-01",
        )
    ]

    write_placeholder_training_data_artifacts(
        exposures=_make_exposures(),
        y_resid=_make_labels()[0],
        y_rank_label=_make_labels()[1],
        output_panel=output_panel,
        output_fold_assignments=output_folds,
        output_summary=output_summary,
        diagnostics_dir=diagnostics_dir,
        folds=folds,
        embargo_trading_days=2,
    )

    summary = json.loads(output_summary.read_text(encoding="utf-8"))
    assignments = pd.read_csv(output_folds)
    embargoed = pd.read_csv(diagnostics_dir / "embargoed_dates.csv")

    assert summary["embargo_trading_days"] == 2
    assert summary["embargoed_date_count"] == 3
    assert set(embargoed["boundary"]) == {"train_to_valid", "valid_to_test"}
    assert set(embargoed["date"]) == {"2024-09-30", "2024-10-01", "2024-12-31"}
    assert set(assignments["date"]) == {"2025-01-01"}
