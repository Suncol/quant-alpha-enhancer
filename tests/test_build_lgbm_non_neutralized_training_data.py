from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analysis.build_lgbm_non_neutralized_training_data as non_neutralized_builder
from analysis.build_lgbm_non_neutralized_training_data import (
    DEFAULT_RAW_RETURN_TARGET_COLUMN,
    build_non_neutralized_training_panel,
    build_raw_feature_table_from_exposures,
    write_non_neutralized_training_data_artifacts,
)


def _make_raw_features() -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
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
                    "amount_k": float(10 + 10 * stock_idx + date_idx),
                    "turnover": float(0.01 + 0.01 * stock_idx + 0.001 * date_idx),
                    "logADV20": float(5.0 + stock_idx + 0.1 * date_idx),
                    "logAmount20": float(6.0 + stock_idx + 0.1 * date_idx),
                    "turnover20": float(0.02 + 0.01 * stock_idx + 0.001 * date_idx),
                    "factor_sss_dx_10_raw": float(1 + stock_idx + 3 * date_idx),
                }
            )
    return pd.DataFrame(rows)


def _make_raw_features_with_kline() -> pd.DataFrame:
    frame = _make_raw_features()
    per_stock_values = {
        "000001": (10.0, 0.0, 100.0),
        "000002": (20.0, 10.0, 200.0),
        "000003": (30.0, 20.0, 300.0),
        "000004": (40.0, 30.0, 400.0),
    }
    close_raw: list[float] = []
    amo_raw: list[float] = []
    vol_raw: list[float] = []
    for row in frame.itertuples(index=False):
        close, amo, vol = per_stock_values[row.stock_code]
        date_offset = 1.0 if pd.Timestamp(row.date) == pd.Timestamp("2024-01-03") else 0.0
        close_raw.append(close + date_offset)
        amo_raw.append(amo + date_offset)
        vol_raw.append(vol + date_offset)
    frame["close_raw"] = close_raw
    frame["amo_raw"] = amo_raw
    frame["vol_raw"] = vol_raw
    return frame


def _make_raw_returns() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [0.11, 0.12, -0.13, np.nan],
            [0.21, 0.22, -0.23, -0.24],
        ],
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        columns=["000001", "000002", "000003", "000004"],
    )


def test_build_raw_feature_table_from_exposures_merges_raw_alpha_without_neutralizing() -> None:
    exposures = _make_raw_features().drop(columns=["factor_sss_dx_10_raw"])
    alpha_values = pd.DataFrame(
        [
            [1.0, 4.0],
            [2.0, 5.0],
            [3.0, 6.0],
            [4.0, 7.0],
        ],
        index=["000001", "000002", "000003", "000004"],
        columns=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )

    raw_features = build_raw_feature_table_from_exposures(
        exposures,
        alpha_values,
        alpha_raw_column="factor_sss_dx_10_raw",
    )

    assert len(raw_features) == len(exposures)
    assert raw_features["factor_sss_dx_10_raw"].tolist() == [1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 7.0]
    assert not any("_neutralized" in column for column in raw_features.columns)


def test_build_non_neutralized_panel_uses_raw_return_target_without_residualization() -> None:
    panel, diagnostics = build_non_neutralized_training_panel(
        raw_features=_make_raw_features(),
        raw_return=_make_raw_returns(),
        winsor_lower=0.0,
        winsor_upper=1.0,
    )

    assert DEFAULT_RAW_RETURN_TARGET_COLUMN == "y_return_hfq_adj_fwd"
    assert DEFAULT_RAW_RETURN_TARGET_COLUMN in panel.columns
    assert "y_resid_fwd" not in panel.columns
    assert "y_rank_label" not in panel.columns
    assert not any("_neutralized" in column for column in panel.columns)

    first_day = panel[panel["date"].eq(pd.Timestamp("2024-01-02"))].sort_values("stock_code")
    assert first_day["stock_code"].tolist() == ["000001", "000002", "000003"]
    assert first_day[DEFAULT_RAW_RETURN_TARGET_COLUMN].tolist() == [0.11, 0.12, -0.13]

    # Ranks prove alpha transforms used the four-stock feature universe before
    # joining labels; a three-stock label universe would give [-1/3, 0, 1/3].
    assert first_day["factor_sss_dx_10_rank"].round(12).tolist() == [-0.375, -0.125, 0.125]

    assert diagnostics["summary"]["return_neutralization"] == {"enabled": False}
    assert diagnostics["summary"]["alpha_signal_neutralization"] == {"enabled": False}
    assert diagnostics["summary"]["target_columns"] == ["y_return_hfq_adj_fwd"]
    assert diagnostics["summary"]["auxiliary_label_columns"] == ["y_return_rank_label"]
    assert diagnostics["summary"]["target_transform"] == (
        "raw_forward_return_no_winsorization_no_residualization"
    )
    assert diagnostics["feature_roles"]["targets"] == ["y_return_hfq_adj_fwd"]
    assert diagnostics["summary"]["dropped_missing_label_rows"] == 1


def test_build_non_neutralized_panel_rejects_overlapping_index_flags() -> None:
    raw_features = _make_raw_features()
    raw_features.loc[0, "is_csi500"] = 1

    try:
        build_non_neutralized_training_panel(
            raw_features=raw_features,
            raw_return=_make_raw_returns(),
            winsor_lower=0.0,
            winsor_upper=1.0,
        )
    except ValueError as exc:
        assert "multiple index membership flags" in str(exc)
    else:
        raise AssertionError("Expected overlapping index membership flags to fail loudly.")


def test_build_non_neutralized_panel_can_exclude_all_index_context_features() -> None:
    raw_features = _make_raw_features().drop(
        columns=["is_csi300", "is_csi500", "is_csi1000", "is_csi2000"]
    )

    panel, diagnostics = build_non_neutralized_training_panel(
        raw_features=raw_features,
        raw_return=_make_raw_returns(),
        include_index_context_features=False,
        winsor_lower=0.0,
        winsor_upper=1.0,
    )

    index_features = {"index_bucket", "is_csi300", "is_csi500", "is_csi1000", "is_csi2000"}
    assert not index_features.intersection(panel.columns)
    assert not index_features.intersection(diagnostics["feature_roles"]["condition_categorical"])
    assert not index_features.intersection(diagnostics["feature_roles"]["condition_continuous"])
    assert diagnostics["summary"]["index_context_features_in_model"] is False


def test_build_non_neutralized_panel_can_include_kline_rank_z_features_without_expanding_signal_role() -> None:
    panel, diagnostics = build_non_neutralized_training_panel(
        raw_features=_make_raw_features_with_kline(),
        raw_return=_make_raw_returns(),
        include_kline_signal_features=True,
        winsor_lower=0.0,
        winsor_upper=1.0,
    )

    expected_kline_features = ["close_rank", "close_z", "amo_rank", "amo_z", "vol_rank", "vol_z"]
    for column in expected_kline_features:
        assert column in panel.columns

    assert diagnostics["feature_roles"]["signal_features"] == [
        "factor_sss_dx_10_rank",
        "factor_sss_dx_10_z",
    ]
    assert diagnostics["feature_roles"]["alpha_placeholder"] == [
        "factor_sss_dx_10_rank",
        "factor_sss_dx_10_z",
    ]
    for column in expected_kline_features:
        assert column in diagnostics["feature_roles"]["condition_continuous"]
        assert column not in diagnostics["feature_roles"]["signal_features"]

    for raw_column in ["close_raw", "amo_raw", "vol_raw"]:
        assert raw_column in diagnostics["feature_roles"]["traceability"]
        assert raw_column in diagnostics["feature_roles"]["excluded_from_model"]

    first_day = panel[panel["date"].eq(pd.Timestamp("2024-01-02"))].sort_values("stock_code")
    assert first_day["stock_code"].tolist() == ["000001", "000002", "000003"]
    assert first_day["close_rank"].round(12).tolist() == [-0.375, -0.125, 0.125]
    assert first_day["amo_rank"].round(12).tolist() == [-0.375, -0.125, 0.125]
    assert first_day["vol_rank"].round(12).tolist() == [-0.375, -0.125, 0.125]
    assert np.isfinite(first_day[["close_z", "amo_z", "vol_z"]].to_numpy()).all()
    assert diagnostics["summary"]["kline_signal_features_in_model"] is True
    assert diagnostics["input_diagnostics"]["include_kline_signal_features"] is True


def test_build_non_neutralized_panel_kline_features_fail_loudly_when_raw_columns_missing() -> None:
    raw_features = _make_raw_features_with_kline().drop(columns=["close_raw"])

    try:
        build_non_neutralized_training_panel(
            raw_features=raw_features,
            raw_return=_make_raw_returns(),
            include_kline_signal_features=True,
            winsor_lower=0.0,
            winsor_upper=1.0,
        )
    except ValueError as exc:
        message = str(exc)
        assert "raw_features missing required columns" in message
        assert "close_raw" in message
    else:
        raise AssertionError("Expected missing Kline raw columns to fail loudly.")


def test_write_non_neutralized_training_data_artifacts(tmp_path: Path) -> None:
    output_panel = tmp_path / "panel.pkl"
    output_folds = tmp_path / "folds.csv"
    output_summary = tmp_path / "training_summary.json"
    diagnostics_dir = tmp_path / "diagnostics"

    summary = write_non_neutralized_training_data_artifacts(
        raw_features=_make_raw_features(),
        raw_return=_make_raw_returns(),
        output_panel=output_panel,
        output_fold_assignments=output_folds,
        output_summary=output_summary,
        diagnostics_dir=diagnostics_dir,
        winsor_lower=0.0,
        winsor_upper=1.0,
        embargo_trading_days=0,
    )

    panel = pd.read_pickle(output_panel)
    folds = pd.read_csv(output_folds)
    loaded_summary = json.loads(output_summary.read_text(encoding="utf-8"))

    assert len(panel) == 7
    assert {"fold_id", "date", "split"}.issubset(folds.columns)
    assert loaded_summary["metadata"]["signal_stage"] == "non_neutralized_real_alpha_raw_return_panel"
    assert loaded_summary["metadata"]["optimization_objective"] == "raw_forward_return"
    assert loaded_summary["metadata"]["neutralization_policy"] == {
        "return_y": "not_neutralized",
        "alpha_signal": "not_neutralized",
    }
    assert loaded_summary["feature_roles"] == summary["feature_roles"]
    assert not Path(loaded_summary["outputs"]["panel"]).is_absolute()
    assert (diagnostics_dir / "feature_standardization_diagnostics.csv").exists()
    assert (diagnostics_dir / "raw_return_distribution_by_date.csv").exists()


def test_write_non_neutralized_training_data_can_use_longer_train_start_without_changing_test_windows(
    tmp_path: Path,
) -> None:
    output_panel = tmp_path / "panel.pkl"
    output_folds = tmp_path / "folds.csv"
    output_summary = tmp_path / "training_summary.json"
    diagnostics_dir = tmp_path / "diagnostics"
    folds = non_neutralized_builder.default_walk_forward_folds_with_train_start("2017-01-01")

    summary = write_non_neutralized_training_data_artifacts(
        raw_features=_make_raw_features(),
        raw_return=_make_raw_returns(),
        output_panel=output_panel,
        output_fold_assignments=output_folds,
        output_summary=output_summary,
        diagnostics_dir=diagnostics_dir,
        folds=folds,
        winsor_lower=0.0,
        winsor_upper=1.0,
        embargo_trading_days=0,
    )

    assert {fold["train_start"] for fold in summary["folds"]} == {"2017-01-01"}
    assert [fold["test_start"] for fold in summary["folds"]] == [
        "2025-01-01",
        "2025-07-01",
        "2026-01-01",
    ]
