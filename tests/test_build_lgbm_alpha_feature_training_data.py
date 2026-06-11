from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.build_lgbm_alpha_feature_training_data import (
    build_alpha_feature_training_panel,
    write_alpha_feature_training_data_artifacts,
)


def _make_context_exposures() -> pd.DataFrame:
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
                }
            )
    return pd.DataFrame(rows)


def _make_labels() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    columns = ["000001", "000002", "000003", "000004"]
    y_resid = pd.DataFrame(
        [
            [0.01, 0.02, -0.01, np.nan],
            [0.02, 0.01, -0.02, -0.03],
        ],
        index=dates,
        columns=columns,
    )
    y_rank = pd.DataFrame(
        [
            [0.333333, 0.0, -0.333333, np.nan],
            [0.375, 0.125, -0.125, -0.375],
        ],
        index=dates,
        columns=columns,
    )
    return y_resid, y_rank


def _stock_by_date(values: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(
        values,
        index=["000001", "000002", "000003", "000004"],
        columns=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )


def _make_signal_values() -> dict[str, pd.DataFrame]:
    return {
        "factor_sss_dx_10": _stock_by_date(
            [
                [1.0, 4.0],
                [2.0, 3.0],
                [3.0, 2.0],
                [4.0, 1.0],
            ]
        ),
        "amo": _stock_by_date(
            [
                [10.0, 40.0],
                [20.0, 30.0],
                [30.0, 20.0],
                [40.0, 10.0],
            ]
        ),
        "close": _stock_by_date(
            [
                [5.0, 8.0],
                [6.0, 7.0],
                [7.0, 6.0],
                [8.0, 5.0],
            ]
        ),
        "vol": _stock_by_date(
            [
                [100.0, 400.0],
                [200.0, 300.0],
                [300.0, 200.0],
                [400.0, 100.0],
            ]
        ),
    }


def test_build_alpha_feature_panel_uses_feature_universe_before_label_join() -> None:
    panel, diagnostics = build_alpha_feature_training_panel(
        exposures=_make_context_exposures(),
        y_resid=_make_labels()[0],
        y_rank_label=_make_labels()[1],
        signal_value_frames=_make_signal_values(),
        winsor_lower=0.0,
        winsor_upper=1.0,
    )

    first_day = panel[panel["date"].eq(pd.Timestamp("2024-01-02"))].sort_values("stock_code")
    assert first_day["stock_code"].tolist() == ["000001", "000002", "000003"]
    assert first_day["factor_sss_dx_10_rank"].round(12).tolist() == [-0.375, -0.125, 0.125]
    assert np.isclose(first_day.loc[first_day["stock_code"].eq("000001"), "factor_sss_dx_10_z"].iloc[0], -1.011736, atol=1e-6)

    assert diagnostics["summary"]["dropped_missing_label_rows"] == 1
    assert diagnostics["summary"]["feature_transform_universe"] == "same_date_context_and_signal_rows_before_label_join"

    expected_signal_features = [
        "factor_sss_dx_10_rank",
        "factor_sss_dx_10_z",
        "amo_rank",
        "amo_z",
        "close_rank",
        "close_z",
        "vol_rank",
        "vol_z",
    ]
    assert diagnostics["feature_roles"]["alpha_placeholder"] == expected_signal_features
    assert diagnostics["feature_roles"]["condition_categorical"] == [
        "industry",
        "board",
        "index_bucket",
        "size_decile",
    ]
    assert diagnostics["feature_roles"]["condition_continuous"] == [
        "is_csi300",
        "is_csi500",
        "is_csi1000",
        "is_csi2000",
        "log_mcap_z",
        "mcap_rank",
        "amount_k_rank",
        "amount_k_z",
        "turnover_rank",
        "turnover_z",
        "logADV20_rank",
        "logADV20_z",
        "logAmount20_rank",
        "logAmount20_z",
        "turnover20_rank",
        "turnover20_z",
    ]
    assert {
        "amount_k",
        "turnover",
        "logADV20",
        "logAmount20",
        "turnover20",
    }.issubset(diagnostics["feature_roles"]["traceability"])
    assert {
        "amount_k",
        "turnover",
        "logADV20",
        "logAmount20",
        "turnover20",
    }.issubset(diagnostics["feature_roles"]["excluded_from_model"])
    expected_liquidity_features = [
        "amount_k_rank",
        "amount_k_z",
        "turnover_rank",
        "turnover_z",
        "logADV20_rank",
        "logADV20_z",
        "logAmount20_rank",
        "logAmount20_z",
        "turnover20_rank",
        "turnover20_z",
    ]
    assert set(expected_signal_features).issubset(panel.columns)
    assert not panel[expected_signal_features].isna().any().any()
    assert set(expected_liquidity_features).issubset(panel.columns)
    assert not panel[expected_liquidity_features].isna().any().any()
    assert first_day["amount_k_rank"].round(12).tolist() == [-0.375, -0.125, 0.125]
    assert set(panel["index_bucket"]) == {"CSI300", "CSI500", "CSI1000", "CSI2000"}
    diagnostic_features = set(diagnostics["feature_standardization"]["feature"])
    assert diagnostic_features == {
        *expected_signal_features,
        *expected_liquidity_features,
        "log_mcap_z",
        "mcap_rank",
    }


def test_build_alpha_feature_panel_drops_invalid_domain_values_after_transforming_by_date() -> None:
    signal_values = _make_signal_values()
    signal_values["close"] = signal_values["close"].copy()
    signal_values["close"].loc["000003", pd.Timestamp("2024-01-03")] = 0.0

    panel, diagnostics = build_alpha_feature_training_panel(
        exposures=_make_context_exposures(),
        y_resid=_make_labels()[0],
        y_rank_label=_make_labels()[1],
        signal_value_frames=signal_values,
        winsor_lower=0.0,
        winsor_upper=1.0,
    )

    invalid_close_row = panel[
        panel["date"].eq(pd.Timestamp("2024-01-03")) & panel["stock_code"].eq("000003")
    ]
    assert invalid_close_row.empty
    assert diagnostics["summary"]["dropped_missing_feature_rows"] == 1
    assert diagnostics["signal_inputs"]["close"]["invalid_domain_count"] == 1


def test_build_alpha_feature_panel_requires_liquidity_context_features() -> None:
    exposures = _make_context_exposures()
    exposures.loc[
        exposures["date"].eq(pd.Timestamp("2024-01-03"))
        & exposures["stock_code"].eq("000004"),
        "turnover20",
    ] = np.nan

    panel, diagnostics = build_alpha_feature_training_panel(
        exposures=exposures,
        y_resid=_make_labels()[0],
        y_rank_label=_make_labels()[1],
        signal_value_frames=_make_signal_values(),
        winsor_lower=0.0,
        winsor_upper=1.0,
    )

    dropped_row = panel[
        panel["date"].eq(pd.Timestamp("2024-01-03")) & panel["stock_code"].eq("000004")
    ]
    assert dropped_row.empty
    assert diagnostics["summary"]["dropped_missing_feature_rows"] == 1


def test_write_alpha_feature_training_data_artifacts(tmp_path: Path) -> None:
    output_panel = tmp_path / "panel.pkl"
    output_folds = tmp_path / "folds.csv"
    output_summary = tmp_path / "summary.json"
    diagnostics_dir = tmp_path / "diagnostics"

    summary = write_alpha_feature_training_data_artifacts(
        exposures=_make_context_exposures(),
        y_resid=_make_labels()[0],
        y_rank_label=_make_labels()[1],
        signal_value_frames=_make_signal_values(),
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
    assert loaded_summary["metadata"]["signal_stage"] == "real_alpha_kline_feature_panel"
    assert loaded_summary["metadata"]["alpha_source"] == "factor_sss_dx_10"
    assert loaded_summary["feature_roles"] == summary["feature_roles"]
    assert not Path(loaded_summary["outputs"]["panel"]).is_absolute()
    assert (diagnostics_dir / "feature_standardization_diagnostics.csv").exists()
    assert (diagnostics_dir / "label_distribution_by_date.csv").exists()
