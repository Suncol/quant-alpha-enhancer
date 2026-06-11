from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from analysis.neutralize_return_y import (
    BasicNeutralizationConfig,
    centered_rank,
    neutralize_return_y_matrix,
)


def _make_exposures(dates: list[str], stock_codes: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    industry_cycle = ["bank", "bank", "tech", "tech", "energy", "energy"]
    board_cycle = ["main", "growth", "main", "growth"]
    for date in dates:
        for idx, stock_code in enumerate(stock_codes):
            rows.append(
                {
                    "date": date,
                    "stock_code": stock_code,
                    "industry": industry_cycle[idx % len(industry_cycle)],
                    "board": board_cycle[idx % len(board_cycle)],
                    "is_csi300": 1 if idx in {0, 1, 2} else 0,
                    "is_csi500": 1 if idx in {3, 4, 5} else 0,
                    "is_csi1000": 1 if idx in {6, 7, 8} else 0,
                    "is_csi2000": 1 if idx in {9, 10} else 0,
                    "market_cap": float(10 + idx * 3),
                }
            )
    return pd.DataFrame(rows)


class ReturnYNeutralizationTests(unittest.TestCase):
    def assert_residual_is_neutral(
        self,
        residual_row: pd.Series,
        exposure_rows: pd.DataFrame,
        *,
        tol: float = 1e-10,
    ) -> None:
        used = exposure_rows.set_index("stock_code").loc[residual_row.dropna().index].copy()
        values = residual_row.dropna().to_numpy(dtype=float)
        self.assertLess(abs(float(values.mean())), tol)

        for column in ["industry", "board"]:
            means = pd.Series(values, index=used.index).groupby(used[column]).mean()
            self.assertLess(float(means.abs().max()), tol)

        size_rank = centered_rank(used["market_cap"]).to_numpy(dtype=float)
        continuous = pd.DataFrame(
            {
                "is_csi300": used["is_csi300"].to_numpy(dtype=float),
                "is_csi500": used["is_csi500"].to_numpy(dtype=float),
                "is_csi1000": used["is_csi1000"].to_numpy(dtype=float),
                "is_csi2000": used["is_csi2000"].to_numpy(dtype=float),
                "size_rank_norm": size_rank,
            },
            index=used.index,
        )
        for column in continuous.columns:
            if continuous[column].std(ddof=0) == 0:
                continue
            exposure = float(np.mean(continuous[column].to_numpy(dtype=float) * values))
            self.assertLess(abs(exposure), tol)

    def test_centered_rank_keeps_missing_values_and_has_zero_mean(self) -> None:
        ranked = centered_rank(pd.Series([10.0, 20.0, 20.0, np.nan]))

        self.assertAlmostEqual(float(ranked.iloc[0]), -1.0 / 3.0)
        self.assertAlmostEqual(float(ranked.iloc[1]), 1.0 / 6.0)
        self.assertAlmostEqual(float(ranked.iloc[2]), 1.0 / 6.0)
        self.assertTrue(np.isnan(float(ranked.iloc[3])))
        self.assertAlmostEqual(float(ranked.dropna().mean()), 0.0)

    def test_basic_neutralization_removes_daily_exposures(self) -> None:
        dates = ["2024-01-02", "2024-01-03"]
        stock_codes = [f"{idx:06d}" for idx in range(1, 13)]
        exposures = _make_exposures(dates, stock_codes)

        returns = pd.DataFrame(index=pd.to_datetime(dates), columns=stock_codes, dtype=float)
        for date in dates:
            day = exposures[exposures["date"] == date].copy()
            size_rank = centered_rank(day["market_cap"]).to_numpy(dtype=float)
            industry_effect = day["industry"].map({"bank": 0.05, "tech": -0.03, "energy": 0.02}).to_numpy()
            board_effect = day["board"].map({"main": 0.01, "growth": -0.01}).to_numpy()
            index_effect = (
                0.02 * day["is_csi300"].to_numpy(dtype=float)
                - 0.015 * day["is_csi500"].to_numpy(dtype=float)
                + 0.005 * day["is_csi1000"].to_numpy(dtype=float)
                - 0.010 * day["is_csi2000"].to_numpy(dtype=float)
            )
            stock_selection = np.array(
                [-0.012, 0.006, 0.011, -0.003, 0.014, -0.009, 0.004, -0.006, 0.008, -0.002, 0.005, -0.016]
            )
            returns.loc[pd.Timestamp(date), stock_codes] = (
                0.001 + industry_effect + board_effect + index_effect + 0.04 * size_rank + stock_selection
            )

        residual, rank_label, diagnostics, summary = neutralize_return_y_matrix(
            returns,
            exposures,
            config=BasicNeutralizationConfig(
                ridge=0.0,
                min_obs=8,
                min_obs_per_column_buffer=0,
                winsor_lower=None,
                winsor_upper=None,
            ),
        )

        self.assertEqual(residual.shape, returns.shape)
        self.assertEqual(rank_label.shape, returns.shape)
        self.assertEqual(list(residual.columns), stock_codes)
        self.assertEqual(summary["date_count"], 2)
        self.assertEqual(summary["processed_date_count"], 2)
        self.assertEqual(int(diagnostics["skipped"].sum()), 0)

        for _, row in diagnostics.iterrows():
            self.assertLess(row["max_abs_industry_mean_after"], 1e-12)
            self.assertLess(row["max_abs_board_mean_after"], 1e-12)
            self.assertLess(row["max_abs_continuous_exposure_after"], 1e-12)
            self.assertLess(abs(row["residual_weighted_mean"]), 1e-12)

        for date in returns.index:
            values = rank_label.loc[date].dropna()
            self.assertAlmostEqual(float(values.mean()), 0.0)
            self.assertGreater(float(values.max()), 0.0)
            self.assertLess(float(values.min()), 0.0)

    def test_start_date_and_missing_exposures_are_explicit(self) -> None:
        dates = ["2023-12-29", "2024-01-02"]
        stock_codes = [f"{idx:06d}" for idx in range(1, 13)]
        returns = pd.DataFrame(index=pd.to_datetime(dates), columns=stock_codes, dtype=float)
        returns.loc[pd.Timestamp("2023-12-29")] = np.linspace(-0.03, 0.03, len(stock_codes))
        returns.loc[pd.Timestamp("2024-01-02")] = np.linspace(0.04, -0.02, len(stock_codes))
        exposures = _make_exposures(dates, stock_codes)
        exposures = exposures[~((exposures["date"] == "2024-01-02") & (exposures["stock_code"] == "000012"))]

        residual, _, diagnostics, summary = neutralize_return_y_matrix(
            returns,
            exposures,
            config=BasicNeutralizationConfig(
                start_date="2024-01-01",
                min_obs=8,
                min_obs_per_column_buffer=0,
                winsor_lower=None,
                winsor_upper=None,
            ),
        )

        self.assertTrue(residual.loc[pd.Timestamp("2023-12-29")].isna().all())
        self.assertTrue(np.isnan(float(residual.loc[pd.Timestamp("2024-01-02"), "000012"])))
        self.assertEqual(summary["processed_date_count"], 1)
        self.assertEqual(summary["skipped_date_count"], 1)

        skipped = diagnostics[diagnostics["date"] == "2023-12-29"].iloc[0]
        processed = diagnostics[diagnostics["date"] == "2024-01-02"].iloc[0]
        self.assertEqual(skipped["skipped_reason"], "before_start_date")
        self.assertEqual(int(processed["missing_exposure_count"]), 1)
        self.assertFalse(bool(processed["skipped"]))

    def test_exact_config_neutralizes_raw_returns_without_winsorizing(self) -> None:
        dates = ["2024-01-02"]
        stock_codes = [f"{idx:06d}" for idx in range(1, 25)]
        exposures = _make_exposures(dates, stock_codes)
        returns = pd.DataFrame(index=pd.to_datetime(dates), columns=stock_codes, dtype=float)
        base = np.linspace(-0.04, 0.04, len(stock_codes))
        base[-1] = 1.25
        returns.loc[pd.Timestamp(dates[0]), stock_codes] = base

        exact_residual, _, exact_diagnostics, _ = neutralize_return_y_matrix(
            returns,
            exposures,
            config=BasicNeutralizationConfig(
                ridge=0.0,
                winsor_lower=None,
                winsor_upper=None,
                min_obs=8,
                min_obs_per_column_buffer=0,
            ),
        )
        winsor_residual, _, _, _ = neutralize_return_y_matrix(
            returns,
            exposures,
            config=BasicNeutralizationConfig(
                winsor_lower=0.01,
                winsor_upper=0.99,
                min_obs=8,
                min_obs_per_column_buffer=0,
            ),
        )

        self.assertFalse(bool(exact_diagnostics.iloc[0]["skipped"]))
        self.assertGreater(
            float(exact_residual.abs().max(axis=1).iloc[0]),
            float(winsor_residual.abs().max(axis=1).iloc[0]),
        )
        self.assert_residual_is_neutral(
            exact_residual.loc[pd.Timestamp(dates[0])],
            exposures[exposures["date"] == dates[0]],
            tol=1e-10,
        )

    def test_default_config_uses_winsorized_approximate_neutralization(self) -> None:
        date = "2024-01-02"
        stock_codes = [f"{idx:06d}" for idx in range(1, 25)]
        exposures = _make_exposures([date], stock_codes)
        returns = pd.DataFrame(index=pd.to_datetime([date]), columns=stock_codes, dtype=float)
        base = np.linspace(-0.04, 0.04, len(stock_codes))
        base[-1] = 1.25
        returns.loc[pd.Timestamp(date), stock_codes] = base

        residual, _, diagnostics, summary = neutralize_return_y_matrix(
            returns,
            exposures,
            config=BasicNeutralizationConfig(min_obs=8, min_obs_per_column_buffer=0),
        )
        exact_residual, _, _, _ = neutralize_return_y_matrix(
            returns,
            exposures,
            config=BasicNeutralizationConfig(
                ridge=0.0,
                winsor_lower=None,
                winsor_upper=None,
                min_obs=8,
                min_obs_per_column_buffer=0,
            ),
        )

        row = diagnostics.iloc[0]
        self.assertFalse(bool(row["skipped"]))
        self.assertEqual(row["neutralization_type"], "approximate_neutralization")
        self.assertEqual(row["neutralization_label"], "近似中性化")
        self.assertEqual(row["input_return_transform"], "winsorized")
        self.assertAlmostEqual(float(row["ridge"]), 1e-8)
        self.assertAlmostEqual(float(row["winsor_lower"]), 0.01)
        self.assertAlmostEqual(float(row["winsor_upper"]), 0.99)
        self.assertEqual(summary["neutralization_type"], "approximate_neutralization")
        self.assertEqual(summary["neutralization_label"], "近似中性化")
        self.assertEqual(summary["input_return_transform"], "winsorized")
        self.assertLess(
            float(residual.abs().max(axis=1).iloc[0]),
            float(exact_residual.abs().max(axis=1).iloc[0]),
        )

    def test_extra_continuous_columns_are_neutralized_when_configured(self) -> None:
        date = "2024-01-02"
        stock_codes = [f"{idx:06d}" for idx in range(1, 25)]
        exposures = _make_exposures([date], stock_codes)
        exposures["logADV20"] = np.sin(np.arange(len(stock_codes), dtype=float))
        returns = pd.DataFrame(index=pd.to_datetime([date]), columns=stock_codes, dtype=float)
        returns.loc[pd.Timestamp(date), stock_codes] = (
            0.02 * centered_rank(exposures["market_cap"]).to_numpy(dtype=float)
            + 0.03 * exposures["logADV20"].to_numpy(dtype=float)
        )

        residual, _, diagnostics, _ = neutralize_return_y_matrix(
            returns,
            exposures,
            config=BasicNeutralizationConfig(
                ridge=0.0,
                winsor_lower=None,
                winsor_upper=None,
                extra_continuous_cols=("logADV20",),
                min_obs=8,
                min_obs_per_column_buffer=0,
            ),
        )

        self.assertFalse(bool(diagnostics.iloc[0]["skipped"]))
        self.assertIn("logADV20", diagnostics.iloc[0]["continuous_columns_used"])
        values = residual.loc[pd.Timestamp(date)].dropna()
        used = exposures.set_index("stock_code").loc[values.index]
        extra_exposure = float(np.mean(used["logADV20"].to_numpy(dtype=float) * values.to_numpy(dtype=float)))
        self.assertLess(abs(extra_exposure), 1e-10)

    def test_size_rank_is_computed_on_used_fit_universe(self) -> None:
        date = "2024-01-02"
        stock_codes = [f"{idx:06d}" for idx in range(1, 31)]
        exposures = _make_exposures([date], stock_codes)
        extra_rows = []
        for idx in range(10):
            extra_rows.append(
                {
                    "date": date,
                    "stock_code": f"9{idx:05d}",
                    "industry": "extra",
                    "board": "extra_board",
                    "is_csi300": 0,
                    "is_csi500": 0,
                    "is_csi1000": 0,
                    "is_csi2000": 1,
                    "market_cap": float(11 + idx * 3),
                }
            )
        exposures = pd.concat([exposures, pd.DataFrame(extra_rows)], ignore_index=True)

        used_exposures = exposures[exposures["stock_code"].isin(stock_codes)].copy()
        used_size_rank = centered_rank(used_exposures["market_cap"]).to_numpy(dtype=float)
        returns = pd.DataFrame(
            [0.03 * used_size_rank],
            index=pd.to_datetime([date]),
            columns=stock_codes,
            dtype=float,
        )

        residual, _, diagnostics, _ = neutralize_return_y_matrix(
            returns,
            exposures,
            config=BasicNeutralizationConfig(
                ridge=0.0,
                winsor_lower=None,
                winsor_upper=None,
                min_obs=8,
                min_obs_per_column_buffer=0,
            ),
        )

        self.assertFalse(bool(diagnostics.iloc[0]["skipped"]))
        self.assertLess(float(residual.abs().max(axis=1).iloc[0]), 1e-12)


if __name__ == "__main__":
    unittest.main()
