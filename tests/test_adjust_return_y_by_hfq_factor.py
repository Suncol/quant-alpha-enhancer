from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from analysis.adjust_return_y_by_hfq_factor import (
    AdjustmentConfig,
    adjust_stock_returns,
    build_trade_date_map,
)


class ReturnYAdjustmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trading_days = pd.DatetimeIndex(
            pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-08",
                    "2024-01-09",
                    "2024-01-10",
                    "2024-01-11",
                ]
            )
        )

    def test_adjustment_uses_signal_day_plus_one_and_plus_two_factors(self) -> None:
        raw = pd.Series(
            [-0.75],
            index=pd.DatetimeIndex(pd.to_datetime(["2024-01-02"])),
            name="000001",
        )
        factors = pd.Series(
            [1.0, 1.0, 4.0],
            index=pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])),
            name="adjustment_factor",
        )
        date_map = build_trade_date_map(raw.index, self.trading_days)

        adjusted, diagnostics = adjust_stock_returns(
            "000001.SZ",
            raw,
            factors,
            date_map,
            AdjustmentConfig(),
        )

        self.assertAlmostEqual(float(adjusted.iloc[0]), 0.0)
        self.assertEqual(diagnostics.iloc[0]["buy_date"], "2024-01-03")
        self.assertEqual(diagnostics.iloc[0]["sell_date"], "2024-01-04")
        self.assertEqual(diagnostics.iloc[0]["adjustment_action"], "adjusted_by_factor_ratio")

    def test_factor_ratio_equal_one_keeps_raw_return(self) -> None:
        raw = pd.Series(
            [0.034, -0.012],
            index=pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03"])),
            name="000001",
        )
        factors = pd.Series(
            [2.0, 2.0, 2.0, 2.0],
            index=pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])),
            name="adjustment_factor",
        )
        date_map = build_trade_date_map(raw.index, self.trading_days)

        adjusted, diagnostics = adjust_stock_returns(
            "000001.SZ",
            raw,
            factors,
            date_map,
            AdjustmentConfig(),
        )

        pd.testing.assert_series_equal(adjusted, raw.astype(float), check_names=False)
        self.assertTrue(diagnostics.empty)

    def test_suspicious_factor_reversal_keeps_raw_by_default(self) -> None:
        raw = pd.Series(
            [0.034448328],
            index=pd.DatetimeIndex(pd.to_datetime(["2024-01-02"])),
            name="000001",
        )
        factors = pd.Series(
            [108.031, 66.3145, 108.031, 108.031],
            index=pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])),
            name="adjustment_factor",
        )
        date_map = build_trade_date_map(raw.index, self.trading_days)

        adjusted, diagnostics = adjust_stock_returns(
            "000001.SZ",
            raw,
            factors,
            date_map,
            AdjustmentConfig(suspicious_policy="keep_raw"),
        )

        self.assertAlmostEqual(float(adjusted.iloc[0]), float(raw.iloc[0]))
        self.assertTrue(bool(diagnostics.iloc[0]["suspicious_factor_reversal_flag"]))
        self.assertEqual(
            diagnostics.iloc[0]["adjustment_action"],
            "kept_raw_suspicious_factor_reversal",
        )

    def test_missing_buy_or_sell_factor_sets_adjusted_return_to_nan(self) -> None:
        raw = pd.Series(
            [0.05],
            index=pd.DatetimeIndex(pd.to_datetime(["2024-01-02"])),
            name="000001",
        )
        factors = pd.Series(
            [1.0],
            index=pd.DatetimeIndex(pd.to_datetime(["2024-01-03"])),
            name="adjustment_factor",
        )
        date_map = build_trade_date_map(raw.index, self.trading_days)

        adjusted, diagnostics = adjust_stock_returns(
            "000001.SZ",
            raw,
            factors,
            date_map,
            AdjustmentConfig(),
        )

        self.assertTrue(np.isnan(float(adjusted.iloc[0])))
        self.assertEqual(diagnostics.iloc[0]["adjustment_action"], "nan_missing_factor")


if __name__ == "__main__":
    unittest.main()
