from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

from analysis.build_stock_daily_metrics_sw_static_dataset import (
    OUTPUT_COLUMNS,
    REQUESTED_DAILY_COLUMNS,
    _path_for_summary,
    build_stock_daily_metrics_sw_static_panel,
    write_stock_daily_metrics_sw_static_dataset,
)


def _write_daily_metrics_dataset(root: Path) -> None:
    daily_root = root / "daily_metrics"
    (daily_root / "2024-01").mkdir(parents=True)
    (daily_root / "2024-02").mkdir(parents=True)
    first = pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "trade_date": "2024-01-02",
                "change": 0.10,
                "pct_change": 1.25,
                "volume": 1000.0,
                "amount": 2000.0,
                "turnover_rate": 0.30,
                "free_float_turnover_rate": 0.60,
                "volume_ratio": 1.10,
                "pe": 5.0,
                "pe_ttm": 5.1,
                "pb": 0.5,
                "ps": 1.2,
                "ps_ttm": 1.3,
                "dividend_yield": 4.0,
                "dividend_yield_ttm": 4.1,
                "total_share_capital": 100.0,
                "float_share_capital": 90.0,
                "free_float_share_capital": 80.0,
                "total_market_cap": 1000.0,
                "float_market_cap": 900.0,
            },
            {
                "symbol": "000002.SZ",
                "trade_date": "2024-01-02",
                "change": -0.20,
                "pct_change": -2.50,
                "volume": 3000.0,
                "amount": 4000.0,
                "turnover_rate": 0.40,
                "free_float_turnover_rate": 0.80,
                "volume_ratio": 0.90,
                "pe": 8.0,
                "pe_ttm": 8.1,
                "pb": 0.8,
                "ps": 2.2,
                "ps_ttm": 2.3,
                "dividend_yield": 2.0,
                "dividend_yield_ttm": 2.1,
                "total_share_capital": 200.0,
                "float_share_capital": 190.0,
                "free_float_share_capital": 180.0,
                "total_market_cap": 2000.0,
                "float_market_cap": 1900.0,
            },
        ]
    )
    second = first.copy()
    second["trade_date"] = "2024-02-01"
    second["change"] = [0.30, -0.40]
    second["volume"] = [1100.0, 3300.0]
    second["amount"] = [2200.0, 4400.0]
    second["turnover_rate"] = [0.31, 0.41]
    second["free_float_turnover_rate"] = [0.61, 0.81]
    first.to_csv(daily_root / "2024-01" / "2024-01-02.csv", index=False)
    second.to_csv(daily_root / "2024-02" / "2024-02-01.csv", index=False)
    pd.DataFrame(
        [
            {"date": "2024-01-02", "file_path": "daily_metrics/2024-01/2024-01-02.csv", "rows": 2, "unique_symbols": 2},
            {"date": "2024-02-01", "file_path": "daily_metrics/2024-02/2024-02-01.csv", "rows": 2, "unique_symbols": 2},
        ]
    ).to_csv(root / "daily_metrics.csv", index=False)


def _write_sw_snapshot(path: Path) -> None:
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "stock_code": "000001.SZ",
                "stock_name": "Ping An Bank",
                "industry_standard": "SW2021",
                "industry_level1_code": "801780.SI",
                "industry_level1": "bank_l1",
                "industry_level2_code": "801782.SI",
                "industry_level2": "joint_stock_bank_l2",
                "industry_level3_code": "851911.SI",
                "industry_level3": "joint_stock_bank_l3",
                "effective_date": "1991-04-03",
                "classification_snapshot_date": "2026-06-03",
                "classification_mode": "static_current_reference",
                "source_file": "source.csv",
                "quality_status": "complete",
            },
            {
                "stock_code": "000002.SZ",
                "stock_name": "Vanke A",
                "industry_standard": "SW2021",
                "industry_level1_code": "801180.SI",
                "industry_level1": "real_estate_l1",
                "industry_level2_code": "801181.SI",
                "industry_level2": "real_estate_development_l2",
                "industry_level3_code": "851811.SI",
                "industry_level3": "residential_development_l3",
                "effective_date": "1991-01-29",
                "classification_snapshot_date": "2026-06-03",
                "classification_mode": "static_current_reference",
                "source_file": "source.csv",
                "quality_status": "complete",
            },
        ]
    ).to_csv(path, index=False)


def _write_listing_board_snapshot(path: Path) -> None:
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "stock_code": "000001.SZ",
                "stock_name": "Ping An Bank",
                "dimension_standard": "A_SHARE_LISTING_BOARD",
                "listing_board_code": "CHINEXT",
                "listing_board": "chinext_board",
                "board_order": 2,
                "exchange_code": "SZSE",
                "exchange_suffix": "SZ",
                "reference_mode": "static_current_reference",
                "quality_status": "complete",
            },
            {
                "stock_code": "000002.SZ",
                "stock_name": "Vanke A",
                "dimension_standard": "A_SHARE_LISTING_BOARD",
                "listing_board_code": "MAIN",
                "listing_board": "main_board",
                "board_order": 1,
                "exchange_code": "SZSE",
                "exchange_suffix": "SZ",
                "reference_mode": "static_current_reference",
                "quality_status": "complete",
            },
        ]
    ).to_csv(path, index=False)


def _make_pit_index_membership_panel() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {"date": "2024-01-02", "stock_code": "000001", "is_csi300": 1, "is_csi500": 0, "is_csi1000": 0, "is_csi2000": 0},
            {"date": "2024-01-02", "stock_code": "000002", "is_csi300": 0, "is_csi500": 0, "is_csi1000": 1, "is_csi2000": 0},
            {"date": "2024-02-01", "stock_code": "000001", "is_csi300": 1, "is_csi500": 0, "is_csi1000": 0, "is_csi2000": 0},
            {"date": "2024-02-01", "stock_code": "000002", "is_csi300": 0, "is_csi500": 1, "is_csi1000": 0, "is_csi2000": 0},
        ]
    )
    for column in ("is_csi300_unknown", "is_csi500_unknown", "is_csi1000_unknown", "is_csi2000_unknown"):
        frame[column] = False
    frame["index_membership_any_unknown"] = False
    frame["index_membership_all_known"] = True
    frame["historical_pit_index_membership"] = True
    return frame


def test_build_panel_attaches_static_sw_static_board_and_pit_index_membership(tmp_path: Path) -> None:
    daily_dir = tmp_path / "ashare_daily_metrics"
    _write_daily_metrics_dataset(daily_dir)
    sw_snapshot = tmp_path / "industry_sw_current_reference" / "current_snapshot.csv"
    board_snapshot = tmp_path / "listing_board_current_reference" / "current_snapshot.csv"
    _write_sw_snapshot(sw_snapshot)
    _write_listing_board_snapshot(board_snapshot)
    index_membership = _make_pit_index_membership_panel()
    universe = pd.DataFrame({"date": ["2024-01-02", "2024-02-01"], "stock_code": ["000001", "000002.SZ"]})

    panel, summary = build_stock_daily_metrics_sw_static_panel(
        daily_metrics_dir=daily_dir,
        sw_current_snapshot=sw_snapshot,
        listing_board_current_snapshot=board_snapshot,
        index_membership_panel=index_membership,
        universe_frame=universe,
    )

    assert list(panel.columns) == list(OUTPUT_COLUMNS)
    assert panel[["date", "stock_code", "symbol"]].to_dict("records") == [
        {"date": pd.Timestamp("2024-01-02"), "stock_code": "000001", "symbol": "000001.SZ"},
        {"date": pd.Timestamp("2024-02-01"), "stock_code": "000002", "symbol": "000002.SZ"},
    ]
    assert panel["change"].tolist() == [0.10, -0.40]
    assert panel[["volume", "amount", "turnover_rate", "free_float_turnover_rate"]].to_numpy().tolist() == [
        [1000.0, 2000.0, 0.30, 0.60],
        [3300.0, 4400.0, 0.41, 0.81],
    ]
    assert panel["sw_l1_name"].tolist() == ["bank_l1", "real_estate_l1"]
    assert panel["historical_pit_industry"].tolist() == [False, False]
    assert panel["listing_board_standard"].tolist() == ["A_SHARE_LISTING_BOARD", "A_SHARE_LISTING_BOARD"]
    assert panel["listing_board_code"].tolist() == ["CHINEXT", "MAIN"]
    assert panel["listing_board_name"].tolist() == ["chinext_board", "main_board"]
    assert panel["listing_board_segment"].tolist() == ["CHINEXT", "SZSE_MAIN"]
    assert panel["listing_board_reference_mode"].tolist() == ["static_current_reference", "static_current_reference"]
    assert panel["historical_pit_listing_board"].tolist() == [False, False]
    assert panel[["is_chinext", "is_star", "is_bse", "is_sse_main", "is_szse_main", "is_main_board"]].to_numpy().tolist() == [
        [True, False, False, False, False, False],
        [False, False, False, False, True, True],
    ]
    assert panel[["is_csi300", "is_csi500", "is_csi1000", "is_csi2000"]].to_numpy().tolist() == [
        [True, False, False, False],
        [False, True, False, False],
    ]
    assert panel["historical_pit_index_membership"].tolist() == [True, True]
    assert panel[["is_csi300_unknown", "is_csi500_unknown", "is_csi1000_unknown", "is_csi2000_unknown"]].to_numpy().tolist() == [
        [False, False, False, False],
        [False, False, False, False],
    ]
    assert panel["index_membership_any_unknown"].tolist() == [False, False]
    assert panel["index_membership_all_known"].tolist() == [True, True]
    assert summary["metadata"]["historical_pit_industry"] is False
    assert summary["metadata"]["historical_pit_listing_board"] is False
    assert summary["metadata"]["historical_pit_index_membership"] is True
    assert summary["metadata"]["listing_board_membership"] == "static_current_reference"
    assert summary["metadata"]["index_membership"] == "historical_pit_date_stock_panel"
    assert summary["metadata"]["index_membership_unknown_flags"] is True
    assert summary["daily_columns"] == list(REQUESTED_DAILY_COLUMNS)
    assert summary["universe"]["mode"] == "provided_date_stock_universe"


def test_build_panel_rejects_universe_date_missing_from_daily_metrics_index(tmp_path: Path) -> None:
    daily_dir = tmp_path / "ashare_daily_metrics"
    _write_daily_metrics_dataset(daily_dir)
    sw_snapshot = tmp_path / "industry_sw_current_reference" / "current_snapshot.csv"
    board_snapshot = tmp_path / "listing_board_current_reference" / "current_snapshot.csv"
    _write_sw_snapshot(sw_snapshot)
    _write_listing_board_snapshot(board_snapshot)
    universe = pd.DataFrame({"date": ["2024-03-01"], "stock_code": ["000001"]})

    try:
        build_stock_daily_metrics_sw_static_panel(
            daily_metrics_dir=daily_dir,
            sw_current_snapshot=sw_snapshot,
            listing_board_current_snapshot=board_snapshot,
            index_membership_panel=_make_pit_index_membership_panel(),
            universe_frame=universe,
        )
    except ValueError as exc:
        assert "missing daily metrics partitions" in str(exc)
        assert "2024-03-01" in str(exc)
    else:
        raise AssertionError("Expected missing daily metric dates to fail loudly.")


def test_build_panel_rejects_missing_static_sw_industry_by_default(tmp_path: Path) -> None:
    daily_dir = tmp_path / "ashare_daily_metrics"
    _write_daily_metrics_dataset(daily_dir)
    sw_snapshot = tmp_path / "industry_sw_current_reference" / "current_snapshot.csv"
    board_snapshot = tmp_path / "listing_board_current_reference" / "current_snapshot.csv"
    _write_sw_snapshot(sw_snapshot)
    _write_listing_board_snapshot(board_snapshot)
    snapshot = pd.read_csv(sw_snapshot)
    snapshot = snapshot[snapshot["stock_code"] != "000002.SZ"]
    snapshot.to_csv(sw_snapshot, index=False)
    universe = pd.DataFrame({"date": ["2024-01-02"], "stock_code": ["000002"]})

    try:
        build_stock_daily_metrics_sw_static_panel(
            daily_metrics_dir=daily_dir,
            sw_current_snapshot=sw_snapshot,
            listing_board_current_snapshot=board_snapshot,
            index_membership_panel=_make_pit_index_membership_panel(),
            universe_frame=universe,
        )
    except ValueError as exc:
        assert "missing static SW industry rows" in str(exc)
        assert "000002.SZ" in str(exc)
    else:
        raise AssertionError("Expected missing static SW industry rows to fail loudly.")


def test_build_panel_rejects_missing_static_listing_board_by_default(tmp_path: Path) -> None:
    daily_dir = tmp_path / "ashare_daily_metrics"
    _write_daily_metrics_dataset(daily_dir)
    sw_snapshot = tmp_path / "industry_sw_current_reference" / "current_snapshot.csv"
    board_snapshot = tmp_path / "listing_board_current_reference" / "current_snapshot.csv"
    _write_sw_snapshot(sw_snapshot)
    _write_listing_board_snapshot(board_snapshot)
    snapshot = pd.read_csv(board_snapshot)
    snapshot = snapshot[snapshot["stock_code"] != "000002.SZ"]
    snapshot.to_csv(board_snapshot, index=False)
    universe = pd.DataFrame({"date": ["2024-01-02"], "stock_code": ["000002"]})

    try:
        build_stock_daily_metrics_sw_static_panel(
            daily_metrics_dir=daily_dir,
            sw_current_snapshot=sw_snapshot,
            listing_board_current_snapshot=board_snapshot,
            index_membership_panel=_make_pit_index_membership_panel(),
            universe_frame=universe,
        )
    except ValueError as exc:
        assert "missing static listing board rows" in str(exc)
        assert "000002.SZ" in str(exc)
    else:
        raise AssertionError("Expected missing static listing board rows to fail loudly.")


def test_build_panel_rejects_missing_pit_index_membership_by_default(tmp_path: Path) -> None:
    daily_dir = tmp_path / "ashare_daily_metrics"
    _write_daily_metrics_dataset(daily_dir)
    sw_snapshot = tmp_path / "industry_sw_current_reference" / "current_snapshot.csv"
    board_snapshot = tmp_path / "listing_board_current_reference" / "current_snapshot.csv"
    _write_sw_snapshot(sw_snapshot)
    _write_listing_board_snapshot(board_snapshot)
    index_membership = _make_pit_index_membership_panel()
    index_membership = index_membership[~((index_membership["date"] == "2024-01-02") & (index_membership["stock_code"] == "000002"))]
    universe = pd.DataFrame({"date": ["2024-01-02"], "stock_code": ["000002"]})

    try:
        build_stock_daily_metrics_sw_static_panel(
            daily_metrics_dir=daily_dir,
            sw_current_snapshot=sw_snapshot,
            listing_board_current_snapshot=board_snapshot,
            index_membership_panel=index_membership,
            universe_frame=universe,
        )
    except ValueError as exc:
        assert "missing PIT index membership rows" in str(exc)
        assert "000002.SZ" in str(exc)
    else:
        raise AssertionError("Expected missing PIT index membership rows to fail loudly.")


def test_build_panel_rejects_unknown_and_true_index_membership_conflict(tmp_path: Path) -> None:
    daily_dir = tmp_path / "ashare_daily_metrics"
    _write_daily_metrics_dataset(daily_dir)
    sw_snapshot = tmp_path / "industry_sw_current_reference" / "current_snapshot.csv"
    board_snapshot = tmp_path / "listing_board_current_reference" / "current_snapshot.csv"
    _write_sw_snapshot(sw_snapshot)
    _write_listing_board_snapshot(board_snapshot)
    index_membership = _make_pit_index_membership_panel()
    index_membership.loc[0, "is_csi300_unknown"] = True
    index_membership.loc[0, "index_membership_any_unknown"] = True
    index_membership.loc[0, "index_membership_all_known"] = False
    universe = pd.DataFrame({"date": ["2024-01-02"], "stock_code": ["000001"]})

    try:
        build_stock_daily_metrics_sw_static_panel(
            daily_metrics_dir=daily_dir,
            sw_current_snapshot=sw_snapshot,
            listing_board_current_snapshot=board_snapshot,
            index_membership_panel=index_membership,
            universe_frame=universe,
        )
    except ValueError as exc:
        assert "cannot be true when" in str(exc)
        assert "is_csi300_unknown" in str(exc)
    else:
        raise AssertionError("Expected conflicting index membership and unknown flags to fail loudly.")


def test_write_dataset_writes_partitions_index_pickle_and_manifest(tmp_path: Path) -> None:
    daily_dir = tmp_path / "ashare_daily_metrics"
    _write_daily_metrics_dataset(daily_dir)
    sw_snapshot = tmp_path / "industry_sw_current_reference" / "current_snapshot.csv"
    board_snapshot = tmp_path / "listing_board_current_reference" / "current_snapshot.csv"
    _write_sw_snapshot(sw_snapshot)
    _write_listing_board_snapshot(board_snapshot)
    universe = pd.DataFrame({"date": ["2024-01-02", "2024-02-01"], "stock_code": ["000001", "000002"]})

    panel, summary = build_stock_daily_metrics_sw_static_panel(
        daily_metrics_dir=daily_dir,
        sw_current_snapshot=sw_snapshot,
        listing_board_current_snapshot=board_snapshot,
        index_membership_panel=_make_pit_index_membership_panel(),
        universe_frame=universe,
    )
    manifest = write_stock_daily_metrics_sw_static_dataset(
        panel=panel,
        summary=summary,
        output_dir=tmp_path / "out",
        archive_path=tmp_path / "out.zip",
    )

    out = tmp_path / "out"
    daily_index = pd.read_csv(out / "daily_features.csv")
    reloaded = pd.read_pickle(out / "stock_daily_metrics_sw_static.pkl")
    written_manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))

    assert (out / "daily_features" / "2024-01" / "2024-01-02.csv").exists()
    assert (out / "daily_features" / "2024-02" / "2024-02-01.csv").exists()
    assert (out / "README.md").exists()
    assert (tmp_path / "out.zip").exists()
    with zipfile.ZipFile(tmp_path / "out.zip") as archive:
        assert sorted(archive.namelist()) == [
            "out/README.md",
            "out/daily_features.csv",
            "out/daily_features/2024-01/2024-01-02.csv",
            "out/daily_features/2024-02/2024-02-01.csv",
            "out/manifest.json",
            "out/stock_daily_metrics_sw_static.pkl",
        ]
    assert daily_index["date"].tolist() == ["2024-01-02", "2024-02-01"]
    assert daily_index["rows"].tolist() == [1, 1]
    pd.testing.assert_frame_equal(reloaded, panel)
    assert written_manifest["metadata"]["artifact_version"] == "stock_daily_metrics_sw_static_v1"
    assert written_manifest["metadata"]["historical_pit_industry"] is False
    assert written_manifest["metadata"]["historical_pit_listing_board"] is False
    assert written_manifest["metadata"]["historical_pit_index_membership"] is True
    assert "listing_board_columns" in written_manifest
    assert "index_membership_columns" in written_manifest
    assert "index_membership_unknown_columns" in written_manifest
    assert written_manifest["missing_static_listing_board_rows"] == 0
    assert written_manifest["missing_pit_index_membership_rows"] == 0
    assert written_manifest["outputs"] == manifest["outputs"]




def test_path_for_summary_does_not_emit_external_absolute_paths(tmp_path: Path) -> None:
    external = (tmp_path / "finfact_io" / "generated_data" / "source.csv").resolve()

    summary_path = _path_for_summary(external)

    assert not Path(summary_path).is_absolute()
    assert str(external) not in summary_path




