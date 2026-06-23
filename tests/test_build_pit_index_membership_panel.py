from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.build_pit_index_membership_panel import (
    INDEX_SPECS,
    build_pit_index_membership_panel,
    write_pit_index_membership_panel,
)


def _write_daily_metrics(root: Path) -> None:
    daily_dir = root / "daily_metrics" / "2024-01"
    daily_dir.mkdir(parents=True)
    rows = pd.DataFrame(
        [
            {"symbol": "000001.SZ", "trade_date": "2024-01-02", "change": 0.1},
            {"symbol": "000002.SZ", "trade_date": "2024-01-02", "change": 0.2},
            {"symbol": "000003.SZ", "trade_date": "2024-01-02", "change": 0.3},
            {"symbol": "000004.SZ", "trade_date": "2024-01-02", "change": 0.4},
        ]
    )
    rows.to_csv(daily_dir / "2024-01-02.csv", index=False)
    pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "file_path": "daily_metrics/2024-01/2024-01-02.csv",
                "rows": 4,
                "unique_symbols": 4,
                "duplicate_key_rows": 0,
                "quality_status": "pass",
            }
        ]
    ).to_csv(root / "daily_metrics.csv", index=False)


def _write_index_asof(
    index_root: Path,
    *,
    index_dir: str,
    index_code: str,
    date: str,
    members: list[str],
    quality_status: str,
    weights: list[float],
) -> None:
    target = index_root / index_dir
    partition = target / "constituent_weights_daily_asof" / f"{date}.csv"
    partition.parent.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "date": [date] * len(members),
            "index_code": [index_code] * len(members),
            "member_symbol": members,
            "weight": weights,
            "weight_unit": ["%"] * len(members),
            "weight_snapshot_date": ["2023-12-29"] * len(members),
            "effective_date": [date] * len(members),
            "days_since_snapshot": [4] * len(members),
            "quality_status": [quality_status] * len(members),
        }
    )
    frame.to_csv(partition, index=False)
    pd.DataFrame(
        [
            {
                "date": date,
                "file_path": f"constituent_weights_daily_asof/{date}.csv",
                "rows": len(members),
                "index_code": index_code,
                "weight_snapshot_date": "2023-12-29",
                "effective_date": date,
                "days_since_snapshot": 4,
                "quality_status": quality_status,
                "weight_sum": sum(weights),
                "duplicate_member_rows": len(members) - len(set(members)),
            }
        ]
    ).to_csv(target / "constituent_weights_daily_asof.csv", index=False)


def _write_empty_index_asof(index_root: Path, *, index_dir: str) -> None:
    target = index_root / index_dir
    target.mkdir(parents=True)
    pd.DataFrame(
        columns=[
            "date",
            "file_path",
            "rows",
            "index_code",
            "weight_snapshot_date",
            "effective_date",
            "days_since_snapshot",
            "quality_status",
            "weight_sum",
            "duplicate_member_rows",
        ]
    ).to_csv(target / "constituent_weights_daily_asof.csv", index=False)


def test_build_pit_index_membership_marks_missing_index_date_unknown(tmp_path: Path) -> None:
    daily_root = tmp_path / "ashare_daily_metrics"
    index_root = tmp_path / "indices"
    _write_daily_metrics(daily_root)
    _write_index_asof(
        index_root,
        index_dir="csi300",
        index_code="000300.SH",
        date="2024-01-02",
        members=["000001.SZ", "000002.SZ", "000003.SZ"],
        quality_status="complete",
        weights=[40.0, 30.0, 30.0],
    )
    _write_empty_index_asof(index_root, index_dir="csi2000")

    panel, quality, summary = build_pit_index_membership_panel(
        daily_metrics_dir=daily_root,
        index_data_root=index_root,
        start="2024-01-01",
        index_specs={
            "is_csi300": INDEX_SPECS["is_csi300"],
            "is_csi2000": INDEX_SPECS["is_csi2000"],
        },
        member_count_tolerances={"is_csi300": (3, 3), "is_csi2000": (3, 3)},
    )

    panel = panel.sort_values("stock_code").reset_index(drop=True)
    assert panel["is_csi300"].tolist() == [True, True, True, False]
    assert panel["is_csi300_unknown"].tolist() == [False, False, False, False]
    assert panel["is_csi2000"].tolist() == [False, False, False, False]
    assert panel["is_csi2000_unknown"].tolist() == [True, True, True, True]
    assert panel["index_membership_any_unknown"].tolist() == [True, True, True, True]
    assert panel["index_membership_all_known"].tolist() == [False, False, False, False]
    assert summary["date_count"] == 1
    assert summary["unknown_row_counts_by_index"]["is_csi2000"] == 4
    assert bool(quality.set_index("membership_flag_column").loc["is_csi300", "reliable"]) is True
    assert bool(quality.set_index("membership_flag_column").loc["is_csi2000", "reliable"]) is False


def test_build_pit_index_membership_rejects_incomplete_weight_data_as_unknown(tmp_path: Path) -> None:
    daily_root = tmp_path / "ashare_daily_metrics"
    index_root = tmp_path / "indices"
    _write_daily_metrics(daily_root)
    _write_index_asof(
        index_root,
        index_dir="csi300",
        index_code="000300.SH",
        date="2024-01-02",
        members=["000001.SZ", "000002.SZ", "000003.SZ"],
        quality_status="incomplete",
        weights=[40.0, 30.0, 30.0],
    )

    panel, quality, summary = build_pit_index_membership_panel(
        daily_metrics_dir=daily_root,
        index_data_root=index_root,
        start="2024-01-01",
        index_specs={"is_csi300": INDEX_SPECS["is_csi300"]},
        member_count_tolerances={"is_csi300": (3, 3)},
    )

    assert panel["is_csi300"].tolist() == [False, False, False, False]
    assert panel["is_csi300_unknown"].tolist() == [True, True, True, True]
    row = quality.iloc[0]
    assert bool(row["reliable"]) is False
    assert "source_quality_status" in row["unreliable_reason"]
    assert summary["unreliable_date_counts_by_index"]["is_csi300"] == 1


def test_build_pit_index_membership_rejects_bad_weight_sum_as_unknown(tmp_path: Path) -> None:
    daily_root = tmp_path / "ashare_daily_metrics"
    index_root = tmp_path / "indices"
    _write_daily_metrics(daily_root)
    _write_index_asof(
        index_root,
        index_dir="csi300",
        index_code="000300.SH",
        date="2024-01-02",
        members=["000001.SZ", "000002.SZ", "000003.SZ"],
        quality_status="complete",
        weights=[20.0, 20.0, 20.0],
    )

    panel, quality, _summary = build_pit_index_membership_panel(
        daily_metrics_dir=daily_root,
        index_data_root=index_root,
        start="2024-01-01",
        index_specs={"is_csi300": INDEX_SPECS["is_csi300"]},
        member_count_tolerances={"is_csi300": (3, 3)},
        weight_sum_tolerance=0.005,
    )

    assert panel["is_csi300"].tolist() == [False, False, False, False]
    assert panel["is_csi300_unknown"].tolist() == [True, True, True, True]
    assert "weight_sum_fraction" in quality.iloc[0]["unreliable_reason"]


def test_write_pit_index_membership_panel_outputs_manifest(tmp_path: Path) -> None:
    daily_root = tmp_path / "ashare_daily_metrics"
    index_root = tmp_path / "indices"
    _write_daily_metrics(daily_root)
    _write_index_asof(
        index_root,
        index_dir="csi300",
        index_code="000300.SH",
        date="2024-01-02",
        members=["000001.SZ", "000002.SZ", "000003.SZ"],
        quality_status="complete",
        weights=[40.0, 30.0, 30.0],
    )
    panel, quality, summary = build_pit_index_membership_panel(
        daily_metrics_dir=daily_root,
        index_data_root=index_root,
        start="2024-01-01",
        index_specs={"is_csi300": INDEX_SPECS["is_csi300"]},
        member_count_tolerances={"is_csi300": (3, 3)},
    )

    manifest = write_pit_index_membership_panel(
        panel=panel,
        quality=quality,
        summary=summary,
        output_dir=tmp_path / "out",
    )

    assert (tmp_path / "out" / "pit_index_membership.pkl").exists()
    assert (tmp_path / "out" / "pit_index_membership.csv").exists()
    assert (tmp_path / "out" / "index_membership_quality_by_date.csv").exists()
    written = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert written["metadata"]["historical_pit_index_membership"] is True
    assert written["metadata"]["unknown_flags_present"] is True
    assert written["row_count"] == manifest["row_count"] == 4


