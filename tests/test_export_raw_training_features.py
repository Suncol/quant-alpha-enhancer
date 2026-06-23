from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.export_raw_training_features import (
    export_raw_training_feature_package,
)


def _make_exposures() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "date": "2024-01-02 15:00:00",
                "stock_code": 1,
                "industry": "bank",
                "board": "SZSE_MAIN",
                "is_csi300": 1,
                "is_csi500": 0,
                "is_csi1000": 0,
                "is_csi2000": 0,
                "market_cap": 100.0,
                "amount_k": 1000.0,
                "turnover": 0.01,
                "logADV20": 5.0,
                "logAmount20": 6.0,
                "turnover20": 0.02,
            },
            {
                "date": "2024-01-02 15:00:00",
                "stock_code": "000002",
                "industry": "tech",
                "board": "CHINEXT",
                "is_csi300": 0,
                "is_csi500": 1,
                "is_csi1000": 0,
                "is_csi2000": 0,
                "market_cap": 200.0,
                "amount_k": 2000.0,
                "turnover": 0.02,
                "logADV20": 6.0,
                "logAmount20": 7.0,
                "turnover20": 0.03,
            },
            {
                "date": "2024-01-03",
                "stock_code": "000001",
                "industry": "bank",
                "board": "SZSE_MAIN",
                "is_csi300": 1,
                "is_csi500": 0,
                "is_csi1000": 0,
                "is_csi2000": 0,
                "market_cap": 110.0,
                "amount_k": 1100.0,
                "turnover": 0.011,
                "logADV20": 5.1,
                "logAmount20": 6.1,
                "turnover20": 0.021,
            },
        ]
    )
    for column in ("is_csi300_unknown", "is_csi500_unknown", "is_csi1000_unknown", "is_csi2000_unknown"):
        frame[column] = False
    frame["index_membership_any_unknown"] = False
    frame["index_membership_all_known"] = True
    frame["historical_pit_index_membership"] = True
    return frame


def _stock_by_date(values: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(
        values,
        index=["000001", "000002"],
        columns=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )


def test_export_raw_training_feature_package_writes_raw_features_and_archive(tmp_path: Path) -> None:
    output_dir = tmp_path / "raw_feature_package"
    archive_path = tmp_path / "raw_feature_package.zip"

    summary = export_raw_training_feature_package(
        exposures=_make_exposures(),
        output_dir=output_dir,
        archive_path=archive_path,
        signal_value_frames={
            "factor_sss_dx_10": _stock_by_date([[0.1, 0.3], [0.2, np.nan]]),
            "close": _stock_by_date([[10.0, 11.0], [20.0, 21.0]]),
        },
        source_notes=["unit test source"],
    )

    raw_features = pd.read_csv(output_dir / "raw_training_features.csv", dtype={"stock_code": "string"})
    constituents = pd.read_csv(output_dir / "index_constituents.csv", dtype={"stock_code": "string"})
    industry_dummy = pd.read_csv(output_dir / "industry_dummy_matrix.csv", dtype={"stock_code": "string"})
    board_dummy = pd.read_csv(output_dir / "board_dummy_matrix.csv", dtype={"stock_code": "string"})
    index_dummy = pd.read_csv(output_dir / "index_dummy_matrix.csv", dtype={"stock_code": "string"})
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert archive_path.exists()
    assert set(summary["outputs"]) == {
        "raw_features_csv",
        "raw_features_pickle",
        "index_constituents_csv",
        "industry_dummy_matrix_csv",
        "board_dummy_matrix_csv",
        "index_dummy_matrix_csv",
        "manifest",
        "readme",
        "archive",
    }
    assert raw_features["date"].tolist() == ["2024-01-02", "2024-01-02", "2024-01-03"]
    assert raw_features["stock_code"].tolist() == ["000001", "000002", "000001"]
    assert raw_features["market_cap"].tolist() == [100.0, 200.0, 110.0]
    assert raw_features["turnover"].tolist() == [0.01, 0.02, 0.011]
    assert raw_features["factor_sss_dx_10_raw"].tolist() == [0.1, 0.2, 0.3]
    assert "log_mcap_z" not in raw_features.columns
    assert "mcap_rank" not in raw_features.columns
    assert "sample_weight" not in raw_features.columns
    assert "y_resid_fwd" not in raw_features.columns

    expected_keys = raw_features[["date", "stock_code"]]
    pd.testing.assert_frame_equal(industry_dummy[["date", "stock_code"]], expected_keys)
    pd.testing.assert_frame_equal(board_dummy[["date", "stock_code"]], expected_keys)
    pd.testing.assert_frame_equal(index_dummy[["date", "stock_code"]], expected_keys)
    assert industry_dummy[["industry__bank", "industry__tech"]].to_numpy().tolist() == [
        [1, 0],
        [0, 1],
        [1, 0],
    ]
    assert board_dummy[["board__CHINEXT", "board__SZSE_MAIN"]].to_numpy().tolist() == [
        [0, 1],
        [1, 0],
        [0, 1],
    ]
    assert index_dummy[["index__CSI300", "index__CSI500", "index__CSI1000", "index__CSI2000", "index__UNKNOWN_INDEX", "index__NON_INDEX"]].to_numpy().tolist() == [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
    ]
    for dummy in [industry_dummy, board_dummy, index_dummy]:
        dummy_values = dummy.drop(columns=["date", "stock_code"])
        assert set(dummy_values.to_numpy().ravel()).issubset({0, 1})
        assert dummy_values.sum(axis=1).tolist() == [1, 1, 1]

    assert constituents.to_dict("records") == [
        {
            "date": "2024-01-02",
            "stock_code": "000001",
            "index_name": "CSI300",
            "membership_flag_column": "is_csi300",
        },
        {
            "date": "2024-01-02",
            "stock_code": "000002",
            "index_name": "CSI500",
            "membership_flag_column": "is_csi500",
        },
        {
            "date": "2024-01-03",
            "stock_code": "000001",
            "index_name": "CSI300",
            "membership_flag_column": "is_csi300",
        },
    ]

    assert manifest["metadata"]["date_index_semantics"] == "signal_generation_date_equals_buy_date_minus_one"
    assert manifest["metadata"]["transform_policy"] == "raw_inputs_only_no_zscore_no_rank_no_winsorization"
    assert manifest["metadata"]["dummy_matrix_policy"] == "industry_board_index_are_exported_as_one_hot_matrices"
    assert manifest["row_count"] == 3
    assert manifest["dummy_matrices"]["industry"]["one_hot_row_sum_min"] == 1
    assert manifest["dummy_matrices"]["industry"]["one_hot_row_sum_max"] == 1
    assert manifest["dummy_matrices"]["board"]["one_hot_row_sum_min"] == 1
    assert manifest["dummy_matrices"]["board"]["one_hot_row_sum_max"] == 1
    assert manifest["dummy_matrices"]["index"]["one_hot_row_sum_min"] == 1
    assert manifest["dummy_matrices"]["index"]["one_hot_row_sum_max"] == 1
    assert manifest["signal_inputs"]["factor_sss_dx_10"]["matched_nonmissing_count"] == 3
    assert manifest["signal_inputs"]["close"]["matched_nonmissing_count"] == 3
    assert manifest["source_notes"] == ["unit test source"]

    with zipfile.ZipFile(archive_path) as archive:
        assert sorted(archive.namelist()) == [
            "raw_feature_package/README.md",
            "raw_feature_package/board_dummy_matrix.csv",
            "raw_feature_package/index_constituents.csv",
            "raw_feature_package/index_dummy_matrix.csv",
            "raw_feature_package/industry_dummy_matrix.csv",
            "raw_feature_package/manifest.json",
            "raw_feature_package/raw_training_features.csv",
            "raw_feature_package/raw_training_features.pkl",
        ]


def test_export_raw_training_feature_package_rejects_duplicate_date_stock_keys(tmp_path: Path) -> None:
    exposures = pd.concat([_make_exposures(), _make_exposures().head(1)], ignore_index=True)

    try:
        export_raw_training_feature_package(
            exposures=exposures,
            output_dir=tmp_path / "out",
            archive_path=tmp_path / "out.zip",
        )
    except ValueError as exc:
        assert "duplicate date, stock_code keys" in str(exc)
    else:
        raise AssertionError("Expected duplicate date/stock exposure keys to fail loudly.")


def test_export_raw_training_feature_package_can_use_training_universe(tmp_path: Path) -> None:
    universe = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "stock_code": ["000002", "000001"],
            "existing_training_feature_z": [1.2, -0.4],
        }
    )

    summary = export_raw_training_feature_package(
        exposures=_make_exposures(),
        output_dir=tmp_path / "out",
        archive_path=tmp_path / "out.zip",
        universe_frame=universe,
    )

    raw_features = pd.read_csv(tmp_path / "out" / "raw_training_features.csv", dtype={"stock_code": "string"})

    assert raw_features[["date", "stock_code"]].to_dict("records") == [
        {"date": "2024-01-02", "stock_code": "000002"},
        {"date": "2024-01-03", "stock_code": "000001"},
    ]
    assert "existing_training_feature_z" not in raw_features.columns
    assert summary["universe"]["mode"] == "provided_date_stock_universe"
    assert summary["universe"]["input_row_count"] == 2
    assert summary["universe"]["dropped_exposure_rows_not_in_universe"] == 1


def test_export_raw_training_feature_package_marks_unknown_index_separately_from_non_index(tmp_path: Path) -> None:
    exposures = _make_exposures()
    unknown = exposures.iloc[[0]].copy()
    unknown["date"] = "2024-01-04"
    unknown["stock_code"] = "000003"
    for column in ["is_csi300", "is_csi500", "is_csi1000", "is_csi2000"]:
        unknown[column] = 0
    unknown["is_csi2000_unknown"] = True
    unknown["index_membership_any_unknown"] = True
    unknown["index_membership_all_known"] = False
    exposures = pd.concat([exposures, unknown], ignore_index=True)

    export_raw_training_feature_package(
        exposures=exposures,
        output_dir=tmp_path / "out",
        archive_path=tmp_path / "out.zip",
    )

    index_dummy = pd.read_csv(tmp_path / "out" / "index_dummy_matrix.csv", dtype={"stock_code": "string"})
    row = index_dummy[index_dummy["stock_code"].eq("000003")].iloc[0]
    assert int(row["index__UNKNOWN_INDEX"]) == 1
    assert int(row["index__NON_INDEX"]) == 0


def test_export_raw_training_feature_package_rejects_non_one_hot_index_flags(tmp_path: Path) -> None:
    exposures = _make_exposures()
    exposures.loc[0, "is_csi500"] = 1

    try:
        export_raw_training_feature_package(
            exposures=exposures,
            output_dir=tmp_path / "out",
            archive_path=tmp_path / "out.zip",
        )
    except ValueError as exc:
        assert "multiple index membership flags" in str(exc)
    else:
        raise AssertionError("Expected overlapping index membership flags to fail loudly.")

