from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.train_lgbm_placeholder_model import (
    _feature_gain_frame,
    _model_tree_diagnostics,
    compute_daily_prediction_metrics,
    train_placeholder_lgbm_models,
)


class RecordingRegressor:
    instances: list["RecordingRegressor"] = []

    def __init__(self, **params: Any) -> None:
        self.params = params
        self.fit_kwargs: dict[str, Any] = {}
        self.predict_calls: list[pd.DataFrame] = []
        self.feature_names_: list[str] = []
        self.feature_importances_: np.ndarray | None = None
        self.best_iteration_: int | None = 7
        RecordingRegressor.instances.append(self)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> "RecordingRegressor":
        self.fit_X = X.copy()
        self.fit_y = y.copy()
        self.fit_kwargs = kwargs
        self.feature_names_ = list(X.columns)
        self.feature_importances_ = np.arange(1, len(X.columns) + 1, dtype=float)
        for column in kwargs.get("categorical_feature", []):
            assert str(X[column].dtype) == "category"
        eval_set = kwargs.get("eval_set", [])
        assert len(eval_set) == 1
        self.valid_X = eval_set[0][0].copy()
        self.valid_y = pd.Series(eval_set[0][1]).copy()
        return self

    def predict(self, X: pd.DataFrame, **_: Any) -> np.ndarray:
        self.predict_calls.append(X.copy())
        numeric = X.select_dtypes(include=[np.number])
        if numeric.empty:
            return np.zeros(len(X), dtype=float)
        return numeric.sum(axis=1).to_numpy(dtype=float) * 0.01


def _make_training_panel() -> pd.DataFrame:
    dates = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
            "2024-01-10",
        ]
    )
    rows: list[dict[str, object]] = []
    for date_idx, date in enumerate(dates):
        for stock_idx, stock_code in enumerate(["000001", "000002", "000003"]):
            split_offset = date_idx * 10
            rows.append(
                {
                    "date": date,
                    "stock_code": stock_code,
                    "industry": ["bank", "tech", "energy"][stock_idx],
                    "board": ["SZSE_MAIN", "CHINEXT", "STAR"][stock_idx],
                    "index_bucket": ["CSI300", "CSI500", "NON_INDEX"][stock_idx],
                    "size_decile": stock_idx,
                    "is_csi300": 1 if stock_idx == 0 else 0,
                    "is_csi500": 1 if stock_idx == 1 else 0,
                    "is_csi1000": 0,
                    "is_csi2000": 0,
                    "log_mcap_z": float(stock_idx - 1),
                    "mcap_rank": [-1 / 3, 0.0, 1 / 3][stock_idx],
                    "alpha_placeholder_turnover20_z": float(stock_idx),
                    "alpha_placeholder_turnover20_rank": [-1 / 3, 0.0, 1 / 3][stock_idx],
                    "alpha_placeholder_logADV20_z": float(2 - stock_idx),
                    "alpha_placeholder_logADV20_rank": [1 / 3, 0.0, -1 / 3][stock_idx],
                    "y_rank_label": float(split_offset + stock_idx),
                    "y_resid_fwd": float(split_offset - stock_idx) / 100.0,
                    "sample_weight": 1.0 / 3.0,
                    "alpha_source": "placeholder_liquidity",
                }
            )
    return pd.DataFrame(rows)


def _make_fold_assignments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold_id": [1, 1, 1, 1, 1, 1],
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-08",
                    "2024-01-09",
                ]
            ),
            "split": ["train", "train", "valid", "valid", "test", "test"],
        }
    )


def _feature_roles() -> dict[str, list[str]]:
    return {
        "alpha_placeholder": [
            "alpha_placeholder_turnover20_z",
            "alpha_placeholder_turnover20_rank",
            "alpha_placeholder_logADV20_z",
            "alpha_placeholder_logADV20_rank",
        ],
        "condition_categorical": ["industry", "board", "index_bucket", "size_decile"],
        "condition_continuous": [
            "is_csi300",
            "is_csi500",
            "is_csi1000",
            "is_csi2000",
            "log_mcap_z",
            "mcap_rank",
        ],
        "targets": ["y_rank_label", "y_resid_fwd"],
    }


REAL_SIGNAL_FEATURES = [
    "factor_sss_dx_10_rank",
    "factor_sss_dx_10_z",
    "amo_rank",
    "amo_z",
    "close_rank",
    "close_z",
    "vol_rank",
    "vol_z",
]

REAL_CATEGORICAL_FEATURES = ["industry", "board", "index_bucket", "size_decile"]
REAL_CONTINUOUS_FEATURES = [
    "is_csi300",
    "is_csi500",
    "is_csi1000",
    "is_csi2000",
    "log_mcap_z",
    "mcap_rank",
]
RAW_RETURN_TARGET = "y_return_hfq_adj_fwd"
RAW_RETURN_RANK_LABEL = "y_return_rank_label"

NEUTRALIZED_SIGNAL_FEATURES = [
    "factor_sss_dx_10_value_neutralized_rank",
    "factor_sss_dx_10_value_neutralized_z",
    "factor_sss_dx_10_rank_neutralized_rank",
    "factor_sss_dx_10_rank_neutralized_z",
    "close_rank",
    "close_z",
]

NEUTRALIZED_ALPHA_MONOTONE_FEATURES = [
    "factor_sss_dx_10_value_neutralized_rank",
    "factor_sss_dx_10_value_neutralized_z",
    "factor_sss_dx_10_rank_neutralized_rank",
    "factor_sss_dx_10_rank_neutralized_z",
]


def _make_real_alpha_feature_panel() -> pd.DataFrame:
    dates = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
        ]
    )
    rows: list[dict[str, object]] = []
    for date_idx, date in enumerate(dates):
        for stock_idx, stock_code in enumerate(["000001", "000002", "000003"]):
            centered = float(stock_idx - 1)
            sample_weight = 10.0 * float(date_idx + 1) * float(stock_idx + 1)
            row: dict[str, object] = {
                "date": date,
                "stock_code": stock_code,
                "industry": ["bank", "tech", "energy"][stock_idx],
                "board": ["SZSE_MAIN", "CHINEXT", "STAR"][stock_idx],
                "index_bucket": ["CSI300", "CSI500", "NON_INDEX"][stock_idx],
                "size_decile": stock_idx,
                "is_csi300": 1 if stock_idx == 0 else 0,
                "is_csi500": 1 if stock_idx == 1 else 0,
                "is_csi1000": 1 if stock_idx == 2 else 0,
                "is_csi2000": 0,
                "log_mcap_z": centered,
                "mcap_rank": [-1 / 3, 0.0, 1 / 3][stock_idx],
                "factor_sss_dx_10_raw": float(100 + date_idx * 10 + stock_idx),
                "amo_raw": float(1000 + date_idx * 10 + stock_idx),
                "close_raw": float(10 + stock_idx),
                "vol_raw": float(10000 + date_idx * 10 + stock_idx),
                "factor_sss_dx_10_raw__rank_input": float(stock_idx),
                "factor_sss_dx_10_raw__z_input": float(stock_idx),
                "amo_raw__rank_input": float(stock_idx),
                "amo_raw__z_input": float(stock_idx),
                "close_raw__rank_input": float(stock_idx),
                "close_raw__z_input": float(stock_idx),
                "vol_raw__rank_input": float(stock_idx),
                "vol_raw__z_input": float(stock_idx),
                "market_cap": float(100 + 10 * stock_idx),
                "log_mcap": float(4.5 + stock_idx),
                "y_rank_label": float(date_idx * 10 + stock_idx),
                "y_resid_fwd": float(date_idx - stock_idx) / 100.0,
                "sample_weight": sample_weight,
                "alpha_source": "factor_sss_dx_10",
            }
            for feature_idx, feature in enumerate(REAL_SIGNAL_FEATURES):
                row[feature] = float((feature_idx + 1) * centered)
            rows.append(row)
    return pd.DataFrame(rows)


def _make_real_alpha_feature_roles(*, use_signal_alias: bool = False) -> dict[str, list[str]]:
    signal_role = "signal_features" if use_signal_alias else "alpha_placeholder"
    return {
        signal_role: REAL_SIGNAL_FEATURES,
        "condition_categorical": REAL_CATEGORICAL_FEATURES,
        "condition_continuous": REAL_CONTINUOUS_FEATURES,
        "traceability": [
            "date",
            "stock_code",
            "alpha_source",
            "market_cap",
            "log_mcap",
            "factor_sss_dx_10_raw",
            "amo_raw",
            "close_raw",
            "vol_raw",
        ],
        "excluded_from_model": [
            "date",
            "stock_code",
            "y_resid_fwd",
            "y_rank_label",
            "sample_weight",
            "alpha_source",
            "market_cap",
            "log_mcap",
            "factor_sss_dx_10_raw",
            "amo_raw",
            "close_raw",
            "vol_raw",
            "factor_sss_dx_10_raw__rank_input",
            "factor_sss_dx_10_raw__z_input",
            "amo_raw__rank_input",
            "amo_raw__z_input",
            "close_raw__rank_input",
            "close_raw__z_input",
            "vol_raw__rank_input",
            "vol_raw__z_input",
        ],
        "targets": ["y_resid_fwd", "y_rank_label"],
    }


def _make_non_neutralized_raw_return_panel() -> pd.DataFrame:
    frame = _make_real_alpha_feature_panel().copy()
    frame[RAW_RETURN_TARGET] = (
        0.02
        + frame["date"].rank(method="dense").astype(float) / 1000.0
        - frame["stock_code"].astype(str).str[-1].astype(float) / 100.0
    )
    frame[RAW_RETURN_RANK_LABEL] = frame.groupby("date")[RAW_RETURN_TARGET].rank(
        method="average",
        pct=True,
    )
    return frame.drop(columns=["y_resid_fwd", "y_rank_label"])


def _make_non_neutralized_raw_return_roles() -> dict[str, list[str]]:
    roles = _make_real_alpha_feature_roles(use_signal_alias=True)
    roles["targets"] = [RAW_RETURN_TARGET]
    roles["excluded_from_model"] = [
        column
        for column in roles["excluded_from_model"]
        if column not in {"y_resid_fwd", "y_rank_label"}
    ] + [RAW_RETURN_TARGET, RAW_RETURN_RANK_LABEL]
    return roles


def _make_neutralized_alpha_feature_panel() -> pd.DataFrame:
    frame = _make_real_alpha_feature_panel()
    source_columns = [
        "factor_sss_dx_10_rank",
        "factor_sss_dx_10_z",
        "factor_sss_dx_10_rank",
        "factor_sss_dx_10_z",
        "close_rank",
        "close_z",
    ]
    for target_column, source_column in zip(
        NEUTRALIZED_SIGNAL_FEATURES,
        source_columns,
        strict=True,
    ):
        frame[target_column] = frame[source_column]
    return frame


def _make_neutralized_alpha_feature_roles() -> dict[str, list[str]]:
    roles = _make_real_alpha_feature_roles()
    roles["alpha_placeholder"] = NEUTRALIZED_SIGNAL_FEATURES
    return roles


def _make_real_alpha_training_metadata() -> dict[str, object]:
    return {
        "signal_stage": "real_alpha_kline_feature_panel",
        "alpha_source": "factor_sss_dx_10",
        "alpha_is_real": True,
        "production_eligible": False,
        "model_form": "p = g(alpha_rank_z_and_kline_rank_z, context)",
        "condition_set": "industry_board_index_size_v1",
        "feature_asof": "same_date_eod_for_close_amount_volume_inputs",
        "label_contract": "forward-looking labels aligned to signal dates",
    }


def _make_real_alpha_fold_assignments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold_id": [1, 1, 1, 1, 1, 1],
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-08",
                    "2024-01-09",
                ]
            ),
            "split": ["train", "train", "valid", "valid", "test", "test"],
        }
    )


def test_train_placeholder_lgbm_models_uses_only_train_for_fit_and_test_for_prediction(
    tmp_path: Path,
) -> None:
    RecordingRegressor.instances.clear()

    result = train_placeholder_lgbm_models(
        panel=_make_training_panel(),
        fold_assignments=_make_fold_assignments(),
        feature_roles=_feature_roles(),
        output_dir=tmp_path,
        model_factory=RecordingRegressor,
        model_params={
            "n_estimators": 10,
            "min_split_gain": 0.01,
            "cat_l2": 3.0,
            "cat_smooth": 7.0,
            "verbosity": -1,
        },
    )

    assert len(RecordingRegressor.instances) == 1
    model = RecordingRegressor.instances[0]
    assert model.params["min_split_gain"] == 0.01
    assert model.params["cat_l2"] == 3.0
    assert model.params["cat_smooth"] == 7.0
    assert model.params["verbosity"] == -1
    assert set(model.fit_y.to_numpy()) == {0.0, -0.01, -0.02, 0.1, 0.09, 0.08}
    assert set(model.valid_y.to_numpy()) == {0.2, 0.19, 0.18, 0.3, 0.29, 0.28}
    assert len(model.fit_X) == 6
    assert len(model.valid_X) == 6
    assert "date" not in model.fit_X.columns
    assert "stock_code" not in model.fit_X.columns
    assert "y_rank_label" not in model.fit_X.columns
    assert "sample_weight" not in model.fit_X.columns
    assert np.isclose(model.fit_kwargs["sample_weight"].mean(), 1.0)
    assert np.isclose(model.fit_kwargs["eval_sample_weight"][0].mean(), 1.0)
    assert np.isclose(model.fit_kwargs["sample_weight"].sum(), 6.0)
    assert np.isclose(model.fit_kwargs["eval_sample_weight"][0].sum(), 6.0)
    assert set(model.fit_kwargs["categorical_feature"]) == {
        "industry",
        "board",
        "index_bucket",
        "size_decile",
    }

    predictions = result["predictions"]
    assert set(predictions["split"]) == {"train", "valid", "test"}
    assert len(predictions[predictions["split"].eq("test")]) == 6
    assert set(predictions.loc[predictions["split"].eq("test"), "y_true"]) == {
        0.4,
        0.39,
        0.38,
        0.5,
        0.49,
        0.48,
    }
    assert {"pred_direct", "pred_context_only", "score_marginal", "score_marginal_z"}.issubset(
        predictions.columns
    )
    assert np.allclose(
        predictions["score_marginal"],
        predictions["pred_direct"] - predictions["pred_context_only"],
    )
    metrics = result["metrics"]
    assert set(metrics["metric_target_col"]) == {"y_resid_fwd"}
    train_predictions = predictions[predictions["split"].eq("train")]
    expected_resid_metrics = compute_daily_prediction_metrics(
        train_predictions,
        target_col="y_resid_fwd",
        pred_col="pred_direct",
    )
    train_metric = metrics[
        metrics["split"].eq("train") & metrics["prediction_col"].eq("pred_direct")
    ].iloc[0]
    assert np.isclose(train_metric["mean_ic"], expected_resid_metrics["mean_ic"])
    assert len(model.predict_calls) == 6
    alpha_columns = _feature_roles()["alpha_placeholder"]
    for context_call in model.predict_calls[1::2]:
        assert np.isclose(context_call[alpha_columns].to_numpy(dtype=float), 0.0).all()
    assert result["summary"]["fold_count"] == 1
    assert (tmp_path / "predictions.csv").exists()
    assert (tmp_path / "metrics_by_fold_split.csv").exists()
    assert not (tmp_path / "feature_importance.csv").exists()
    assert (tmp_path / "detailed_metrics" / "evaluation_summary.json").exists()
    assert (tmp_path / "detailed_metrics" / "feature_gain_by_fold.csv").exists()
    assert (tmp_path / "detailed_metrics" / "feature_gain_summary.csv").exists()
    assert (tmp_path / "detailed_metrics" / "feature_gain_role_summary.csv").exists()
    assert (tmp_path / "detailed_metrics" / "feature_gain_diagnostics.json").exists()
    assert (tmp_path / "charts" / "overall_ic_rankic.png").exists()
    assert (tmp_path / "charts" / "example_score_return_paths.png").exists()
    assert (tmp_path / "charts" / "feature_gain_top.png").exists()
    assert (tmp_path / "training_summary.json").exists()
    summary = json.loads((tmp_path / "training_summary.json").read_text(encoding="utf-8"))
    assert summary["target_col"] == "y_resid_fwd"
    assert summary["fit_target_col"] == "y_resid_fwd"
    assert summary["evaluation_target_col"] == "y_resid_fwd"
    assert summary["metadata"]["model_form"] == "p = g(alpha_placeholder, c)"
    assert summary["metadata"]["alpha_is_real"] is False
    assert summary["model_diagnostics"]["available"] is False
    assert "feature_importance" not in summary["outputs"]
    assert "detailed_metrics" in summary["outputs"]
    assert "charts" in summary["outputs"]
    evaluation_summary = json.loads(
        (tmp_path / "detailed_metrics" / "evaluation_summary.json").read_text(encoding="utf-8")
    )
    assert "feature_gain_summary" in evaluation_summary["outputs"]
    assert "feature_gain_top" in evaluation_summary["charts"]
    assert "feature_gain" in result
    assert "feature_importance" not in result
    metadata = json.loads((tmp_path / "models" / "fold_1_metadata.json").read_text(encoding="utf-8"))
    assert "score_marginal_z" in metadata["score_columns"]


def test_train_real_alpha_feature_panel_uses_signal_kline_and_context_contract(
    tmp_path: Path,
) -> None:
    RecordingRegressor.instances.clear()

    result = train_placeholder_lgbm_models(
        panel=_make_real_alpha_feature_panel(),
        fold_assignments=_make_real_alpha_fold_assignments(),
        feature_roles=_make_real_alpha_feature_roles(use_signal_alias=True),
        output_dir=tmp_path,
        model_factory=RecordingRegressor,
        model_params={"n_estimators": 10, "verbosity": -1},
        training_metadata=_make_real_alpha_training_metadata(),
    )

    model = RecordingRegressor.instances[0]
    expected_features = REAL_SIGNAL_FEATURES + REAL_CATEGORICAL_FEATURES + REAL_CONTINUOUS_FEATURES
    assert model.feature_names_ == expected_features
    forbidden_features = {
        "date",
        "stock_code",
        "y_rank_label",
        "y_resid_fwd",
        "sample_weight",
        "alpha_source",
        "market_cap",
        "log_mcap",
        "factor_sss_dx_10_raw",
        "amo_raw",
        "close_raw",
        "vol_raw",
        "factor_sss_dx_10_raw__rank_input",
        "factor_sss_dx_10_raw__z_input",
        "amo_raw__rank_input",
        "amo_raw__z_input",
        "close_raw__rank_input",
        "close_raw__z_input",
        "vol_raw__rank_input",
        "vol_raw__z_input",
    }
    assert forbidden_features.isdisjoint(model.fit_X.columns)
    assert set(model.fit_kwargs["categorical_feature"]) == set(REAL_CATEGORICAL_FEATURES)

    for direct_call, context_call in zip(model.predict_calls[0::2], model.predict_calls[1::2], strict=True):
        assert np.isclose(context_call[REAL_SIGNAL_FEATURES].to_numpy(dtype=float), 0.0).all()
        assert np.isclose(
            context_call[REAL_CONTINUOUS_FEATURES].to_numpy(dtype=float),
            direct_call[REAL_CONTINUOUS_FEATURES].to_numpy(dtype=float),
        ).all()
        for column in REAL_CATEGORICAL_FEATURES:
            assert context_call[column].equals(direct_call[column])

    assert result["summary"]["signal_features"] == REAL_SIGNAL_FEATURES
    assert result["summary"]["context_only_zeroed_features"] == REAL_SIGNAL_FEATURES
    assert result["summary"]["metadata"]["signal_feature_role"] == "signal_features"
    assert result["summary"]["metadata"]["context_only_policy"] == "zero_signal_features_at_centered_neutral"


def test_train_real_alpha_can_fit_and_evaluate_non_neutralized_raw_return_target(
    tmp_path: Path,
) -> None:
    RecordingRegressor.instances.clear()
    panel = _make_non_neutralized_raw_return_panel()

    result = train_placeholder_lgbm_models(
        panel=panel,
        fold_assignments=_make_real_alpha_fold_assignments(),
        feature_roles=_make_non_neutralized_raw_return_roles(),
        output_dir=tmp_path,
        target_col=RAW_RETURN_TARGET,
        model_factory=RecordingRegressor,
        model_params={"n_estimators": 10, "verbosity": -1},
        training_metadata={
            **_make_real_alpha_training_metadata(),
            "signal_stage": "non_neutralized_real_alpha_raw_return_panel",
            "optimization_objective": "raw_forward_return",
            "target_col": RAW_RETURN_TARGET,
            "neutralization_policy": {
                "return_y": "not_neutralized",
                "alpha_signal": "not_neutralized",
            },
        },
    )

    model = RecordingRegressor.instances[0]
    train_expected = (
        panel[panel["date"].isin(pd.to_datetime(["2024-01-02", "2024-01-03"]))]
        .sort_values(["date", "stock_code"])[RAW_RETURN_TARGET]
        .to_numpy(dtype=float)
    )
    valid_expected = (
        panel[panel["date"].isin(pd.to_datetime(["2024-01-04", "2024-01-05"]))]
        .sort_values(["date", "stock_code"])[RAW_RETURN_TARGET]
        .to_numpy(dtype=float)
    )
    assert np.allclose(model.fit_y.to_numpy(dtype=float), train_expected)
    assert np.allclose(model.valid_y.to_numpy(dtype=float), valid_expected)
    assert RAW_RETURN_TARGET not in model.fit_X.columns
    assert RAW_RETURN_RANK_LABEL not in model.fit_X.columns
    assert "y_resid_fwd" not in model.fit_X.columns

    predictions = result["predictions"]
    assert RAW_RETURN_TARGET in predictions.columns
    assert "y_true" in predictions.columns
    assert "y_resid_fwd" not in predictions.columns
    assert np.allclose(predictions[RAW_RETURN_TARGET], predictions["y_true"])
    assert set(result["metrics"]["metric_target_col"]) == {RAW_RETURN_TARGET}

    train_predictions = predictions[predictions["split"].eq("train")]
    expected_raw_metrics = compute_daily_prediction_metrics(
        train_predictions,
        target_col=RAW_RETURN_TARGET,
        pred_col="pred_direct",
    )
    train_metric = result["metrics"][
        result["metrics"]["split"].eq("train")
        & result["metrics"]["prediction_col"].eq("pred_direct")
    ].iloc[0]
    assert np.isclose(train_metric["mean_ic"], expected_raw_metrics["mean_ic"])

    summary = json.loads((tmp_path / "training_summary.json").read_text(encoding="utf-8"))
    assert summary["target_col"] == RAW_RETURN_TARGET
    assert summary["fit_target_col"] == RAW_RETURN_TARGET
    assert summary["evaluation_target_col"] == RAW_RETURN_TARGET
    assert summary["detailed_evaluation"]["metric_contract"]["target_col"] == RAW_RETURN_TARGET
    assert summary["detailed_evaluation"]["metric_contract"]["spread_target_col"] == RAW_RETURN_TARGET
    assert summary["metadata"]["optimization_objective"] == "raw_forward_return"
    assert summary["metadata"]["neutralization_policy"] == {
        "return_y": "not_neutralized",
        "alpha_signal": "not_neutralized",
    }

    evaluation_summary = json.loads(
        (tmp_path / "detailed_metrics" / "evaluation_summary.json").read_text(encoding="utf-8")
    )
    assert evaluation_summary["metric_contract"]["target_col"] == RAW_RETURN_TARGET
    assert evaluation_summary["metric_contract"]["spread_target_col"] == RAW_RETURN_TARGET
    metadata = json.loads((tmp_path / "models" / "fold_1_metadata.json").read_text(encoding="utf-8"))
    assert metadata["target_col"] == RAW_RETURN_TARGET
    assert metadata["fit_target_col"] == RAW_RETURN_TARGET
    assert metadata["evaluation_target_col"] == RAW_RETURN_TARGET


def test_train_monotone_neutralized_alpha_constrains_only_neutralized_alpha_features(
    tmp_path: Path,
) -> None:
    RecordingRegressor.instances.clear()

    result = train_placeholder_lgbm_models(
        panel=_make_neutralized_alpha_feature_panel(),
        fold_assignments=_make_real_alpha_fold_assignments(),
        feature_roles=_make_neutralized_alpha_feature_roles(),
        output_dir=tmp_path,
        model_factory=RecordingRegressor,
        model_params={"n_estimators": 10, "verbosity": -1},
        training_metadata=_make_real_alpha_training_metadata(),
        monotone_neutralized_alpha=True,
    )

    expected_features = NEUTRALIZED_SIGNAL_FEATURES + REAL_CATEGORICAL_FEATURES + REAL_CONTINUOUS_FEATURES
    expected_constraints = [
        1 if feature in NEUTRALIZED_ALPHA_MONOTONE_FEATURES else 0
        for feature in expected_features
    ]

    model = RecordingRegressor.instances[0]
    assert model.feature_names_ == expected_features
    assert model.params["monotone_constraints"] == expected_constraints
    assert result["summary"]["model_params"]["monotone_constraints"] == expected_constraints
    assert result["summary"]["monotone_constraints"] == {
        "enabled": True,
        "policy": "positive_on_neutralized_alpha_rank_z_only",
        "positive_constraint_value": 1,
        "negative_constraint_value": -1,
        "constrained_features": NEUTRALIZED_ALPHA_MONOTONE_FEATURES,
        "constraints_by_feature": dict(zip(expected_features, expected_constraints, strict=True)),
        "semantic_note": (
            "LightGBM monotone constraints enforce partial monotonicity while all other "
            "features are fixed; they do not preserve global cross-context ranking."
        ),
    }

    metadata = json.loads((tmp_path / "models" / "fold_1_metadata.json").read_text(encoding="utf-8"))
    assert metadata["model_params"]["monotone_constraints"] == expected_constraints
    assert metadata["monotone_constraints"] == result["summary"]["monotone_constraints"]


def test_train_monotone_signal_features_constrains_raw_alpha_signal_features(
    tmp_path: Path,
) -> None:
    RecordingRegressor.instances.clear()
    feature_roles = _make_non_neutralized_raw_return_roles()
    feature_roles["signal_features"] = ["factor_sss_dx_10_rank", "factor_sss_dx_10_z"]

    result = train_placeholder_lgbm_models(
        panel=_make_non_neutralized_raw_return_panel(),
        fold_assignments=_make_real_alpha_fold_assignments(),
        feature_roles=feature_roles,
        output_dir=tmp_path,
        target_col=RAW_RETURN_TARGET,
        model_factory=RecordingRegressor,
        model_params={"n_estimators": 10, "verbosity": -1},
        training_metadata=_make_real_alpha_training_metadata(),
        monotone_signal_features="positive",
    )

    expected_features = [
        "factor_sss_dx_10_rank",
        "factor_sss_dx_10_z",
        *REAL_CATEGORICAL_FEATURES,
        *REAL_CONTINUOUS_FEATURES,
    ]
    expected_constraints = [1, 1, *([0] * (len(expected_features) - 2))]

    model = RecordingRegressor.instances[0]
    assert model.feature_names_ == expected_features
    assert model.params["monotone_constraints"] == expected_constraints
    assert result["summary"]["model_params"]["monotone_constraints"] == expected_constraints
    assert result["summary"]["monotone_constraints"] == {
        "enabled": True,
        "policy": "positive_on_signal_features",
        "positive_constraint_value": 1,
        "negative_constraint_value": -1,
        "constrained_features": ["factor_sss_dx_10_rank", "factor_sss_dx_10_z"],
        "constraints_by_feature": dict(zip(expected_features, expected_constraints, strict=True)),
        "semantic_note": (
            "LightGBM monotone constraints enforce partial monotonicity while all other "
            "features are fixed; they do not preserve global cross-context ranking."
        ),
    }

    metadata = json.loads((tmp_path / "models" / "fold_1_metadata.json").read_text(encoding="utf-8"))
    assert metadata["model_params"]["monotone_constraints"] == expected_constraints
    assert metadata["monotone_constraints"] == result["summary"]["monotone_constraints"]


def test_train_can_exclude_model_features_before_contract_and_monotone_constraints(
    tmp_path: Path,
) -> None:
    RecordingRegressor.instances.clear()
    excluded_features = ["is_csi300", "is_csi500", "is_csi1000", "is_csi2000"]
    feature_roles = _make_non_neutralized_raw_return_roles()
    feature_roles["signal_features"] = ["factor_sss_dx_10_rank", "factor_sss_dx_10_z"]

    result = train_placeholder_lgbm_models(
        panel=_make_non_neutralized_raw_return_panel(),
        fold_assignments=_make_real_alpha_fold_assignments(),
        feature_roles=feature_roles,
        output_dir=tmp_path,
        target_col=RAW_RETURN_TARGET,
        model_factory=RecordingRegressor,
        model_params={"n_estimators": 10, "verbosity": -1},
        training_metadata=_make_real_alpha_training_metadata(),
        monotone_signal_features="positive",
        exclude_features=excluded_features,
    )

    expected_features = [
        "factor_sss_dx_10_rank",
        "factor_sss_dx_10_z",
        *REAL_CATEGORICAL_FEATURES,
        "log_mcap_z",
        "mcap_rank",
    ]
    expected_constraints = [1, 1, *([0] * (len(expected_features) - 2))]

    model = RecordingRegressor.instances[0]
    assert model.feature_names_ == expected_features
    assert all(feature not in model.fit_X.columns for feature in excluded_features)
    assert model.params["monotone_constraints"] == expected_constraints
    assert result["summary"]["feature_columns"] == expected_features
    assert result["summary"]["condition_continuous_features"] == ["log_mcap_z", "mcap_rank"]
    assert result["summary"]["context_features"] == [
        *REAL_CATEGORICAL_FEATURES,
        "log_mcap_z",
        "mcap_rank",
    ]
    assert result["summary"]["feature_exclusion"] == {
        "requested_exclude_features": excluded_features,
        "excluded_model_features": excluded_features,
    }
    assert all(
        feature not in result["summary"]["monotone_constraints"]["constraints_by_feature"]
        for feature in excluded_features
    )

    metadata = json.loads((tmp_path / "models" / "fold_1_metadata.json").read_text(encoding="utf-8"))
    assert metadata["feature_columns"] == expected_features
    assert metadata["feature_exclusion"] == result["summary"]["feature_exclusion"]


def test_train_monotone_signal_features_rejects_manual_constraint_source_conflict(
    tmp_path: Path,
) -> None:
    feature_roles = _make_non_neutralized_raw_return_roles()
    feature_roles["signal_features"] = ["factor_sss_dx_10_rank", "factor_sss_dx_10_z"]

    try:
        train_placeholder_lgbm_models(
            panel=_make_non_neutralized_raw_return_panel(),
            fold_assignments=_make_real_alpha_fold_assignments(),
            feature_roles=feature_roles,
            output_dir=tmp_path,
            target_col=RAW_RETURN_TARGET,
            model_factory=RecordingRegressor,
            model_params={
                "n_estimators": 10,
                "verbosity": -1,
                "monotone_constraints": [0] * 12,
            },
            training_metadata=_make_real_alpha_training_metadata(),
            monotone_signal_features="positive",
        )
    except ValueError as exc:
        assert "model_params must not define monotone_constraints" in str(exc)
    else:
        raise AssertionError("Expected conflicting monotone constraint sources to raise ValueError")


def test_train_real_alpha_uses_fold_local_weight_normalization(
    tmp_path: Path,
) -> None:
    RecordingRegressor.instances.clear()
    panel = _make_real_alpha_feature_panel()

    train_rows = panel[panel["date"].isin(pd.to_datetime(["2024-01-02", "2024-01-03"]))]
    valid_rows = panel[panel["date"].isin(pd.to_datetime(["2024-01-04", "2024-01-05"]))]
    expected_train_weights = (
        train_rows["sample_weight"].astype(float) / train_rows["sample_weight"].astype(float).mean()
    ).to_numpy(dtype=float)
    expected_valid_weights = (
        valid_rows["sample_weight"].astype(float) / valid_rows["sample_weight"].astype(float).mean()
    ).to_numpy(dtype=float)

    train_placeholder_lgbm_models(
        panel=panel,
        fold_assignments=_make_real_alpha_fold_assignments(),
        feature_roles=_make_real_alpha_feature_roles(),
        output_dir=tmp_path,
        model_factory=RecordingRegressor,
        model_params={"n_estimators": 10, "verbosity": -1},
        training_metadata=_make_real_alpha_training_metadata(),
    )

    model = RecordingRegressor.instances[0]
    assert np.allclose(model.fit_kwargs["sample_weight"], expected_train_weights)
    assert np.allclose(model.fit_kwargs["eval_sample_weight"][0], expected_valid_weights)


def test_train_real_alpha_rejects_target_not_declared_in_feature_roles(tmp_path: Path) -> None:
    try:
        train_placeholder_lgbm_models(
            panel=_make_real_alpha_feature_panel(),
            fold_assignments=_make_real_alpha_fold_assignments(),
            feature_roles=_make_real_alpha_feature_roles(),
            output_dir=tmp_path,
            model_factory=RecordingRegressor,
            model_params={"n_estimators": 10, "verbosity": -1},
            training_metadata=_make_real_alpha_training_metadata(),
            target_col="not_a_declared_target",
        )
    except ValueError as exc:
        assert "not declared in feature_roles['targets']" in str(exc)
    else:
        raise AssertionError("Expected undeclared target column to raise ValueError")


def test_train_rejects_leakage_columns_in_real_alpha_feature_roles(tmp_path: Path) -> None:
    feature_roles = _make_real_alpha_feature_roles()
    feature_roles["alpha_placeholder"] = [*feature_roles["alpha_placeholder"], "y_resid_fwd"]

    try:
        train_placeholder_lgbm_models(
            panel=_make_real_alpha_feature_panel(),
            fold_assignments=_make_real_alpha_fold_assignments(),
            feature_roles=feature_roles,
            output_dir=tmp_path,
            model_factory=RecordingRegressor,
            model_params={"n_estimators": 10, "verbosity": -1},
            training_metadata=_make_real_alpha_training_metadata(),
        )
    except ValueError as exc:
        assert "Leakage-prone columns cannot be model features" in str(exc)
    else:
        raise AssertionError("Expected leakage-prone feature role to raise ValueError")


def test_train_rejects_fold_assignments_with_future_leakage_order(tmp_path: Path) -> None:
    fold_assignments = _make_real_alpha_fold_assignments()
    fold_assignments.loc[fold_assignments["split"].eq("train"), "date"] = pd.to_datetime(
        ["2024-01-04", "2024-01-05"]
    )
    fold_assignments.loc[fold_assignments["split"].eq("valid"), "date"] = pd.to_datetime(
        ["2024-01-02", "2024-01-03"]
    )

    try:
        train_placeholder_lgbm_models(
            panel=_make_real_alpha_feature_panel(),
            fold_assignments=fold_assignments,
            feature_roles=_make_real_alpha_feature_roles(),
            output_dir=tmp_path,
            model_factory=RecordingRegressor,
            model_params={"n_estimators": 10, "verbosity": -1},
            training_metadata=_make_real_alpha_training_metadata(),
        )
    except ValueError as exc:
        assert "train dates that are not strictly before validation dates" in str(exc)
    else:
        raise AssertionError("Expected future-leaking fold order to raise ValueError")


def test_train_placeholder_lgbm_models_preserves_real_alpha_training_metadata(
    tmp_path: Path,
) -> None:
    RecordingRegressor.instances.clear()

    result = train_placeholder_lgbm_models(
        panel=_make_training_panel(),
        fold_assignments=_make_fold_assignments(),
        feature_roles=_feature_roles(),
        output_dir=tmp_path,
        model_factory=RecordingRegressor,
        model_params={"n_estimators": 10, "verbosity": -1},
        training_metadata={
            "signal_stage": "real_alpha_kline_feature_panel",
            "alpha_source": "factor_sss_dx_10",
            "alpha_is_real": True,
            "production_eligible": False,
            "model_form": "p = g(alpha_rank_z_and_kline_rank_z, context)",
        },
    )

    assert result["summary"]["metadata"]["signal_stage"] == "real_alpha_kline_feature_panel"
    assert result["summary"]["metadata"]["alpha_source"] == "factor_sss_dx_10"
    assert result["summary"]["metadata"]["alpha_is_real"] is True
    assert result["summary"]["metadata"]["model_form"] == "p = g(alpha_rank_z_and_kline_rank_z, context)"
    assert result["summary"]["metadata"]["training_role"] == "direct_and_marginal_signal_probe"
    assert all("placeholder-liquidity" not in note for note in result["summary"]["notes"])


def test_compute_daily_prediction_metrics_returns_ic_and_rankic() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "y_true": [1.0, 2.0, 3.0, 2.0, 1.0],
            "prediction": [1.0, 2.0, 3.0, 1.0, 2.0],
        }
    )

    metrics = compute_daily_prediction_metrics(frame)

    assert metrics["date_count"] == 2
    assert np.isclose(metrics["mean_ic"], 0.0)
    assert np.isclose(metrics["mean_rankic"], 0.0)
    assert np.isclose(metrics["ic_by_date"][pd.Timestamp("2024-01-02")], 1.0)
    assert np.isclose(metrics["rankic_by_date"][pd.Timestamp("2024-01-03")], -1.0)


def test_model_tree_diagnostics_summarizes_lightgbm_dump() -> None:
    class Booster:
        def dump_model(self) -> dict[str, object]:
            return {
                "tree_info": [
                    {
                        "num_leaves": 1,
                        "tree_structure": {"leaf_index": 0},
                    },
                    {
                        "num_leaves": 3,
                        "tree_structure": {
                            "split_index": 0,
                            "left_child": {"leaf_index": 0},
                            "right_child": {
                                "split_index": 1,
                                "left_child": {"leaf_index": 1},
                                "right_child": {"leaf_index": 2},
                            },
                        },
                    },
                ]
            }

    class Model:
        booster_ = Booster()

    diagnostics = _model_tree_diagnostics(Model())

    assert diagnostics["available"] is True
    assert diagnostics["tree_count"] == 2
    assert diagnostics["one_leaf_tree_count"] == 1
    assert np.isclose(diagnostics["one_leaf_tree_share"], 0.5)
    assert np.isclose(diagnostics["mean_leaf_count"], 2.0)
    assert diagnostics["max_tree_depth"] == 2


def test_feature_gain_frame_uses_best_iteration_when_supported() -> None:
    class Booster:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def feature_importance(self, *, importance_type: str, iteration: int | None = None) -> np.ndarray:
            self.calls.append({"importance_type": importance_type, "iteration": iteration})
            if importance_type == "gain":
                return np.array([5.0, 1.0])
            return np.array([3.0, 2.0])

    class Model:
        def __init__(self) -> None:
            self.booster_ = Booster()
            self.best_iteration_ = 4

    model = Model()
    frame = _feature_gain_frame(model, ["alpha", "context"], fold_id=2)

    assert model.booster_.calls == [
        {"importance_type": "gain", "iteration": 4},
        {"importance_type": "split", "iteration": 4},
    ]
    assert frame["fold_id"].tolist() == [2, 2]
    assert frame["feature"].tolist() == ["alpha", "context"]
    assert frame["importance_gain"].tolist() == [5.0, 1.0]
    assert frame["importance_split"].tolist() == [3.0, 2.0]
