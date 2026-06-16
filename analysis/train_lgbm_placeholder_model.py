from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


DEFAULT_MODEL_PARAMS: dict[str, Any] = {
    "objective": "huber",
    "learning_rate": 0.03,
    "n_estimators": 5000,
    "num_leaves": 31,
    "max_depth": 5,
    "min_child_samples": 500,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 2.0,
    "min_split_gain": 0.0,
    "max_bin": 255,
    "cat_smooth": 15,
    "cat_l2": 3,
    "importance_type": "gain",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 1,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train fold-wise LGBMRegressor models for p = g(signal, context)."
    )
    parser.add_argument("--training-panel", required=True, type=Path)
    parser.add_argument("--fold-assignments", required=True, type=Path)
    parser.add_argument("--training-data-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-col", default="y_resid_fwd")
    parser.add_argument("--objective", default=DEFAULT_MODEL_PARAMS["objective"])
    parser.add_argument("--learning-rate", default=DEFAULT_MODEL_PARAMS["learning_rate"], type=float)
    parser.add_argument("--n-estimators", default=DEFAULT_MODEL_PARAMS["n_estimators"], type=int)
    parser.add_argument("--num-leaves", default=DEFAULT_MODEL_PARAMS["num_leaves"], type=int)
    parser.add_argument("--max-depth", default=DEFAULT_MODEL_PARAMS["max_depth"], type=int)
    parser.add_argument("--min-child-samples", default=DEFAULT_MODEL_PARAMS["min_child_samples"], type=int)
    parser.add_argument("--subsample", default=DEFAULT_MODEL_PARAMS["subsample"], type=float)
    parser.add_argument("--subsample-freq", default=DEFAULT_MODEL_PARAMS["subsample_freq"], type=int)
    parser.add_argument("--colsample-bytree", default=DEFAULT_MODEL_PARAMS["colsample_bytree"], type=float)
    parser.add_argument("--reg-alpha", default=DEFAULT_MODEL_PARAMS["reg_alpha"], type=float)
    parser.add_argument("--reg-lambda", default=DEFAULT_MODEL_PARAMS["reg_lambda"], type=float)
    parser.add_argument("--min-split-gain", default=DEFAULT_MODEL_PARAMS["min_split_gain"], type=float)
    parser.add_argument("--max-bin", default=DEFAULT_MODEL_PARAMS["max_bin"], type=int)
    parser.add_argument("--cat-smooth", default=DEFAULT_MODEL_PARAMS["cat_smooth"], type=float)
    parser.add_argument("--cat-l2", default=DEFAULT_MODEL_PARAMS["cat_l2"], type=float)
    parser.add_argument("--n-jobs", default=DEFAULT_MODEL_PARAMS["n_jobs"], type=int)
    parser.add_argument("--verbosity", default=DEFAULT_MODEL_PARAMS["verbosity"], type=int)
    parser.add_argument("--early-stopping-rounds", default=200, type=int)
    parser.add_argument(
        "--monotone-neutralized-alpha",
        action="store_true",
        help=(
            "Apply +1 LightGBM monotone constraints only to neutralized alpha "
            "rank/z signal features; all context and non-neutralized signal features remain unconstrained."
        ),
    )
    parser.add_argument(
        "--monotone-signal-features",
        default="none",
        choices=["none", "positive", "negative"],
        help=(
            "Apply a LightGBM monotone constraint to all model signal features. "
            "Use 'positive' for raw alpha features where higher signal should not lower prediction "
            "with all other features fixed."
        ),
    )
    parser.add_argument(
        "--monotone-positive-features",
        nargs="*",
        default=None,
        help="Additional numeric model features to constrain with +1 monotonicity.",
    )
    parser.add_argument(
        "--monotone-negative-features",
        nargs="*",
        default=None,
        help="Additional numeric model features to constrain with -1 monotonicity.",
    )
    parser.add_argument(
        "--exclude-features",
        nargs="*",
        default=None,
        help="Model feature columns to remove from feature_roles before training.",
    )
    parser.add_argument("--fold-ids", nargs="*", type=int, default=None)
    args = parser.parse_args()

    panel = pd.read_pickle(args.training_panel)
    fold_assignments = pd.read_csv(args.fold_assignments, parse_dates=["date"])
    training_data_summary = json.loads(args.training_data_summary.read_text(encoding="utf-8"))
    feature_roles = training_data_summary["feature_roles"]
    model_params = {
        **DEFAULT_MODEL_PARAMS,
        "objective": args.objective,
        "learning_rate": args.learning_rate,
        "n_estimators": args.n_estimators,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_child_samples": args.min_child_samples,
        "subsample": args.subsample,
        "subsample_freq": args.subsample_freq,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "min_split_gain": args.min_split_gain,
        "max_bin": args.max_bin,
        "cat_smooth": args.cat_smooth,
        "cat_l2": args.cat_l2,
        "n_jobs": args.n_jobs,
        "verbosity": args.verbosity,
    }

    result = train_placeholder_lgbm_models(
        panel=panel,
        fold_assignments=fold_assignments,
        feature_roles=feature_roles,
        output_dir=args.output_dir,
        target_col=args.target_col,
        model_params=model_params,
        early_stopping_rounds=args.early_stopping_rounds,
        fold_ids=args.fold_ids,
        training_metadata=training_data_summary.get("metadata"),
        training_data_summary=training_data_summary,
        training_data_summary_path=args.training_data_summary,
        monotone_neutralized_alpha=args.monotone_neutralized_alpha,
        monotone_signal_features=args.monotone_signal_features,
        monotone_positive_features=args.monotone_positive_features,
        monotone_negative_features=args.monotone_negative_features,
        exclude_features=args.exclude_features,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


def train_placeholder_lgbm_models(
    *,
    panel: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    feature_roles: dict[str, list[str]],
    output_dir: Path,
    target_col: str = "y_resid_fwd",
    model_params: dict[str, Any] | None = None,
    model_factory: Callable[..., Any] | None = None,
    early_stopping_rounds: int = 200,
    fold_ids: list[int] | None = None,
    training_metadata: dict[str, Any] | None = None,
    training_data_summary: dict[str, Any] | None = None,
    training_data_summary_path: Path | None = None,
    monotone_neutralized_alpha: bool = False,
    monotone_signal_features: str = "none",
    monotone_positive_features: list[str] | None = None,
    monotone_negative_features: list[str] | None = None,
    exclude_features: list[str] | None = None,
) -> dict[str, Any]:
    prepared_panel = _prepare_panel(panel)
    prepared_folds = _prepare_fold_assignments(fold_assignments)
    effective_feature_roles, feature_exclusion = _feature_roles_after_exclusion(
        feature_roles,
        exclude_features=exclude_features,
    )
    feature_contract = _feature_contract(effective_feature_roles)
    model_feature_roles = feature_contract["feature_roles"]
    features = feature_contract["feature_columns"]
    signal_features = feature_contract["signal_features"]
    categorical_features = feature_contract["condition_categorical"]
    continuous_features = feature_contract["condition_continuous"]
    context_features = feature_contract["context_features"]
    _validate_training_inputs(
        prepared_panel,
        prepared_folds,
        features,
        categorical_features,
        target_col,
        target_columns=feature_contract["targets"],
    )
    evaluation_target_col = target_col

    if model_factory is None:
        model_factory = _lightgbm_model_factory
    managed_monotone_requested = (
        monotone_neutralized_alpha
        or monotone_signal_features != "none"
        or bool(monotone_positive_features)
        or bool(monotone_negative_features)
    )
    if managed_monotone_requested and model_params and "monotone_constraints" in model_params:
        raise ValueError(
            "model_params must not define monotone_constraints when "
            "a managed monotone constraint policy is requested; use one monotonicity source so "
            "the training contract is explicit."
        )
    if monotone_neutralized_alpha and (
        monotone_signal_features != "none"
        or bool(monotone_positive_features)
        or bool(monotone_negative_features)
    ):
        raise ValueError(
            "monotone_neutralized_alpha cannot be combined with monotone_signal_features or explicit "
            "monotone feature lists; choose one managed monotone constraint policy."
        )
    params = dict(DEFAULT_MODEL_PARAMS)
    if model_params:
        params.update(model_params)
    monotone_policy = "none"
    monotone_enabled = False
    if monotone_neutralized_alpha:
        monotone_constraints = _neutralized_alpha_monotone_constraints(
            features,
            signal_features=signal_features,
        )
        if not any(monotone_constraints):
            raise ValueError(
                "monotone_neutralized_alpha=True requires at least one signal feature ending "
                "with '_neutralized_rank' or '_neutralized_z'."
            )
        params["monotone_constraints"] = monotone_constraints
        monotone_policy = "positive_on_neutralized_alpha_rank_z_only"
        monotone_enabled = True
    elif managed_monotone_requested:
        monotone_constraints, monotone_policy = _signal_monotone_constraints(
            features,
            signal_features=signal_features,
            categorical_features=categorical_features,
            monotone_signal_features=monotone_signal_features,
            monotone_positive_features=monotone_positive_features or [],
            monotone_negative_features=monotone_negative_features or [],
        )
        if not any(monotone_constraints):
            raise ValueError("Managed monotone constraint policy produced no non-zero constraints.")
        params["monotone_constraints"] = monotone_constraints
        monotone_enabled = True
    elif "monotone_constraints" in params:
        monotone_policy = "model_params_manual"
        monotone_enabled = any(int(value) != 0 for value in params["monotone_constraints"])
    monotone_constraint_summary = _monotone_constraint_summary(
        features,
        params.get("monotone_constraints"),
        enabled=monotone_enabled,
        policy=monotone_policy,
    )

    selected_fold_ids = sorted(prepared_folds["fold_id"].unique().tolist())
    if fold_ids is not None:
        requested = set(int(fold_id) for fold_id in fold_ids)
        selected_fold_ids = [fold_id for fold_id in selected_fold_ids if fold_id in requested]
    if not selected_fold_ids:
        raise ValueError("No fold ids selected for training.")

    summary_metadata = _metadata_from_training_summary(training_data_summary, training_metadata)
    run_metadata = _training_metadata(
        summary_metadata,
        signal_feature_role=feature_contract["signal_feature_role"],
        context_only_policy="zero_signal_features_at_centered_neutral",
    )
    source_training_data_summary = (
        _training_data_summary_provenance(training_data_summary_path)
        if training_data_summary_path is not None
        else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    prediction_frames: list[pd.DataFrame] = []
    metrics_records: list[dict[str, Any]] = []
    feature_gain_frames: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, Any]] = []

    for fold_id in selected_fold_ids:
        fold_dates = prepared_folds[prepared_folds["fold_id"].eq(fold_id)].copy()
        split_frames = {
            split: _panel_for_split(prepared_panel, fold_dates, split)
            for split in ("train", "valid", "test")
        }
        for split, frame in split_frames.items():
            if frame.empty:
                raise ValueError(f"Fold {fold_id} has no {split} rows.")

        train_frame, valid_frame, test_frame = (
            split_frames["train"],
            split_frames["valid"],
            split_frames["test"],
        )
        category_maps = _fit_category_maps(train_frame, categorical_features)
        train_X = _make_feature_matrix(train_frame, features, categorical_features, category_maps)
        valid_X = _make_feature_matrix(valid_frame, features, categorical_features, category_maps)
        test_X = _make_feature_matrix(test_frame, features, categorical_features, category_maps)
        y_train = train_frame[target_col].astype(float)
        y_valid = valid_frame[target_col].astype(float)

        model = model_factory(**params)
        fit_kwargs = {
            "sample_weight": _normalized_model_weights(train_frame["sample_weight"], split_name="train"),
            "eval_set": [(valid_X, y_valid)],
            "eval_names": ["valid"],
            "eval_sample_weight": [
                _normalized_model_weights(valid_frame["sample_weight"], split_name="valid")
            ],
            "feature_name": features,
            "categorical_feature": categorical_features,
        }
        callbacks = _lightgbm_callbacks(early_stopping_rounds)
        if callbacks:
            fit_kwargs["callbacks"] = callbacks
        model.fit(train_X, y_train, **fit_kwargs)

        model_path = _save_model(model, model_dir / f"fold_{fold_id}_model")
        feature_gain_frames.append(_feature_gain_frame(model, features, fold_id))
        tree_diagnostics = _model_tree_diagnostics(model)

        for split, frame, matrix in [
            ("train", train_frame, train_X),
            ("valid", valid_frame, valid_X),
            ("test", test_frame, test_X),
        ]:
            direct_predictions = _predict_model(model, matrix)
            context_matrix = _make_context_only_matrix(matrix, signal_features)
            context_predictions = _predict_model(model, context_matrix)
            marginal_score = direct_predictions - context_predictions
            prediction_data = {
                "fold_id": int(fold_id),
                "split": split,
                "date": frame["date"].to_numpy(),
                "stock_code": frame["stock_code"].astype(str).to_numpy(),
                target_col: frame[target_col].astype(float).to_numpy(),
                "y_true": frame[target_col].astype(float).to_numpy(),
                "sample_weight": frame["sample_weight"].astype(float).to_numpy(),
                "pred_direct": direct_predictions,
                "pred_context_only": context_predictions,
                "score_marginal": marginal_score,
            }
            if target_col != "y_resid_fwd" and "y_resid_fwd" in frame:
                prediction_data["y_resid_fwd"] = frame["y_resid_fwd"].astype(float).to_numpy()
            prediction_frame = pd.DataFrame(prediction_data)
            prediction_frame["score_marginal_z"] = (
                prediction_frame.groupby("date", group_keys=False)["score_marginal"]
                .transform(_robust_zscore_series)
                .astype(float)
            )
            prediction_frames.append(prediction_frame)
            metric_target_col = evaluation_target_col
            for prediction_col in ("pred_direct", "score_marginal", "score_marginal_z"):
                split_metrics = compute_daily_prediction_metrics(
                    prediction_frame,
                    target_col=metric_target_col,
                    pred_col=prediction_col,
                )
                metrics_records.append(
                    {
                        "fold_id": int(fold_id),
                        "split": split,
                        "prediction_col": prediction_col,
                        "metric_target_col": metric_target_col,
                        **_metrics_for_csv(split_metrics),
                    }
                )

        fold_summary = {
            "fold_id": int(fold_id),
            "best_iteration": _best_iteration(model),
            "best_score": _json_safe(getattr(model, "best_score_", None)),
            "model_path": _path_for_summary(model_path),
            "rows": {split: int(len(frame)) for split, frame in split_frames.items()},
            "date_ranges": {
                split: {
                    "start": _date_to_string(frame["date"].min()),
                    "end": _date_to_string(frame["date"].max()),
                    "date_count": int(frame["date"].nunique()),
                }
                for split, frame in split_frames.items()
            },
            "category_maps": category_maps,
            "tree_diagnostics": tree_diagnostics,
        }
        fold_summaries.append(fold_summary)
        (model_dir / f"fold_{fold_id}_metadata.json").write_text(
            json.dumps(
                {
                    "fold": fold_summary,
                    "feature_roles": model_feature_roles,
                    "feature_columns": features,
                    "signal_features": signal_features,
                    "alpha_features": signal_features,
                    "context_features": context_features,
                    "categorical_features": categorical_features,
                    "continuous_features": continuous_features,
                    "context_only_zeroed_features": signal_features,
                    "target_col": target_col,
                    "fit_target_col": target_col,
                    "evaluation_target_col": evaluation_target_col,
                    "training_metadata": run_metadata,
                    "source_training_data_summary": source_training_data_summary,
                    "model_params": params,
                    "monotone_constraints": monotone_constraint_summary,
                    "feature_exclusion": feature_exclusion,
                    "score_columns": [
                        "pred_direct",
                        "pred_context_only",
                        "score_marginal",
                        "score_marginal_z",
                    ],
                    "notes": [
                        "Category maps are fitted on the fold train split only.",
                        "Unseen validation or test categories are mapped to UNKNOWN.",
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    predictions_output = pd.concat(prediction_frames, ignore_index=True)
    metrics_output = pd.DataFrame(metrics_records).sort_values(
        ["fold_id", "split", "prediction_col"]
    )
    feature_gain_output = pd.concat(feature_gain_frames, ignore_index=True)

    _write_csv_with_iso_dates(predictions_output, output_dir / "predictions.csv")
    _write_csv_with_iso_dates(metrics_output, output_dir / "metrics_by_fold_split.csv")

    try:
        from analysis.evaluate_lgbm_training_metrics import write_model_evaluation_artifacts
    except ModuleNotFoundError:
        from evaluate_lgbm_training_metrics import write_model_evaluation_artifacts

    detailed_metrics_dir = output_dir / "detailed_metrics"
    charts_dir = output_dir / "charts"
    report_path = output_dir / "training_report.html"
    detailed_evaluation_summary = write_model_evaluation_artifacts(
        predictions=predictions_output,
        training_panel=prepared_panel,
        output_dir=detailed_metrics_dir,
        score_cols=["pred_direct", "score_marginal", "score_marginal_z"],
        target_col=evaluation_target_col,
        spread_target_col=evaluation_target_col,
        top_bottom_quantiles=10,
        condition_bucket_count=3,
        min_group_obs=2,
        charts_dir=charts_dir,
        report_path=report_path,
        feature_gain=feature_gain_output,
        feature_roles=model_feature_roles,
        feature_columns=features,
    )

    summary = {
        "metadata": run_metadata,
        "fold_count": int(len(selected_fold_ids)),
        "target_col": target_col,
        "fit_target_col": target_col,
        "evaluation_target_col": evaluation_target_col,
        "feature_roles": model_feature_roles,
        "feature_columns": features,
        "signal_features": signal_features,
        "alpha_features": signal_features,
        "context_features": context_features,
        "condition_categorical_features": categorical_features,
        "condition_continuous_features": continuous_features,
        "categorical_features": categorical_features,
        "context_only_zeroed_features": signal_features,
        "source_training_data_summary": source_training_data_summary,
        "score_columns": ["pred_direct", "pred_context_only", "score_marginal", "score_marginal_z"],
        "model_params": params,
        "monotone_constraints": monotone_constraint_summary,
        "feature_exclusion": feature_exclusion,
        "early_stopping_rounds": int(early_stopping_rounds),
        "folds": fold_summaries,
        "model_diagnostics": _summarize_tree_diagnostics(fold_summaries),
        "metrics": metrics_output.to_dict("records"),
        "detailed_evaluation": {
            "primary_score_col": detailed_evaluation_summary["primary_score_col"],
            "metric_contract": detailed_evaluation_summary["metric_contract"],
            "row_counts": detailed_evaluation_summary["row_counts"],
            "group_dimensions": detailed_evaluation_summary["group_dimensions"],
        },
        "outputs": {
            "predictions": _path_for_summary(output_dir / "predictions.csv"),
            "metrics": _path_for_summary(output_dir / "metrics_by_fold_split.csv"),
            "models": _path_for_summary(model_dir),
            "detailed_metrics": _path_for_summary(detailed_metrics_dir),
            "charts": _path_for_summary(charts_dir),
            "report": _path_for_summary(report_path),
        },
        "notes": _training_notes(run_metadata, target_col=evaluation_target_col),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "predictions": predictions_output,
        "metrics": metrics_output,
        "feature_gain": feature_gain_output,
        "summary": summary,
    }


def _metadata_from_training_summary(
    training_data_summary: dict[str, Any] | None,
    training_metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if training_data_summary is None:
        return training_metadata
    metadata = training_data_summary.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    return training_metadata


def _training_metadata(
    training_metadata: dict[str, Any] | None,
    *,
    signal_feature_role: str,
    context_only_policy: str,
) -> dict[str, Any]:
    if training_metadata is None:
        metadata = {
            "signal_stage": "placeholder_alpha_probe",
            "alpha_source": "placeholder_liquidity",
            "alpha_is_real": False,
            "production_eligible": False,
            "model_form": "p = g(alpha_placeholder, c)",
            "training_role": "direct_and_marginal_placeholder_probe",
        }
    else:
        metadata = {str(key): _json_safe(value) for key, value in training_metadata.items()}
    if "training_role" not in metadata:
        metadata["training_role"] = (
            "direct_and_marginal_signal_probe"
            if metadata.get("alpha_is_real") is True
            else "direct_and_marginal_placeholder_probe"
        )
    metadata["signal_feature_role"] = signal_feature_role
    metadata["context_only_policy"] = context_only_policy
    return metadata


def _training_notes(metadata: dict[str, Any], *, target_col: str) -> list[str]:
    notes = [
        "Each fold is trained only on dates marked train in fold_assignments.",
        "Validation dates are used only as eval_set for early stopping.",
        "Test dates are used only for prediction and metrics.",
        "score_marginal = pred_direct - pred_context_only subtracts a zero-signal baseline prediction.",
        f"Compact metrics, detailed evaluation, and charts use {target_col} as the realized target.",
        "The current centered signal rank/z convention uses neutral signal values of 0.0.",
    ]
    if metadata.get("alpha_is_real") is True:
        notes.append(
            "Context-only prediction sets signal features to the centered neutral value 0.0 and preserves context features."
        )
    else:
        notes.append("This is a placeholder-liquidity probe, not a production alpha model.")
    return notes


def _training_data_summary_provenance(path: Path) -> dict[str, Any]:
    return {
        "path": _path_for_summary(path),
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_daily_prediction_metrics(
    predictions: pd.DataFrame,
    *,
    target_col: str = "y_true",
    pred_col: str = "prediction",
) -> dict[str, Any]:
    required = {"date", target_col, pred_col}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing required columns: {sorted(missing)}")

    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    ic_by_date: dict[pd.Timestamp, float] = {}
    rankic_by_date: dict[pd.Timestamp, float] = {}
    for date, group in frame.groupby("date", sort=True):
        y = pd.to_numeric(group[target_col], errors="coerce")
        p = pd.to_numeric(group[pred_col], errors="coerce")
        valid = np.isfinite(y.to_numpy(dtype=float)) & np.isfinite(p.to_numpy(dtype=float))
        if int(valid.sum()) < 2:
            continue
        yv = y.loc[valid]
        pv = p.loc[valid]
        ic_value = _safe_corr(yv, pv)
        rankic_value = _safe_corr(yv.rank(method="average"), pv.rank(method="average"))
        if np.isfinite(ic_value):
            ic_by_date[pd.Timestamp(date)] = ic_value
        if np.isfinite(rankic_value):
            rankic_by_date[pd.Timestamp(date)] = rankic_value

    ic_values = _finite_values(ic_by_date.values())
    rankic_values = _finite_values(rankic_by_date.values())
    return {
        "date_count": int(len(ic_values)),
        "mean_ic": float(np.nanmean(ic_values)) if len(ic_values) else np.nan,
        "std_ic": float(np.nanstd(ic_values, ddof=0)) if len(ic_values) else np.nan,
        "mean_rankic": float(np.nanmean(rankic_values)) if len(rankic_values) else np.nan,
        "std_rankic": float(np.nanstd(rankic_values, ddof=0)) if len(rankic_values) else np.nan,
        "ic_by_date": ic_by_date,
        "rankic_by_date": rankic_by_date,
    }


def _prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["stock_code"] = frame["stock_code"].astype(str).str.zfill(6)
    frame = frame[frame["date"].notna()].copy()
    if frame.duplicated(["date", "stock_code"]).any():
        raise ValueError("Training panel contains duplicate (date, stock_code) keys.")
    return frame


def _prepare_fold_assignments(fold_assignments: pd.DataFrame) -> pd.DataFrame:
    frame = fold_assignments.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["fold_id"] = frame["fold_id"].astype(int)
    frame["split"] = frame["split"].astype(str)
    frame = frame[frame["date"].notna()].copy()
    if frame.duplicated(["fold_id", "date"]).any():
        raise ValueError("Fold assignments contain duplicate (fold_id, date) keys.")
    split_counts = frame.groupby(["fold_id", "date"])["split"].nunique()
    if not split_counts.empty and int(split_counts.max()) > 1:
        raise ValueError("A fold assigns the same date to multiple splits.")
    unknown_splits = sorted(set(frame["split"]).difference({"train", "valid", "test"}))
    if unknown_splits:
        raise ValueError(f"Unknown split labels: {unknown_splits}")
    frame = frame.sort_values(["fold_id", "date"]).reset_index(drop=True)
    _validate_fold_time_order(frame)
    return frame


def _validate_fold_time_order(frame: pd.DataFrame) -> None:
    required_splits = ("train", "valid", "test")
    for fold_id, group in frame.groupby("fold_id", sort=True):
        split_dates = {
            split: group.loc[group["split"].eq(split), "date"].sort_values()
            for split in required_splits
        }
        missing_splits = [split for split, dates in split_dates.items() if dates.empty]
        if missing_splits:
            raise ValueError(f"Fold {int(fold_id)} is missing split assignment(s): {missing_splits}")
        if split_dates["train"].max() >= split_dates["valid"].min():
            raise ValueError(
                f"Fold {int(fold_id)} has train dates that are not strictly before validation dates."
            )
        if split_dates["valid"].max() >= split_dates["test"].min():
            raise ValueError(
                f"Fold {int(fold_id)} has validation dates that are not strictly before test dates."
            )


def _feature_columns(feature_roles: dict[str, list[str]]) -> list[str]:
    return _feature_contract(feature_roles)["feature_columns"]


def _neutralized_alpha_monotone_constraints(
    features: list[str],
    *,
    signal_features: list[str],
) -> list[int]:
    signal_feature_set = set(signal_features)
    return [
        1 if feature in signal_feature_set and _is_neutralized_alpha_rank_z_feature(feature) else 0
        for feature in features
    ]


def _is_neutralized_alpha_rank_z_feature(feature: str) -> bool:
    return feature.endswith("_neutralized_rank") or feature.endswith("_neutralized_z")


def _signal_monotone_constraints(
    features: list[str],
    *,
    signal_features: list[str],
    categorical_features: list[str],
    monotone_signal_features: str,
    monotone_positive_features: list[str],
    monotone_negative_features: list[str],
) -> tuple[list[int], str]:
    if monotone_signal_features not in {"none", "positive", "negative"}:
        raise ValueError(
            "monotone_signal_features must be one of 'none', 'positive', or 'negative'."
        )
    feature_set = set(features)
    categorical_set = set(categorical_features)
    constraints_by_feature = {feature: 0 for feature in features}

    if monotone_signal_features in {"positive", "negative"}:
        signal_constraint = 1 if monotone_signal_features == "positive" else -1
        for feature in signal_features:
            _assign_monotone_constraint(
                constraints_by_feature,
                feature,
                signal_constraint,
                feature_set=feature_set,
                categorical_set=categorical_set,
            )

    for feature in monotone_positive_features:
        _assign_monotone_constraint(
            constraints_by_feature,
            str(feature),
            1,
            feature_set=feature_set,
            categorical_set=categorical_set,
        )
    for feature in monotone_negative_features:
        _assign_monotone_constraint(
            constraints_by_feature,
            str(feature),
            -1,
            feature_set=feature_set,
            categorical_set=categorical_set,
        )

    if monotone_signal_features == "positive" and not monotone_positive_features and not monotone_negative_features:
        policy = "positive_on_signal_features"
    elif monotone_signal_features == "negative" and not monotone_positive_features and not monotone_negative_features:
        policy = "negative_on_signal_features"
    elif monotone_signal_features == "none":
        policy = "explicit_feature_monotone_constraints"
    else:
        policy = "signal_features_plus_explicit_feature_constraints"

    return [constraints_by_feature[feature] for feature in features], policy


def _assign_monotone_constraint(
    constraints_by_feature: dict[str, int],
    feature: str,
    constraint: int,
    *,
    feature_set: set[str],
    categorical_set: set[str],
) -> None:
    if feature not in feature_set:
        raise ValueError(f"Monotone constraint feature {feature!r} is not in feature_columns.")
    if feature in categorical_set:
        raise ValueError(f"Categorical features cannot have monotone constraints: {feature!r}.")
    if feature.startswith("y_") or feature in {"date", "stock_code", "sample_weight", "y_true"}:
        raise ValueError(f"Leakage-prone columns cannot have monotone constraints: {feature!r}.")
    existing = constraints_by_feature[feature]
    if existing not in {0, constraint}:
        raise ValueError(
            f"Feature {feature!r} has conflicting monotone constraints: {existing} and {constraint}."
        )
    constraints_by_feature[feature] = int(constraint)


def _monotone_constraint_summary(
    features: list[str],
    constraints: Any,
    *,
    enabled: bool,
    policy: str,
) -> dict[str, Any]:
    if constraints is None:
        constraint_values = [0 for _ in features]
    else:
        constraint_values = [int(value) for value in constraints]
    if len(constraint_values) != len(features):
        raise ValueError(
            "monotone_constraints length must match feature_columns length: "
            f"{len(constraint_values)} != {len(features)}."
        )
    constrained_features = [
        feature
        for feature, constraint in zip(features, constraint_values, strict=True)
        if constraint != 0
    ]
    return {
        "enabled": bool(enabled),
        "policy": policy,
        "positive_constraint_value": 1,
        "negative_constraint_value": -1,
        "constrained_features": constrained_features,
        "constraints_by_feature": dict(zip(features, constraint_values, strict=True)),
        "semantic_note": (
            "LightGBM monotone constraints enforce partial monotonicity while all other "
            "features are fixed; they do not preserve global cross-context ranking."
        ),
    }


def _feature_roles_after_exclusion(
    feature_roles: dict[str, list[str]],
    *,
    exclude_features: list[str] | None,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    requested = _dedupe_feature_list(exclude_features or [])
    roles = {str(role): _feature_role_list(feature_roles, str(role)) for role in feature_roles}
    if not requested:
        return roles, {
            "requested_exclude_features": [],
            "excluded_model_features": [],
        }

    model_role_names = (
        "signal_features",
        "alpha_placeholder",
        "condition_categorical",
        "condition_continuous",
    )
    model_features = {
        feature
        for role_name in model_role_names
        for feature in _feature_role_list(feature_roles, role_name)
    }
    missing = [feature for feature in requested if feature not in model_features]
    if missing:
        raise ValueError(f"exclude_features are not declared model features: {missing}")

    excluded_set = set(requested)
    for role_name in model_role_names:
        if role_name in roles:
            roles[role_name] = [
                feature for feature in roles[role_name] if feature not in excluded_set
            ]

    excluded_from_model = roles.get("excluded_from_model", [])
    for feature in requested:
        if feature not in excluded_from_model:
            excluded_from_model.append(feature)
    roles["excluded_from_model"] = excluded_from_model

    return roles, {
        "requested_exclude_features": requested,
        "excluded_model_features": requested,
    }


def _dedupe_feature_list(features: list[str]) -> list[str]:
    result: list[str] = []
    for feature in features:
        text = str(feature).strip()
        if text and text not in result:
            result.append(text)
    return result


def _feature_contract(feature_roles: dict[str, list[str]]) -> dict[str, Any]:
    legacy_signal_features = _feature_role_list(feature_roles, "alpha_placeholder")
    signal_alias_features = _feature_role_list(feature_roles, "signal_features")
    if legacy_signal_features and signal_alias_features and legacy_signal_features != signal_alias_features:
        raise ValueError(
            "feature_roles['signal_features'] and feature_roles['alpha_placeholder'] "
            "must match when both are provided."
        )

    signal_features = signal_alias_features or legacy_signal_features
    signal_feature_role = "signal_features" if signal_alias_features else "alpha_placeholder"
    if not signal_features:
        raise ValueError("feature_roles must declare signal columns in 'signal_features' or 'alpha_placeholder'.")

    categorical_features = _feature_role_list(feature_roles, "condition_categorical")
    continuous_features = _feature_role_list(feature_roles, "condition_continuous")
    context_features = categorical_features + continuous_features
    targets = _feature_role_list(feature_roles, "targets")

    _validate_role_uniqueness(
        {
            signal_feature_role: signal_features,
            "condition_categorical": categorical_features,
            "condition_continuous": continuous_features,
        }
    )

    columns: list[str] = []
    for role_columns in (signal_features, categorical_features, continuous_features):
        for column in role_columns:
            if column not in columns:
                columns.append(column)
    return {
        "feature_roles": _model_feature_roles(feature_roles, signal_feature_role),
        "feature_columns": columns,
        "signal_features": signal_features,
        "signal_feature_role": signal_feature_role,
        "condition_categorical": categorical_features,
        "condition_continuous": continuous_features,
        "context_features": context_features,
        "targets": targets,
    }


def _feature_role_list(feature_roles: dict[str, list[str]], role: str) -> list[str]:
    return [str(column) for column in feature_roles.get(role, [])]


def _model_feature_roles(feature_roles: dict[str, list[str]], signal_feature_role: str) -> dict[str, list[str]]:
    roles = {str(role): _feature_role_list(feature_roles, str(role)) for role in feature_roles}
    if signal_feature_role == "signal_features":
        roles.pop("alpha_placeholder", None)
    else:
        roles.pop("signal_features", None)
    return roles


def _validate_role_uniqueness(role_columns: dict[str, list[str]]) -> None:
    owner_by_column: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}
    for role, columns in role_columns.items():
        seen_in_role: set[str] = set()
        for column in columns:
            if column in seen_in_role:
                duplicates.setdefault(column, []).append(role)
            seen_in_role.add(column)
            existing_role = owner_by_column.get(column)
            if existing_role is not None and existing_role != role:
                duplicates.setdefault(column, [existing_role]).append(role)
            else:
                owner_by_column[column] = role
    if duplicates:
        details = {column: sorted(set(roles)) for column, roles in duplicates.items()}
        raise ValueError(f"Feature columns must be assigned to exactly one model role: {details}")


def _validate_training_inputs(
    panel: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    features: list[str],
    categorical_features: list[str],
    target_col: str,
    *,
    target_columns: list[str],
) -> None:
    if target_columns and target_col not in target_columns:
        raise ValueError(f"Target column {target_col!r} is not declared in feature_roles['targets'].")
    if target_col not in panel.columns:
        raise ValueError(f"Target column {target_col!r} is missing from panel.")
    missing_features = sorted(set(features).difference(panel.columns))
    if missing_features:
        raise ValueError(f"Feature columns are missing from panel: {missing_features}")
    missing_categorical = sorted(set(categorical_features).difference(features))
    if missing_categorical:
        raise ValueError(f"Categorical features are not in the feature set: {missing_categorical}")
    target_like_columns = {str(column) for column in panel.columns if str(column).startswith("y_")}
    forbidden = {
        "date",
        "stock_code",
        "sample_weight",
        "y_true",
        target_col,
        *target_columns,
        *target_like_columns,
    }
    leakage = sorted(set(features).intersection(forbidden))
    if leakage:
        raise ValueError(f"Leakage-prone columns cannot be model features: {leakage}")
    if "sample_weight" not in panel.columns:
        raise ValueError("Training panel must contain sample_weight.")
    if fold_assignments.empty:
        raise ValueError("Fold assignments are empty.")


def _panel_for_split(panel: pd.DataFrame, fold_dates: pd.DataFrame, split: str) -> pd.DataFrame:
    dates = fold_dates.loc[fold_dates["split"].eq(split), ["date"]].drop_duplicates()
    frame = panel.merge(dates, on="date", how="inner", validate="many_to_one")
    return frame.sort_values(["date", "stock_code"]).reset_index(drop=True)


def _fit_category_maps(train_frame: pd.DataFrame, categorical_features: list[str]) -> dict[str, list[str]]:
    maps: dict[str, list[str]] = {}
    for column in categorical_features:
        values = train_frame[column].astype("string").fillna("UNKNOWN")
        categories = sorted(set(values.astype(str).tolist()) | {"UNKNOWN"})
        maps[column] = categories
    return maps


def _make_feature_matrix(
    frame: pd.DataFrame,
    features: list[str],
    categorical_features: list[str],
    category_maps: dict[str, list[str]],
) -> pd.DataFrame:
    matrix = frame[features].copy()
    for column in categorical_features:
        known = set(category_maps[column])
        values = matrix[column].astype("string").fillna("UNKNOWN").astype(str)
        values = values.where(values.isin(known), "UNKNOWN")
        matrix[column] = pd.Categorical(values, categories=category_maps[column])
    continuous_features = [column for column in features if column not in categorical_features]
    for column in continuous_features:
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce").astype(float)
    if matrix[continuous_features].isna().any().any():
        missing = matrix[continuous_features].isna().sum()
        missing = missing[missing.gt(0)].to_dict()
        raise ValueError(f"Continuous feature matrix contains missing values: {missing}")
    return matrix


def _normalized_model_weights(weights: pd.Series, *, split_name: str) -> np.ndarray:
    normalized = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    if normalized.size == 0:
        raise ValueError(f"{split_name} sample_weight is empty.")
    if not np.isfinite(normalized).all():
        raise ValueError(f"{split_name} sample_weight contains non-finite values.")
    if np.any(normalized < 0.0):
        raise ValueError(f"{split_name} sample_weight contains negative values.")
    mean_weight = float(np.mean(normalized))
    if mean_weight <= 0.0:
        raise ValueError(f"{split_name} sample_weight must have positive mean.")
    return normalized / mean_weight


def _make_context_only_matrix(matrix: pd.DataFrame, signal_features: list[str]) -> pd.DataFrame:
    context = matrix.copy()
    missing = sorted(set(signal_features).difference(context.columns))
    if missing:
        raise ValueError(f"Signal columns are missing from feature matrix: {missing}")
    for column in signal_features:
        context[column] = 0.0
    return context


def _predict_model(model: Any, matrix: pd.DataFrame) -> np.ndarray:
    best_iteration = _best_iteration(model)
    if best_iteration is not None and best_iteration > 0:
        try:
            return np.asarray(model.predict(matrix, num_iteration=best_iteration), dtype=float)
        except TypeError:
            pass
    return np.asarray(model.predict(matrix), dtype=float)


def _lightgbm_model_factory(**params: Any) -> Any:
    try:
        from lightgbm import LGBMRegressor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LightGBM is required for training. Install project dependencies first."
        ) from exc
    return LGBMRegressor(**params)


def _lightgbm_callbacks(early_stopping_rounds: int) -> list[Any]:
    if early_stopping_rounds <= 0:
        return []
    try:
        import lightgbm as lgb
    except ModuleNotFoundError:
        return []
    return [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=0),
    ]


def _save_model(model: Any, base_path: Path) -> Path:
    booster = getattr(model, "booster_", None)
    if booster is not None and hasattr(booster, "save_model"):
        path = base_path.with_suffix(".txt")
        best_iteration = _best_iteration(model)
        try:
            booster.save_model(str(path), num_iteration=best_iteration)
        except TypeError:
            booster.save_model(str(path))
        return path
    path = base_path.with_suffix(".pkl")
    with path.open("wb") as file:
        pickle.dump(model, file)
    return path


def _feature_gain_frame(model: Any, features: list[str], fold_id: int) -> pd.DataFrame:
    booster = getattr(model, "booster_", None)
    if booster is not None and hasattr(booster, "feature_importance"):
        best_iteration = _best_iteration(model)
        try:
            importance_gain = np.asarray(
                _booster_feature_importance(booster, importance_type="gain", iteration=best_iteration),
                dtype=float,
            )
            importance_split = np.asarray(
                _booster_feature_importance(booster, importance_type="split", iteration=best_iteration),
                dtype=float,
            )
        except TypeError:
            importance_gain = np.asarray(booster.feature_importance(), dtype=float)
            importance_split = np.full(len(features), np.nan)
    else:
        raw_importance = getattr(model, "feature_importances_", None)
        if raw_importance is None:
            raw_importance = np.full(len(features), np.nan)
        importance_gain = np.asarray(raw_importance, dtype=float)
        importance_split = np.full(len(features), np.nan)
    return pd.DataFrame(
        {
            "fold_id": int(fold_id),
            "feature": features,
            "importance_gain": importance_gain,
            "importance_split": importance_split,
            "importance": importance_gain,
        }
    )


def _booster_feature_importance(booster: Any, *, importance_type: str, iteration: int | None) -> Any:
    if iteration is not None and iteration > 0:
        try:
            return booster.feature_importance(importance_type=importance_type, iteration=iteration)
        except TypeError:
            pass
    return booster.feature_importance(importance_type=importance_type)


def _model_tree_diagnostics(model: Any) -> dict[str, Any]:
    booster = getattr(model, "booster_", None)
    if booster is None or not hasattr(booster, "dump_model"):
        return {"available": False}
    try:
        model_dump = booster.dump_model()
    except Exception as exc:  # pragma: no cover - depends on LightGBM internals.
        return {"available": False, "error": str(exc)}

    tree_info = model_dump.get("tree_info", []) if isinstance(model_dump, dict) else []
    leaf_counts: list[int] = []
    split_counts: list[int] = []
    tree_depths: list[int] = []
    for tree in tree_info:
        if not isinstance(tree, dict):
            continue
        leaf_count = _to_int_or_none(tree.get("num_leaves"))
        if leaf_count is not None:
            leaf_counts.append(leaf_count)
        structure = tree.get("tree_structure")
        depth, split_count = _tree_structure_stats(structure)
        tree_depths.append(depth)
        split_counts.append(split_count)

    leaf_values = np.asarray(leaf_counts, dtype=float)
    split_values = np.asarray(split_counts, dtype=float)
    depth_values = np.asarray(tree_depths, dtype=float)
    tree_count = int(len(tree_info))
    one_leaf_tree_count = int(np.sum(leaf_values <= 1)) if len(leaf_values) else 0
    return {
        "available": True,
        "tree_count": tree_count,
        "leaf_count_observations": int(len(leaf_counts)),
        "min_leaf_count": int(np.nanmin(leaf_values)) if len(leaf_values) else None,
        "max_leaf_count": int(np.nanmax(leaf_values)) if len(leaf_values) else None,
        "mean_leaf_count": float(np.nanmean(leaf_values)) if len(leaf_values) else None,
        "median_leaf_count": float(np.nanmedian(leaf_values)) if len(leaf_values) else None,
        "one_leaf_tree_count": one_leaf_tree_count,
        "one_leaf_tree_share": float(one_leaf_tree_count / len(leaf_values)) if len(leaf_values) else None,
        "min_split_count": int(np.nanmin(split_values)) if len(split_values) else None,
        "max_split_count": int(np.nanmax(split_values)) if len(split_values) else None,
        "mean_split_count": float(np.nanmean(split_values)) if len(split_values) else None,
        "min_tree_depth": int(np.nanmin(depth_values)) if len(depth_values) else None,
        "max_tree_depth": int(np.nanmax(depth_values)) if len(depth_values) else None,
        "mean_tree_depth": float(np.nanmean(depth_values)) if len(depth_values) else None,
    }


def _tree_structure_stats(node: Any, depth: int = 0) -> tuple[int, int]:
    if not isinstance(node, dict):
        return depth, 0
    if "split_index" not in node:
        return depth, 0
    left_depth, left_splits = _tree_structure_stats(node.get("left_child"), depth + 1)
    right_depth, right_splits = _tree_structure_stats(node.get("right_child"), depth + 1)
    return max(left_depth, right_depth), 1 + left_splits + right_splits


def _summarize_tree_diagnostics(fold_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = [
        fold_summary.get("tree_diagnostics", {})
        for fold_summary in fold_summaries
        if fold_summary.get("tree_diagnostics", {}).get("available") is True
    ]
    if not diagnostics:
        return {"available": False}
    tree_count = int(sum(int(item.get("tree_count") or 0) for item in diagnostics))
    one_leaf_tree_count = int(sum(int(item.get("one_leaf_tree_count") or 0) for item in diagnostics))
    mean_leaf_values = _finite_values(
        item.get("mean_leaf_count") for item in diagnostics if item.get("mean_leaf_count") is not None
    )
    mean_depth_values = _finite_values(
        item.get("mean_tree_depth") for item in diagnostics if item.get("mean_tree_depth") is not None
    )
    max_leaf_values = _finite_values(
        item.get("max_leaf_count") for item in diagnostics if item.get("max_leaf_count") is not None
    )
    max_depth_values = _finite_values(
        item.get("max_tree_depth") for item in diagnostics if item.get("max_tree_depth") is not None
    )
    return {
        "available": True,
        "fold_count": int(len(diagnostics)),
        "tree_count": tree_count,
        "one_leaf_tree_count": one_leaf_tree_count,
        "one_leaf_tree_share": float(one_leaf_tree_count / tree_count) if tree_count else None,
        "mean_leaf_count_across_folds": float(np.nanmean(mean_leaf_values)) if len(mean_leaf_values) else None,
        "max_leaf_count_across_folds": int(np.nanmax(max_leaf_values)) if len(max_leaf_values) else None,
        "mean_tree_depth_across_folds": float(np.nanmean(mean_depth_values)) if len(mean_depth_values) else None,
        "max_tree_depth_across_folds": int(np.nanmax(max_depth_values)) if len(max_depth_values) else None,
    }


def _to_int_or_none(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _best_iteration(model: Any) -> int | None:
    value = getattr(model, "best_iteration_", None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    left_values = left.to_numpy(dtype=float)
    right_values = right.to_numpy(dtype=float)
    if np.nanstd(left_values) <= 1e-12 or np.nanstd(right_values) <= 1e-12:
        return np.nan
    return float(np.corrcoef(left_values, right_values)[0, 1])


def _finite_values(values: Any) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def _robust_zscore_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    finite_mask = np.isfinite(numeric.to_numpy(dtype=float))
    valid = numeric.loc[finite_mask]
    if valid.empty:
        return pd.Series(np.zeros(len(numeric), dtype=float), index=values.index)
    valid_values = valid.to_numpy(dtype=float)
    median = float(np.nanmedian(valid_values))
    mad = float(np.nanmedian(np.abs(valid_values - median)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        return pd.Series(np.zeros(len(numeric), dtype=float), index=values.index)
    z = ((numeric - median) / scale).clip(-5.0, 5.0)
    return z.fillna(0.0)


def _metrics_for_csv(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "date_count": metrics["date_count"],
        "mean_ic": metrics["mean_ic"],
        "std_ic": metrics["std_ic"],
        "mean_rankic": metrics["mean_rankic"],
        "std_rankic": metrics["std_rankic"],
    }


def _write_csv_with_iso_dates(frame: pd.DataFrame, path: Path) -> None:
    output = frame.copy()
    for column in output.columns:
        if is_datetime64_any_dtype(output[column]):
            output[column] = pd.to_datetime(output[column]).dt.date.astype(str)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, encoding="utf-8-sig")


def _date_to_string(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def _path_for_summary(path: Path) -> str:
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.name


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return _date_to_string(value)
    return value


if __name__ == "__main__":
    main()
