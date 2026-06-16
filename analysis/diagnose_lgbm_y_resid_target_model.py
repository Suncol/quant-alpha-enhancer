from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252.0
RISK_CATEGORICAL = ["industry", "board", "index_bucket", "size_decile"]
RISK_CONTINUOUS = ["is_csi300", "is_csi500", "is_csi1000", "is_csi2000", "log_mcap_z", "mcap_rank"]
RAW_ALPHA_COL = "factor_sss_dx_10_raw"
NEUTRAL_ALPHA_COL = "factor_sss_dx_10_value_neutralized_raw"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose the y_resid_fwd-target LGBM model against old rank-label outputs."
    )
    parser.add_argument(
        "--training-dir",
        default=Path("analysis/outputs/training_data/alpha_value_rank_neutralized_strict_no_liquidity_v1"),
        type=Path,
    )
    parser.add_argument(
        "--new-model-dir",
        default=Path("analysis/outputs/lgbm_alpha_value_rank_neutralized_strict_no_liquidity_y_resid_depth7_leaves127_min250_reg_v1"),
        type=Path,
    )
    parser.add_argument(
        "--old-model-dir",
        default=Path("analysis/outputs/lgbm_alpha_value_rank_neutralized_strict_no_liquidity_depth7_leaves127_min250_reg_v1"),
        type=Path,
    )
    parser.add_argument(
        "--return-y",
        default=Path("analysis/outputs/return_y_hfq_adjusted/return_y_hfq_adj.pkl"),
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default=Path("analysis/outputs/lgbm_y_resid_target_diagnostics_v1"),
        type=Path,
    )
    args = parser.parse_args()

    summary = run_diagnostics(
        training_dir=args.training_dir,
        new_model_dir=args.new_model_dir,
        old_model_dir=args.old_model_dir,
        return_y_path=args.return_y,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary["run_status"], ensure_ascii=False, indent=2))


def run_diagnostics(
    *,
    training_dir: Path,
    new_model_dir: Path,
    old_model_dir: Path,
    return_y_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    training_summary = _read_json(training_dir / "training_summary.json")
    new_summary = _read_json(new_model_dir / "training_summary.json")
    old_summary = _read_json(old_model_dir / "training_summary.json")

    contract = {
        "branch_goal_target_col": "y_resid_fwd",
        "new_model_target_col": new_summary["target_col"],
        "new_model_fit_target_col": new_summary["fit_target_col"],
        "new_model_evaluation_target_col": new_summary["evaluation_target_col"],
        "old_model_target_col": old_summary["target_col"],
        "feature_columns": new_summary["feature_columns"],
        "liquidity_like_features_in_model": [
            column
            for column in new_summary["feature_columns"]
            if any(token in column.lower() for token in ["turnover", "adv", "amount", "amo", "vol"])
        ],
        "alpha_signal_neutralization": training_summary["metadata"].get("alpha_signal_neutralization"),
        "test_split_rows": sum(
            int(item["row_count"]) for item in training_summary["split_summary"] if item["split"] == "test"
        ),
        "test_split_dates": sum(
            int(item["date_count"]) for item in training_summary["split_summary"] if item["split"] == "test"
        ),
    }

    frame = _load_test_prediction_frame(
        training_dir=training_dir,
        new_model_dir=new_model_dir,
        return_y_path=return_y_path,
    )
    frame["raw_alpha_value"] = pd.to_numeric(frame[RAW_ALPHA_COL], errors="coerce")
    frame["neutralized_alpha_value"] = pd.to_numeric(frame[NEUTRAL_ALPHA_COL], errors="coerce")

    exposure_daily, residual_scores = _daily_r2_and_residual_scores(
        frame,
        ["pred_direct", "pred_context_only", "score_marginal_z", "raw_alpha_value", "neutralized_alpha_value"],
    )
    frame = frame.merge(residual_scores, on=["date", "stock_code"], how="left", validate="one_to_one")
    exposure_summary = _summarize_r2(exposure_daily)

    score_cols = [
        "pred_direct",
        "pred_context_only",
        "score_marginal",
        "score_marginal_z",
        "raw_alpha_value",
        "neutralized_alpha_value",
        "pred_direct__neutralized_score",
        "score_marginal_z__neutralized_score",
    ]
    performance_daily = _daily_performance(frame, score_cols, ["y_resid_fwd", "return_y_hfq_adj"])
    performance_summary = _summarize_daily_performance(performance_daily)

    model_metric_compare = pd.concat(
        [
            _weighted_metric_from_training_summary(old_model_dir, "old_y_rank_label_model"),
            _weighted_metric_from_training_summary(new_model_dir, "new_y_resid_fwd_model"),
        ],
        ignore_index=True,
    )

    bucket_direct = _stitched_bucket_summary(
        new_model_dir / "rank_bucket_nav_pred_direct",
        "pred_direct",
        output_dir,
    )
    bucket_marginal = _stitched_bucket_summary(
        new_model_dir / "rank_bucket_nav_score_marginal_z",
        "score_marginal_z",
        output_dir,
    )

    style_rows: list[dict[str, Any]] = []
    industry_extremes: list[pd.DataFrame] = []
    for score_label, nav_dir in [
        ("pred_direct", new_model_dir / "rank_bucket_nav_pred_direct"),
        ("score_marginal_z", new_model_dir / "rank_bucket_nav_score_marginal_z"),
    ]:
        style, extremes = _top_bucket_style(nav_dir, score_label)
        style_rows.append(style)
        industry_extremes.append(extremes)
    style_summary = pd.DataFrame(style_rows)
    industry_extreme_summary = pd.concat(industry_extremes, ignore_index=True)

    feature_gain = pd.read_csv(new_model_dir / "detailed_metrics" / "feature_gain_summary.csv")
    role_gain = pd.read_csv(new_model_dir / "detailed_metrics" / "feature_gain_role_summary.csv")

    _write_csv(performance_daily, output_dir / "score_performance_daily.csv")
    _write_csv(performance_summary, output_dir / "score_performance_summary.csv")
    _write_csv(exposure_daily, output_dir / "score_exposure_daily.csv")
    _write_csv(exposure_summary, output_dir / "score_exposure_summary.csv")
    _write_csv(model_metric_compare, output_dir / "model_metric_comparison_from_training_summaries.csv")
    _write_csv(style_summary, output_dir / "top_bucket_style_summary.csv")
    _write_csv(industry_extreme_summary, output_dir / "top_bucket_industry_active_extremes.csv")
    _write_csv(feature_gain, output_dir / "feature_gain_summary.csv")
    _write_csv(role_gain, output_dir / "feature_gain_role_summary.csv")

    summary = {
        "run_status": {
            "output_dir": output_dir.as_posix(),
            "test_rows": int(len(frame)),
            "test_dates": int(frame["date"].nunique()),
            "new_target_col": contract["new_model_target_col"],
            "liquidity_like_features_in_model": contract["liquidity_like_features_in_model"],
        },
        "contract": contract,
        "sample": {
            "test_rows": int(len(frame)),
            "test_dates": int(frame["date"].nunique()),
            "test_start": str(frame["date"].min().date()),
            "test_end": str(frame["date"].max().date()),
            "return_nonmissing_count": int(frame["return_y_hfq_adj"].notna().sum()),
        },
        "model_metric_comparison_selected": {
            "old_pred_direct": _row_for(
                model_metric_compare,
                model_label="old_y_rank_label_model",
                prediction_col="pred_direct",
            ),
            "new_pred_direct": _row_for(
                model_metric_compare,
                model_label="new_y_resid_fwd_model",
                prediction_col="pred_direct",
            ),
            "old_score_marginal_z": _row_for(
                model_metric_compare,
                model_label="old_y_rank_label_model",
                prediction_col="score_marginal_z",
            ),
            "new_score_marginal_z": _row_for(
                model_metric_compare,
                model_label="new_y_resid_fwd_model",
                prediction_col="score_marginal_z",
            ),
        },
        "performance_selected": {
            key: _row_for(performance_summary, target_col=target_col, score_col=score_col)
            for key, target_col, score_col in [
                ("new_pred_direct_vs_y_resid", "y_resid_fwd", "pred_direct"),
                ("new_score_marginal_z_vs_y_resid", "y_resid_fwd", "score_marginal_z"),
                ("new_pred_context_only_vs_y_resid", "y_resid_fwd", "pred_context_only"),
                ("raw_alpha_vs_y_resid", "y_resid_fwd", "raw_alpha_value"),
                ("neutralized_alpha_vs_y_resid", "y_resid_fwd", "neutralized_alpha_value"),
                ("new_pred_direct_neutralized_score_vs_y_resid", "y_resid_fwd", "pred_direct__neutralized_score"),
                (
                    "new_score_marginal_z_neutralized_score_vs_y_resid",
                    "y_resid_fwd",
                    "score_marginal_z__neutralized_score",
                ),
                ("new_pred_direct_vs_raw_return", "return_y_hfq_adj", "pred_direct"),
                ("new_score_marginal_z_vs_raw_return", "return_y_hfq_adj", "score_marginal_z"),
            ]
        },
        "score_exposure_selected": {
            key: _row_for(exposure_summary, score_col=score_col)
            for key, score_col in [
                ("new_pred_direct", "pred_direct"),
                ("new_pred_context_only", "pred_context_only"),
                ("new_score_marginal_z", "score_marginal_z"),
                ("raw_alpha_value", "raw_alpha_value"),
                ("neutralized_alpha_value", "neutralized_alpha_value"),
            ]
        },
        "bucket_selected": {
            "pred_direct_bucket_01": _row_for(bucket_direct, bucket=1),
            "pred_direct_bucket_31": _row_for(bucket_direct, bucket=31),
            "score_marginal_z_bucket_01": _row_for(bucket_marginal, bucket=1),
            "score_marginal_z_bucket_31": _row_for(bucket_marginal, bucket=31),
        },
        "top_bucket_style_selected": style_summary.to_dict("records"),
        "feature_gain_top10": _top_feature_gain(feature_gain),
        "feature_gain_by_role": role_gain.to_dict("records"),
        "outputs": {
            "score_performance_summary": (output_dir / "score_performance_summary.csv").as_posix(),
            "score_exposure_summary": (output_dir / "score_exposure_summary.csv").as_posix(),
            "model_metric_comparison": (output_dir / "model_metric_comparison_from_training_summaries.csv").as_posix(),
            "stitched_bucket_summary_pred_direct": (output_dir / "stitched_bucket_summary_pred_direct.csv").as_posix(),
            "stitched_bucket_summary_score_marginal_z": (
                output_dir / "stitched_bucket_summary_score_marginal_z.csv"
            ).as_posix(),
            "top_bucket_style_summary": (output_dir / "top_bucket_style_summary.csv").as_posix(),
            "top_bucket_industry_active_extremes": (output_dir / "top_bucket_industry_active_extremes.csv").as_posix(),
            "feature_gain_summary": (output_dir / "feature_gain_summary.csv").as_posix(),
        },
    }
    (output_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _load_test_prediction_frame(
    *,
    training_dir: Path,
    new_model_dir: Path,
    return_y_path: Path,
) -> pd.DataFrame:
    usecols = [
        "fold_id",
        "split",
        "date",
        "stock_code",
        "pred_direct",
        "pred_context_only",
        "score_marginal",
        "score_marginal_z",
        "y_resid_fwd",
        "y_true",
    ]
    pred = pd.read_csv(
        new_model_dir / "predictions.csv",
        usecols=usecols,
        dtype={"stock_code": "string", "split": "string"},
        parse_dates=["date"],
    )
    pred = pred[pred["split"].astype(str).eq("test")].copy()
    pred["stock_code"] = pred["stock_code"].map(_normalize_stock_code)
    pred["date"] = pd.to_datetime(pred["date"]).dt.normalize()

    panel_cols = [
        "date",
        "stock_code",
        *RISK_CATEGORICAL,
        *RISK_CONTINUOUS,
        RAW_ALPHA_COL,
        NEUTRAL_ALPHA_COL,
    ]
    panel = pd.read_pickle(training_dir / "training_panel.pkl")[panel_cols].copy()
    panel["stock_code"] = panel["stock_code"].map(_normalize_stock_code)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = panel[panel["date"].isin(pred["date"].unique())].copy()

    frame = pred.merge(panel, on=["date", "stock_code"], how="inner", validate="one_to_one")
    if len(frame) != len(pred):
        raise RuntimeError(f"Prediction/panel merge lost rows: pred={len(pred)} merged={len(frame)}")

    return_y = pd.read_pickle(return_y_path)
    return_y.index = pd.to_datetime(return_y.index).normalize()
    return_y = return_y.loc[sorted(frame["date"].unique())]
    returns = return_y.stack(future_stack=True).rename("return_y_hfq_adj").reset_index()
    returns.columns = ["date", "stock_code", "return_y_hfq_adj"]
    returns["stock_code"] = returns["stock_code"].map(_normalize_stock_code)
    frame = frame.merge(returns, on=["date", "stock_code"], how="left", validate="one_to_one")
    missing_returns = int(frame["return_y_hfq_adj"].isna().sum())
    if missing_returns:
        raise RuntimeError(f"Missing return_y_hfq_adj after merge: {missing_returns}")
    return frame


def _daily_performance(
    frame: pd.DataFrame,
    score_cols: list[str],
    target_cols: list[str],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for target_col in target_cols:
        for date, group in frame.groupby("date", sort=True):
            target = pd.to_numeric(group[target_col], errors="coerce")
            for score_col in score_cols:
                score = pd.to_numeric(group[score_col], errors="coerce")
                mask = _finite_pair(score, target)
                valid = group.loc[mask, [score_col, target_col]].copy()
                if len(valid) < 2:
                    continue
                spread = _top_bottom_decile_spread(valid, score_col=score_col, target_col=target_col)
                records.append(
                    {
                        "target_col": target_col,
                        "score_col": score_col,
                        "date": pd.Timestamp(date),
                        "obs_count": int(len(valid)),
                        "ic": _safe_corr(valid[score_col], valid[target_col]),
                        "rank_ic": _safe_corr(
                            valid[score_col].rank(method="average"),
                            valid[target_col].rank(method="average"),
                        ),
                        "top_bottom_decile_spread": spread,
                    }
                )
    return pd.DataFrame(records)


def _summarize_daily_performance(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (target_col, score_col), group in daily.groupby(["target_col", "score_col"], sort=True):
        rows.append(
            {
                "target_col": target_col,
                "score_col": score_col,
                "date_count": int(group["date"].nunique()),
                "mean_obs_count": float(group["obs_count"].mean()),
                "mean_ic": _nanmean(group["ic"]),
                "ic_std": _nanstd(group["ic"]),
                "ic_positive_rate": _positive_rate(group["ic"]),
                "mean_rankic": _nanmean(group["rank_ic"]),
                "rankic_std": _nanstd(group["rank_ic"]),
                "rankic_positive_rate": _positive_rate(group["rank_ic"]),
                "mean_top_bottom_decile_spread": _nanmean(group["top_bottom_decile_spread"]),
                "spread_positive_rate": _positive_rate(group["top_bottom_decile_spread"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["target_col", "score_col"]).reset_index(drop=True)


def _daily_r2_and_residual_scores(
    frame: pd.DataFrame,
    score_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    residual_frames: list[pd.DataFrame] = []
    for date, group in frame.groupby("date", sort=True):
        group = group.copy()
        x = _build_risk_matrix(group)
        residual_part = group[["date", "stock_code"]].copy()
        for score_col in score_cols:
            y = pd.to_numeric(group[score_col], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(y) & np.isfinite(x).all(axis=1)
            rank = int(np.linalg.matrix_rank(x[finite])) if int(finite.sum()) else 0
            if int(finite.sum()) <= x.shape[1] or np.nanstd(y[finite]) <= 1e-12:
                records.append(
                    {
                        "date": pd.Timestamp(date),
                        "score_col": score_col,
                        "row_count": int(len(group)),
                        "obs_count": int(finite.sum()),
                        "risk_rank": rank,
                        "r2": np.nan,
                        "adj_r2": np.nan,
                    }
                )
                residual_part[f"{score_col}__neutralized_score"] = np.nan
                continue

            beta, *_ = np.linalg.lstsq(x[finite], y[finite], rcond=None)
            fitted = x @ beta
            resid = y - fitted
            y_mean = float(np.mean(y[finite]))
            sst = float(np.sum((y[finite] - y_mean) ** 2))
            sse = float(np.sum((y[finite] - fitted[finite]) ** 2))
            r2 = 1.0 - sse / sst if sst > 1e-20 else np.nan
            n = int(finite.sum())
            p = rank - 1
            adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1) if np.isfinite(r2) and n > p + 1 else np.nan
            records.append(
                {
                    "date": pd.Timestamp(date),
                    "score_col": score_col,
                    "row_count": int(len(group)),
                    "obs_count": n,
                    "risk_rank": rank,
                    "r2": r2,
                    "adj_r2": adj_r2,
                }
            )
            residual_part[f"{score_col}__neutralized_score"] = resid
        residual_frames.append(residual_part)
    return pd.DataFrame(records), pd.concat(residual_frames, ignore_index=True)


def _summarize_r2(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for score_col, group in daily.groupby("score_col", sort=True):
        values = pd.to_numeric(group["r2"], errors="coerce").dropna().to_numpy(dtype=float)
        adj_values = pd.to_numeric(group["adj_r2"], errors="coerce").dropna().to_numpy(dtype=float)
        rows.append(
            {
                "score_col": score_col,
                "date_count": int(len(values)),
                "mean_r2": float(np.mean(values)) if len(values) else np.nan,
                "median_r2": float(np.median(values)) if len(values) else np.nan,
                "p90_r2": float(np.quantile(values, 0.9)) if len(values) else np.nan,
                "mean_adj_r2": float(np.mean(adj_values)) if len(adj_values) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("score_col").reset_index(drop=True)


def _weighted_metric_from_training_summary(path: Path, label: str) -> pd.DataFrame:
    summary = _read_json(path / "training_summary.json")
    rows = [{"model_label": label, **record} for record in summary["metrics"] if record["split"] == "test"]
    frame = pd.DataFrame(rows)
    output_rows: list[dict[str, Any]] = []
    for prediction_col, group in frame.groupby("prediction_col", sort=True):
        weights = group["date_count"].to_numpy(dtype=float)
        output_rows.append(
            {
                "model_label": label,
                "prediction_col": prediction_col,
                "date_count": int(weights.sum()),
                "weighted_mean_ic": float(np.average(group["mean_ic"], weights=weights)),
                "weighted_mean_rankic": float(np.average(group["mean_rankic"], weights=weights)),
            }
        )
    return pd.DataFrame(output_rows)


def _stitched_bucket_summary(nav_dir: Path, score_label: str, output_dir: Path) -> pd.DataFrame:
    daily = pd.read_csv(nav_dir / "rank_bucket_daily_returns.csv", parse_dates=["signal_date"])
    rows: list[dict[str, Any]] = []
    for (bucket_index, bucket_label), group in daily.groupby(["bucket_index", "bucket_label"], sort=True):
        sorted_group = group.sort_values(["signal_date", "fold_id"]).copy()
        returns = pd.to_numeric(sorted_group["bucket_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        date_count = int(len(returns))
        stitched_nav = np.cumprod(1.0 + returns) if date_count else np.array([], dtype=float)
        nav_end = float(stitched_nav[-1]) if date_count else np.nan
        mean_return = float(np.mean(returns)) if date_count else np.nan
        std_return = float(np.std(returns, ddof=0)) if date_count else np.nan
        annualized_return = (
            float(nav_end ** (TRADING_DAYS_PER_YEAR / date_count) - 1.0)
            if date_count and np.isfinite(nav_end) and nav_end > 0.0
            else np.nan
        )
        sharpe = (
            float(mean_return / std_return * math.sqrt(TRADING_DAYS_PER_YEAR))
            if date_count > 1 and std_return > 1e-12
            else np.nan
        )
        path = np.concatenate([[1.0], stitched_nav]) if date_count else np.array([], dtype=float)
        drawdowns = path / np.maximum.accumulate(path) - 1.0 if date_count else np.array([], dtype=float)
        rows.append(
            {
                "score_label": score_label,
                "bucket": int(bucket_index),
                "bucket_label": str(bucket_label),
                "date_count": date_count,
                "mean_daily_return": mean_return,
                "ann_return": annualized_return,
                "nav_end": nav_end,
                "sharpe": sharpe,
                "pos_rate": float(np.mean(returns > 0.0)) if date_count else np.nan,
                "max_drawdown": float(np.min(drawdowns)) if date_count else np.nan,
                "longest_underwater_days": _longest_underwater_days(pd.Series(stitched_nav)),
                "mean_selected_count": float(sorted_group["selected_count"].mean()),
                "mean_valid_return_count": float(sorted_group["valid_return_count"].mean()),
            }
        )
    output = pd.DataFrame(rows).sort_values("bucket").reset_index(drop=True)
    _write_csv(output, output_dir / f"stitched_bucket_summary_{score_label}.csv")
    return output


def _top_bucket_style(nav_dir: Path, score_label: str) -> tuple[dict[str, Any], pd.DataFrame]:
    industry = pd.read_csv(nav_dir / "rank_bucket_composition_industry_summary.csv")
    size = pd.read_csv(nav_dir / "rank_bucket_composition_size_summary.csv")
    top_industry = industry[industry["bucket_index"].astype(int).eq(1)].copy()
    top_size = size[size["bucket_index"].astype(int).eq(1)].copy()
    top_industry = (
        top_industry.groupby("industry", as_index=False)
        .agg(
            mean_bucket_weight=("mean_bucket_weight", "mean"),
            mean_universe_weight=("mean_universe_weight", "mean"),
            mean_active_weight=("mean_active_weight", "mean"),
            date_count=("date_count", "sum"),
        )
    )
    top_size = (
        top_size.groupby(["metric", "segment"], as_index=False)
        .agg(
            mean_bucket_value=("mean_bucket_value", "mean"),
            mean_universe_value=("mean_universe_value", "mean"),
            mean_active_value=("mean_active_value", "mean"),
            date_count=("date_count", "sum"),
        )
    )
    size_weight = top_size[top_size["metric"].eq("size_decile_weight")]

    def metric_value(metric: str) -> float:
        row = top_size[top_size["metric"].eq(metric) & top_size["segment"].astype(str).eq("all")]
        return float(row["mean_active_value"].iloc[0]) if not row.empty else np.nan

    sorted_industry = top_industry.sort_values("mean_active_weight", ascending=False)
    extremes = pd.concat([sorted_industry.head(10), sorted_industry.tail(10)], ignore_index=True)
    extremes.insert(0, "score_label", score_label)
    return (
        {
            "score_label": score_label,
            "top_bucket_industry_active_l1_half": float(top_industry["mean_active_weight"].abs().sum() / 2.0),
            "top_bucket_max_abs_industry_active": float(top_industry["mean_active_weight"].abs().max()),
            "top_bucket_size_decile_active_l1_half": (
                float(size_weight["mean_active_value"].abs().sum() / 2.0) if not size_weight.empty else np.nan
            ),
            "top_bucket_active_log_mcap_z": metric_value("log_mcap_z"),
            "top_bucket_active_mcap_rank": metric_value("mcap_rank"),
        },
        extremes,
    )


def _build_risk_matrix(group: pd.DataFrame) -> np.ndarray:
    parts = [np.ones((len(group), 1), dtype=float)]
    for column in RISK_CATEGORICAL:
        dummies = pd.get_dummies(group[column].astype("string").fillna("UNKNOWN"), prefix=column, dtype=float)
        if not dummies.empty:
            parts.append(dummies.to_numpy(dtype=float))
    continuous = group[RISK_CONTINUOUS].apply(pd.to_numeric, errors="coerce").astype(float).fillna(0.0)
    parts.append(continuous.to_numpy(dtype=float))
    return np.column_stack(parts)


def _top_bottom_decile_spread(frame: pd.DataFrame, *, score_col: str, target_col: str) -> float:
    if frame[score_col].nunique(dropna=True) < 2 or len(frame) < 10:
        return np.nan
    valid = frame[[score_col, target_col]].copy()
    valid["_bucket"] = pd.qcut(valid[score_col].rank(method="first"), q=10, labels=False, duplicates="drop")
    bottom = valid[valid["_bucket"].eq(valid["_bucket"].min())]
    top = valid[valid["_bucket"].eq(valid["_bucket"].max())]
    if top.empty or bottom.empty or top["_bucket"].iloc[0] == bottom["_bucket"].iloc[0]:
        return np.nan
    return float(top[target_col].mean() - bottom[target_col].mean())


def _longest_underwater_days(nav_values: pd.Series) -> int:
    values = pd.to_numeric(nav_values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return 0
    high = 1.0
    current = 0
    longest = 0
    for value in values:
        if value >= high - 1e-12:
            high = max(high, float(value))
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return int(longest)


def _top_feature_gain(feature_gain: pd.DataFrame) -> list[dict[str, Any]]:
    sort_col = "mean_importance_gain" if "mean_importance_gain" in feature_gain.columns else "importance_gain_mean"
    if sort_col not in feature_gain.columns:
        return feature_gain.head(10).to_dict("records")
    return feature_gain.sort_values(sort_col, ascending=False).head(10).to_dict("records")


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    mask = _finite_pair(left, right)
    if int(mask.sum()) < 2:
        return np.nan
    left_values = pd.to_numeric(left.loc[mask], errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(right.loc[mask], errors="coerce").to_numpy(dtype=float)
    if np.nanstd(left_values) <= 1e-12 or np.nanstd(right_values) <= 1e-12:
        return np.nan
    return float(np.corrcoef(left_values, right_values)[0, 1])


def _finite_pair(left: pd.Series, right: pd.Series) -> pd.Series:
    left_values = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    return pd.Series(np.isfinite(left_values) & np.isfinite(right_values), index=left.index)


def _nanmean(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.mean(finite)) if len(finite) else np.nan


def _nanstd(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.std(finite, ddof=0)) if len(finite) else np.nan


def _positive_rate(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.mean(finite > 0.0)) if len(finite) else np.nan


def _row_for(frame: pd.DataFrame, **conditions: Any) -> dict[str, Any]:
    mask = pd.Series(True, index=frame.index)
    for key, value in conditions.items():
        mask &= frame[key].astype(str).eq(str(value))
    if not mask.any():
        return {}
    return _json_safe(frame.loc[mask].iloc[0].to_dict())


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value).date())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _normalize_stock_code(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if "." in text:
        prefix = text.split(".", 1)[0]
        if prefix.isdigit():
            text = prefix
    if text.isdigit():
        return text.zfill(6)
    return text


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    output = frame.copy()
    for column in output.columns:
        if pd.api.types.is_datetime64_any_dtype(output[column]):
            output[column] = pd.to_datetime(output[column]).dt.date.astype(str)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
