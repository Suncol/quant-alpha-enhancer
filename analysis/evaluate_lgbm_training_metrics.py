from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


DEFAULT_SCORE_COLS = ["pred_direct", "score_marginal", "score_marginal_z"]
SPLIT_ORDER = {"train": 1, "valid": 2, "test": 3}
QUANTILE_BUCKET_LABELS = {
    3: ["low", "mid", "high"],
    5: ["q1", "q2", "q3", "q4", "q5"],
    10: [f"d{i}" for i in range(1, 11)],
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute detailed fold/split/score metrics for LGBM training predictions."
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--training-panel", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--score-cols", nargs="*", default=None)
    parser.add_argument("--target-col", default="y_resid_fwd")
    parser.add_argument("--spread-target-col", default=None)
    parser.add_argument("--top-bottom-quantiles", default=10, type=int)
    parser.add_argument("--condition-bucket-count", default=3, type=int)
    parser.add_argument("--min-group-obs", default=2, type=int)
    args = parser.parse_args()

    predictions = _read_frame(args.predictions)
    training_panel = _read_frame(args.training_panel)
    score_cols = args.score_cols or _default_score_cols(predictions)
    summary = write_model_evaluation_artifacts(
        predictions=predictions,
        training_panel=training_panel,
        output_dir=args.output_dir,
        score_cols=score_cols,
        target_col=args.target_col,
        spread_target_col=args.spread_target_col or args.target_col,
        top_bottom_quantiles=args.top_bottom_quantiles,
        condition_bucket_count=args.condition_bucket_count,
        min_group_obs=args.min_group_obs,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def evaluate_model_predictions(
    *,
    predictions: pd.DataFrame,
    training_panel: pd.DataFrame,
    score_cols: list[str] | None = None,
    target_col: str = "y_resid_fwd",
    spread_target_col: str | None = None,
    top_bottom_quantiles: int = 10,
    condition_bucket_count: int = 3,
    min_group_obs: int = 2,
) -> dict[str, pd.DataFrame]:
    score_cols = score_cols or _default_score_cols(predictions)
    spread_target_col = spread_target_col or target_col
    frame = _prepare_evaluation_frame(
        predictions=predictions,
        training_panel=training_panel,
        score_cols=score_cols,
        target_col=target_col,
        spread_target_col=spread_target_col,
        condition_bucket_count=condition_bucket_count,
    )
    group_dimensions = _available_group_dimensions(frame)

    daily_ic = _compute_daily_ic(frame, score_cols, target_col)
    overall_metrics = _summarize_daily_ic(daily_ic, frame)
    top_bottom_by_date = _compute_top_bottom_spread(
        frame,
        score_cols=score_cols,
        target_col=spread_target_col,
        quantiles=top_bottom_quantiles,
    )
    top_bottom_summary = _summarize_top_bottom_spread(top_bottom_by_date)
    group_ic_by_date = _compute_group_ic_by_date(
        frame,
        score_cols=score_cols,
        target_col=target_col,
        group_dimensions=group_dimensions,
        min_group_obs=min_group_obs,
    )
    group_ic_summary = _summarize_group_ic(group_ic_by_date)

    return {
        "evaluation_frame": frame,
        "daily_ic_by_fold_split_score": daily_ic,
        "overall_metrics": overall_metrics,
        "top_bottom_spread_by_date": top_bottom_by_date,
        "top_bottom_spread_summary": top_bottom_summary,
        "group_ic_by_date": group_ic_by_date,
        "group_ic_summary": group_ic_summary,
    }


def write_model_evaluation_artifacts(
    *,
    predictions: pd.DataFrame,
    training_panel: pd.DataFrame,
    output_dir: Path,
    score_cols: list[str] | None = None,
    target_col: str = "y_resid_fwd",
    spread_target_col: str | None = None,
    top_bottom_quantiles: int = 10,
    condition_bucket_count: int = 3,
    min_group_obs: int = 2,
    charts_dir: Path | None = None,
    report_path: Path | None = None,
    feature_gain: pd.DataFrame | None = None,
    feature_roles: dict[str, list[str]] | None = None,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    score_cols = score_cols or _default_score_cols(predictions)
    spread_target_col = spread_target_col or target_col
    result = evaluate_model_predictions(
        predictions=predictions,
        training_panel=training_panel,
        score_cols=score_cols,
        target_col=target_col,
        spread_target_col=spread_target_col,
        top_bottom_quantiles=top_bottom_quantiles,
        condition_bucket_count=condition_bucket_count,
        min_group_obs=min_group_obs,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "overall_metrics": output_dir / "overall_metrics_by_fold_split_score.csv",
        "daily_ic": output_dir / "daily_ic_by_fold_split_score.csv",
        "top_bottom_by_date": output_dir / "top_bottom_spread_by_date.csv",
        "top_bottom_summary": output_dir / "top_bottom_spread_summary.csv",
        "group_ic_by_date": output_dir / "group_ic_by_date.csv",
        "group_ic_summary": output_dir / "group_ic_summary.csv",
        "summary": output_dir / "evaluation_summary.json",
    }
    _write_csv_with_iso_dates(result["overall_metrics"], output_paths["overall_metrics"])
    _write_csv_with_iso_dates(result["daily_ic_by_fold_split_score"], output_paths["daily_ic"])
    _write_csv_with_iso_dates(result["top_bottom_spread_by_date"], output_paths["top_bottom_by_date"])
    _write_csv_with_iso_dates(result["top_bottom_spread_summary"], output_paths["top_bottom_summary"])
    _write_csv_with_iso_dates(result["group_ic_by_date"], output_paths["group_ic_by_date"])
    _write_csv_with_iso_dates(result["group_ic_summary"], output_paths["group_ic_summary"])

    feature_gain_result: dict[str, Any] | None = None
    if feature_gain is not None:
        feature_gain_result = build_feature_gain_artifacts(
            feature_gain=feature_gain,
            feature_roles=feature_roles or {},
            feature_columns=feature_columns,
        )
        feature_gain_paths = {
            "feature_gain_by_fold": output_dir / "feature_gain_by_fold.csv",
            "feature_gain_summary": output_dir / "feature_gain_summary.csv",
            "feature_gain_role_summary": output_dir / "feature_gain_role_summary.csv",
            "feature_gain_diagnostics": output_dir / "feature_gain_diagnostics.json",
        }
        _write_csv_with_iso_dates(
            feature_gain_result["feature_gain_by_fold"],
            feature_gain_paths["feature_gain_by_fold"],
        )
        _write_csv_with_iso_dates(
            feature_gain_result["feature_gain_summary"],
            feature_gain_paths["feature_gain_summary"],
        )
        _write_csv_with_iso_dates(
            feature_gain_result["feature_gain_role_summary"],
            feature_gain_paths["feature_gain_role_summary"],
        )
        feature_gain_paths["feature_gain_diagnostics"].write_text(
            json.dumps(feature_gain_result["feature_gain_diagnostics"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output_paths.update(feature_gain_paths)

    chart_outputs: dict[str, str] = {}
    if charts_dir is not None:
        chart_paths = write_evaluation_charts(
            result,
            charts_dir,
            primary_score_col="score_marginal_z" if "score_marginal_z" in score_cols else score_cols[-1],
            target_col=target_col,
            feature_gain_result=feature_gain_result,
        )
        chart_outputs = {key: _path_for_summary(path) for key, path in chart_paths.items()}

    row_counts = {
        "evaluation_frame": int(len(result["evaluation_frame"])),
        "overall_metrics": int(len(result["overall_metrics"])),
        "daily_ic": int(len(result["daily_ic_by_fold_split_score"])),
        "top_bottom_by_date": int(len(result["top_bottom_spread_by_date"])),
        "group_ic_by_date": int(len(result["group_ic_by_date"])),
        "group_ic_summary": int(len(result["group_ic_summary"])),
    }
    if feature_gain_result is not None:
        row_counts.update(
            {
                "feature_gain_by_fold": int(len(feature_gain_result["feature_gain_by_fold"])),
                "feature_gain_summary": int(len(feature_gain_result["feature_gain_summary"])),
                "feature_gain_role_summary": int(len(feature_gain_result["feature_gain_role_summary"])),
            }
        )

    summary = {
        "schema_version": "lgbm_training_metrics_v1",
        "metric_contract": {
            "ic_aggregation": "mean_of_daily_cross_section_ic",
            "rank_ic_aggregation": "mean_of_daily_cross_section_rank_ic",
            "top_bottom_definition": "mean(target in top score quantile) - mean(target in bottom score quantile)",
            "top_bottom_quantiles": int(top_bottom_quantiles),
            "condition_bucket_count": int(condition_bucket_count),
            "condition_buckets_are_daily": True,
            "min_group_obs": int(min_group_obs),
            "target_col": target_col,
            "spread_target_col": spread_target_col,
            "score_cols": score_cols,
            "split_order": SPLIT_ORDER,
        },
        "primary_score_col": "score_marginal_z" if "score_marginal_z" in score_cols else score_cols[-1],
        "group_dimensions": _available_group_dimensions(result["evaluation_frame"]),
        "row_counts": row_counts,
        "outputs": {key: _path_for_summary(path) for key, path in output_paths.items()},
        "charts": chart_outputs,
        "notes": [
            "IC and RankIC are computed within each trading-date cross section before aggregation.",
            "Top-bottom groups are formed from the evaluated score, never from future returns.",
            "Industry, size, ADV, and turnover diagnostics are evaluated as conditional context dimensions when present.",
            "Feature gain diagnostics are derived from trained model split/gain statistics and do not use future returns.",
            "Placeholder-alpha runs remain non-production research pipeline checks unless replaced by a real alpha.",
        ],
    }
    if report_path is not None:
        write_evaluation_html_report(
            result,
            summary,
            report_path,
            chart_outputs=chart_outputs,
        )
        summary["outputs"]["report"] = _path_for_summary(report_path)
    output_paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def write_evaluation_charts(
    result: dict[str, pd.DataFrame],
    output_dir: Path,
    *,
    primary_score_col: str,
    target_col: str,
    max_example_symbols: int = 6,
    feature_gain_result: dict[str, Any] | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _use_static_chart_theme()
    chart_paths = {
        "overall_ic_rankic": output_dir / "overall_ic_rankic.png",
        "daily_ic_rankic": output_dir / "daily_ic_rankic.png",
        "top_bottom_spread": output_dir / "top_bottom_spread.png",
        "conditional_group_ic": output_dir / "conditional_group_ic.png",
        "example_score_return_paths": output_dir / "example_score_return_paths.png",
    }
    _plot_overall_ic_rankic(
        result["overall_metrics"],
        chart_paths["overall_ic_rankic"],
        primary_score_col=primary_score_col,
    )
    _plot_daily_ic_rankic(
        result["daily_ic_by_fold_split_score"],
        chart_paths["daily_ic_rankic"],
        primary_score_col=primary_score_col,
    )
    _plot_top_bottom_spread(
        result["top_bottom_spread_summary"],
        chart_paths["top_bottom_spread"],
        primary_score_col=primary_score_col,
    )
    _plot_conditional_group_ic(
        result["group_ic_summary"],
        chart_paths["conditional_group_ic"],
        primary_score_col=primary_score_col,
    )
    _plot_example_score_return_paths(
        result["evaluation_frame"],
        chart_paths["example_score_return_paths"],
        primary_score_col=primary_score_col,
        target_col=target_col,
        max_symbols=max_example_symbols,
    )
    if feature_gain_result is not None:
        feature_chart_paths = {
            "feature_gain_top": output_dir / "feature_gain_top.png",
            "feature_gain_by_role": output_dir / "feature_gain_by_role.png",
            "feature_gain_fold_heatmap": output_dir / "feature_gain_fold_heatmap.png",
        }
        _plot_feature_gain_top(
            feature_gain_result["feature_gain_summary"],
            feature_chart_paths["feature_gain_top"],
        )
        _plot_feature_gain_by_role(
            feature_gain_result["feature_gain_role_summary"],
            feature_chart_paths["feature_gain_by_role"],
        )
        _plot_feature_gain_fold_heatmap(
            feature_gain_result["feature_gain_by_fold"],
            feature_chart_paths["feature_gain_fold_heatmap"],
        )
        chart_paths.update(feature_chart_paths)
    return chart_paths


def build_feature_gain_artifacts(
    *,
    feature_gain: pd.DataFrame,
    feature_roles: dict[str, list[str]] | None = None,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    frame, diagnostics = _prepare_feature_gain_by_fold(
        feature_gain=feature_gain,
        feature_roles=feature_roles or {},
        feature_columns=feature_columns,
    )
    summary = _summarize_feature_gain(frame)
    role_summary = _summarize_feature_gain_roles(summary)
    diagnostics.update(
        {
            "summary_row_count": int(len(summary)),
            "role_summary_row_count": int(len(role_summary)),
            "role_counts": {
                str(role): int(count)
                for role, count in summary["feature_role"].value_counts(sort=False).items()
            }
            if "feature_role" in summary.columns
            else {},
        }
    )
    return {
        "feature_gain_by_fold": frame,
        "feature_gain_summary": summary,
        "feature_gain_role_summary": role_summary,
        "feature_gain_diagnostics": diagnostics,
    }


def write_evaluation_html_report(
    result: dict[str, pd.DataFrame],
    summary: dict[str, Any],
    report_path: Path,
    *,
    chart_outputs: dict[str, str],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    primary_score_col = str(summary.get("primary_score_col", "score_marginal_z"))
    key_metrics = _report_key_metrics(result["overall_metrics"], primary_score_col)
    metric_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['split'])}</td>"
        f"<td>{html.escape(row['fold_id'])}</td>"
        f"<td>{_format_float(row['mean_ic'])}</td>"
        f"<td>{_format_float(row['mean_rankic'])}</td>"
        f"<td>{_format_float(row['ic_positive_rate'])}</td>"
        f"<td>{_format_int(row['date_count'])}</td>"
        "</tr>"
        for row in key_metrics
    )
    if not metric_rows:
        metric_rows = (
            "<tr><td colspan=\"6\">No finite primary-score metrics were available.</td></tr>"
        )

    chart_sections = "\n".join(
        _report_chart_block(report_path, chart_key, chart_path)
        for chart_key, chart_path in chart_outputs.items()
    )
    group_dimensions = ", ".join(summary.get("group_dimensions", [])) or "none"
    outputs = summary.get("outputs", {})
    core_output_keys = [
        "overall_metrics",
        "top_bottom_summary",
        "group_ic_summary",
        "feature_gain_summary",
        "feature_gain_role_summary",
    ]
    core_outputs = [
        f"<code>{html.escape(str(outputs[key]))}</code>"
        for key in core_output_keys
        if key in outputs
    ]
    core_outputs_text = ", ".join(core_outputs) if core_outputs else "none"
    report_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>LGBM Placeholder Training Evaluation</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1f2937;
      --muted: #667085;
      --line: #d9e2ef;
      --panel: #ffffff;
      --bg: #f7f9fc;
      --accent: #315fbe;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      width: min(1180px, calc(100% - 48px));
      margin: 28px auto 48px;
    }}
    h1 {{
      font-size: 30px;
      margin: 0 0 4px;
      letter-spacing: 0;
    }}
    h2 {{
      font-size: 18px;
      margin: 28px 0 12px;
    }}
    p, li {{
      color: var(--muted);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    img {{
      display: block;
      width: 100%;
      max-width: 1120px;
      margin: 8px 0 24px;
      border: 1px solid var(--line);
      background: var(--panel);
    }}
    code {{
      background: #eef2f7;
      padding: 1px 5px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>LGBM Placeholder Training Evaluation</h1>
    <p>
      Fold-wise evaluation for <code>{html.escape(primary_score_col)}</code>.
      IC and RankIC are daily cross-sectional metrics before cross-day aggregation.
    </p>

    <h2>Key Metrics</h2>
    <table>
      <thead>
        <tr>
          <th>Split</th>
          <th>Fold</th>
          <th>Mean IC</th>
          <th>Mean RankIC</th>
          <th>IC Positive Rate</th>
          <th>Date Count</th>
        </tr>
      </thead>
      <tbody>
        {metric_rows}
      </tbody>
    </table>

    <h2>Charts</h2>
    {chart_sections}

    <h2>Method Notes</h2>
    <ul>
      <li>Evaluation target: <code>{html.escape(str(summary["metric_contract"]["target_col"]))}</code>; top-bottom spread target: <code>{html.escape(str(summary["metric_contract"]["spread_target_col"]))}</code>.</li>
      <li>Conditional diagnostics currently cover: {html.escape(group_dimensions)}.</li>
      <li>Core CSV outputs: {core_outputs_text}.</li>
      <li>Placeholder-alpha runs are research pipeline checks unless the placeholder signal is replaced by a real alpha input.</li>
    </ul>
  </main>
</body>
</html>
"""
    report_path.write_text(report_html, encoding="utf-8")


def _use_static_chart_theme() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "#f7f9fc",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#d7deea",
            "axes.labelcolor": "#283142",
            "axes.titlecolor": "#202938",
            "xtick.color": "#667085",
            "ytick.color": "#667085",
            "grid.color": "#dce5f2",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "#f7f9fc",
            "savefig.dpi": 160,
        }
    )


def _plot_overall_ic_rankic(
    overall_metrics: pd.DataFrame,
    path: Path,
    *,
    primary_score_col: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    _chart_title(
        fig,
        "Overall IC / RankIC by fold and split",
        "Mean values are computed after daily cross-sectional IC aggregation.",
    )
    if overall_metrics.empty:
        _plot_empty(ax, "No overall metrics available")
        _save_chart(fig, path)
        return

    frame = overall_metrics[overall_metrics["score_col"].eq(primary_score_col)].copy()
    if frame.empty:
        frame = overall_metrics.copy()
    frame = frame.sort_values(["fold_id", "split_order", "score_col"])
    frame["label"] = (
        "F"
        + frame["fold_id"].astype(str)
        + " "
        + frame["split"].astype(str)
    )
    x = np.arange(len(frame), dtype=float)
    width = 0.38
    ax.bar(x - width / 2, frame["mean_ic"].astype(float), width, label="IC", color="#315fbe")
    ax.bar(x + width / 2, frame["mean_rankic"].astype(float), width, label="RankIC", color="#d17a34")
    ax.axhline(0.0, color="#27384c", linewidth=0.9)
    ax.set_ylabel("Mean daily correlation")
    ax.set_xticks(x)
    ax.set_xticklabels(frame["label"], rotation=45, ha="right")
    ax.legend(frameon=False, loc="best")
    ax.grid(axis="y")
    _save_chart(fig, path)


def _plot_daily_ic_rankic(
    daily_ic: pd.DataFrame,
    path: Path,
    *,
    primary_score_col: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    _chart_title(
        fig,
        "Daily IC path for primary score",
        "Each point is one fold/split/date cross section.",
    )
    frame = daily_ic[daily_ic["score_col"].eq(primary_score_col)].copy()
    if "test" in set(frame["split"].astype(str)):
        frame = frame[frame["split"].eq("test")].copy()
    if frame.empty:
        _plot_empty(ax, f"No daily IC data for {primary_score_col}")
        _save_chart(fig, path)
        return

    frame = frame.sort_values(["fold_id", "split_order", "date"])
    colors = {"train": "#315fbe", "valid": "#6bb33f", "test": "#d17a34"}
    grouped_paths = list(frame.groupby(["fold_id", "split"], sort=True))
    fold_colors = ["#315fbe", "#d17a34", "#6bb33f", "#8057b5", "#2f8c8c"]
    use_fold_colors = frame["split"].astype(str).nunique() == 1
    for path_index, ((fold_id, split), group) in enumerate(grouped_paths):
        group = group.sort_values("date")
        label_prefix = f"F{fold_id} {split}"
        color = fold_colors[path_index % len(fold_colors)] if use_fold_colors else colors.get(str(split), "#667085")
        ax.plot(
            group["date"],
            group["ic"].astype(float),
            marker="o",
            markersize=3.2,
            linewidth=1.5,
            color=color,
            alpha=0.9,
            label=f"{label_prefix} IC",
        )
        ax.plot(
            group["date"],
            group["rank_ic"].astype(float),
            marker="s",
            markersize=3.0,
            linewidth=1.2,
            linestyle="--",
            color=color,
            alpha=0.65,
            label=f"{label_prefix} RankIC",
        )
    ax.axhline(0.0, color="#27384c", linewidth=0.9)
    ax.set_ylabel("Daily correlation")
    ax.set_xlabel("Date")
    ax.legend(frameon=False, ncols=2, fontsize=8.2)
    ax.grid(axis="y")
    fig.autofmt_xdate()
    _save_chart(fig, path)


def _plot_top_bottom_spread(
    spread_summary: pd.DataFrame,
    path: Path,
    *,
    primary_score_col: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    _chart_title(
        fig,
        "Top-minus-bottom realized return spread",
        "Spread uses the requested evaluation return target within each date.",
    )
    frame = spread_summary[spread_summary["score_col"].eq(primary_score_col)].copy()
    if frame.empty:
        _plot_empty(ax, f"No spread data for {primary_score_col}")
        _save_chart(fig, path)
        return

    frame = frame.sort_values(["fold_id", "split_order"])
    labels = "F" + frame["fold_id"].astype(str) + " " + frame["split"].astype(str)
    values = frame["mean_spread"].astype(float).to_numpy()
    colors = ["#315fbe" if value >= 0 else "#b64d4d" for value in values]
    ax.bar(np.arange(len(frame)), values, color=colors, width=0.66)
    ax.axhline(0.0, color="#27384c", linewidth=0.9)
    ax.set_xticks(np.arange(len(frame)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean top-bottom spread")
    ax.grid(axis="y")
    _save_chart(fig, path)


def _plot_conditional_group_ic(
    group_summary: pd.DataFrame,
    path: Path,
    *,
    primary_score_col: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    _chart_title(
        fig,
        "Conditional IC by diagnostic group",
        "Bars aggregate group-level mean IC within each diagnostic dimension.",
    )
    frame = group_summary[group_summary["score_col"].eq(primary_score_col)].copy()
    if "test" in set(frame["split"].astype(str)):
        frame = frame[frame["split"].eq("test")].copy()
    if frame.empty:
        _plot_empty(ax, f"No conditional group IC data for {primary_score_col}")
        _save_chart(fig, path)
        return

    grouped = (
        frame.groupby("group_dimension", sort=True)
        .agg(
            mean_ic=("mean_ic", "mean"),
            mean_rankic=("mean_rankic", "mean"),
            group_count=("group_value", "nunique"),
        )
        .reset_index()
        .sort_values("mean_ic")
    )
    y = np.arange(len(grouped), dtype=float)
    width = 0.35
    ax.barh(y - width / 2, grouped["mean_ic"].astype(float), height=width, label="IC", color="#315fbe")
    ax.barh(
        y + width / 2,
        grouped["mean_rankic"].astype(float),
        height=width,
        label="RankIC",
        color="#6bb33f",
    )
    labels = [
        f"{dimension} ({int(group_count)})"
        for dimension, group_count in zip(grouped["group_dimension"], grouped["group_count"])
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0.0, color="#27384c", linewidth=0.9)
    ax.set_xlabel("Mean group-level correlation")
    ax.legend(frameon=False, loc="best")
    ax.grid(axis="x")
    _save_chart(fig, path)


def _plot_example_score_return_paths(
    frame: pd.DataFrame,
    path: Path,
    *,
    primary_score_col: str,
    target_col: str,
    max_symbols: int,
) -> None:
    import matplotlib.pyplot as plt

    required = {"date", "stock_code", "split", primary_score_col, target_col}
    if frame.empty or not required.issubset(frame.columns):
        fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
        _chart_title(
            fig,
            "Example symbol score and target paths",
            f"Series are z-scored per symbol; target column: {target_col}.",
        )
        _plot_empty(ax, "No example path data available")
        _save_chart(fig, path)
        return

    sample = frame.copy()
    if "test" in set(sample["split"].astype(str)):
        sample = sample[sample["split"].eq("test")].copy()
    sample = sample[np.isfinite(pd.to_numeric(sample[primary_score_col], errors="coerce"))].copy()
    sample = sample[np.isfinite(pd.to_numeric(sample[target_col], errors="coerce"))].copy()
    if sample.empty:
        fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
        _chart_title(
            fig,
            "Example symbol score and target paths",
            f"Series are z-scored per symbol; target column: {target_col}.",
        )
        _plot_empty(ax, "No finite example path data available")
        _save_chart(fig, path)
        return

    symbol_order = (
        sample.groupby("stock_code")
        .agg(
            row_count=("date", "size"),
            abs_score=(
                primary_score_col,
                lambda value: float(np.nanmean(np.abs(pd.to_numeric(value, errors="coerce")))),
            ),
        )
        .reset_index()
        .sort_values(["row_count", "abs_score", "stock_code"], ascending=[False, False, True])
    )
    symbols = symbol_order["stock_code"].head(max_symbols).astype(str).tolist()
    if not symbols:
        fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
        _chart_title(
            fig,
            "Example symbol score and residual-return paths",
            "Series are z-scored per symbol to compare shape rather than scale.",
        )
        _plot_empty(ax, "No symbols selected for example paths")
        _save_chart(fig, path)
        return

    column_count = 2 if len(symbols) > 1 else 1
    row_count = int(np.ceil(len(symbols) / column_count))
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(11.5, 2.7 * row_count + 1.2),
        constrained_layout=False,
        squeeze=False,
    )
    fig.subplots_adjust(top=0.88, hspace=0.50, wspace=0.12, left=0.06, right=0.985, bottom=0.07)
    _chart_title(
        fig,
        "Example symbol score and target paths",
        f"Each panel z-scores one symbol's primary score and realized target ({target_col}).",
    )
    flat_axes = list(axes.ravel())
    for ax in flat_axes[len(symbols):]:
        ax.set_visible(False)

    for ax, symbol in zip(flat_axes, symbols):
        group = sample[sample["stock_code"].astype(str).eq(symbol)].sort_values("date").copy()
        if group.empty:
            continue
        score_z = _zscore_array(pd.to_numeric(group[primary_score_col], errors="coerce").to_numpy(dtype=float))
        return_z = _zscore_array(pd.to_numeric(group[target_col], errors="coerce").to_numpy(dtype=float))
        ax.plot(
            group["date"],
            score_z,
            marker="o",
            linewidth=1.5,
            markersize=3.2,
            color="#315fbe",
            alpha=0.88,
            label="score",
        )
        ax.plot(
            group["date"],
            return_z,
            marker="x",
            linewidth=1.1,
            linestyle="--",
            markersize=3.2,
            color="#d17a34",
            alpha=0.62,
            label="return",
        )
        ax.axhline(0.0, color="#27384c", linewidth=0.8)
        ax.set_title(str(symbol), fontsize=10.5, loc="left")
        ax.set_ylabel("z-score")
        ax.grid(axis="y")
    flat_axes[0].legend(frameon=False, loc="upper right")
    for ax in flat_axes[: len(symbols)]:
        ax.tick_params(axis="x", rotation=25)
    _save_chart(fig, path)


def _prepare_feature_gain_by_fold(
    *,
    feature_gain: pd.DataFrame,
    feature_roles: dict[str, list[str]],
    feature_columns: list[str] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"fold_id", "feature", "importance_gain"}
    missing = sorted(required.difference(feature_gain.columns))
    if missing:
        raise ValueError(f"Feature gain frame is missing required columns: {missing}")

    frame = feature_gain.copy()
    if "importance_split" not in frame.columns:
        frame["importance_split"] = np.nan
    frame["fold_id"] = pd.to_numeric(frame["fold_id"], errors="raise").astype(int)
    frame["feature"] = frame["feature"].astype(str)

    expected_features = list(feature_columns) if feature_columns is not None else sorted(frame["feature"].unique())
    if not expected_features:
        raise ValueError("feature_columns must not be empty when feature gain is provided.")
    expected_set = set(expected_features)
    unexpected = sorted(set(frame["feature"]).difference(expected_set))
    if unexpected:
        raise ValueError(f"Feature gain contains unknown features: {unexpected}")

    duplicate_rows = frame.duplicated(["fold_id", "feature"])
    if duplicate_rows.any():
        sample = frame.loc[duplicate_rows, ["fold_id", "feature"]].head(5).to_dict("records")
        raise ValueError(f"Feature gain contains duplicate fold/feature rows: {sample}")

    for fold_id, group in frame.groupby("fold_id", sort=True):
        observed = set(group["feature"])
        missing_features = sorted(expected_set.difference(observed))
        extra_features = sorted(observed.difference(expected_set))
        if missing_features or extra_features:
            raise ValueError(
                f"Fold {fold_id} feature gain set does not match feature_columns. "
                f"missing={missing_features}, extra={extra_features}"
            )

    gain_values = pd.to_numeric(frame["importance_gain"], errors="coerce").to_numpy(dtype=float)
    finite_gain = np.isfinite(gain_values)
    if np.any(gain_values[finite_gain] < 0.0):
        raise ValueError("Feature gain contains negative importance_gain values.")
    nonfinite_gain_count = int((~finite_gain).sum())
    frame["importance_gain"] = np.where(finite_gain, gain_values, 0.0)

    split_values = pd.to_numeric(frame["importance_split"], errors="coerce").to_numpy(dtype=float)
    finite_split = np.isfinite(split_values)
    if np.any(split_values[finite_split] < 0.0):
        raise ValueError("Feature gain contains negative importance_split values.")
    nonfinite_split_count = int((~finite_split).sum())
    frame["importance_split"] = np.where(finite_split, split_values, 0.0)

    role_map = _feature_role_map(feature_roles, expected_features)
    frame["feature_role"] = frame["feature"].map(role_map).fillna("unknown").astype(str)

    gain_total = frame.groupby("fold_id")["importance_gain"].transform("sum")
    split_total = frame.groupby("fold_id")["importance_split"].transform("sum")
    gain_total_values = gain_total.to_numpy(dtype=float)
    split_total_values = split_total.to_numpy(dtype=float)
    gain_values_clean = frame["importance_gain"].to_numpy(dtype=float)
    split_values_clean = frame["importance_split"].to_numpy(dtype=float)
    frame["gain_share_in_fold"] = np.divide(
        gain_values_clean,
        gain_total_values,
        out=np.zeros(len(frame), dtype=float),
        where=gain_total_values > 0.0,
    )
    frame["split_share_in_fold"] = np.divide(
        split_values_clean,
        split_total_values,
        out=np.zeros(len(frame), dtype=float),
        where=split_total_values > 0.0,
    )
    frame["gain_rank_in_fold"] = (
        frame.groupby("fold_id")["importance_gain"].rank(method="min", ascending=False).astype(int)
    )
    frame["split_rank_in_fold"] = (
        frame.groupby("fold_id")["importance_split"].rank(method="min", ascending=False).astype(int)
    )
    frame = frame.sort_values(["fold_id", "importance_gain", "feature"], ascending=[True, False, True])
    frame["cumulative_gain_share_in_fold"] = frame.groupby("fold_id")["gain_share_in_fold"].cumsum()

    gain_by_fold = frame.groupby("fold_id")["importance_gain"].sum()
    split_by_fold = frame.groupby("fold_id")["importance_split"].sum()
    diagnostics = {
        "schema_version": "feature_gain_diagnostics_v1",
        "fold_count": int(frame["fold_id"].nunique()),
        "feature_count": int(len(expected_features)),
        "row_count": int(len(frame)),
        "nonfinite_gain_replaced_with_zero": nonfinite_gain_count,
        "nonfinite_split_replaced_with_zero": nonfinite_split_count,
        "zero_total_gain_folds": [int(fold_id) for fold_id, value in gain_by_fold.items() if float(value) <= 0.0],
        "zero_total_split_folds": [int(fold_id) for fold_id, value in split_by_fold.items() if float(value) <= 0.0],
        "feature_columns": expected_features,
        "notes": [
            "Gain shares are normalized within each fold.",
            "Feature gain is model-derived and does not use realized future returns.",
        ],
    }
    return frame[
        [
            "fold_id",
            "feature",
            "feature_role",
            "importance_gain",
            "importance_split",
            "gain_share_in_fold",
            "split_share_in_fold",
            "gain_rank_in_fold",
            "split_rank_in_fold",
            "cumulative_gain_share_in_fold",
        ]
    ].reset_index(drop=True), diagnostics


def _feature_role_map(feature_roles: dict[str, list[str]], feature_columns: list[str]) -> dict[str, str]:
    ignored_roles = {"traceability", "excluded_from_model", "targets"}
    feature_set = set(feature_columns)
    mapping: dict[str, str] = {}
    conflicts: dict[str, list[str]] = {}
    for role, columns in feature_roles.items():
        if role in ignored_roles:
            continue
        for feature in columns:
            feature_text = str(feature)
            if feature_text not in feature_set:
                continue
            existing = mapping.get(feature_text)
            if existing is not None and existing != role:
                conflicts.setdefault(feature_text, [existing]).append(role)
            else:
                mapping[feature_text] = role
    if conflicts:
        details = {feature: sorted(set(roles)) for feature, roles in conflicts.items()}
        raise ValueError(f"Feature(s) assigned to multiple feature roles: {details}")
    return mapping


def _summarize_feature_gain(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "feature_role",
                "fold_count",
                "total_gain",
                "mean_gain",
                "mean_gain_share",
                "std_gain_share",
                "mean_rank",
                "best_rank",
                "worst_rank",
                "total_split",
                "mean_split_share",
            ]
        )
    records: list[dict[str, Any]] = []
    for (feature, role), group in frame.groupby(["feature", "feature_role"], sort=True):
        gain_shares = group["gain_share_in_fold"].to_numpy(dtype=float)
        records.append(
            {
                "feature": str(feature),
                "feature_role": str(role),
                "fold_count": int(group["fold_id"].nunique()),
                "total_gain": float(group["importance_gain"].sum()),
                "mean_gain": float(group["importance_gain"].mean()),
                "mean_gain_share": float(group["gain_share_in_fold"].mean()),
                "std_gain_share": float(np.std(gain_shares, ddof=0)) if len(gain_shares) else 0.0,
                "mean_rank": float(group["gain_rank_in_fold"].mean()),
                "best_rank": int(group["gain_rank_in_fold"].min()),
                "worst_rank": int(group["gain_rank_in_fold"].max()),
                "total_split": float(group["importance_split"].sum()),
                "mean_split_share": float(group["split_share_in_fold"].mean()),
            }
        )
    return (
        pd.DataFrame(records)
        .sort_values(["total_gain", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _summarize_feature_gain_roles(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "feature_role",
        "feature_count",
        "total_gain",
        "gain_share",
        "total_split",
        "split_share",
        "top_feature",
        "top_feature_gain_share",
    ]
    if summary.empty:
        return pd.DataFrame(columns=columns)
    total_gain = float(summary["total_gain"].sum())
    total_split = float(summary["total_split"].sum())
    records: list[dict[str, Any]] = []
    for role, group in summary.groupby("feature_role", sort=True):
        role_gain = float(group["total_gain"].sum())
        role_split = float(group["total_split"].sum())
        top = group.sort_values(["total_gain", "feature"], ascending=[False, True]).iloc[0]
        records.append(
            {
                "feature_role": str(role),
                "feature_count": int(group["feature"].nunique()),
                "total_gain": role_gain,
                "gain_share": role_gain / total_gain if total_gain > 0.0 else 0.0,
                "total_split": role_split,
                "split_share": role_split / total_split if total_split > 0.0 else 0.0,
                "top_feature": str(top["feature"]),
                "top_feature_gain_share": float(top["total_gain"]) / role_gain if role_gain > 0.0 else 0.0,
            }
        )
    return pd.DataFrame(records, columns=columns).sort_values(
        ["gain_share", "feature_role"], ascending=[False, True]
    ).reset_index(drop=True)


def _plot_feature_gain_top(summary: pd.DataFrame, path: Path, *, max_features: int = 20) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.8, 6.2), constrained_layout=True)
    _chart_title(
        fig,
        "Top feature gain shares",
        "Mean gain share is normalized within each fold before cross-fold aggregation.",
    )
    if summary.empty or float(summary["mean_gain_share"].sum()) <= 0.0:
        _plot_empty(ax, "No positive feature gain data available")
        _save_chart(fig, path)
        return
    frame = summary.sort_values(["mean_gain_share", "feature"], ascending=[False, True]).head(max_features)
    frame = frame.sort_values("mean_gain_share", ascending=True)
    ax.barh(frame["feature"], frame["mean_gain_share"], color="#315fbe")
    ax.set_xlabel("Mean gain share")
    ax.grid(axis="x")
    _save_chart(fig, path)


def _plot_feature_gain_by_role(role_summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.6, 5.2), constrained_layout=True)
    _chart_title(
        fig,
        "Feature gain by role",
        "Gain and split shares show whether the model relies on alpha or context features.",
    )
    if role_summary.empty:
        _plot_empty(ax, "No feature role data available")
        _save_chart(fig, path)
        return
    frame = role_summary.sort_values("gain_share", ascending=False)
    x = np.arange(len(frame), dtype=float)
    width = 0.36
    ax.bar(x - width / 2, frame["gain_share"], width, label="Gain share", color="#315fbe")
    ax.bar(x + width / 2, frame["split_share"], width, label="Split share", color="#d17a34")
    ax.set_xticks(x)
    ax.set_xticklabels(frame["feature_role"], rotation=25, ha="right")
    ax.set_ylabel("Share")
    ax.legend(frameon=False)
    ax.grid(axis="y")
    _save_chart(fig, path)


def _plot_feature_gain_fold_heatmap(
    by_fold: pd.DataFrame,
    path: Path,
    *,
    max_features: int = 20,
) -> None:
    import matplotlib.pyplot as plt

    if by_fold.empty:
        fig, ax = plt.subplots(figsize=(9.6, 5.2), constrained_layout=True)
        _chart_title(fig, "Feature gain fold heatmap", "Gain shares by fold for top features.")
        _plot_empty(ax, "No feature gain data available")
        _save_chart(fig, path)
        return

    top_features = (
        by_fold.groupby("feature")["importance_gain"]
        .sum()
        .sort_values(ascending=False)
        .head(max_features)
        .index.tolist()
    )
    matrix = (
        by_fold[by_fold["feature"].isin(top_features)]
        .pivot_table(index="feature", columns="fold_id", values="gain_share_in_fold", aggfunc="sum", fill_value=0.0)
        .loc[top_features]
    )
    fig_height = max(5.2, 0.32 * len(matrix) + 2.0)
    fig, ax = plt.subplots(figsize=(8.8, fig_height), constrained_layout=True)
    _chart_title(fig, "Feature gain fold heatmap", "Darker cells indicate larger within-fold gain share.")
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="Blues")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels([f"F{fold_id}" for fold_id in matrix.columns])
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Feature")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="Gain share")
    _save_chart(fig, path)


def _report_key_metrics(overall_metrics: pd.DataFrame, primary_score_col: str) -> list[dict[str, Any]]:
    if overall_metrics.empty:
        return []
    frame = overall_metrics[overall_metrics["score_col"].eq(primary_score_col)].copy()
    if frame.empty:
        frame = overall_metrics.copy()
    frame = frame.sort_values(["split_order", "fold_id", "score_col"])
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "split": str(row.get("split", "")),
                "fold_id": str(row.get("fold_id", "")),
                "mean_ic": row.get("mean_ic", np.nan),
                "mean_rankic": row.get("mean_rankic", np.nan),
                "ic_positive_rate": row.get("ic_positive_rate", np.nan),
                "date_count": row.get("date_count", np.nan),
            }
        )
    return rows


def _report_chart_block(report_path: Path, chart_key: str, chart_path_text: str) -> str:
    chart_path = Path(chart_path_text)
    report_dir = report_path.parent
    if not chart_path.is_absolute():
        report_relative = report_dir / chart_path
        cwd_relative = Path.cwd() / chart_path
        if report_relative.exists():
            chart_path = report_relative
        elif cwd_relative.exists():
            chart_path = cwd_relative
        else:
            chart_path = report_relative
    try:
        src = chart_path.resolve().relative_to(report_dir.resolve()).as_posix()
    except (OSError, ValueError):
        src = chart_path.name
    title = chart_key.replace("_", " ").title()
    return f"<h3>{html.escape(title)}</h3>\n<img src=\"{html.escape(src)}\" alt=\"{html.escape(title)}\">"


def _chart_title(fig: Any, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.01, y=1.03, ha="left", va="bottom", fontsize=15, fontweight="bold")
    fig.text(0.01, 0.98, subtitle, ha="left", va="bottom", fontsize=10.5, color="#667085")


def _plot_empty(ax: Any, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", color="#667085", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def _save_chart(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def _zscore_array(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    output = np.full(values.shape, np.nan, dtype=float)
    if not finite.any():
        return output
    finite_values = values[finite]
    std = float(np.nanstd(finite_values))
    if std <= 1e-12:
        output[finite] = 0.0
        return output
    output[finite] = (finite_values - float(np.nanmean(finite_values))) / std
    return output


def _format_float(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(numeric):
        return ""
    return f"{numeric:.6f}"


def _format_int(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(numeric):
        return ""
    return str(int(numeric))


def _prepare_evaluation_frame(
    *,
    predictions: pd.DataFrame,
    training_panel: pd.DataFrame,
    score_cols: list[str],
    target_col: str,
    spread_target_col: str,
    condition_bucket_count: int,
) -> pd.DataFrame:
    pred = _normalize_keys(predictions)
    panel = _normalize_keys(training_panel)
    _validate_prediction_columns(pred, score_cols, target_col, spread_target_col)
    if pred.duplicated(["fold_id", "split", "date", "stock_code"]).any():
        raise ValueError("Predictions contain duplicate (fold_id, split, date, stock_code) rows.")
    if panel.duplicated(["date", "stock_code"]).any():
        raise ValueError("Training panel contains duplicate (date, stock_code) rows.")

    context_cols = _context_columns(panel)
    check_cols = _consistency_check_columns(panel, pred, target_col, spread_target_col)
    context = panel[["date", "stock_code", *context_cols, *check_cols]].copy()
    context = context.rename(columns={column: f"{column}__panel" for column in check_cols})
    merged = pred.merge(
        context,
        on=["date", "stock_code"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    missing_context = merged["_merge"].ne("both")
    if missing_context.any():
        sample = merged.loc[missing_context, ["date", "stock_code"]].head(5).to_dict("records")
        raise ValueError(f"Predictions could not be matched to training panel context: {sample}")
    merged = merged.drop(columns=["_merge"])
    _validate_consistent_columns(merged, check_cols)
    merged = merged.drop(columns=[f"{column}__panel" for column in check_cols])

    for column in [target_col, spread_target_col, *score_cols]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    if "split" in merged:
        merged["split"] = merged["split"].astype(str)
        merged["split_order"] = merged["split"].map(SPLIT_ORDER).fillna(99).astype(int)
    else:
        merged["split_order"] = 99
    _add_daily_condition_buckets(merged, condition_bucket_count)
    return merged.sort_values(["fold_id", "split_order", "date", "stock_code"]).reset_index(drop=True)


def _compute_daily_ic(frame: pd.DataFrame, score_cols: list[str], target_col: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (fold_id, split, date), group in frame.groupby(["fold_id", "split", "date"], sort=True):
        target = pd.to_numeric(group[target_col], errors="coerce")
        for score_col in score_cols:
            score = pd.to_numeric(group[score_col], errors="coerce")
            valid = _finite_pair_mask(score, target)
            obs_count = int(valid.sum())
            score_valid = score.loc[valid]
            target_valid = target.loc[valid]
            records.append(
                {
                    "fold_id": int(fold_id),
                    "split": split,
                    "split_order": _split_order(split),
                    "score_col": score_col,
                    "date": pd.Timestamp(date),
                    "row_count": int(len(group)),
                    "obs_count": obs_count,
                    "ic": _safe_corr(score_valid, target_valid),
                    "rank_ic": _safe_corr(
                        score_valid.rank(method="average"),
                        target_valid.rank(method="average"),
                    ),
                }
            )
    if not records:
        return pd.DataFrame(
            columns=[
                "fold_id",
                "split",
                "split_order",
                "score_col",
                "date",
                "row_count",
                "obs_count",
                "ic",
                "rank_ic",
            ]
        )
    return pd.DataFrame(records).sort_values(["fold_id", "split_order", "score_col", "date"])


def _summarize_daily_ic(daily_ic: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    if daily_ic.empty:
        return _empty_overall_metrics()
    row_summary = (
        frame.groupby(["fold_id", "split"], sort=True)
        .agg(total_row_count=("stock_code", "size"), total_sample_weight=("sample_weight", "sum")
             if "sample_weight" in frame.columns else ("stock_code", "size"))
        .reset_index()
    )
    records: list[dict[str, Any]] = []
    for key, group in daily_ic.groupby(["fold_id", "split", "score_col"], sort=True):
        fold_id, split, score_col = key
        ic_values = _finite_values(group["ic"])
        rank_values = _finite_values(group["rank_ic"])
        row_info = row_summary[
            row_summary["fold_id"].eq(fold_id) & row_summary["split"].eq(split)
        ].iloc[0]
        records.append(
            {
                "fold_id": int(fold_id),
                "split": split,
                "split_order": _split_order(split),
                "score_col": score_col,
                "date_count": int(len(ic_values)),
                "rank_ic_date_count": int(len(rank_values)),
                "row_count": int(row_info["total_row_count"]),
                "sample_weight_sum": float(row_info["total_sample_weight"]),
                "mean_ic": _nanmean(ic_values),
                "std_ic": _nanstd(ic_values),
                "ic_ir": _annualized_ir(ic_values),
                "ic_positive_rate": _positive_rate(ic_values),
                "mean_rankic": _nanmean(rank_values),
                "std_rankic": _nanstd(rank_values),
                "rankic_ir": _annualized_ir(rank_values),
                "rankic_positive_rate": _positive_rate(rank_values),
            }
        )
    output = pd.DataFrame(records)
    output["ic_mean"] = output["mean_ic"]
    output["rank_ic_mean"] = output["mean_rankic"]
    return output.sort_values(["fold_id", "split_order", "score_col"]).reset_index(drop=True)


def _compute_top_bottom_spread(
    frame: pd.DataFrame,
    *,
    score_cols: list[str],
    target_col: str,
    quantiles: int,
) -> pd.DataFrame:
    if quantiles < 2:
        raise ValueError("top_bottom_quantiles must be at least 2.")
    records: list[dict[str, Any]] = []
    for (fold_id, split, date), group in frame.groupby(["fold_id", "split", "date"], sort=True):
        target = pd.to_numeric(group[target_col], errors="coerce")
        for score_col in score_cols:
            score = pd.to_numeric(group[score_col], errors="coerce")
            valid = group.loc[_finite_pair_mask(score, target), [score_col, target_col]].copy()
            n_obs = int(len(valid))
            if valid[score_col].nunique(dropna=True) < 2:
                continue
            actual_quantiles = min(int(quantiles), n_obs)
            if actual_quantiles < 2:
                continue
            valid["_bucket"] = pd.qcut(
                valid[score_col].rank(method="first"),
                q=actual_quantiles,
                labels=False,
                duplicates="drop",
            )
            bottom = valid[valid["_bucket"].eq(valid["_bucket"].min())]
            top = valid[valid["_bucket"].eq(valid["_bucket"].max())]
            if top.empty or bottom.empty or top["_bucket"].iloc[0] == bottom["_bucket"].iloc[0]:
                continue
            top_mean = float(top[target_col].mean())
            bottom_mean = float(bottom[target_col].mean())
            records.append(
                {
                    "fold_id": int(fold_id),
                    "split": split,
                    "split_order": _split_order(split),
                    "score_col": score_col,
                    "date": pd.Timestamp(date),
                    "target_col": target_col,
                    "quantiles_requested": int(quantiles),
                    "quantiles_used": int(actual_quantiles),
                    "obs_count": n_obs,
                    "top_count": int(len(top)),
                    "bottom_count": int(len(bottom)),
                    "top_mean": top_mean,
                    "bottom_mean": bottom_mean,
                    "spread": top_mean - bottom_mean,
                }
            )
    if not records:
        return pd.DataFrame(
            columns=[
                "fold_id",
                "split",
                "split_order",
                "score_col",
                "date",
                "target_col",
                "quantiles_requested",
                "quantiles_used",
                "obs_count",
                "top_count",
                "bottom_count",
                "top_mean",
                "bottom_mean",
                "spread",
            ]
        )
    return pd.DataFrame(records).sort_values(["fold_id", "split_order", "score_col", "date"])


def _summarize_top_bottom_spread(spread_by_date: pd.DataFrame) -> pd.DataFrame:
    if spread_by_date.empty:
        return pd.DataFrame(
            columns=[
                "fold_id",
                "split",
                "split_order",
                "score_col",
                "date_count",
                "mean_spread",
                "std_spread",
                "spread_ir",
                "spread_positive_rate",
                "mean_top_count",
                "mean_bottom_count",
            ]
        )
    records: list[dict[str, Any]] = []
    for key, group in spread_by_date.groupby(["fold_id", "split", "score_col"], sort=True):
        fold_id, split, score_col = key
        spread_values = _finite_values(group["spread"])
        records.append(
            {
                "fold_id": int(fold_id),
                "split": split,
                "split_order": _split_order(split),
                "score_col": score_col,
                "date_count": int(len(spread_values)),
                "mean_spread": _nanmean(spread_values),
                "std_spread": _nanstd(spread_values),
                "spread_ir": _annualized_ir(spread_values),
                "spread_positive_rate": _positive_rate(spread_values),
                "mean_top_count": float(group["top_count"].mean()),
                "mean_bottom_count": float(group["bottom_count"].mean()),
            }
        )
    return pd.DataFrame(records).sort_values(["fold_id", "split_order", "score_col"]).reset_index(drop=True)


def _compute_group_ic_by_date(
    frame: pd.DataFrame,
    *,
    score_cols: list[str],
    target_col: str,
    group_dimensions: list[str],
    min_group_obs: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for dimension in group_dimensions:
        grouped = frame[frame[dimension].notna()].groupby(
            ["fold_id", "split", "date", dimension],
            sort=True,
        )
        for (fold_id, split, date, group_value), group in grouped:
            target = pd.to_numeric(group[target_col], errors="coerce")
            for score_col in score_cols:
                score = pd.to_numeric(group[score_col], errors="coerce")
                valid = _finite_pair_mask(score, target)
                obs_count = int(valid.sum())
                if obs_count < min_group_obs:
                    continue
                score_valid = score.loc[valid]
                target_valid = target.loc[valid]
                ic = _safe_corr(score_valid, target_valid)
                rank_ic = _safe_corr(
                    score_valid.rank(method="average"),
                    target_valid.rank(method="average"),
                )
                if not np.isfinite(ic) and not np.isfinite(rank_ic):
                    continue
                records.append(
                    {
                        "fold_id": int(fold_id),
                        "split": split,
                        "split_order": _split_order(split),
                        "score_col": score_col,
                        "date": pd.Timestamp(date),
                        "group_dimension": dimension,
                        "group_value": str(group_value),
                        "row_count": int(len(group)),
                        "obs_count": obs_count,
                        "ic": ic,
                        "rank_ic": rank_ic,
                    }
                )
    if not records:
        return pd.DataFrame(
            columns=[
                "fold_id",
                "split",
                "split_order",
                "score_col",
                "date",
                "group_dimension",
                "group_value",
                "row_count",
                "obs_count",
                "ic",
                "rank_ic",
            ]
        )
    return pd.DataFrame(records).sort_values(
        ["fold_id", "split_order", "score_col", "group_dimension", "group_value", "date"]
    )


def _summarize_group_ic(group_ic_by_date: pd.DataFrame) -> pd.DataFrame:
    if group_ic_by_date.empty:
        return pd.DataFrame(
            columns=[
                "fold_id",
                "split",
                "split_order",
                "score_col",
                "group_dimension",
                "group_value",
                "date_count",
                "row_count",
                "obs_count",
                "mean_ic",
                "std_ic",
                "ic_ir",
                "mean_rankic",
                "std_rankic",
                "rankic_ir",
            ]
        )
    records: list[dict[str, Any]] = []
    for key, group in group_ic_by_date.groupby(
        ["fold_id", "split", "score_col", "group_dimension", "group_value"],
        sort=True,
    ):
        fold_id, split, score_col, dimension, value = key
        ic_values = _finite_values(group["ic"])
        rank_values = _finite_values(group["rank_ic"])
        records.append(
            {
                "fold_id": int(fold_id),
                "split": split,
                "split_order": _split_order(split),
                "score_col": score_col,
                "group_dimension": dimension,
                "group_value": str(value),
                "date_count": int(len(ic_values)),
                "rank_ic_date_count": int(len(rank_values)),
                "row_count": int(group["row_count"].sum()),
                "obs_count": int(group["obs_count"].sum()),
                "mean_ic": _nanmean(ic_values),
                "std_ic": _nanstd(ic_values),
                "ic_ir": _annualized_ir(ic_values),
                "mean_rankic": _nanmean(rank_values),
                "std_rankic": _nanstd(rank_values),
                "rankic_ir": _annualized_ir(rank_values),
            }
        )
    output = pd.DataFrame(records)
    output["ic_mean"] = output["mean_ic"]
    output["rank_ic_mean"] = output["mean_rankic"]
    return output.sort_values(
        ["fold_id", "split_order", "score_col", "group_dimension", "group_value"]
    ).reset_index(drop=True)


def _add_daily_condition_buckets(frame: pd.DataFrame, bucket_count: int) -> None:
    if "mcap_rank" in frame.columns:
        frame["mcap_tercile"] = _daily_quantile_bucket(frame, "mcap_rank", bucket_count)
    elif "log_mcap" in frame.columns:
        frame["mcap_tercile"] = _daily_quantile_bucket(frame, "log_mcap", bucket_count)
    elif "market_cap" in frame.columns:
        frame["mcap_tercile"] = _daily_quantile_bucket(frame, "market_cap", bucket_count)
    if "logADV20" in frame.columns:
        frame["adv_tercile"] = _daily_quantile_bucket(frame, "logADV20", bucket_count)
    if "turnover20" in frame.columns:
        frame["turnover_tercile"] = _daily_quantile_bucket(frame, "turnover20", bucket_count)


def _daily_quantile_bucket(frame: pd.DataFrame, column: str, bucket_count: int) -> pd.Series:
    labels = QUANTILE_BUCKET_LABELS.get(bucket_count, [f"q{i}" for i in range(1, bucket_count + 1)])
    output = pd.Series(pd.NA, index=frame.index, dtype="string")
    values = pd.to_numeric(frame[column], errors="coerce")
    for _, index in frame.groupby("date", sort=False).groups.items():
        group_values = values.loc[index]
        valid = np.isfinite(group_values.to_numpy(dtype=float))
        valid_index = group_values.index[valid]
        if len(valid_index) < 2:
            continue
        q = min(bucket_count, len(valid_index))
        group_labels = labels[:q] if q == bucket_count else [f"q{i}" for i in range(1, q + 1)]
        try:
            bucket_codes = pd.qcut(
                group_values.loc[valid_index].rank(method="first"),
                q=q,
                labels=group_labels,
                duplicates="drop",
            )
        except ValueError:
            continue
        output.loc[valid_index] = bucket_codes.astype("string")
    return output


def _available_group_dimensions(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "industry",
        "board",
        "index_bucket",
        "size_decile",
        "mcap_tercile",
        "adv_tercile",
        "turnover_tercile",
    ]
    return [column for column in preferred if column in frame.columns and frame[column].notna().any()]


def _context_columns(panel: pd.DataFrame) -> list[str]:
    candidates = [
        "industry",
        "board",
        "index_bucket",
        "size_decile",
        "mcap_rank",
        "log_mcap",
        "market_cap",
        "logADV20",
        "turnover20",
    ]
    columns: list[str] = []
    for column in candidates:
        if column in panel.columns and column not in columns:
            columns.append(column)
    return columns


def _consistency_check_columns(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    target_col: str,
    spread_target_col: str,
) -> list[str]:
    candidates = [target_col, spread_target_col, "y_rank_label", "y_resid_fwd", "sample_weight"]
    columns: list[str] = []
    for column in candidates:
        if column in panel.columns and column in predictions.columns and column not in columns:
            columns.append(column)
    return columns


def _validate_consistent_columns(frame: pd.DataFrame, check_cols: list[str]) -> None:
    for column in check_cols:
        panel_column = f"{column}__panel"
        left = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(frame[panel_column], errors="coerce").to_numpy(dtype=float)
        left_finite = np.isfinite(left)
        right_finite = np.isfinite(right)
        if not np.array_equal(left_finite, right_finite):
            raise ValueError(f"Prediction column {column!r} is inconsistent with training panel.")
        if left_finite.any() and not np.allclose(left[left_finite], right[right_finite], atol=1e-10, rtol=1e-10):
            raise ValueError(f"Prediction column {column!r} is inconsistent with training panel.")


def _validate_prediction_columns(
    frame: pd.DataFrame,
    score_cols: list[str],
    target_col: str,
    spread_target_col: str,
) -> None:
    required = {"fold_id", "split", "date", "stock_code", *score_cols, target_col, spread_target_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Predictions are missing required columns: {missing}")


def _default_score_cols(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in DEFAULT_SCORE_COLS if column in frame.columns]
    if not columns:
        raise ValueError(f"No default score columns found. Expected one of {DEFAULT_SCORE_COLS}.")
    return columns


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce").dt.normalize()
    output["stock_code"] = output["stock_code"].map(_normalize_stock_code)
    output = output[output["date"].notna() & output["stock_code"].ne("")].copy()
    return output


def _normalize_stock_code(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text.isdigit():
        return text.zfill(6)
    return text


def _finite_pair_mask(left: pd.Series, right: pd.Series) -> pd.Series:
    left_values = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    return pd.Series(np.isfinite(left_values) & np.isfinite(right_values), index=left.index)


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 2 or len(right) < 2:
        return np.nan
    left_values = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(left_values) & np.isfinite(right_values)
    left_values = left_values[valid]
    right_values = right_values[valid]
    if len(left_values) < 2:
        return np.nan
    if np.nanstd(left_values) <= 1e-12 or np.nanstd(right_values) <= 1e-12:
        return np.nan
    return float(np.corrcoef(left_values, right_values)[0, 1])


def _finite_values(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def _nanmean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else np.nan


def _nanstd(values: np.ndarray) -> float:
    return float(np.std(values, ddof=0)) if len(values) else np.nan


def _annualized_ir(values: np.ndarray) -> float:
    if len(values) < 2:
        return np.nan
    std = float(np.std(values, ddof=0))
    if std <= 1e-12:
        return np.nan
    return float(np.mean(values) / std * np.sqrt(252.0))


def _positive_rate(values: np.ndarray) -> float:
    return float(np.mean(values > 0.0)) if len(values) else np.nan


def _split_order(split: Any) -> int:
    return SPLIT_ORDER.get(str(split), 99)


def _empty_overall_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "fold_id",
            "split",
            "split_order",
            "score_col",
            "date_count",
            "rank_ic_date_count",
            "row_count",
            "sample_weight_sum",
            "mean_ic",
            "std_ic",
            "ic_ir",
            "ic_positive_rate",
            "mean_rankic",
            "std_rankic",
            "rankic_ir",
            "rankic_positive_rate",
            "ic_mean",
            "rank_ic_mean",
        ]
    )


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    return pd.read_csv(path, dtype={"stock_code": "string"})


def _write_csv_with_iso_dates(frame: pd.DataFrame, path: Path) -> None:
    output = frame.copy()
    for column in output.columns:
        if is_datetime64_any_dtype(output[column]):
            output[column] = pd.to_datetime(output[column]).dt.date.astype(str)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, encoding="utf-8-sig")


def _path_for_summary(path: Path) -> str:
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.name


if __name__ == "__main__":
    main()
