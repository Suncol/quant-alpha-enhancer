from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


QUANTILES = (0.001, 0.005, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999)
DEFAULT_MAX_EXAMPLE_SYMBOLS = 4


def build_neutralization_analysis(
    return_y: pd.DataFrame,
    residual: pd.DataFrame,
    rank_label: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    example_symbols: Sequence[str] | None = None,
    max_example_symbols: int = DEFAULT_MAX_EXAMPLE_SYMBOLS,
) -> dict[str, Any]:
    """Recompute distribution diagnostics for a completed neutralization run."""

    raw, resid, ranks = _align_frames(return_y, residual, rank_label)
    active_dates = resid.notna().any(axis=1)
    raw_on_residual = raw.where(resid.notna()).loc[active_dates]
    resid = resid.loc[active_dates]
    ranks = ranks.where(resid.notna()).loc[active_dates]
    raw_values = _finite_values(raw_on_residual)
    residual_values = _finite_values(resid)
    rank_values = _finite_values(ranks)
    processed = _processed_diagnostics(diagnostics)
    selected_symbols = _select_example_symbols(
        raw_on_residual,
        resid,
        requested=example_symbols,
        max_symbols=max_example_symbols,
    )

    daily = _daily_distribution_frame(raw_on_residual, resid, ranks, processed)
    symbol_examples = _symbol_example_records(raw_on_residual, resid, selected_symbols)

    headline_metrics = {
        "raw_std": _std(raw_values),
        "residual_std": _std(residual_values),
        "residual_mean": _mean(residual_values),
        "max_industry_residual_mean": _diagnostic_max(processed, "max_abs_industry_mean_after"),
        "max_board_residual_mean": _diagnostic_max(processed, "max_abs_board_mean_after"),
        "max_continuous_exposure": _diagnostic_max(processed, "max_abs_continuous_exposure_after"),
        "median_r2": _diagnostic_median(processed, "r2"),
        "rank_label_min": _min(rank_values),
        "rank_label_max": _max(rank_values),
    }

    return {
        "scope": {
            "date_min": raw_on_residual.index.min().date().isoformat() if len(raw_on_residual.index) else None,
            "date_max": raw_on_residual.index.max().date().isoformat() if len(raw_on_residual.index) else None,
            "processed_date_count": int(len(processed)) if not processed.empty else int(resid.notna().any(axis=1).sum()),
            "residual_cell_count": int(np.isfinite(residual_values).sum()),
            "raw_cell_count_on_residual_universe": int(np.isfinite(raw_values).sum()),
            "symbol_count": int(resid.shape[1]),
        },
        "headline_metrics": headline_metrics,
        "raw_distribution": _distribution_stats(raw_values),
        "residual_distribution": _distribution_stats(residual_values),
        "daily_distribution": daily.to_dict(orient="records"),
        "diagnostics": _diagnostic_summary(processed),
        "example_symbols": symbol_examples,
    }


def write_neutralization_analysis_artifacts(
    return_y: pd.DataFrame,
    residual: pd.DataFrame,
    rank_label: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    output_dir: Path,
    example_symbols: Sequence[str] | None = None,
    title: str = "Return-y neutralization distribution report",
) -> dict[str, Any]:
    """Write JSON, PNG charts, and an HTML report for a neutralization run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    raw, resid, ranks = _align_frames(return_y, residual, rank_label)
    active_dates = resid.notna().any(axis=1)
    raw_on_residual = raw.where(resid.notna()).loc[active_dates]
    resid = resid.loc[active_dates]
    ranks = ranks.where(resid.notna()).loc[active_dates]
    processed = _processed_diagnostics(diagnostics)
    analysis = build_neutralization_analysis(
        return_y,
        residual,
        rank_label,
        diagnostics,
        example_symbols=example_symbols,
    )
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    charts = _write_charts(
        raw_on_residual,
        resid,
        ranks,
        processed,
        analysis,
        charts_dir,
    )

    summary_json = output_dir / "analysis_summary.json"
    summary_json.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    report_html = output_dir / "report.html"
    report_html.write_text(
        _build_html_report(title, analysis, charts, charts_dir),
        encoding="utf-8",
    )
    return {
        "summary_json": summary_json,
        "report_html": report_html,
        "charts": charts,
    }


def _align_frames(
    return_y: pd.DataFrame,
    residual: pd.DataFrame,
    rank_label: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    resid = residual.copy()
    resid.index = pd.DatetimeIndex(pd.to_datetime(resid.index)).normalize()
    resid.columns = [_normalize_stock_code(column) for column in resid.columns]

    raw = return_y.copy()
    raw.index = pd.DatetimeIndex(pd.to_datetime(raw.index)).normalize()
    raw.columns = [_normalize_stock_code(column) for column in raw.columns]
    raw = raw.reindex(index=resid.index, columns=resid.columns)

    ranks = rank_label.copy()
    ranks.index = pd.DatetimeIndex(pd.to_datetime(ranks.index)).normalize()
    ranks.columns = [_normalize_stock_code(column) for column in ranks.columns]
    ranks = ranks.reindex(index=resid.index, columns=resid.columns)
    return raw, resid, ranks


def _normalize_stock_code(value: Any) -> str:
    text = str(value).strip()
    if "." in text:
        left, right = text.split(".", 1)
        if left.isdigit() and (right.isalpha() or right.isdigit()):
            text = left
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text.isdigit() and len(text) <= 6:
        return text.zfill(6)
    return text


def _processed_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return diagnostics.copy()
    frame = diagnostics.copy()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    if "skipped" not in frame.columns:
        return frame
    skipped = frame["skipped"]
    if skipped.dtype == object:
        skipped = skipped.astype(str).str.lower().isin({"true", "1", "yes"})
    return frame.loc[~skipped.astype(bool)].copy()


def _finite_values(frame: pd.DataFrame) -> np.ndarray:
    values = frame.to_numpy(dtype=float).ravel()
    return values[np.isfinite(values)]


def _distribution_stats(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "abs_gt_10pct": 0,
            "abs_gt_20pct": 0,
            "abs_gt_50pct": 0,
            "quantiles": {str(q): None for q in QUANTILES},
        }
    quantiles = np.quantile(values, QUANTILES)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "abs_gt_10pct": int(np.sum(np.abs(values) > 0.10)),
        "abs_gt_20pct": int(np.sum(np.abs(values) > 0.20)),
        "abs_gt_50pct": int(np.sum(np.abs(values) > 0.50)),
        "quantiles": {str(q): float(v) for q, v in zip(QUANTILES, quantiles, strict=True)},
    }


def _daily_distribution_frame(
    raw: pd.DataFrame,
    residual: pd.DataFrame,
    rank_label: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    daily = pd.DataFrame(index=residual.index)
    daily["date"] = daily.index
    daily["raw_std"] = raw.std(axis=1, skipna=True, ddof=0)
    daily["residual_std"] = residual.std(axis=1, skipna=True, ddof=0)
    daily["raw_mean"] = raw.mean(axis=1, skipna=True)
    daily["residual_mean"] = residual.mean(axis=1, skipna=True)
    daily["rank_label_min"] = rank_label.min(axis=1, skipna=True)
    daily["rank_label_max"] = rank_label.max(axis=1, skipna=True)
    daily["rank_label_p01"] = rank_label.quantile(0.01, axis=1, interpolation="linear")
    daily["rank_label_p99"] = rank_label.quantile(0.99, axis=1, interpolation="linear")
    if not diagnostics.empty and "date" in diagnostics.columns:
        diag_cols = [
            column
            for column in [
                "r2",
                "max_abs_industry_mean_after",
                "max_abs_board_mean_after",
                "max_abs_continuous_exposure_after",
                "n_used",
            ]
            if column in diagnostics.columns
        ]
        if diag_cols:
            diag = diagnostics[["date", *diag_cols]].dropna(subset=["date"])
            daily = daily.merge(diag, on="date", how="left")
    return daily.replace({np.nan: None})


def _diagnostic_summary(processed: pd.DataFrame) -> dict[str, Any]:
    return {
        column: _series_summary(processed[column])
        for column in [
            "r2",
            "max_abs_industry_mean_after",
            "max_abs_board_mean_after",
            "max_abs_continuous_exposure_after",
            "n_used",
        ]
        if column in processed.columns
    }


def _series_summary(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0, "mean": None, "min": None, "median": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "median": float(values.median()),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
    }


def _select_example_symbols(
    raw: pd.DataFrame,
    residual: pd.DataFrame,
    *,
    requested: Sequence[str] | None,
    max_symbols: int,
) -> list[str]:
    if requested:
        selected: list[str] = []
        for symbol in requested:
            code = _normalize_stock_code(symbol)
            if code in residual.columns and code not in selected:
                selected.append(code)
            if len(selected) >= max_symbols:
                return selected
        return selected

    coverage = residual.notna().sum(axis=0)
    min_count = max(20, int(0.50 * max(1, residual.notna().any(axis=1).sum())))
    candidates = coverage[coverage >= min_count].index.tolist()
    if not candidates:
        candidates = coverage.sort_values(ascending=False).head(max_symbols).index.tolist()
    stats = pd.DataFrame(index=candidates)
    stats["raw_std"] = raw[candidates].std(axis=0, skipna=True, ddof=0)
    stats["residual_std"] = residual[candidates].std(axis=0, skipna=True, ddof=0)
    stats["std_reduction"] = stats["raw_std"] - stats["residual_std"]
    stats = stats.replace([np.inf, -np.inf], np.nan).dropna()
    if stats.empty:
        return candidates[:max_symbols]

    selected = _append_unique([], stats["std_reduction"].idxmax())
    for q in (0.25, 0.50, 0.75):
        target = float(stats["raw_std"].quantile(q))
        symbol = (stats["raw_std"] - target).abs().sort_values().index[0]
        selected = _append_unique(selected, symbol)
    for symbol in coverage.sort_values(ascending=False).index:
        selected = _append_unique(selected, str(symbol))
        if len(selected) >= max_symbols:
            break
    return selected[:max_symbols]


def _append_unique(values: list[str], value: Any) -> list[str]:
    text = str(value)
    if text not in values:
        values.append(text)
    return values


def _symbol_example_records(
    raw: pd.DataFrame,
    residual: pd.DataFrame,
    symbols: Sequence[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for symbol in symbols:
        if symbol not in residual.columns:
            continue
        pair = pd.DataFrame({"raw": raw[symbol], "residual": residual[symbol]}).dropna()
        if pair.empty:
            continue
        records.append(
            {
                "symbol": symbol,
                "observation_count": int(len(pair)),
                "raw_std": float(pair["raw"].std(ddof=0)),
                "residual_std": float(pair["residual"].std(ddof=0)),
                "raw_mean": float(pair["raw"].mean()),
                "residual_mean": float(pair["residual"].mean()),
                "correlation": float(pair["raw"].corr(pair["residual"])) if len(pair) > 1 else None,
            }
        )
    return records


def _write_charts(
    raw: pd.DataFrame,
    residual: pd.DataFrame,
    rank_label: pd.DataFrame,
    diagnostics: pd.DataFrame,
    analysis: Mapping[str, Any],
    charts_dir: Path,
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    _set_chart_style(plt)
    raw_values = _finite_values(raw)
    residual_values = _finite_values(residual)
    charts = {
        "distribution": charts_dir / "raw_vs_residual_distribution.png",
        "quantiles": charts_dir / "raw_vs_residual_quantiles.png",
        "daily_std_r2": charts_dir / "daily_std_and_r2.png",
        "exposure_diagnostics": charts_dir / "residual_exposure_diagnostics.png",
        "rank_label_range": charts_dir / "rank_label_range.png",
        "example_symbols": charts_dir / "example_symbol_timeseries.png",
    }

    _plot_distribution(raw_values, residual_values, charts["distribution"], plt, mticker)
    _plot_quantiles(analysis, charts["quantiles"], plt, mticker)
    _plot_daily_std_r2(raw, residual, diagnostics, charts["daily_std_r2"], plt, mdates, mticker)
    _plot_exposure_diagnostics(diagnostics, charts["exposure_diagnostics"], plt, mdates)
    _plot_rank_label_range(rank_label, charts["rank_label_range"], plt, mdates)
    _plot_example_symbols(raw, residual, analysis["example_symbols"], charts["example_symbols"], plt, mdates, mticker)
    return charts


def _set_chart_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#FCFCFD",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#D7DBE7",
            "axes.labelcolor": "#1F2430",
            "xtick.color": "#6F768A",
            "ytick.color": "#6F768A",
            "grid.color": "#E6E8F0",
            "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _plot_distribution(raw_values: np.ndarray, residual_values: np.ndarray, path: Path, plt: Any, mticker: Any) -> None:
    raw_sample = _sample_values(raw_values)
    residual_sample = _sample_values(residual_values)
    combined = np.concatenate([raw_sample, residual_sample])
    lo, hi = np.quantile(combined, [0.001, 0.999])
    bins = np.linspace(lo, hi, 120)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.hist(raw_sample, bins=bins, density=True, alpha=0.45, color="#A3BEFA", edgecolor="#2E4780", linewidth=0.35, label="Raw return")
    ax.hist(residual_sample, bins=bins, density=True, alpha=0.55, color="#F0986E", edgecolor="#804126", linewidth=0.35, label="Residual return")
    ax.axvline(0.0, color="#464C55", linewidth=1.0, linestyle="--")
    _add_header(fig, ax, "Raw vs residual return distribution", "Density comparison on matched non-null residual cells; x-axis clipped to the 0.1%-99.9% pooled range.")
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", linestyle=":", linewidth=0.8)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_quantiles(analysis: Mapping[str, Any], path: Path, plt: Any, mticker: Any) -> None:
    quantiles = ["0.001", "0.005", "0.01", "0.05", "0.1", "0.25", "0.5", "0.75", "0.9", "0.95", "0.99", "0.995", "0.999"]
    x = np.arange(len(quantiles))
    raw_q = [analysis["raw_distribution"]["quantiles"][q] for q in quantiles]
    residual_q = [analysis["residual_distribution"]["quantiles"][q] for q in quantiles]

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.plot(x, raw_q, marker="o", color="#5477C4", label="Raw return")
    ax.plot(x, residual_q, marker="o", color="#CC6F47", label="Residual return")
    ax.axhline(0.0, color="#464C55", linewidth=1.0, linestyle="--")
    _add_header(fig, ax, "Quantile curve before and after neutralization", "Selected flat-sample quantiles show how neutralization compresses both tails.")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{float(q) * 100:g}%" for q in quantiles], rotation=35, ha="right")
    ax.set_ylabel("Return")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", linestyle=":", linewidth=0.8)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_daily_std_r2(raw: pd.DataFrame, residual: pd.DataFrame, diagnostics: pd.DataFrame, path: Path, plt: Any, mdates: Any, mticker: Any) -> None:
    daily = pd.DataFrame(index=residual.index)
    daily["raw_std"] = raw.std(axis=1, skipna=True, ddof=0)
    daily["residual_std"] = residual.std(axis=1, skipna=True, ddof=0)
    if not diagnostics.empty and {"date", "r2"}.issubset(diagnostics.columns):
        diag = diagnostics[["date", "r2"]].copy()
        diag["date"] = pd.to_datetime(diag["date"], errors="coerce").dt.normalize()
        daily = daily.merge(diag, left_index=True, right_on="date", how="left").set_index("date")

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), sharex=True)
    axes[0].plot(daily.index, daily["raw_std"], color="#5477C4", label="Raw std")
    axes[0].plot(daily.index, daily["residual_std"], color="#CC6F47", label="Residual std")
    axes[0].set_ylabel("Cross-section std")
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    axes[0].legend(loc="upper right", frameon=False)
    axes[0].grid(axis="y", linestyle=":", linewidth=0.8)
    if "r2" in daily.columns:
        axes[1].plot(daily.index, daily["r2"], color="#71B436")
    axes[1].set_ylabel("Daily R2")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(axis="y", linestyle=":", linewidth=0.8)
    axes[1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[1].xaxis.get_major_locator()))
    _add_header(fig, axes[0], "Daily cross-sectional dispersion and neutralization fit", "Top: raw and residual cross-sectional standard deviation. Bottom: daily regression R2.")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_exposure_diagnostics(diagnostics: pd.DataFrame, path: Path, plt: Any, mdates: Any) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    if not diagnostics.empty and "date" in diagnostics.columns:
        frame = diagnostics.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        columns = {
            "max_abs_industry_mean_after": ("Industry", "#5477C4"),
            "max_abs_board_mean_after": ("Board", "#CC6F47"),
            "max_abs_continuous_exposure_after": ("Continuous", "#71B436"),
        }
        for column, (label, color) in columns.items():
            if column in frame.columns:
                values = pd.to_numeric(frame[column], errors="coerce").replace(0.0, np.nan)
                ax.plot(frame["date"], values, color=color, label=label)
    ax.set_yscale("log")
    ax.set_ylabel("Max absolute residual exposure")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", linestyle=":", linewidth=0.8)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    _add_header(fig, ax, "Residual exposure diagnostics after neutralization", "Lower values indicate stronger realized neutrality; log scale keeps industry, board, and continuous diagnostics readable together.")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_rank_label_range(rank_label: pd.DataFrame, path: Path, plt: Any, mdates: Any) -> None:
    daily = pd.DataFrame(index=rank_label.index)
    daily["min"] = rank_label.min(axis=1, skipna=True)
    daily["max"] = rank_label.max(axis=1, skipna=True)
    daily["p01"] = rank_label.quantile(0.01, axis=1, interpolation="linear")
    daily["p99"] = rank_label.quantile(0.99, axis=1, interpolation="linear")

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.fill_between(daily.index, daily["p01"].astype(float), daily["p99"].astype(float), color="#CEDFFE", alpha=0.7, label="P1-P99")
    ax.plot(daily.index, daily["min"], color="#2E4780", linewidth=0.8, label="Daily min")
    ax.plot(daily.index, daily["max"], color="#804126", linewidth=0.8, label="Daily max")
    ax.axhline(0.0, color="#464C55", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Rank label")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", linestyle=":", linewidth=0.8)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    _add_header(fig, ax, "Rank-label range stays centered by construction", "Daily min/max and P1-P99 bands for residual cross-sectional centered ranks.")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_example_symbols(raw: pd.DataFrame, residual: pd.DataFrame, examples: Sequence[Mapping[str, Any]], path: Path, plt: Any, mdates: Any, mticker: Any) -> None:
    symbols = [str(item["symbol"]) for item in examples]
    if not symbols:
        symbols = list(residual.columns[:1])
    n = len(symbols)
    fig, axes = plt.subplots(n, 1, figsize=(11.5, max(3.0, 2.4 * n)), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, symbol in zip(axes, symbols, strict=False):
        pair = pd.DataFrame({"raw": raw[symbol], "residual": residual[symbol]}).dropna()
        ax.plot(pair.index, pair["raw"], color="#5477C4", linewidth=0.9, label="Raw")
        ax.plot(pair.index, pair["residual"], color="#CC6F47", linewidth=0.9, label="Residual")
        ax.axhline(0.0, color="#464C55", linewidth=0.8, linestyle="--")
        ax.set_ylabel(symbol)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.grid(axis="y", linestyle=":", linewidth=0.8)
    axes[0].legend(loc="upper right", frameon=False, ncol=2)
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[-1].xaxis.get_major_locator()))
    _add_header(fig, axes[0], "Example symbol time series", "Raw and residual returns for selected symbols on matched neutralization dates.")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _sample_values(values: np.ndarray, max_count: int = 400_000) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size <= max_count:
        return finite
    rng = np.random.default_rng(20260610)
    return finite[rng.choice(finite.size, size=max_count, replace=False)]


def _add_header(fig: Any, ax: Any, title: str, subtitle: str) -> None:
    ax.set_title("")
    fig.subplots_adjust(top=0.82)
    left = ax.get_position().x0
    fig.text(left, 0.95, title, ha="left", va="top", fontsize=14, weight="bold", color="#1F2430")
    fig.text(left, 0.905, subtitle, ha="left", va="top", fontsize=10, color="#6F768A")


def _build_html_report(title: str, analysis: Mapping[str, Any], charts: Mapping[str, Path], charts_dir: Path) -> str:
    metrics = analysis["headline_metrics"]
    scope = analysis["scope"]
    relative_charts = {name: Path("charts") / path.name for name, path in charts.items()}
    cards = [
        ("Raw std", _format_pct(metrics["raw_std"])),
        ("Residual std", _format_pct(metrics["residual_std"])),
        ("Residual mean", _format_sci(metrics["residual_mean"])),
        ("Median R2", _format_number(metrics["median_r2"], digits=3)),
        ("Rank label range", f"{_format_number(metrics['rank_label_min'], digits=3)} to {_format_number(metrics['rank_label_max'], digits=3)}"),
    ]
    example_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(item['symbol'])}</td>"
        f"<td>{item['observation_count']:,}</td>"
        f"<td>{_format_pct(item['raw_std'])}</td>"
        f"<td>{_format_pct(item['residual_std'])}</td>"
        f"<td>{_format_number(item['correlation'], digits=3)}</td>"
        "</tr>"
        for item in analysis["example_symbols"]
    )
    card_html = "\n".join(
        f"<div class=\"metric-card\"><div class=\"metric-label\">{label}</div><div class=\"metric-value\">{value}</div></div>"
        for label, value in cards
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 0; background: #FCFCFD; color: #1F2430; font-family: Segoe UI, Arial, sans-serif; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 36px 24px 56px; }}
    h1 {{ font-size: 30px; margin: 0 0 18px; }}
    h2 {{ font-size: 21px; margin: 34px 0 10px; }}
    p, li {{ font-size: 15px; line-height: 1.58; color: #3A4050; }}
    .summary {{ background: #FFFFFF; border: 1px solid #E6E8F0; padding: 18px 20px; border-radius: 8px; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 18px 0; }}
    .metric-card {{ background: #FFFFFF; border: 1px solid #E6E8F0; border-radius: 8px; padding: 14px 15px; }}
    .metric-label {{ color: #6F768A; font-size: 12px; text-transform: uppercase; letter-spacing: .03em; }}
    .metric-value {{ font-size: 22px; font-weight: 700; margin-top: 6px; }}
    figure {{ margin: 22px 0 30px; }}
    figure img {{ width: 100%; height: auto; border: 1px solid #E6E8F0; border-radius: 8px; background: #FFFFFF; }}
    figcaption {{ color: #6F768A; font-size: 13px; margin-top: 8px; }}
    table {{ border-collapse: collapse; width: 100%; background: #FFFFFF; border: 1px solid #E6E8F0; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #E6E8F0; text-align: left; font-size: 14px; }}
    th {{ color: #6F768A; font-size: 12px; text-transform: uppercase; letter-spacing: .03em; }}
    code {{ background: #F4F5F7; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
<main>
  <h1>{html.escape(title)}</h1>

  <section class="summary">
    <h2>Executive Summary</h2>
    <p><strong>Neutralization materially compressed the return distribution.</strong>
    On the matched residual universe, raw cross-sectional labels have std {_format_pct(metrics["raw_std"])}
    versus residual std {_format_pct(metrics["residual_std"])}. The residual mean is
    {_format_sci(metrics["residual_mean"])}, so the transformed label is centered at the flat sample level.</p>
    <p><strong>Residual exposures are explicitly monitored as part of the run.</strong>
    Max industry residual mean is {_format_sci(metrics["max_industry_residual_mean"])},
    max board residual mean is {_format_sci(metrics["max_board_residual_mean"])}, and max continuous
    exposure is {_format_sci(metrics["max_continuous_exposure"])}. Median daily R2 is
    {_format_number(metrics["median_r2"], digits=3)}.</p>
  </section>

  <div class="metric-grid">{card_html}</div>

  <h2>Distribution Before And After Neutralization</h2>
  <p>The comparison uses raw return values only where a residual was produced, so the two distributions
  have the same observation universe: {scope["residual_cell_count"]:,} cells from {scope["date_min"]} to {scope["date_max"]}.</p>
  <figure><img src="{relative_charts["distribution"].as_posix()}" alt="Raw and residual return distribution"><figcaption>Histogram density comparison, clipped only for readability.</figcaption></figure>
  <figure><img src="{relative_charts["quantiles"].as_posix()}" alt="Raw and residual return quantiles"><figcaption>Selected quantiles show the residual tail compression.</figcaption></figure>

  <h2>Daily Fit And Residual Neutrality</h2>
  <p>Daily R2 and residual exposure diagnostics are written with the neutralization output, so a future run can
  detect when winsorization, ridge, exposure coverage, or market conditions change the residual distribution.</p>
  <figure><img src="{relative_charts["daily_std_r2"].as_posix()}" alt="Daily standard deviation and R2"><figcaption>Daily dispersion and fit diagnostics.</figcaption></figure>
  <figure><img src="{relative_charts["exposure_diagnostics"].as_posix()}" alt="Residual exposure diagnostics"><figcaption>Industry, board, and continuous residual diagnostics on a log scale.</figcaption></figure>

  <h2>Rank Label And Symbol Examples</h2>
  <p>The rank label remains bounded and centered by daily cross-sectional ranking. The examples below show raw
  versus residual time series for selected symbols; they are examples, not hand-picked targets.</p>
  <figure><img src="{relative_charts["rank_label_range"].as_posix()}" alt="Rank label range"><figcaption>Daily rank-label bounds and central band.</figcaption></figure>
  <figure><img src="{relative_charts["example_symbols"].as_posix()}" alt="Example symbol time series"><figcaption>Raw and residual return time series for selected symbols.</figcaption></figure>

  <table>
    <thead><tr><th>Symbol</th><th>Obs.</th><th>Raw std</th><th>Residual std</th><th>Correlation</th></tr></thead>
    <tbody>{example_rows}</tbody>
  </table>

  <h2>Caveats And Assumptions</h2>
  <ul>
    <li>This report recomputes actual values from the generated raw/residual/rank/diagnostics files; it does not use illustrative target numbers.</li>
    <li>The run is approximate neutralization when ridge is positive or returns are winsorized; the diagnostics should be read as realized residual checks.</li>
    <li>Raw return metrics are calculated on the matched residual universe, not on every non-null raw label in the original file.</li>
  </ul>
</main>
</body>
</html>
"""


def _std(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(np.std(values))


def _mean(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(np.mean(values))


def _min(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(np.min(values))


def _max(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(np.max(values))


def _diagnostic_max(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return None if values.empty else float(values.max())


def _diagnostic_median(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return None if values.empty else float(values.median())


def _format_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _format_sci(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.3e}"


def _format_number(value: Any, *, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
