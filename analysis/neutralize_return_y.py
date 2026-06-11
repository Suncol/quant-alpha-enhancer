from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


DEFAULT_INDEX_COLUMNS = ("is_csi300", "is_csi500", "is_csi1000", "is_csi2000")


@dataclass(frozen=True)
class BasicNeutralizationConfig:
    """Configuration for basic return-y neutralization.

    The basic design neutralizes future returns against industry, board,
    CSI 300/500/1000/2000 membership flags, and cross-sectional market-cap rank.
    """

    date_col: str = "date"
    stock_col: str = "stock_code"
    industry_col: str = "industry"
    board_col: str = "board"
    market_cap_col: str = "market_cap"
    size_rank_col: str = "size_rank_norm"
    index_cols: tuple[str, ...] = DEFAULT_INDEX_COLUMNS
    extra_continuous_cols: tuple[str, ...] = ()
    start_date: str | pd.Timestamp | None = None
    ridge: float = 1e-8
    winsor_lower: float | None = 0.01
    winsor_upper: float | None = 0.99
    min_obs: int = 30
    min_obs_per_column_buffer: int = 10
    eps: float = 1e-12


def centered_rank(values: Sequence[float] | pd.Series | np.ndarray) -> pd.Series:
    """Return centered mid-ranks in roughly [-0.5, 0.5], keeping missing values.

    For n finite observations, the transform is (rank - 0.5) / n - 0.5.
    This uses average ranks for ties and has exact zero finite-sample mean.
    """

    if isinstance(values, pd.Series):
        numeric = pd.to_numeric(values, errors="coerce")
        result = pd.Series(np.nan, index=values.index, dtype=float)
    else:
        numeric = pd.Series(pd.to_numeric(pd.Series(values), errors="coerce"), dtype=float)
        result = pd.Series(np.nan, index=numeric.index, dtype=float)

    finite = np.isfinite(numeric.to_numpy(dtype=float, copy=False))
    n = int(finite.sum())
    if n == 0:
        return result

    ranks = numeric[finite].rank(method="average")
    result.loc[finite] = (ranks - 0.5) / float(n) - 0.5
    return result


def neutralize_return_y_matrix(
    return_y: pd.DataFrame,
    exposures: pd.DataFrame,
    *,
    config: BasicNeutralizationConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Neutralize a date x stock return matrix using basic daily exposures.

    `return_y` must be a wide DataFrame with dates as rows and stock codes as
    columns. `exposures` must be a long DataFrame with one row per date-stock
    observation and the columns defined by `BasicNeutralizationConfig`.
    """

    cfg = config or BasicNeutralizationConfig()
    _validate_config(cfg)

    returns = _prepare_return_y(return_y)
    exposure_frame = prepare_basic_exposures(exposures, config=cfg)
    exposure_by_date = {
        pd.Timestamp(date).normalize(): group.reset_index(drop=True)
        for date, group in exposure_frame.groupby(cfg.date_col, sort=False)
    }

    residual = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns, dtype=float)
    rank_label = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns, dtype=float)
    diagnostics: list[dict[str, Any]] = []
    start_date = _normalize_optional_timestamp(cfg.start_date)
    neutralization_type = _neutralization_type(cfg)
    input_return_transform = _input_return_transform(cfg)

    for date in returns.index:
        date_key = pd.Timestamp(date).normalize()
        row = pd.to_numeric(returns.loc[date], errors="coerce")
        finite_return = np.isfinite(row.to_numpy(dtype=float, copy=False))
        finite_return_count = int(finite_return.sum())

        base_diag = {
            "date": date_key.date().isoformat(),
            "neutralization_type": neutralization_type,
            "neutralization_label": _neutralization_label(neutralization_type),
            "input_return_transform": input_return_transform,
            "ridge": float(cfg.ridge),
            "winsor_lower": np.nan if cfg.winsor_lower is None else float(cfg.winsor_lower),
            "winsor_upper": np.nan if cfg.winsor_upper is None else float(cfg.winsor_upper),
            "n_total": int(returns.shape[1]),
            "n_valid_return": finite_return_count,
            "n_valid_exposure": 0,
            "n_used": 0,
            "n_columns": 0,
            "missing_exposure_count": finite_return_count,
            "dropped_constant_continuous_count": 0,
            "skipped": True,
            "skipped_reason": "",
            "ridge_solver": None,
            "r2": np.nan,
            "return_weighted_mean": np.nan,
            "residual_weighted_mean": np.nan,
            "residual_weighted_std": np.nan,
            "max_abs_industry_mean_after": np.nan,
            "max_abs_board_mean_after": np.nan,
            "max_abs_continuous_exposure_after": np.nan,
            "continuous_columns_used": "",
        }

        if start_date is not None and date_key < start_date:
            base_diag["skipped_reason"] = "before_start_date"
            diagnostics.append(base_diag)
            continue

        day_exposure = exposure_by_date.get(date_key)
        if day_exposure is None:
            base_diag["skipped_reason"] = "missing_exposure_date"
            diagnostics.append(base_diag)
            continue

        day = pd.DataFrame(
            {
                cfg.stock_col: returns.columns.astype(str),
                "return_y": row.to_numpy(dtype=float, copy=False),
            }
        )
        exposure_cols = [
            cfg.stock_col,
            cfg.industry_col,
            cfg.board_col,
            *cfg.index_cols,
            cfg.market_cap_col,
            cfg.size_rank_col,
            *cfg.extra_continuous_cols,
        ]
        day = day.merge(
            day_exposure[exposure_cols],
            on=cfg.stock_col,
            how="left",
            validate="one_to_one",
        )

        valid = _valid_daily_rows(day, config=cfg)
        valid &= np.isfinite(day["return_y"].to_numpy(dtype=float, copy=False))
        used = day.loc[valid].copy()
        used[cfg.size_rank_col] = centered_rank(used[cfg.market_cap_col]).to_numpy(dtype=float)
        n_used = int(len(used))
        base_diag["n_valid_exposure"] = int(
            _valid_daily_rows(day, config=cfg).sum()
        )
        base_diag["n_used"] = n_used
        base_diag["missing_exposure_count"] = int(finite_return_count - n_used)

        if n_used < cfg.min_obs:
            base_diag["skipped_reason"] = "too_few_valid_observations"
            diagnostics.append(base_diag)
            continue

        design, design_meta = _build_daily_design(used, config=cfg)
        n_columns = int(design.shape[1])
        base_diag["n_columns"] = n_columns
        base_diag["dropped_constant_continuous_count"] = int(
            design_meta["dropped_constant_continuous_count"]
        )
        base_diag["continuous_columns_used"] = ",".join(design_meta["continuous_columns"])

        if n_used <= n_columns + cfg.min_obs_per_column_buffer:
            base_diag["skipped_reason"] = "too_few_observations_for_design"
            diagnostics.append(base_diag)
            continue

        y = _winsorize_array(used["return_y"].to_numpy(dtype=float), config=cfg)
        weights = np.full(n_used, 1.0 / float(n_used), dtype=float)
        beta, solver_name = _solve_weighted_ridge(
            design,
            y,
            weights,
            ridge=cfg.ridge,
            eps=cfg.eps,
        )
        fitted = design @ beta
        residual_values = y - fitted

        used_codes = used[cfg.stock_col].astype(str).to_numpy()
        residual.loc[date_key, used_codes] = residual_values
        rank_label.loc[date_key, used_codes] = centered_rank(
            pd.Series(residual_values, index=used_codes)
        ).to_numpy(dtype=float)

        day_diag = _daily_diagnostics(
            used,
            y,
            residual_values,
            weights,
            design_meta,
            config=cfg,
        )
        base_diag.update(day_diag)
        base_diag["skipped"] = False
        base_diag["skipped_reason"] = ""
        base_diag["ridge_solver"] = solver_name
        diagnostics.append(base_diag)

    diagnostics_frame = pd.DataFrame(diagnostics)
    summary = _build_summary(
        returns=returns,
        diagnostics=diagnostics_frame,
        exposure_frame=exposure_frame,
        config=cfg,
    )
    return residual, rank_label, diagnostics_frame, summary


def prepare_basic_exposures(
    exposures: pd.DataFrame,
    *,
    config: BasicNeutralizationConfig | None = None,
) -> pd.DataFrame:
    """Validate and normalize basic exposure inputs."""

    cfg = config or BasicNeutralizationConfig()
    required = [
        cfg.date_col,
        cfg.stock_col,
        cfg.industry_col,
        cfg.board_col,
        *cfg.index_cols,
        *cfg.extra_continuous_cols,
        cfg.market_cap_col,
    ]
    missing = [column for column in required if column not in exposures.columns]
    if missing:
        raise ValueError(f"exposures missing required columns: {missing}")

    frame = exposures.copy()
    frame[cfg.date_col] = _normalize_datetime_index(frame[cfg.date_col])
    frame[cfg.stock_col] = frame[cfg.stock_col].map(_normalize_stock_code)
    frame = frame[frame[cfg.date_col].notna() & frame[cfg.stock_col].ne("")]

    for column in (cfg.industry_col, cfg.board_col):
        frame[column] = frame[column].map(_normalize_category)

    for column in (*cfg.index_cols, *cfg.extra_continuous_cols):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame[cfg.market_cap_col] = pd.to_numeric(frame[cfg.market_cap_col], errors="coerce")
    frame = frame.drop_duplicates(subset=[cfg.date_col, cfg.stock_col], keep="last")
    frame = frame.sort_values([cfg.date_col, cfg.stock_col]).reset_index(drop=True)
    frame[cfg.size_rank_col] = np.nan

    for _, idx in frame.groupby(cfg.date_col, sort=False).groups.items():
        market_cap = frame.loc[idx, cfg.market_cap_col].astype(float)
        positive_market_cap = market_cap.where(np.isfinite(market_cap) & (market_cap > 0.0))
        frame.loc[idx, cfg.size_rank_col] = centered_rank(positive_market_cap).to_numpy(dtype=float)

    return frame


def read_table(path: Path) -> pd.DataFrame:
    """Read a DataFrame from pickle, parquet, or csv."""

    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        table = pd.read_pickle(path)
    elif suffix == ".parquet":
        table = pd.read_parquet(path)
    elif suffix == ".csv":
        table = pd.read_csv(path, dtype={"stock_code": str})
    else:
        raise ValueError(f"Unsupported table suffix: {path.suffix!r}")
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(table)!r}")
    return table


def _prepare_return_y(return_y: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(return_y, pd.DataFrame):
        raise TypeError(f"return_y must be a pandas DataFrame, got {type(return_y)!r}")
    frame = return_y.copy()
    frame.index = _normalize_datetime_index(frame.index)
    if frame.index.hasnans:
        raise ValueError("return_y index contains unparseable dates.")
    if not frame.index.is_unique:
        raise ValueError("return_y index contains duplicate dates after normalization.")
    normalized_columns = [_normalize_stock_code(column) for column in frame.columns]
    if len(set(normalized_columns)) != len(normalized_columns):
        raise ValueError("return_y columns contain duplicate stock codes after normalization.")
    frame.columns = normalized_columns
    return frame.sort_index()


def _validate_config(config: BasicNeutralizationConfig) -> None:
    if config.ridge < 0:
        raise ValueError("ridge must be nonnegative.")
    if config.min_obs <= 0:
        raise ValueError("min_obs must be positive.")
    if config.min_obs_per_column_buffer < 0:
        raise ValueError("min_obs_per_column_buffer must be nonnegative.")
    if config.winsor_lower is not None and not 0.0 <= config.winsor_lower < 1.0:
        raise ValueError("winsor_lower must be in [0, 1).")
    if config.winsor_upper is not None and not 0.0 < config.winsor_upper <= 1.0:
        raise ValueError("winsor_upper must be in (0, 1].")
    if (
        config.winsor_lower is not None
        and config.winsor_upper is not None
        and config.winsor_lower >= config.winsor_upper
    ):
        raise ValueError("winsor_lower must be smaller than winsor_upper.")
    if config.eps <= 0:
        raise ValueError("eps must be positive.")


def _neutralization_type(config: BasicNeutralizationConfig) -> str:
    return "exact_neutralization" if config.ridge <= config.eps else "approximate_neutralization"


def _neutralization_label(neutralization_type: str) -> str:
    if neutralization_type == "approximate_neutralization":
        return "近似中性化"
    return "精确中性化"


def _input_return_transform(config: BasicNeutralizationConfig) -> str:
    if config.winsor_lower is None and config.winsor_upper is None:
        return "raw"
    return "winsorized"


def _normalize_datetime_index(values: Sequence[Any] | pd.Index | pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(list(values), errors="coerce")).normalize()


def _normalize_optional_timestamp(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Cannot parse start_date: {value!r}")
    return pd.Timestamp(parsed).normalize()


def _normalize_stock_code(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
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


def _normalize_category(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    return text if text else None


def _valid_daily_rows(day: pd.DataFrame, *, config: BasicNeutralizationConfig) -> np.ndarray:
    valid = day[config.industry_col].notna().to_numpy(copy=True)
    valid &= day[config.board_col].notna().to_numpy()
    market_cap = day[config.market_cap_col].to_numpy(dtype=float, copy=False)
    valid &= np.isfinite(market_cap) & (market_cap > 0.0)
    valid &= np.isfinite(day[config.size_rank_col].to_numpy(dtype=float, copy=False))
    for column in config.index_cols:
        valid &= np.isfinite(day[column].to_numpy(dtype=float, copy=False))
    for column in config.extra_continuous_cols:
        valid &= np.isfinite(day[column].to_numpy(dtype=float, copy=False))
    return valid


def _build_daily_design(
    day: pd.DataFrame,
    *,
    config: BasicNeutralizationConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    blocks: list[np.ndarray] = [np.ones((len(day), 1), dtype=float)]
    categorical_columns: list[str] = []

    for column in (config.industry_col, config.board_col):
        values = day[column].astype(str)
        categories = sorted(values.unique().tolist())
        if len(categories) <= 1:
            continue
        kept = categories[:-1]
        for category in kept:
            blocks.append((values.to_numpy() == category).astype(float).reshape(-1, 1))
            categorical_columns.append(f"{column}[{category}]")

    continuous_columns: list[str] = []
    dropped_constant = 0
    for column in (*config.index_cols, config.size_rank_col, *config.extra_continuous_cols):
        values = day[column].to_numpy(dtype=float, copy=False)
        if _weighted_std(values, None, config.eps) <= config.eps:
            dropped_constant += 1
            continue
        blocks.append(values.reshape(-1, 1))
        continuous_columns.append(column)

    design = np.column_stack(blocks)
    return design, {
        "categorical_columns": categorical_columns,
        "continuous_columns": continuous_columns,
        "dropped_constant_continuous_count": dropped_constant,
    }


def _winsorize_array(values: np.ndarray, *, config: BasicNeutralizationConfig) -> np.ndarray:
    y = values.astype(float, copy=True)
    if config.winsor_lower is not None:
        lo = float(np.quantile(y, config.winsor_lower))
        y = np.maximum(y, lo)
    if config.winsor_upper is not None:
        hi = float(np.quantile(y, config.winsor_upper))
        y = np.minimum(y, hi)
    return y


def _solve_weighted_ridge(
    design: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    ridge: float,
    eps: float,
) -> tuple[np.ndarray, str]:
    sqrt_w = np.sqrt(weights)
    if ridge <= eps:
        beta, *_ = np.linalg.lstsq(design * sqrt_w[:, None], y * sqrt_w, rcond=None)
        return beta, "lstsq"

    lhs = design.T @ (weights[:, None] * design)
    rhs = design.T @ (weights * y)
    penalty = np.full(lhs.shape[0], ridge, dtype=float)
    if penalty.size:
        penalty[0] = 0.0
    lhs = lhs + np.diag(penalty)
    try:
        return np.linalg.solve(lhs, rhs), "solve"
    except np.linalg.LinAlgError:
        beta, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        return beta, "lstsq_normal_equation"


def _daily_diagnostics(
    day: pd.DataFrame,
    y: np.ndarray,
    residual: np.ndarray,
    weights: np.ndarray,
    design_meta: dict[str, Any],
    *,
    config: BasicNeutralizationConfig,
) -> dict[str, Any]:
    y_mean = float(weights @ y)
    centered_y = y - y_mean
    residual_mean = float(weights @ residual)
    residual_std = float(np.sqrt(weights @ ((residual - residual_mean) ** 2)))
    total_var = float(weights @ (centered_y * centered_y))
    residual_var = float(weights @ (residual * residual))
    r2 = np.nan if total_var <= config.eps else 1.0 - residual_var / total_var

    continuous_columns = list(design_meta["continuous_columns"])
    if continuous_columns:
        continuous = day[continuous_columns].to_numpy(dtype=float)
        continuous_exposure = weights @ (continuous * residual[:, None])
        max_abs_continuous = float(np.max(np.abs(continuous_exposure)))
    else:
        max_abs_continuous = 0.0

    return {
        "r2": float(r2) if np.isfinite(r2) else np.nan,
        "return_weighted_mean": y_mean,
        "residual_weighted_mean": residual_mean,
        "residual_weighted_std": residual_std,
        "max_abs_industry_mean_after": _max_abs_group_mean(
            day[config.industry_col].astype(str).to_numpy(),
            residual,
            weights,
        ),
        "max_abs_board_mean_after": _max_abs_group_mean(
            day[config.board_col].astype(str).to_numpy(),
            residual,
            weights,
        ),
        "max_abs_continuous_exposure_after": max_abs_continuous,
    }


def _max_abs_group_mean(values: np.ndarray, x: np.ndarray, weights: np.ndarray) -> float:
    max_abs = 0.0
    for value in np.unique(values):
        mask = values == value
        group_weight = float(np.sum(weights[mask]))
        if group_weight <= 0.0:
            continue
        group_mean = float(np.sum(weights[mask] * x[mask]) / group_weight)
        max_abs = max(max_abs, abs(group_mean))
    return float(max_abs)


def _weighted_std(values: np.ndarray, weights: np.ndarray | None, eps: float) -> float:
    arr = values.astype(float, copy=False)
    if weights is None:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return 0.0
        return float(np.std(finite))
    mean = float(weights @ arr)
    std = float(np.sqrt(weights @ ((arr - mean) ** 2)))
    return 0.0 if std <= eps else std


def _build_summary(
    *,
    returns: pd.DataFrame,
    diagnostics: pd.DataFrame,
    exposure_frame: pd.DataFrame,
    config: BasicNeutralizationConfig,
) -> dict[str, Any]:
    skipped = diagnostics["skipped"].astype(bool) if not diagnostics.empty else pd.Series(dtype=bool)
    processed_count = int((~skipped).sum()) if not diagnostics.empty else 0
    skipped_count = int(skipped.sum()) if not diagnostics.empty else 0
    return {
        "shape": {"rows": int(returns.shape[0]), "columns": int(returns.shape[1])},
        "neutralization_type": _neutralization_type(config),
        "neutralization_label": _neutralization_label(_neutralization_type(config)),
        "input_return_transform": _input_return_transform(config),
        "date_count": int(returns.shape[0]),
        "processed_date_count": processed_count,
        "skipped_date_count": skipped_count,
        "date_min": returns.index.min().date().isoformat() if len(returns.index) else None,
        "date_max": returns.index.max().date().isoformat() if len(returns.index) else None,
        "exposure_rows": int(len(exposure_frame)),
        "config": asdict(config),
        "skipped_reason_counts": (
            diagnostics.loc[skipped, "skipped_reason"].value_counts(dropna=False).to_dict()
            if not diagnostics.empty
            else {}
        ),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, tuple):
        return list(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Neutralize adjusted return_y labels against industry, board, "
            "CSI membership flags, and normalized market-cap rank."
        )
    )
    parser.add_argument("--return-y-adjusted", required=True, type=Path)
    parser.add_argument("--exposures", required=True, type=Path)
    parser.add_argument("--output-residual", required=True, type=Path)
    parser.add_argument("--output-rank-label", required=True, type=Path)
    parser.add_argument("--output-diagnostics", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--output-analysis-dir", type=Path)
    parser.add_argument("--example-symbols", nargs="*", default=())
    parser.add_argument("--start-date")
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--winsor-lower", type=float, default=0.01)
    parser.add_argument("--winsor-upper", type=float, default=0.99)
    parser.add_argument("--extra-continuous-cols", nargs="*", default=())
    parser.add_argument("--min-obs", type=int, default=30)
    parser.add_argument("--min-obs-per-column-buffer", type=int, default=10)
    args = parser.parse_args()

    return_y = read_table(args.return_y_adjusted)
    exposures = read_table(args.exposures)
    config = BasicNeutralizationConfig(
        start_date=args.start_date,
        ridge=args.ridge,
        winsor_lower=args.winsor_lower,
        winsor_upper=args.winsor_upper,
        extra_continuous_cols=tuple(args.extra_continuous_cols),
        min_obs=args.min_obs,
        min_obs_per_column_buffer=args.min_obs_per_column_buffer,
    )
    residual, rank_label, diagnostics, summary = neutralize_return_y_matrix(
        return_y,
        exposures,
        config=config,
    )

    args.output_residual.parent.mkdir(parents=True, exist_ok=True)
    args.output_rank_label.parent.mkdir(parents=True, exist_ok=True)
    args.output_diagnostics.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)

    residual.to_pickle(args.output_residual)
    rank_label.to_pickle(args.output_rank_label)
    diagnostics.to_csv(args.output_diagnostics, index=False, encoding="utf-8-sig")

    if args.output_analysis_dir is not None:
        try:
            from analysis.return_y_neutralization_analysis import (
                write_neutralization_analysis_artifacts,
            )
        except ModuleNotFoundError:
            from return_y_neutralization_analysis import (
                write_neutralization_analysis_artifacts,
            )

        artifacts = write_neutralization_analysis_artifacts(
            return_y,
            residual,
            rank_label,
            diagnostics,
            output_dir=args.output_analysis_dir,
            example_symbols=tuple(args.example_symbols),
        )
        summary["analysis_outputs"] = {
            "summary_json": artifacts["summary_json"],
            "report_html": artifacts["report_html"],
            "charts": artifacts["charts"],
        }

    args.output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
