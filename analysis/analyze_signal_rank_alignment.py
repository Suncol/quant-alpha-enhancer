from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    from analysis.evaluate_lgbm_rank_bucket_nav import (
        _finite_values,
        _normalize_stock_code,
        _path_for_summary,
        _write_csv_with_iso_dates,
    )
    from analysis.summarize_rank_bucket_periods import PeriodSpec, parse_period_spec
except ModuleNotFoundError:  # Allows direct execution via python analysis/script.py.
    from evaluate_lgbm_rank_bucket_nav import (
        _finite_values,
        _normalize_stock_code,
        _path_for_summary,
        _write_csv_with_iso_dates,
    )
    from summarize_rank_bucket_periods import PeriodSpec, parse_period_spec


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    predictions: pd.DataFrame
    score_col: str


@dataclass(frozen=True)
class CandidatePathSpec:
    label: str
    path: Path
    score_col: str


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a reference alpha signal rank with one or more prediction-score "
            "rankings over daily cross sections."
        )
    )
    parser.add_argument("--reference-panel", required=True, type=Path)
    parser.add_argument("--reference-col", required=True)
    parser.add_argument(
        "--candidate",
        nargs="+",
        required=True,
        help="Candidate spec formatted as label|predictions_path|score_col.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--periods", nargs="+", required=True)
    parser.add_argument("--rank-bucket-count", default=31, type=int)
    parser.add_argument("--splits", nargs="*", default=["test"])
    args = parser.parse_args(argv)

    reference_panel = pd.read_pickle(args.reference_panel)
    candidates = [
        CandidateSpec(
            label=spec.label,
            predictions=_read_predictions(spec.path),
            score_col=spec.score_col,
        )
        for spec in (_parse_candidate_spec(value) for value in args.candidate)
    ]
    summary = write_signal_rank_alignment_artifacts(
        reference_panel=reference_panel,
        candidates=candidates,
        reference_col=args.reference_col,
        periods=[parse_period_spec(value) for value in args.periods],
        output_dir=args.output_dir,
        rank_bucket_count=args.rank_bucket_count,
        splits=args.splits,
    )
    if argv is None:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def compare_signal_rank_alignment(
    *,
    reference_panel: pd.DataFrame,
    candidates: Sequence[CandidateSpec],
    reference_col: str,
    periods: Sequence[PeriodSpec],
    rank_bucket_count: int = 31,
    splits: Sequence[str] | None = ("test",),
) -> dict[str, pd.DataFrame]:
    if rank_bucket_count < 1:
        raise ValueError("rank_bucket_count must be positive.")
    if not candidates:
        raise ValueError("At least one candidate must be supplied.")
    if not periods:
        raise ValueError("At least one period must be supplied.")

    reference = _prepare_reference_panel(reference_panel, reference_col=reference_col)
    daily_frames = [
        _compute_candidate_daily_alignment(
            reference,
            _prepare_candidate_predictions(candidate.predictions, candidate.score_col),
            candidate_label=candidate.label,
            candidate_score_col=candidate.score_col,
            reference_col=reference_col,
            rank_bucket_count=rank_bucket_count,
            splits=splits,
        )
        for candidate in candidates
    ]
    daily = _concat_frames(daily_frames)
    period_summary = summarize_alignment_periods(
        daily,
        periods=periods,
        splits=splits,
    )
    return {
        "daily_alignment": daily,
        "period_summary": period_summary,
    }


def write_signal_rank_alignment_artifacts(
    *,
    reference_panel: pd.DataFrame,
    candidates: Sequence[CandidateSpec],
    reference_col: str,
    periods: Sequence[PeriodSpec],
    output_dir: Path,
    rank_bucket_count: int = 31,
    splits: Sequence[str] | None = ("test",),
) -> dict[str, Any]:
    result = compare_signal_rank_alignment(
        reference_panel=reference_panel,
        candidates=candidates,
        reference_col=reference_col,
        periods=periods,
        rank_bucket_count=rank_bucket_count,
        splits=splits,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "daily_alignment": output_dir / "signal_rank_alignment_daily.csv",
        "period_summary": output_dir / "signal_rank_alignment_period_summary.csv",
        "evaluation_summary": output_dir / "signal_rank_alignment_summary.json",
    }
    _write_csv_with_iso_dates(result["daily_alignment"], paths["daily_alignment"])
    _write_csv_with_iso_dates(result["period_summary"], paths["period_summary"])
    summary = {
        "schema_version": "signal_rank_alignment_v1",
        "metric_contract": {
            "reference_col": reference_col,
            "join_key": ["date", "stock_code"],
            "candidate_key": ["fold_id", "split", "date", "stock_code"],
            "rank_scope": "daily_cross_section_within_fold_split",
            "rank_order": "descending",
            "top_tail_definition": "bucket 01 from daily equal-count rank_bucket_count",
            "top_overlap_rate": "top_overlap_count / top_count",
            "spearman_rank_corr": "Pearson correlation of average ranks within each date",
        },
        "rank_bucket_count": int(rank_bucket_count),
        "splits": list(splits) if splits is not None else None,
        "periods": [
            {
                "label": str(period.label),
                "start": period.start_ts.date().isoformat(),
                "end": period.end_ts.date().isoformat(),
            }
            for period in periods
        ],
        "candidates": [
            {
                "label": str(candidate.label),
                "score_col": str(candidate.score_col),
            }
            for candidate in candidates
        ],
        "row_counts": {
            "daily_alignment": int(len(result["daily_alignment"])),
            "period_summary": int(len(result["period_summary"])),
        },
        "outputs": {key: _path_for_summary(path) for key, path in paths.items()},
    }
    paths["evaluation_summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def summarize_alignment_periods(
    daily_alignment: pd.DataFrame,
    *,
    periods: Sequence[PeriodSpec],
    splits: Sequence[str] | None = ("test",),
) -> pd.DataFrame:
    if daily_alignment.empty:
        return _empty_period_summary()
    frame = daily_alignment.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise").dt.normalize()
    frame["split"] = frame["split"].astype(str)
    if splits is not None:
        split_set = {str(split) for split in splits}
        frame = frame[frame["split"].isin(split_set)].copy()

    assigned_frames: list[pd.DataFrame] = []
    for period in periods:
        mask = frame["signal_date"].between(period.start_ts, period.end_ts, inclusive="both")
        period_frame = frame.loc[mask].copy()
        if period_frame.empty:
            continue
        period_frame["period_label"] = str(period.label)
        period_frame["period_start"] = period.start_ts
        period_frame["period_end"] = period.end_ts
        assigned_frames.append(period_frame)
    if not assigned_frames:
        return _empty_period_summary()

    assigned = pd.concat(assigned_frames, ignore_index=True)
    records: list[dict[str, Any]] = []
    group_cols = [
        "period_label",
        "period_start",
        "period_end",
        "candidate_label",
        "candidate_score_col",
    ]
    for key, group in assigned.groupby(group_cols, sort=True):
        period_label, period_start, period_end, candidate_label, candidate_score_col = key
        spearman = _finite_values(group["spearman_rank_corr"])
        overlap = _finite_values(group["top_overlap_rate"])
        records.append(
            {
                "period_label": str(period_label),
                "period_start": pd.Timestamp(period_start),
                "period_end": pd.Timestamp(period_end),
                "actual_start": pd.Timestamp(group["signal_date"].min()),
                "actual_end": pd.Timestamp(group["signal_date"].max()),
                "candidate_label": str(candidate_label),
                "candidate_score_col": str(candidate_score_col),
                "date_count": int(len(group)),
                "mean_spearman_rank_corr": _nanmean(spearman),
                "median_spearman_rank_corr": _nanmedian(spearman),
                "mean_top_overlap_rate": _nanmean(overlap),
                "median_top_overlap_rate": _nanmedian(overlap),
                "mean_top_overlap_count": _nanmean(group["top_overlap_count"]),
                "mean_top_count": _nanmean(group["top_count"]),
                "mean_eligible_count": _nanmean(group["eligible_count"]),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["period_start", "period_label", "candidate_label"]
    ).reset_index(drop=True)


def _compute_candidate_daily_alignment(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    candidate_label: str,
    candidate_score_col: str,
    reference_col: str,
    rank_bucket_count: int,
    splits: Sequence[str] | None,
) -> pd.DataFrame:
    merged = candidate.merge(
        reference,
        on=["date", "stock_code"],
        how="left",
        validate="many_to_one",
    )
    missing_reference = merged[reference_col].isna()
    finite_candidate = np.isfinite(
        pd.to_numeric(merged[candidate_score_col], errors="coerce").to_numpy(dtype=float)
    )
    bad = missing_reference.to_numpy(dtype=bool) & finite_candidate
    if bad.any():
        sample = merged.loc[bad, ["fold_id", "split", "date", "stock_code"]].head(10).copy()
        sample["date"] = pd.to_datetime(sample["date"]).dt.date.astype(str)
        raise ValueError(
            "Candidate predictions contain finite-score rows without matching reference scores: "
            f"{sample.to_dict('records')}"
        )
    if splits is not None:
        split_set = {str(split) for split in splits}
        merged = merged[merged["split"].isin(split_set)].copy()

    records: list[dict[str, Any]] = []
    for (fold_id, split, signal_date), group in merged.groupby(["fold_id", "split", "date"], sort=True):
        ref = pd.to_numeric(group[reference_col], errors="coerce")
        cand = pd.to_numeric(group[candidate_score_col], errors="coerce")
        finite = np.isfinite(ref.to_numpy(dtype=float)) & np.isfinite(cand.to_numpy(dtype=float))
        aligned = group.loc[finite, ["stock_code"]].copy()
        aligned[reference_col] = ref.loc[finite].astype(float)
        aligned[candidate_score_col] = cand.loc[finite].astype(float)
        aligned = aligned.sort_values("stock_code").reset_index(drop=True)
        eligible_count = int(len(aligned))
        if eligible_count < 2:
            spearman = np.nan
        else:
            spearman = float(aligned[reference_col].corr(aligned[candidate_score_col], method="spearman"))

        top_count = _top_bucket_count(eligible_count, rank_bucket_count)
        ref_top = _top_stock_set(aligned, reference_col, top_count)
        cand_top = _top_stock_set(aligned, candidate_score_col, top_count)
        overlap_count = int(len(ref_top.intersection(cand_top)))
        overlap_rate = float(overlap_count / top_count) if top_count > 0 else np.nan
        records.append(
            {
                "fold_id": int(fold_id),
                "split": str(split),
                "signal_date": pd.Timestamp(signal_date),
                "candidate_label": str(candidate_label),
                "candidate_score_col": str(candidate_score_col),
                "reference_col": str(reference_col),
                "eligible_count": eligible_count,
                "top_count": top_count,
                "top_overlap_count": overlap_count,
                "top_overlap_rate": overlap_rate,
                "spearman_rank_corr": spearman,
            }
        )
    if not records:
        return _empty_daily_alignment()
    return pd.DataFrame(records).sort_values(
        ["fold_id", "split", "signal_date", "candidate_label"]
    ).reset_index(drop=True)


def _prepare_reference_panel(reference_panel: pd.DataFrame, *, reference_col: str) -> pd.DataFrame:
    required = {"date", "stock_code", reference_col}
    missing = sorted(required.difference(reference_panel.columns))
    if missing:
        raise ValueError(f"Reference panel is missing required columns: {missing}")
    frame = reference_panel[["date", "stock_code", reference_col]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["stock_code"] = frame["stock_code"].map(_normalize_stock_code)
    invalid = frame["date"].isna() | frame["stock_code"].eq("")
    if invalid.any():
        sample = frame.loc[invalid, ["date", "stock_code"]].head(5)
        raise ValueError(f"Reference panel contains invalid keys: {sample.to_dict('records')}")
    duplicate_keys = ["date", "stock_code"]
    if frame.duplicated(duplicate_keys).any():
        sample = frame.loc[frame.duplicated(duplicate_keys, keep=False), duplicate_keys].head(5)
        raise ValueError(f"Reference panel contains duplicate date/stock_code rows: {sample.to_dict('records')}")
    frame[reference_col] = pd.to_numeric(frame[reference_col], errors="coerce")
    return frame.sort_values(duplicate_keys).reset_index(drop=True)


def _prepare_candidate_predictions(predictions: pd.DataFrame, score_col: str) -> pd.DataFrame:
    required = {"fold_id", "split", "date", "stock_code", score_col}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Candidate predictions are missing required columns: {missing}")
    frame = predictions[["fold_id", "split", "date", "stock_code", score_col]].copy()
    frame["fold_id"] = pd.to_numeric(frame["fold_id"], errors="raise").astype(int)
    frame["split"] = frame["split"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["stock_code"] = frame["stock_code"].map(_normalize_stock_code)
    invalid = frame["date"].isna() | frame["stock_code"].eq("")
    if invalid.any():
        sample = frame.loc[invalid, ["fold_id", "split", "date", "stock_code"]].head(5)
        raise ValueError(f"Candidate predictions contain invalid keys: {sample.to_dict('records')}")
    duplicate_keys = ["fold_id", "split", "date", "stock_code"]
    if frame.duplicated(duplicate_keys).any():
        sample = frame.loc[frame.duplicated(duplicate_keys, keep=False), duplicate_keys].head(5)
        raise ValueError(f"Candidate predictions contain duplicate rows: {sample.to_dict('records')}")
    frame[score_col] = pd.to_numeric(frame[score_col], errors="coerce")
    return frame.sort_values(duplicate_keys).reset_index(drop=True)


def _read_predictions(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    return pd.read_csv(path, dtype={"stock_code": "string"})


def _parse_candidate_spec(value: str) -> CandidatePathSpec:
    parts = value.split("|")
    if len(parts) != 3:
        raise ValueError("Candidate specs must be formatted as label|predictions_path|score_col.")
    label, path, score_col = (part.strip() for part in parts)
    if not label or not path or not score_col:
        raise ValueError("Candidate label, path, and score_col must be non-empty.")
    return CandidatePathSpec(label=label, path=Path(path), score_col=score_col)


def _top_bucket_count(eligible_count: int, rank_bucket_count: int) -> int:
    if eligible_count <= 0:
        return 0
    return int(len(np.array_split(np.arange(eligible_count), rank_bucket_count)[0]))


def _top_stock_set(frame: pd.DataFrame, score_col: str, top_count: int) -> set[str]:
    if top_count <= 0:
        return set()
    ranked = frame.sort_values(
        [score_col, "stock_code"],
        ascending=[False, True],
        kind="mergesort",
    )
    return set(ranked.head(top_count)["stock_code"].astype(str).tolist())


def _concat_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    materialized = [frame for frame in frames if frame is not None and not frame.empty]
    if not materialized:
        return pd.DataFrame()
    return pd.concat(materialized, ignore_index=True)


def _nanmean(values: Any) -> float:
    finite = _finite_values(values)
    return float(np.mean(finite)) if len(finite) else np.nan


def _nanmedian(values: Any) -> float:
    finite = _finite_values(values)
    return float(np.median(finite)) if len(finite) else np.nan


def _empty_daily_alignment() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "fold_id",
            "split",
            "signal_date",
            "candidate_label",
            "candidate_score_col",
            "reference_col",
            "eligible_count",
            "top_count",
            "top_overlap_count",
            "top_overlap_rate",
            "spearman_rank_corr",
        ]
    )


def _empty_period_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "period_label",
            "period_start",
            "period_end",
            "actual_start",
            "actual_end",
            "candidate_label",
            "candidate_score_col",
            "date_count",
            "mean_spearman_rank_corr",
            "median_spearman_rank_corr",
            "mean_top_overlap_rate",
            "median_top_overlap_rate",
            "mean_top_overlap_count",
            "mean_top_count",
            "mean_eligible_count",
        ]
    )


if __name__ == "__main__":
    main()
