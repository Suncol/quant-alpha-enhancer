from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence


DEFAULT_OUTPUT_SPLIT = "full_chronological"
DEFAULT_OUTPUT_FOLD_ID = 0
DEFAULT_SCORE_COL = "pred_direct"
SELECTION_POLICY = "first_fold_train_valid_test_then_later_fold_tests"


@dataclass(frozen=True)
class SourceSegment:
    fold_id: int
    split: str
    start: date
    end: date
    expected_date_count: int

    @property
    def key(self) -> tuple[int, str]:
        return (self.fold_id, self.split)

    def contains(self, value: date) -> bool:
        return self.start <= value <= self.end

    def to_manifest_record(self, *, actual_date_count: int, row_count: int) -> dict[str, Any]:
        record = asdict(self)
        record["start"] = self.start.isoformat()
        record["end"] = self.end.isoformat()
        record["actual_date_count"] = int(actual_date_count)
        record["row_count"] = int(row_count)
        return record


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description=(
            "Build a single non-overlapping chronological predictions file for "
            "rank-bucket NAV evaluation from rolling-fold LGBM predictions."
        )
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--training-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--score-col", default=DEFAULT_SCORE_COL)
    parser.add_argument("--output-fold-id", default=DEFAULT_OUTPUT_FOLD_ID, type=int)
    parser.add_argument("--output-split", default=DEFAULT_OUTPUT_SPLIT)
    args = parser.parse_args(argv)

    summary = build_chronological_predictions(
        predictions_path=args.predictions,
        training_summary_path=args.training_summary,
        output_dir=args.output_dir,
        score_col=args.score_col,
        output_fold_id=args.output_fold_id,
        output_split=args.output_split,
    )
    if argv is None:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_chronological_predictions(
    *,
    predictions_path: Path,
    training_summary_path: Path,
    output_dir: Path,
    score_col: str = DEFAULT_SCORE_COL,
    output_fold_id: int = DEFAULT_OUTPUT_FOLD_ID,
    output_split: str = DEFAULT_OUTPUT_SPLIT,
) -> dict[str, Any]:
    segments = source_segments_from_training_summary(training_summary_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_predictions = output_dir / "predictions.csv"
    manifest_path = output_dir / "selection_manifest.json"

    manifest = write_selected_predictions(
        predictions_path=predictions_path,
        output_predictions_path=output_predictions,
        segments=segments,
        score_col=score_col,
        output_fold_id=output_fold_id,
        output_split=output_split,
    )
    manifest.update(
        {
            "schema_version": "chronological_rank_bucket_predictions_v1",
            "selection_policy": SELECTION_POLICY,
            "source_predictions": _path_for_summary(predictions_path),
            "training_summary": _path_for_summary(training_summary_path),
            "output_predictions": _path_for_summary(output_predictions),
            "manifest": _path_for_summary(manifest_path),
            "notes": [
                "The first fold contributes train, valid, and test dates.",
                "Later folds contribute only test dates, so earlier dates are not replaced by models trained on later data.",
                "The resulting file is one synthetic fold/split path for NAV evaluation; source_fold_id and source_split preserve provenance.",
                "Fold 1 train rows are in-sample and fold 1 valid rows are validation-period predictions, not pure out-of-sample test results.",
            ],
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def source_segments_from_training_summary(training_summary_path: Path) -> list[SourceSegment]:
    summary = json.loads(
        training_summary_path.read_text(encoding="utf-8"),
        parse_constant=lambda _constant: None,
    )
    folds = sorted(summary.get("folds", []), key=lambda item: int(item["fold_id"]))
    if not folds:
        raise ValueError("Training summary does not contain any fold records.")

    segments: list[SourceSegment] = []
    for fold_position, fold in enumerate(folds):
        fold_id = int(fold["fold_id"])
        date_ranges = fold.get("date_ranges") or {}
        splits = ("train", "valid", "test") if fold_position == 0 else ("test",)
        for split in splits:
            if split not in date_ranges:
                raise ValueError(f"Fold {fold_id} is missing date range for split {split!r}.")
            range_record = date_ranges[split]
            segments.append(
                SourceSegment(
                    fold_id=fold_id,
                    split=split,
                    start=_parse_date(range_record["start"]),
                    end=_parse_date(range_record["end"]),
                    expected_date_count=int(range_record["date_count"]),
                )
            )
    _validate_non_overlapping_segments(segments)
    return segments


def write_selected_predictions(
    *,
    predictions_path: Path,
    output_predictions_path: Path,
    segments: Sequence[SourceSegment],
    score_col: str,
    output_fold_id: int,
    output_split: str,
) -> dict[str, Any]:
    if not segments:
        raise ValueError("At least one source segment is required.")

    segments_by_key: dict[tuple[int, str], SourceSegment] = {segment.key: segment for segment in segments}
    selected_dates: dict[tuple[int, str], set[str]] = {segment.key: set() for segment in segments}
    row_counts: dict[tuple[int, str], int] = {segment.key: 0 for segment in segments}
    selected_keys: set[tuple[str, str]] = set()

    temp_path = output_predictions_path.with_suffix(output_predictions_path.suffix + ".tmp")
    try:
        with predictions_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise ValueError("Predictions file has no header row.")
            fieldnames = _normalized_fieldnames(reader.fieldnames)
            required = {"fold_id", "split", "date", "stock_code", score_col}
            missing = sorted(required.difference(fieldnames))
            if missing:
                raise ValueError(f"Predictions are missing required columns: {missing}")

            output_fieldnames = list(fieldnames)
            for provenance_col in ("source_fold_id", "source_split"):
                if provenance_col not in output_fieldnames:
                    output_fieldnames.append(provenance_col)

            with temp_path.open("w", encoding="utf-8-sig", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=output_fieldnames)
                writer.writeheader()
                for raw_row in reader:
                    row = _normalize_row_keys(raw_row)
                    fold_id = int(row["fold_id"])
                    split = str(row["split"])
                    segment = segments_by_key.get((fold_id, split))
                    if segment is None:
                        continue
                    signal_date = _parse_date(row["date"])
                    if not segment.contains(signal_date):
                        continue

                    stock_code = _normalize_stock_code(row["stock_code"])
                    if not stock_code:
                        continue
                    duplicate_key = (signal_date.isoformat(), stock_code)
                    if duplicate_key in selected_keys:
                        raise ValueError(
                            "Selected predictions contain duplicate date/stock_code rows after "
                            f"fold stitching: {duplicate_key}"
                        )
                    selected_keys.add(duplicate_key)

                    selected_dates[segment.key].add(signal_date.isoformat())
                    row_counts[segment.key] += 1
                    row["source_fold_id"] = row["fold_id"]
                    row["source_split"] = row["split"]
                    row["fold_id"] = str(output_fold_id)
                    row["split"] = output_split
                    writer.writerow(row)

        segment_records: list[dict[str, Any]] = []
        for segment in segments:
            actual_date_count = len(selected_dates[segment.key])
            if actual_date_count != segment.expected_date_count:
                raise ValueError(
                    "Selected date coverage does not match training_summary for "
                    f"fold {segment.fold_id} {segment.split}: expected "
                    f"{segment.expected_date_count}, got {actual_date_count}."
                )
            segment_records.append(
                segment.to_manifest_record(
                    actual_date_count=actual_date_count,
                    row_count=row_counts[segment.key],
                )
            )
        temp_path.replace(output_predictions_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    all_dates = sorted({value for values in selected_dates.values() for value in values})
    return {
        "score_col": score_col,
        "output_fold_id": int(output_fold_id),
        "output_split": output_split,
        "selected_segments": segment_records,
        "row_counts": {
            "selected_prediction_rows": int(sum(row_counts.values())),
            "selected_unique_date_stock_rows": int(len(selected_keys)),
            "selected_dates": int(len(all_dates)),
        },
        "date_range": {
            "start": all_dates[0] if all_dates else None,
            "end": all_dates[-1] if all_dates else None,
        },
    }


def _validate_non_overlapping_segments(segments: Sequence[SourceSegment]) -> None:
    ordered = sorted(segments, key=lambda segment: (segment.start, segment.end, segment.fold_id))
    for previous, current in zip(ordered, ordered[1:]):
        if current.start <= previous.end:
            raise ValueError(
                "Source segments overlap in calendar time: "
                f"fold {previous.fold_id} {previous.split} {previous.start}..{previous.end} and "
                f"fold {current.fold_id} {current.split} {current.start}..{current.end}."
            )


def _normalized_fieldnames(fieldnames: Sequence[str]) -> list[str]:
    return [fieldname.lstrip("\ufeff") for fieldname in fieldnames]


def _normalize_row_keys(row: dict[str, str]) -> dict[str, str]:
    return {str(key).lstrip("\ufeff"): value for key, value in row.items()}


def _normalize_stock_code(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if "." in text:
        prefix = text.split(".", 1)[0]
        if prefix.isdigit():
            text = prefix
    if text.isdigit():
        return text.zfill(6)
    return text


def _parse_date(value: object) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def _path_for_summary(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    main()
