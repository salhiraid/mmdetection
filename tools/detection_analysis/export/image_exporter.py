"""Annotated image export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from tools.detection_analysis.models import (
    AnalysisBundle,
    GroundTruthMatchRecord,
    ImageRecord,
    MatchRecord,
)
from tools.detection_analysis.visualization.box_renderer import render_image


MIN_VISUALIZATION_CONFIDENCE = 0.35


def export_annotated_image(image: ImageRecord, matches: Iterable[MatchRecord], output_path: str | Path) -> None:
    """Render and save one annotated image."""

    rendered = render_image(image, list(matches))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)


def export_bundle_images(bundle: AnalysisBundle, output_dir: str | Path) -> None:
    """Save full-resolution prediction-only and problem-only overlays.

    Prediction overlays never contain ground-truth boxes. Problem overlays contain
    erroneous predictions and false-negative ground truths. Predictions below
    0.35 (or the analysis threshold when it is higher) are omitted from both.
    """

    out = Path(output_dir)
    confidence = max(
        MIN_VISUALIZATION_CONFIDENCE,
        float(bundle.config.get("confidence_threshold", MIN_VISUALIZATION_CONFIDENCE)),
    )
    for detector in bundle.detectors:
        predictions_by_image = _by_image([
            match for match in detector.prediction_matches
            if match.score >= confidence
        ])
        problems_by_image = _by_image([
            match for match in detector.prediction_matches
            if match.score >= confidence and match.status != "true_positive"
        ])
        false_negatives_by_image = _by_image([
            match for match in detector.ground_truth_matches
            if match.status == "false_negative"
        ])

        for image in bundle.dataset.images:
            filename = _output_filename(image)
            predictions = predictions_by_image.get(image.image_id, [])
            prediction_image = render_image(
                image,
                predictions,
                label_mode="class",
                color_mode="class",
                show_iou=False,
                show_detector=False,
            )
            _save_image(
                prediction_image,
                out / detector.detector_name / "predictions_only" / filename,
            )

            problems = problems_by_image.get(image.image_id, [])
            false_negatives = false_negatives_by_image.get(image.image_id, [])
            if problems or false_negatives:
                problem_image = render_image(
                    image,
                    problems,
                    false_negatives,
                    label_mode="class_problem",
                    color_mode="error",
                    show_iou=True,
                    show_detector=False,
                )
                _save_image(
                    problem_image,
                    out / detector.detector_name / "problems_only" / filename,
                )


def _by_image(records: Iterable[MatchRecord | GroundTruthMatchRecord]) -> dict[str, List]:
    grouped: dict[str, List] = {}
    for record in records:
        grouped.setdefault(record.image_id, []).append(record)
    return grouped


def _output_filename(image: ImageRecord) -> str:
    """Return a stable, collision-resistant filename with a supported suffix."""

    source = Path(image.file_name).name
    suffix = Path(source).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
        source = f"{Path(source).stem}.png"
    return f"{image.image_index:06d}_{source}"


def _save_image(image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
