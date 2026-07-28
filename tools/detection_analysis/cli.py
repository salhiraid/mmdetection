"""Command-line orchestration for detection analysis."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Sequence

from tools.detection_analysis.config import AnalysisConfig
from tools.detection_analysis.evaluation.coco_evaluator import evaluate_coco
from tools.detection_analysis.evaluation.custom_metrics import compute_custom_metrics
from tools.detection_analysis.evaluation.detector_comparison import compare_detectors
from tools.detection_analysis.evaluation.matcher import match_detector
from tools.detection_analysis.export.csv_exporter import export_bundle_csv
from tools.detection_analysis.export.html_report import write_html_report
from tools.detection_analysis.export.image_exporter import export_bundle_images
from tools.detection_analysis.loaders.dataset_loader import load_dataset
from tools.detection_analysis.loaders.result_adapter import normalize_result_file
from tools.detection_analysis.models import AnalysisBundle, DetectorAnalysis
from tools.detection_analysis.utils.serialization import dump_json


def build_parser() -> argparse.ArgumentParser:
    """Create the shared CLI parser."""

    parser = argparse.ArgumentParser(description="Analyze object detection results.")
    parser.add_argument("--config", help="MMDetection config containing test/val dataset.")
    parser.add_argument("--ann-file", help="COCO annotation file for manual dataset loading.")
    parser.add_argument("--img-prefix", help="Image directory/prefix for manual COCO loading.")
    parser.add_argument("--dataset-type", default="CocoDataset", help="Manual dataset type. Currently CocoDataset.")
    parser.add_argument("--classes", nargs="*", help="Class names for manual loading.")
    parser.add_argument("--results", nargs="+", required=False, help="Detector result .pkl/.json files.")
    parser.add_argument("--names", nargs="*", help="Optional detector display names.")
    parser.add_argument("--output-dir", default="work_dirs/detection_analysis/latest", help="Directory for saved analysis.")
    parser.add_argument("--split", default="test", choices=["test", "val"], help="Config dataloader split.")
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--tp-iou-threshold", type=float, default=0.5)
    parser.add_argument("--localization-iou-min", type=float, default=0.1)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=0.5)
    parser.add_argument("--classification-iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections-per-image", type=int, default=100)
    parser.add_argument("--class-agnostic-matching", action="store_true")
    parser.add_argument("--include-crowd", action="store_true", help="Do not ignore crowd annotations during matching.")
    parser.add_argument("--skip-coco", action="store_true", help="Skip pycocotools COCO metric computation.")
    return parser


def config_from_args(args: argparse.Namespace) -> AnalysisConfig:
    """Create an ``AnalysisConfig`` from parsed arguments."""

    cfg = AnalysisConfig(
        confidence_threshold=args.confidence_threshold,
        tp_iou_threshold=args.tp_iou_threshold,
        localization_iou_min=args.localization_iou_min,
        duplicate_iou_threshold=args.duplicate_iou_threshold,
        classification_iou_threshold=args.classification_iou_threshold,
        max_detections_per_image=args.max_detections_per_image,
        class_agnostic_matching=args.class_agnostic_matching,
        ignore_crowd=not args.include_crowd,
    )
    cfg.validate()
    return cfg


def run_analysis_from_args(args: argparse.Namespace) -> AnalysisBundle:
    """Run preprocessing from argparse inputs."""

    if not args.results:
        raise ValueError("--results is required for preprocessing.")
    names = _detector_names(args.results, args.names)
    cfg = config_from_args(args)
    dataset = load_dataset(
        config_path=args.config,
        ann_file=args.ann_file,
        img_prefix=args.img_prefix,
        dataset_type=args.dataset_type,
        classes=args.classes,
        split=args.split,
        analysis_config=cfg,
    )
    detectors: List[DetectorAnalysis] = []
    for result_path, name in zip(args.results, names):
        predictions = normalize_result_file(result_path, dataset, detector_name=name)
        pred_matches, gt_matches = match_detector(predictions, dataset.ground_truths, dataset.images, cfg)
        metrics = compute_custom_metrics(pred_matches, gt_matches, dataset.classes, len(dataset.images))
        coco_metrics = {} if args.skip_coco else evaluate_coco(dataset, predictions)
        detectors.append(DetectorAnalysis(
            detector_name=name,
            predictions=predictions,
            prediction_matches=pred_matches,
            ground_truth_matches=gt_matches,
            metrics=metrics,
            coco_metrics=coco_metrics,
        ))
    bundle = AnalysisBundle(
        dataset=dataset,
        detectors=detectors,
        config=asdict(cfg),
        comparison=compare_detectors(detectors) if len(detectors) > 1 else {},
    )
    save_analysis(bundle, args.output_dir)
    return bundle


def save_analysis(bundle: AnalysisBundle, output_dir: str | Path) -> None:
    """Persist a full analysis directory."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump_json(out / "analysis.json", bundle)
    dump_json(out / "dataset.json", bundle.dataset)
    dump_json(out / "config.json", bundle.config)
    dump_json(out / "comparison.json", bundle.comparison)
    for detector in bundle.detectors:
        det_dir = out / "detectors" / detector.detector_name
        dump_json(det_dir / "predictions.json", detector.predictions)
        dump_json(det_dir / "prediction_matches.json", detector.prediction_matches)
        dump_json(det_dir / "ground_truth_matches.json", detector.ground_truth_matches)
        dump_json(det_dir / "metrics.json", detector.metrics)
        dump_json(det_dir / "coco_metrics.json", detector.coco_metrics)
    export_bundle_csv(bundle, out / "exports")
    export_bundle_images(bundle, out / "visualizations")
    write_html_report(bundle, out / "report.html")


def _detector_names(results: Sequence[str], names: Optional[Sequence[str]]) -> List[str]:
    if names:
        if len(names) != len(results):
            raise ValueError("--names length must match --results length.")
        return [str(n) for n in names]
    return [Path(path).stem for path in results]


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the analysis CLI."""

    args = build_parser().parse_args(argv)
    bundle = run_analysis_from_args(args)
    print(f"Wrote analysis for {len(bundle.detectors)} detector(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
