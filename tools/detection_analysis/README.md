# Detection Analysis Tool

This directory contains a standalone object detection evaluation and error-analysis framework for MMDetection projects. It is isolated under `tools/detection_analysis/` and does not modify detector training, inference, or MMDetection source code.

## Purpose

The tool loads one or more saved detector result files, loads the matching test dataset, normalizes predictions, matches predictions to ground truth, computes COCO and custom metrics, compares detectors, visualizes errors image by image, and exports reusable analysis artifacts.

## Installation

From the MMDetection repository root:

```bash
pip install -r tools/detection_analysis/requirements.txt
pip install -e .
```

The core preprocessing logic uses MMDetection, MMEngine, NumPy, Pandas, Pillow, and optional `pycocotools`. The GUI additionally uses Streamlit and Plotly.

## Supported Inputs

Dataset inputs:

- MMDetection config with `test_dataloader` or `val_dataloader`.
- Manual COCO JSON annotation file with an optional image prefix and class list.

Result inputs:

- MMDetection `.pkl` files loaded through MMEngine.
- MMDetection 3.x `DetDataSample` or dict results with `pred_instances`.
- Dicts containing `bboxes`, `scores`, and `labels`.
- Older MMDetection per-image, per-class `Nx5` arrays.
- COCO JSON prediction lists containing `image_id`, `bbox`, `score`, and `category_id`.

All formats are normalized to records with `image_id`, `bbox_xyxy`, `score`, `class_id`, `class_name`, and `detector_name`.

## Preprocessing Examples

One detector:

```bash
python tools/detection_analysis/analyze.py \
  --config configs/my_detector_config.py \
  --results results/detector_a.pkl \
  --output-dir work_dirs/detection_analysis/run_001
```

Two detectors:

```bash
python tools/detection_analysis/analyze.py \
  --config configs/my_detector_config.py \
  --results results/detector_a.pkl results/detector_b.pkl \
  --names Detector_A Detector_B \
  --output-dir work_dirs/detection_analysis/run_002
```

Multiple detectors:

```bash
python tools/detection_analysis/analyze.py \
  --config configs/my_detector_config.py \
  --results results/a.pkl results/b.pkl results/c.pkl \
  --names A B C \
  --output-dir work_dirs/detection_analysis/run_003
```

Manual COCO dataset definition:

```bash
python tools/detection_analysis/analyze.py \
  --ann-file /path/to/annotations.json \
  --img-prefix /path/to/images \
  --dataset-type CocoDataset \
  --results results/detector_a.pkl \
  --output-dir work_dirs/detection_analysis/manual_run
```

Useful threshold flags:

```bash
--confidence-threshold 0.35
--tp-iou-threshold 0.5
--localization-iou-min 0.1
--duplicate-iou-threshold 0.5
--classification-iou-threshold 0.5
--max-detections-per-image 100
```

## GUI Launch

Load a preprocessed analysis directory:

```bash
streamlit run tools/detection_analysis/app.py -- \
  --analysis-dir work_dirs/detection_analysis/run_001
```

Run analysis and open the GUI from raw files:

```bash
streamlit run tools/detection_analysis/app.py -- \
  --config configs/my_detector_config.py \
  --results results/detector_a.pkl results/detector_b.pkl \
  --names Detector_A Detector_B \
  --output-dir work_dirs/detection_analysis/gui_run
```

Direct launch also delegates to Streamlit:

```bash
python tools/detection_analysis/app.py \
  --analysis-dir work_dirs/detection_analysis/run_001
```

## GUI Pages

- Dataset overview: image count, annotation count, class distribution, object-size distribution, metadata, and prediction counts.
- Detector overview: COCO metrics, precision/recall/F1, TP/FP/FN, error counts, per-class metrics, confidence and IoU distributions.
- Detector comparison: shared metrics table, rankings, metric deltas, per-image disagreement states, and comparison charts.
- Image browser: image overlays with ground truth, predictions, false negatives, filters by detector, class, image ID/name, and error type.
- Error explorer: sortable/filterable prediction and ground-truth tables.
- Class analysis: selected-class metrics across detectors.
- Confusion matrix: GT classes, predicted classes, background, and missed detections with raw/row/column normalization.
- Export and reporting: JSON and CSV downloads.

## Error Categories

Prediction categories are mutually exclusive:

- `true_positive`: same-class unmatched ground truth with IoU at or above `tp_iou_threshold`.
- `background_false_positive`: no relevant overlap with any active ground truth.
- `classification_error`: good localization to a different-class ground truth.
- `localization_error`: same-class overlap is present but below the TP threshold.
- `classification_and_localization_error`: different-class overlap is present but below the classification threshold and above the localization minimum.
- `duplicate_detection`: a lower-confidence prediction overlaps a ground-truth object already claimed by a higher-confidence prediction.
- `matched_ignored_annotation`: prediction overlaps an ignored annotation.
- `matched_crowd_annotation`: prediction overlaps a crowd annotation when crowd annotations are ignored.
- `low_confidence_prediction`: score is below `confidence_threshold`.
- `invalid_prediction`: malformed box or a box fully outside the image.

Ground-truth categories:

- `correctly_detected`
- `false_negative`
- `detected_with_wrong_class`
- `poorly_localized`
- `only_duplicate_predictions`
- `ignored`
- `crowd_annotation`

## Matching Priority

Predictions are sorted by confidence per image. Matching is greedy and one-to-one. The primary category decision order is:

1. Invalid box.
2. Low confidence.
3. Ignored annotation overlap.
4. Crowd annotation overlap.
5. Same-class true-positive match.
6. Duplicate match to an already claimed ground truth.
7. High-IoU different-class match.
8. Same-class localization error.
9. Different-class classification and localization error.
10. Background false positive.

This order prevents one prediction from receiving conflicting primary categories.

## Output Directory

`analyze.py` writes:

- `analysis.json`: complete machine-readable report.
- `dataset.json`: normalized image and ground-truth records.
- `config.json`: threshold configuration for reproducibility.
- `comparison.json`: multi-detector comparison.
- `detectors/<name>/predictions.json`
- `detectors/<name>/prediction_matches.json`
- `detectors/<name>/ground_truth_matches.json`
- `detectors/<name>/metrics.json`
- `detectors/<name>/coco_metrics.json`
- `exports/global_metrics.csv`
- `exports/per_class_metrics.csv`
- `exports/per_image_analysis.csv`
- `exports/*_prediction_matches.csv`
- `exports/*_ground_truth_matches.csv`
- `report.html`
- `visualizations/<detector>/predictions_only/`: original-resolution images
  containing prediction overlays only (no ground-truth boxes).
- `visualizations/<detector>/problems_only/`: original-resolution images that
  have erroneous predictions or false negatives, containing only those problems.

Both visualization exports omit predictions below confidence `0.35`. If the
analysis confidence threshold is higher, the higher threshold is used instead.

## Tests

Run the synthetic unit tests from the repository root:

```bash
pytest tools/detection_analysis/tests
```

The tests cover true positives, duplicate detections, classification errors, localization errors, classification+localization errors, background false positives, false negatives, ignored and crowd annotations, invalid boxes, empty predictions, empty ground truth, multiple detectors, per-class metrics, confusion matrix generation, and result-format conversion.

## Compatibility Notes

This implementation targets the repository's MMDetection 3.3.0 layout first. It uses MMEngine config loading, MMDetection registry initialization, and dataset construction without loading model weights. The result adapter also supports common MMDetection 2.x-style per-class arrays when they are saved as per-image lists.

For unusual custom result pickles, the adapter needs enough information to recover `bboxes`, `scores`, and `labels` per image. If a result file stores transformed boxes, non-contiguous label IDs, or custom category mappings, verify the config dataset class order or use COCO JSON predictions with matching category metadata.

## Troubleshooting

- If the prediction count does not match the dataset image count, confirm the result file was generated on the same split as the config.
- If class names are numeric, the dataset metadata did not expose `classes`; pass a config with metainfo or use manual `--classes`.
- If COCO metrics are unavailable, install `pycocotools` or use the custom metrics produced by the core analyzer.
- If images do not render in the GUI, check that config `data_root` and `data_prefix` resolve correctly, or provide `--img-prefix` for manual datasets.
- If Streamlit is not installed, preprocessing still works with `analyze.py`; install GUI dependencies from this directory's requirements file to use `app.py`.
