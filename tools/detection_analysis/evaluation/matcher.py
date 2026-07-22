"""Greedy prediction-to-ground-truth matching."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from tools.detection_analysis.config import AnalysisConfig
from tools.detection_analysis.constants import (
    ERROR_BACKGROUND,
    ERROR_CLASSIFICATION,
    ERROR_CLASSIFICATION_LOCALIZATION,
    ERROR_CROWD,
    ERROR_DUPLICATE,
    ERROR_IGNORED,
    ERROR_INVALID,
    ERROR_LOCALIZATION,
    ERROR_LOW_CONFIDENCE,
    ERROR_TRUE_POSITIVE,
    GT_CORRECT,
    GT_CROWD,
    GT_DUPLICATE_ONLY,
    GT_FALSE_NEGATIVE,
    GT_IGNORED,
    GT_POOR_LOCALIZATION,
    GT_WRONG_CLASS,
)
from tools.detection_analysis.models import GroundTruth, GroundTruthMatchRecord, ImageRecord, MatchRecord, Prediction
from tools.detection_analysis.utils.bbox import iou, is_valid_box


def match_detector(
    predictions: Iterable[Prediction],
    ground_truths: Iterable[GroundTruth],
    images: Iterable[ImageRecord],
    config: Optional[AnalysisConfig] = None,
) -> Tuple[List[MatchRecord], List[GroundTruthMatchRecord]]:
    """Assign a single primary status to every prediction and ground-truth box.

    Decision order for predictions:
    invalid box, low confidence, ignored/crowd overlap, true positive,
    duplicate, classification error, localization error,
    classification+localization error, background false positive.
    """

    cfg = config or AnalysisConfig()
    cfg.validate()
    image_by_id = {img.image_id: img for img in images}
    all_predictions = list(predictions)
    all_ground_truths = list(ground_truths)
    all_images = list(images)
    gt_by_image: Dict[str, List[GroundTruth]] = defaultdict(list)
    pred_by_image: Dict[str, List[Prediction]] = defaultdict(list)
    for gt in all_ground_truths:
        gt_by_image[gt.image_id].append(gt)
    for pred in all_predictions:
        pred_by_image[pred.image_id].append(pred)

    pred_records: List[MatchRecord] = []
    gt_records: List[GroundTruthMatchRecord] = []
    detector_name = all_predictions[0].detector_name if all_predictions else "detector"

    for image in all_images:
        gts = gt_by_image.get(image.image_id, [])
        preds = sorted(
            pred_by_image.get(image.image_id, []),
            key=lambda p: p.score,
            reverse=True,
        )[: cfg.max_detections_per_image]
        matched_gt_to_pred: Dict[str, Prediction] = {}
        gt_candidate_records: Dict[str, List[MatchRecord]] = defaultdict(list)

        for pred in preds:
            rec = _classify_prediction(pred, gts, image_by_id.get(pred.image_id), matched_gt_to_pred, cfg)
            pred_records.append(rec)
            if rec.matched_gt_id:
                gt_candidate_records[rec.matched_gt_id].append(rec)
            if rec.status == ERROR_TRUE_POSITIVE and rec.matched_gt_id:
                matched_gt_to_pred[rec.matched_gt_id] = pred

        for gt in gts:
            status, rec = _classify_gt(gt, gt_candidate_records.get(gt.gt_id, []), matched_gt_to_pred.get(gt.gt_id), detector_name)
            gt_records.append(rec)

    return pred_records, gt_records


def _classify_prediction(
    pred: Prediction,
    gts: List[GroundTruth],
    image: Optional[ImageRecord],
    matched_gt_to_pred: Dict[str, Prediction],
    cfg: AnalysisConfig,
) -> MatchRecord:
    width = image.width if image else None
    height = image.height if image else None
    if not is_valid_box(pred.bbox_xyxy, width=width, height=height):
        return _record(pred, ERROR_INVALID, valid=False)
    if pred.score < cfg.confidence_threshold:
        return _record(pred, ERROR_LOW_CONFIDENCE)

    ignored_gt, ignored_iou = _best_gt(pred, [gt for gt in gts if gt.ignored], class_filter=None)
    if ignored_gt and ignored_iou >= cfg.tp_iou_threshold:
        return _record(pred, ERROR_IGNORED, ignored_gt, ignored_iou)
    crowd_gt, crowd_iou = _best_gt(pred, [gt for gt in gts if gt.crowd], class_filter=None)
    if cfg.ignore_crowd and crowd_gt and crowd_iou >= cfg.tp_iou_threshold:
        return _record(pred, ERROR_CROWD, crowd_gt, crowd_iou)

    active_gts = [gt for gt in gts if not gt.ignored and not (cfg.ignore_crowd and gt.crowd)]
    same_gt, same_iou = _best_gt(pred, active_gts, class_filter=pred.class_id if not cfg.class_agnostic_matching else None)
    if same_gt and same_iou >= cfg.tp_iou_threshold:
        if same_gt.gt_id not in matched_gt_to_pred:
            return _record(pred, ERROR_TRUE_POSITIVE, same_gt, same_iou)
        return _record(pred, ERROR_DUPLICATE, same_gt, same_iou)

    if same_gt and same_gt.gt_id in matched_gt_to_pred and same_iou >= cfg.duplicate_iou_threshold:
        return _record(pred, ERROR_DUPLICATE, same_gt, same_iou)

    other_gt, other_iou = _best_gt(pred, active_gts, class_filter=None, exclude_class=pred.class_id)
    if other_gt and other_iou >= cfg.classification_iou_threshold:
        return _record(pred, ERROR_CLASSIFICATION, other_gt, other_iou)
    if same_gt and same_iou >= cfg.localization_iou_min:
        return _record(pred, ERROR_LOCALIZATION, same_gt, same_iou)
    if other_gt and other_iou >= cfg.localization_iou_min:
        return _record(pred, ERROR_CLASSIFICATION_LOCALIZATION, other_gt, other_iou)
    return _record(pred, ERROR_BACKGROUND)


def _best_gt(
    pred: Prediction,
    gts: List[GroundTruth],
    class_filter: Optional[int],
    exclude_class: Optional[int] = None,
) -> Tuple[Optional[GroundTruth], float]:
    best_gt = None
    best_iou = 0.0
    for gt in gts:
        if class_filter is not None and gt.class_id != class_filter:
            continue
        if exclude_class is not None and gt.class_id == exclude_class:
            continue
        value = iou(pred.bbox_xyxy, gt.bbox_xyxy)
        if value > best_iou:
            best_iou = value
            best_gt = gt
    return best_gt, best_iou


def _record(
    pred: Prediction,
    status: str,
    gt: Optional[GroundTruth] = None,
    value: float = 0.0,
    valid: bool = True,
) -> MatchRecord:
    return MatchRecord(
        detector_name=pred.detector_name,
        image_id=pred.image_id,
        image_index=pred.image_index,
        prediction_id=pred.prediction_id,
        status=status,
        score=pred.score,
        pred_class_id=pred.class_id,
        pred_class_name=pred.class_name,
        pred_bbox=pred.bbox_xyxy,
        iou=float(value),
        matched_gt_id=gt.gt_id if gt else None,
        gt_class_id=gt.class_id if gt else None,
        gt_class_name=gt.class_name if gt else None,
        gt_bbox=gt.bbox_xyxy if gt else None,
        object_size=gt.object_size if gt else None,
        valid=valid,
    )


def _classify_gt(
    gt: GroundTruth,
    candidates: List[MatchRecord],
    matched_pred: Optional[Prediction],
    detector_name: str,
) -> Tuple[str, GroundTruthMatchRecord]:
    if gt.ignored:
        status = GT_IGNORED
        chosen = None
    elif gt.crowd:
        status = GT_CROWD
        chosen = None
    elif matched_pred:
        status = GT_CORRECT
        chosen = next((c for c in candidates if c.prediction_id == matched_pred.prediction_id), None)
    elif any(c.status == ERROR_CLASSIFICATION for c in candidates):
        status = GT_WRONG_CLASS
        chosen = max((c for c in candidates if c.status == ERROR_CLASSIFICATION), key=lambda c: c.iou)
    elif any(c.status == ERROR_LOCALIZATION for c in candidates):
        status = GT_POOR_LOCALIZATION
        chosen = max((c for c in candidates if c.status == ERROR_LOCALIZATION), key=lambda c: c.iou)
    elif candidates and all(c.status == ERROR_DUPLICATE for c in candidates):
        status = GT_DUPLICATE_ONLY
        chosen = max(candidates, key=lambda c: c.iou)
    else:
        status = GT_FALSE_NEGATIVE
        chosen = None
    return status, GroundTruthMatchRecord(
        detector_name=detector_name,
        image_id=gt.image_id,
        image_index=gt.image_index,
        gt_id=gt.gt_id,
        status=status,
        gt_class_id=gt.class_id,
        gt_class_name=gt.class_name,
        gt_bbox=gt.bbox_xyxy,
        object_size=gt.object_size,
        matched_prediction_id=chosen.prediction_id if chosen else None,
        pred_class_id=chosen.pred_class_id if chosen else None,
        pred_class_name=chosen.pred_class_name if chosen else None,
        score=chosen.score if chosen else None,
        iou=chosen.iou if chosen else 0.0,
    )
