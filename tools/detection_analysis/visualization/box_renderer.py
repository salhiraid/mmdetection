"""Pillow-based image overlay renderer."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from PIL import Image, ImageDraw, ImageFont

from tools.detection_analysis.models import GroundTruthMatchRecord, ImageRecord, MatchRecord
from tools.detection_analysis.utils.bbox import clip_box
from tools.detection_analysis.visualization.colors import color_for_class, color_for_status


def render_image(
    image: ImageRecord,
    prediction_matches: Iterable[MatchRecord] = (),
    ground_truth_matches: Iterable[GroundTruthMatchRecord] = (),
    label_mode: str = "class_problem",
    color_mode: str = "error",
    show_scores: bool = True,
    show_iou: bool = True,
    show_detector: bool = True,
) -> Image.Image:
    """Render predictions and ground truth overlays on an image."""

    if image.img_path and Path(image.img_path).exists():
        img = Image.open(image.img_path).convert("RGB")
    else:
        width = image.width or 960
        height = image.height or 640
        img = Image.new("RGB", (width, height), "#f5f5f5")
    draw = ImageDraw.Draw(img)
    line_width = max(2, int(min(img.size) / 250))
    for gt in ground_truth_matches:
        color = color_for_class(gt.gt_class_id) if color_mode == "class" else color_for_status(gt.status)
        _draw_box(draw, gt.gt_bbox, img.size, color, line_width, _gt_label(gt, label_mode))
    for pred in prediction_matches:
        label = _prediction_label(pred, label_mode, show_scores, show_iou, show_detector)
        color = color_for_class(pred.pred_class_id) if color_mode == "class" else color_for_status(pred.status)
        _draw_box(draw, pred.pred_bbox, img.size, color, line_width, label)
    return img


def _prediction_label(
    pred: MatchRecord,
    label_mode: str,
    show_scores: bool,
    show_iou: bool,
    show_detector: bool,
) -> str:
    parts: List[str] = []
    if show_detector:
        parts.append(pred.detector_name)
    if label_mode == "class":
        parts.append(pred.pred_class_name)
    elif label_mode == "problem":
        parts.append(pred.status)
    elif label_mode == "class_problem":
        parts.extend([pred.pred_class_name, pred.status])
    if show_scores:
        parts.append(f"{pred.score:.2f}")
    if show_iou:
        parts.append(f"IoU {pred.iou:.2f}")
    return " | ".join(parts)


def _gt_label(gt: GroundTruthMatchRecord, label_mode: str) -> str:
    if label_mode == "none":
        return ""
    if label_mode == "problem":
        return gt.status
    if label_mode == "class":
        return f"GT {gt.gt_class_name}"
    return f"GT {gt.gt_class_name} | {gt.status}"


def _draw_box(draw: ImageDraw.ImageDraw, box, image_size, color: str, line_width: int, label: str) -> None:
    width, height = image_size
    x1, y1, x2, y2 = clip_box(box, width, height)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
    if not label:
        return
    font = ImageFont.load_default()
    label = _fit_label(label, max(40, int(x2 - x1)))
    bbox = draw.textbbox((x1, y1), label, font=font)
    text_h = bbox[3] - bbox[1]
    text_w = bbox[2] - bbox[0]
    y_text = y1 - text_h - 4 if y1 - text_h - 4 > 0 else y1 + 2
    draw.rectangle([x1, y_text, x1 + text_w + 6, y_text + text_h + 4], fill=color)
    draw.text((x1 + 3, y_text + 2), label, fill="white", font=font)


def _fit_label(label: str, max_pixels: int) -> str:
    max_chars = max(12, max_pixels // 6)
    return label if len(label) <= max_chars else label[: max_chars - 1] + "..."
