"""Pillow-based image overlay renderer."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from PIL import Image, ImageDraw, ImageFont

from tools.detection_analysis.models import GroundTruthMatchRecord, ImageRecord, MatchRecord
from tools.detection_analysis.utils.bbox import clip_box
from tools.detection_analysis.visualization.colors import color_for_status


def render_image(
    image: ImageRecord,
    prediction_matches: Iterable[MatchRecord] = (),
    ground_truth_matches: Iterable[GroundTruthMatchRecord] = (),
    show_labels: bool = True,
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
        _draw_box(draw, gt.gt_bbox, img.size, "#1f77b4", line_width, f"GT {gt.gt_class_name} | {gt.status}" if show_labels else "")
    for pred in prediction_matches:
        parts: List[str] = []
        if show_detector:
            parts.append(pred.detector_name)
        if show_labels:
            parts.append(pred.pred_class_name)
        if show_scores:
            parts.append(f"{pred.score:.2f}")
        if show_iou:
            parts.append(f"IoU {pred.iou:.2f}")
        parts.append(pred.status)
        _draw_box(draw, pred.pred_bbox, img.size, color_for_status(pred.status), line_width, " | ".join(parts))
    return img


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

