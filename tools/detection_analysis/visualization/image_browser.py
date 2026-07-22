"""Filtering helpers for the interactive image browser."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from tools.detection_analysis.models import ImageRecord, MatchRecord


def filter_images(
    images: Sequence[ImageRecord],
    matches: Iterable[MatchRecord],
    error_types: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    min_errors: int = 0,
    disagreeing_ids: Sequence[str] | None = None,
) -> List[ImageRecord]:
    """Return images matching common browser filters."""

    by_image: Dict[str, List[MatchRecord]] = {}
    for match in matches:
        by_image.setdefault(match.image_id, []).append(match)
    disagreeing = set(disagreeing_ids or [])
    out = []
    for image in images:
        image_matches = by_image.get(image.image_id, [])
        if error_types and not any(m.status in error_types for m in image_matches):
            continue
        if class_names and not any(m.pred_class_name in class_names or m.gt_class_name in class_names for m in image_matches):
            continue
        errors = sum(1 for m in image_matches if m.status != "true_positive")
        if errors < min_errors:
            continue
        if disagreeing_ids is not None and image.image_id not in disagreeing:
            continue
        out.append(image)
    return out

