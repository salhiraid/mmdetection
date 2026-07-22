"""Annotated image export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tools.detection_analysis.models import ImageRecord, MatchRecord
from tools.detection_analysis.visualization.box_renderer import render_image


def export_annotated_image(image: ImageRecord, matches: Iterable[MatchRecord], output_path: str | Path) -> None:
    """Render and save one annotated image."""

    rendered = render_image(image, list(matches))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)

