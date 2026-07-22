"""Stable visualization colors."""

from __future__ import annotations

from tools.detection_analysis.constants import ERROR_COLORS


def color_for_status(status: str) -> str:
    """Return a stable hex color for a prediction or ground-truth status."""

    return ERROR_COLORS.get(status, "#333333")

