"""Stable visualization colors."""

from __future__ import annotations

from tools.detection_analysis.constants import ERROR_COLORS

CLASS_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1",
    "#9c755f", "#edc948", "#76b7b2", "#af7aa1", "#ff9da7",
]


def color_for_status(status: str) -> str:
    """Return a stable hex color for a prediction or ground-truth status."""

    return ERROR_COLORS.get(status, "#333333")


def color_for_class(class_id: int | None) -> str:
    """Return a stable, repeatable color for a class id."""

    if class_id is None:
        return "#333333"
    return CLASS_COLORS[int(class_id) % len(CLASS_COLORS)]
