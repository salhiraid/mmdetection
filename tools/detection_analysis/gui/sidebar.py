"""Streamlit sidebar controls."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List

from tools.detection_analysis.config import AnalysisConfig


def render_sidebar(st) -> Dict[str, Any]:
    """Render shared sidebar controls and return values."""

    st.sidebar.header("Inputs")
    analysis_dir = st.sidebar.text_input("Analysis directory")
    config = st.sidebar.text_input("MMDetection config")
    ann_file = st.sidebar.text_input("COCO annotation file")
    img_prefix = st.sidebar.text_input("Image prefix")
    results = st.sidebar.text_area("Result files", help="One path per line")
    names = st.sidebar.text_input("Detector names", help="Comma-separated, optional")
    st.sidebar.header("Thresholds")
    cfg = AnalysisConfig(
        confidence_threshold=st.sidebar.slider("Confidence", 0.0, 1.0, 0.05, 0.01),
        tp_iou_threshold=st.sidebar.slider("TP IoU", 0.0, 1.0, 0.5, 0.01),
        localization_iou_min=st.sidebar.slider("Localization IoU min", 0.0, 1.0, 0.1, 0.01),
        duplicate_iou_threshold=st.sidebar.slider("Duplicate IoU", 0.0, 1.0, 0.5, 0.01),
        classification_iou_threshold=st.sidebar.slider("Classification IoU", 0.0, 1.0, 0.5, 0.01),
    )
    return {
        "analysis_dir": analysis_dir.strip(),
        "config": config.strip(),
        "ann_file": ann_file.strip(),
        "img_prefix": img_prefix.strip(),
        "results": [line.strip() for line in results.splitlines() if line.strip()],
        "names": [part.strip() for part in names.split(",") if part.strip()],
        "config_obj": cfg,
        "run": st.sidebar.button("Run analysis"),
    }

