"""CSV and table exports."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from tools.detection_analysis.models import AnalysisBundle


def export_bundle_csv(bundle: AnalysisBundle, output_dir: str | Path) -> None:
    """Write global, per-class, per-image, and per-box CSV files."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    global_rows = []
    per_class_rows = []
    per_image_rows = []
    for detector in bundle.detectors:
        row = {"detector": detector.detector_name}
        row.update({k: v for k, v in detector.metrics.items() if not isinstance(v, (dict, list))})
        row.update({k: v for k, v in detector.coco_metrics.items() if isinstance(v, (int, float))})
        global_rows.append(row)
        for class_name, values in detector.metrics.get("per_class", {}).items():
            per_class_rows.append({"detector": detector.detector_name, "class_name": class_name, **values})
        for image_id, values in detector.metrics.get("per_image", {}).items():
            per_image_rows.append({"detector": detector.detector_name, "image_id": image_id, **values})
        pd.DataFrame([asdict(p) for p in detector.prediction_matches]).to_csv(out / f"{detector.detector_name}_prediction_matches.csv", index=False)
        pd.DataFrame([asdict(g) for g in detector.ground_truth_matches]).to_csv(out / f"{detector.detector_name}_ground_truth_matches.csv", index=False)
    pd.DataFrame(global_rows).to_csv(out / "global_metrics.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(out / "per_class_metrics.csv", index=False)
    pd.DataFrame(per_image_rows).to_csv(out / "per_image_analysis.csv", index=False)
    if bundle.comparison.get("metrics_table"):
        pd.DataFrame(bundle.comparison["metrics_table"]).to_csv(out / "detector_comparison.csv", index=False)

