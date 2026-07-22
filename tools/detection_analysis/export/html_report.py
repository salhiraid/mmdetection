"""Self-contained HTML report generation."""

from __future__ import annotations

from pathlib import Path

from tools.detection_analysis.models import AnalysisBundle
from tools.detection_analysis.utils.serialization import to_plain


def write_html_report(bundle: AnalysisBundle, path: str | Path) -> None:
    """Write a compact HTML report with metrics and error summaries."""

    data = to_plain(bundle)
    rows = []
    for det in data["detectors"]:
        metrics = det["metrics"]
        rows.append(
            f"<tr><td>{det['detector_name']}</td><td>{metrics.get('precision', 0):.4f}</td>"
            f"<td>{metrics.get('recall', 0):.4f}</td><td>{metrics.get('f1', 0):.4f}</td>"
            f"<td>{det.get('coco_metrics', {}).get('mAP', 'n/a')}</td></tr>"
        )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Detection Analysis Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f3f5f7; }}
    code {{ background: #f3f5f7; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Detection Analysis Report</h1>
  <h2>Dataset Summary</h2>
  <p>{len(data['dataset']['images'])} images, {len(data['dataset']['ground_truths'])} annotations, {len(data['dataset']['classes'])} classes.</p>
  <h2>Detector Summaries</h2>
  <table><thead><tr><th>Detector</th><th>Precision</th><th>Recall</th><th>F1</th><th>mAP</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
  <h2>Configuration</h2>
  <pre>{data['config']}</pre>
</body>
</html>
"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(html, encoding="utf-8")

