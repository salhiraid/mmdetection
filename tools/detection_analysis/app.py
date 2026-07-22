"""Streamlit GUI for object detection analysis."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.detection_analysis.cli import run_analysis_from_args
from tools.detection_analysis.evaluation.confusion_matrix import build_confusion_matrix
from tools.detection_analysis.evaluation.pr_curves import precision_recall_curve
from tools.detection_analysis.gui.sidebar import render_sidebar
from tools.detection_analysis.models import AnalysisBundle
from tools.detection_analysis.utils.serialization import load_json, to_plain
from tools.detection_analysis.visualization.box_renderer import render_image
from tools.detection_analysis.visualization.charts import metric_bar, status_counts_bar


def _is_streamlit_runtime() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def _launch_streamlit() -> None:
    args = [sys.executable, "-m", "streamlit", "run", __file__, "--", *sys.argv[1:]]
    raise SystemExit(subprocess.call(args))


def main() -> None:
    """Render the Streamlit app."""

    import pandas as pd
    import streamlit as st

    st.set_page_config(page_title="Detection Analysis", layout="wide")
    st.title("Detection Analysis")
    defaults = _parse_initial_args()
    controls = render_sidebar(st)
    bundle_data: Dict[str, Any] | None = None

    analysis_dir = controls["analysis_dir"] or defaults.get("analysis_dir")
    if analysis_dir:
        path = Path(analysis_dir) / "analysis.json"
        if path.exists():
            bundle_data = load_json(path)
        else:
            st.error(f"analysis.json not found in {analysis_dir}")

    if controls["run"] or (not bundle_data and defaults.get("results")):
        args = Namespace(
            config=controls["config"] or defaults.get("config"),
            ann_file=controls["ann_file"] or defaults.get("ann_file"),
            img_prefix=controls["img_prefix"] or defaults.get("img_prefix"),
            dataset_type="CocoDataset",
            classes=None,
            results=controls["results"] or defaults.get("results"),
            names=controls["names"] or defaults.get("names"),
            output_dir=defaults.get("output_dir") or "work_dirs/detection_analysis/gui_run",
            split="test",
            confidence_threshold=controls["config_obj"].confidence_threshold,
            tp_iou_threshold=controls["config_obj"].tp_iou_threshold,
            localization_iou_min=controls["config_obj"].localization_iou_min,
            duplicate_iou_threshold=controls["config_obj"].duplicate_iou_threshold,
            classification_iou_threshold=controls["config_obj"].classification_iou_threshold,
            max_detections_per_image=100,
            class_agnostic_matching=False,
            include_crowd=False,
            skip_coco=False,
        )
        with st.spinner("Running analysis..."):
            bundle = run_analysis_from_args(args)
            bundle_data = to_plain(bundle)

    if not bundle_data:
        st.info("Load an existing analysis directory or provide dataset and result files in the sidebar.")
        return

    _render_tabs(st, pd, bundle_data)


def _parse_initial_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--analysis-dir")
    parser.add_argument("--config")
    parser.add_argument("--ann-file")
    parser.add_argument("--img-prefix")
    parser.add_argument("--results", nargs="*")
    parser.add_argument("--names", nargs="*")
    parser.add_argument("--output-dir")
    args, _ = parser.parse_known_args()
    return {k: v for k, v in vars(args).items() if v}


def _render_tabs(st, pd, bundle: Dict[str, Any]) -> None:
    tabs = st.tabs([
        "Dataset overview",
        "Detector overview",
        "Detector comparison",
        "Image browser",
        "Error explorer",
        "Class analysis",
        "Confusion matrix",
        "Export and reporting",
    ])
    with tabs[0]:
        _dataset_page(st, pd, bundle)
    with tabs[1]:
        _overview_page(st, pd, bundle)
    with tabs[2]:
        _comparison_page(st, pd, bundle)
    with tabs[3]:
        _image_browser_page(st, pd, bundle)
    with tabs[4]:
        _error_explorer_page(st, pd, bundle)
    with tabs[5]:
        _class_analysis_page(st, pd, bundle)
    with tabs[6]:
        _confusion_page(st, pd, bundle)
    with tabs[7]:
        _report_page(st, pd, bundle)


def _dataset_page(st, pd, bundle: Dict[str, Any]) -> None:
    dataset = bundle["dataset"]
    st.metric("Images", len(dataset["images"]))
    st.metric("Annotations", len(dataset["ground_truths"]))
    st.metric("Classes", len(dataset["classes"]))
    gt_df = pd.DataFrame(dataset["ground_truths"])
    if not gt_df.empty:
        st.plotly_chart(status_counts_bar(gt_df["class_name"].value_counts().to_dict(), "Class distribution"), use_container_width=True)
        st.plotly_chart(status_counts_bar(gt_df["object_size"].value_counts().to_dict(), "Object-size distribution"), use_container_width=True)
    pred_counts = [{"detector": d["detector_name"], "predictions": len(d["predictions"])} for d in bundle["detectors"]]
    st.dataframe(pd.DataFrame(pred_counts), use_container_width=True)
    st.json(dataset.get("metadata", {}), expanded=False)


def _overview_page(st, pd, bundle: Dict[str, Any]) -> None:
    rows = []
    for det in bundle["detectors"]:
        row = {"detector": det["detector_name"], **{k: v for k, v in det["metrics"].items() if isinstance(v, (int, float))}}
        row.update({k: v for k, v in det.get("coco_metrics", {}).items() if isinstance(v, (int, float))})
        rows.append(row)
        st.subheader(det["detector_name"])
        st.dataframe(pd.DataFrame([row]), use_container_width=True)
        st.plotly_chart(status_counts_bar(det["metrics"].get("prediction_status_counts", {}), "Prediction status counts"), use_container_width=True)
        st.dataframe(pd.DataFrame(det["metrics"].get("per_class", {})).T, use_container_width=True)
    if rows:
        st.plotly_chart(metric_bar(rows, "f1"), use_container_width=True)


def _comparison_page(st, pd, bundle: Dict[str, Any]) -> None:
    comparison = bundle.get("comparison") or {}
    rows = comparison.get("metrics_table", [])
    if not rows:
        st.info("Load at least two detectors for comparison.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    metric = st.selectbox("Metric", [c for c in rows[0] if c != "detector"], index=1)
    st.plotly_chart(metric_bar(rows, metric), use_container_width=True)
    st.dataframe(pd.DataFrame(comparison.get("metric_differences", [])), use_container_width=True)
    image_states = pd.DataFrame([{"image_id": k, **v} for k, v in comparison.get("image_states", {}).items()])
    st.dataframe(image_states, use_container_width=True)


def _image_browser_page(st, pd, bundle: Dict[str, Any]) -> None:
    images = bundle["dataset"]["images"]
    image_by_id = {img["image_id"]: img for img in images}
    detectors = {d["detector_name"]: d for d in bundle["detectors"]}
    selected = st.multiselect("Overlay detectors", list(detectors), default=list(detectors)[:1])
    error_types = sorted({m["status"] for d in detectors.values() for m in d["prediction_matches"]})
    selected_errors = st.multiselect("Error types", error_types, default=error_types)
    classes = bundle["dataset"]["classes"]
    selected_classes = st.multiselect("Classes", classes)
    search = st.text_input("Image name or ID")
    browse_col, display_col, label_col, color_col = st.columns(4)
    with browse_col:
        browse_mode = st.radio("Browse list", ["Images", "Problems"], horizontal=True)
    with display_col:
        display_method = st.radio("Display boxes", ["Predictions", "Problems only"], horizontal=True)
    with label_col:
        default_label = "Problem" if display_method == "Problems only" else "Class"
        label_choice = st.selectbox("Box label", ["Class", "Problem", "Class + problem", "None"], index=["Class", "Problem", "Class + problem", "None"].index(default_label))
    with color_col:
        default_color = "Error / problem" if display_method == "Problems only" else "Class"
        color_choice = st.selectbox("Box colors", ["Class", "Error / problem"], index=["Class", "Error / problem"].index(default_color))
    detail_col_a, detail_col_b, detail_col_c = st.columns(3)
    with detail_col_a:
        show_detector_name = st.checkbox("Show detector name", value=len(selected) > 1)
    with detail_col_b:
        show_scores = st.checkbox("Show scores", value=True)
    with detail_col_c:
        show_iou = st.checkbox("Show IoU", value=True)
    hide_correct = display_method == "Problems only"

    all_selected_matches = [
        m for name in selected for m in detectors[name]["prediction_matches"]
    ]
    matches_by_image: Dict[str, List[Dict[str, Any]]] = {}
    for match in all_selected_matches:
        matches_by_image.setdefault(match["image_id"], []).append(match)
    gt_by_image: Dict[str, List[Dict[str, Any]]] = {}
    for name in selected:
        for gt_match in detectors[name]["ground_truth_matches"]:
            gt_by_image.setdefault(gt_match["image_id"], []).append(gt_match)

    candidates = []
    for img in images:
        if search and search not in img["image_id"] and search.lower() not in img["file_name"].lower():
            continue
        matches = matches_by_image.get(img["image_id"], [])
        false_negatives = [g for g in gt_by_image.get(img["image_id"], []) if g["status"] == "false_negative"]
        shown_matches = [m for m in matches if m["status"] in selected_errors]
        if hide_correct:
            shown_matches = [m for m in shown_matches if m["status"] != "true_positive"]
        if selected_errors and not any(m["status"] in selected_errors for m in matches) and not false_negatives:
            continue
        if selected_classes and not (
            any(m["pred_class_name"] in selected_classes or m.get("gt_class_name") in selected_classes for m in matches)
            or any(g["gt_class_name"] in selected_classes for g in false_negatives)
        ):
            continue
        errors = [m for m in shown_matches if m["status"] != "true_positive"]
        if display_method == "Problems only" and not errors and not false_negatives:
            continue
        top_statuses = [m["status"] for m in errors] + (["false_negative"] if false_negatives else [])
        candidates.append({
            **img,
            "predictions": len(matches),
            "shown_predictions": len(shown_matches),
            "errors": len(errors),
            "false_negatives": len(false_negatives),
            "false_positives": sum(1 for m in errors if m.get("matched_gt_id") is None),
            "duplicates": sum(1 for m in errors if m["status"] == "duplicate_detection"),
            "top_statuses": ", ".join(pd.Series(top_statuses).value_counts().head(3).index.tolist()) if top_statuses else "clean",
        })
    if not candidates:
        st.info("No images match the filters.")
        return

    problem_rows = _problem_rows(candidates, matches_by_image, gt_by_image, selected_errors, selected_classes)
    rows = problem_rows if browse_mode == "Problems" and problem_rows else candidates
    if browse_mode == "Problems" and not problem_rows:
        st.info("No individual problem rows match the current filters; showing filtered images instead.")
        rows = candidates

    selected_pos = _fast_browser_controls(st, pd, rows, browse_mode)
    image = image_by_id[rows[selected_pos]["image_id"]]

    matches = []
    gt_matches = []
    for name in selected:
        matches.extend([
            m for m in detectors[name]["prediction_matches"]
            if m["image_id"] == image["image_id"]
            and m["status"] in selected_errors
            and (not hide_correct or m["status"] != "true_positive")
        ])
        gt_matches.extend([g for g in detectors[name]["ground_truth_matches"] if g["image_id"] == image["image_id"]])
    from tools.detection_analysis.models import GroundTruthMatchRecord, ImageRecord, MatchRecord
    label_mode = {
        "Class": "class",
        "Problem": "problem",
        "Class + problem": "class_problem",
        "None": "none",
    }[label_choice]
    color_mode = "class" if color_choice == "Class" else "error"
    rendered_gt_matches = [
        GroundTruthMatchRecord(**g) for g in gt_matches
        if display_method == "Problems only" and g["status"] == "false_negative"
    ]
    rendered = render_image(
        ImageRecord(**image),
        [MatchRecord(**m) for m in matches],
        rendered_gt_matches,
        label_mode=label_mode,
        color_mode=color_mode,
        show_scores=show_scores,
        show_iou=show_iou,
        show_detector=show_detector_name,
    )
    st.caption(f"{selected_pos + 1} / {len(rows)} in {browse_mode.lower()} view - image {image['image_id']} - {image['file_name']}")
    st.image(rendered, use_container_width=True)
    st.dataframe(pd.DataFrame(matches), use_container_width=True)


def _fast_browser_controls(st, pd, rows: List[Dict[str, Any]], browse_mode: str) -> int:
    """Render fast scrolling, row selection, and thumbnail controls."""

    key_prefix = f"browser_{browse_mode.lower()}"
    index_key = f"{key_prefix}_index"
    if index_key not in st.session_state or st.session_state[index_key] >= len(rows):
        st.session_state[index_key] = 0

    col_prev, col_slider, col_next, col_jump = st.columns([1, 6, 1, 2])
    with col_prev:
        if st.button("Previous", use_container_width=True):
            st.session_state[index_key] = max(0, st.session_state[index_key] - 1)
    with col_next:
        if st.button("Next", use_container_width=True):
            st.session_state[index_key] = min(len(rows) - 1, st.session_state[index_key] + 1)
    with col_slider:
        st.session_state[index_key] = st.slider(
            "Fast scroll",
            min_value=0,
            max_value=len(rows) - 1,
            value=int(st.session_state[index_key]),
            format="%d",
        )
    with col_jump:
        jump = st.number_input("Jump", min_value=1, max_value=len(rows), value=int(st.session_state[index_key]) + 1)
        st.session_state[index_key] = int(jump) - 1

    table_df = pd.DataFrame(rows).copy()
    preferred = [
        "image_id", "file_name", "detector_name", "status", "pred_class_name",
        "gt_class_name", "score", "iou", "errors", "predictions",
        "false_negatives", "false_positives", "duplicates", "top_statuses",
    ]
    visible_cols = [c for c in preferred if c in table_df.columns]
    table_df = table_df[visible_cols]
    event = st.dataframe(
        table_df,
        use_container_width=True,
        height=360,
        selection_mode="single-row",
        on_select="rerun",
        key=f"{key_prefix}_table",
    )
    selected_rows = event.selection.rows if hasattr(event, "selection") else []
    if selected_rows:
        st.session_state[index_key] = int(selected_rows[0])

    if browse_mode == "Images" and st.checkbox("Show scrollable thumbnails", value=False):
        page_size = st.slider("Thumbnails per page", 12, 96, 36, 12)
        page_count = max(1, (len(rows) + page_size - 1) // page_size)
        page = st.number_input("Thumbnail page", min_value=1, max_value=page_count, value=1)
        start = (int(page) - 1) * page_size
        with st.container(height=520):
            for row_start in range(start, min(start + page_size, len(rows)), 4):
                cols = st.columns(4)
                for offset, col in enumerate(cols):
                    pos = row_start + offset
                    if pos >= min(start + page_size, len(rows)):
                        continue
                    row = rows[pos]
                    with col:
                        if row.get("img_path"):
                            st.image(row["img_path"], use_container_width=True)
                        if st.button(f"{pos + 1}: {row['file_name']}", key=f"{key_prefix}_thumb_{pos}", use_container_width=True):
                            st.session_state[index_key] = pos
                        st.caption(f"errors {row.get('errors', 0)} | preds {row.get('predictions', 0)}")

    return int(st.session_state[index_key])


def _problem_rows(
    candidates: List[Dict[str, Any]],
    matches_by_image: Dict[str, List[Dict[str, Any]]],
    gt_by_image: Dict[str, List[Dict[str, Any]]],
    selected_errors: List[str],
    selected_classes: List[str],
) -> List[Dict[str, Any]]:
    """Flatten filtered predictions into scrollable problem rows."""

    candidate_ids = {img["image_id"] for img in candidates}
    file_by_image = {img["image_id"]: img["file_name"] for img in candidates}
    rows = []
    for image_id in candidate_ids:
        for match in matches_by_image.get(image_id, []):
            if match["status"] not in selected_errors:
                continue
            if match["status"] == "true_positive":
                continue
            if selected_classes and match["pred_class_name"] not in selected_classes and match.get("gt_class_name") not in selected_classes:
                continue
            rows.append({
                "image_id": image_id,
                "file_name": file_by_image[image_id],
                "detector_name": match["detector_name"],
                "status": match["status"],
                "pred_class_name": match["pred_class_name"],
                "gt_class_name": match.get("gt_class_name"),
                "score": match["score"],
                "iou": match["iou"],
                "prediction_id": match["prediction_id"],
            })
        for gt_match in gt_by_image.get(image_id, []):
            if gt_match["status"] != "false_negative":
                continue
            if selected_classes and gt_match["gt_class_name"] not in selected_classes:
                continue
            rows.append({
                "image_id": image_id,
                "file_name": file_by_image[image_id],
                "detector_name": gt_match["detector_name"],
                "status": "false_negative",
                "pred_class_name": None,
                "gt_class_name": gt_match["gt_class_name"],
                "score": None,
                "iou": gt_match["iou"],
                "prediction_id": None,
            })
    return sorted(rows, key=lambda row: (row["image_id"], -(row["score"] or 0.0), row["status"]))


def _error_explorer_page(st, pd, bundle: Dict[str, Any]) -> None:
    detector = st.selectbox("Detector", [d["detector_name"] for d in bundle["detectors"]])
    det = next(d for d in bundle["detectors"] if d["detector_name"] == detector)
    pred_df = pd.DataFrame(det["prediction_matches"])
    gt_df = pd.DataFrame(det["ground_truth_matches"])
    status = st.multiselect("Prediction status", sorted(pred_df["status"].unique()) if not pred_df.empty else [])
    if status:
        pred_df = pred_df[pred_df["status"].isin(status)]
    st.dataframe(pred_df, use_container_width=True)
    st.dataframe(gt_df, use_container_width=True)


def _class_analysis_page(st, pd, bundle: Dict[str, Any]) -> None:
    cls = st.selectbox("Class", bundle["dataset"]["classes"])
    rows = []
    for det in bundle["detectors"]:
        values = det["metrics"].get("per_class", {}).get(cls, {})
        rows.append({"detector": det["detector_name"], **values})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _confusion_page(st, pd, bundle: Dict[str, Any]) -> None:
    detector = st.selectbox("Detector for matrix", [d["detector_name"] for d in bundle["detectors"]])
    normalize = st.radio("Normalization", ["raw", "row", "column"], horizontal=True)
    det = next(d for d in bundle["detectors"] if d["detector_name"] == detector)
    from tools.detection_analysis.models import GroundTruthMatchRecord, MatchRecord
    matrix = build_confusion_matrix(
        [MatchRecord(**m) for m in det["prediction_matches"]],
        [GroundTruthMatchRecord(**g) for g in det["ground_truth_matches"]],
        bundle["dataset"]["classes"],
        normalize=normalize,
    )
    st.dataframe(matrix, use_container_width=True)


def _report_page(st, pd, bundle: Dict[str, Any]) -> None:
    st.download_button("Download JSON report", json.dumps(bundle, indent=2), "detection_analysis_report.json", "application/json")
    rows = bundle.get("comparison", {}).get("metrics_table", [])
    if rows:
        st.download_button("Download comparison CSV", pd.DataFrame(rows).to_csv(index=False), "detector_comparison.csv", "text/csv")


if __name__ == "__main__":
    if _is_streamlit_runtime():
        main()
    else:
        _launch_streamlit()
