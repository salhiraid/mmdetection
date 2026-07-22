"""Dataset loading from MMDetection configs or manual COCO-style inputs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from tools.detection_analysis.config import AnalysisConfig
from tools.detection_analysis.models import AnalysisDataset, GroundTruth, ImageRecord
from tools.detection_analysis.utils.bbox import area, object_size, to_xyxy
from tools.detection_analysis.utils.validation import DatasetLoadError


def _classes_from_metainfo(metainfo: Dict[str, Any]) -> List[str]:
    classes = metainfo.get("classes") or metainfo.get("CLASSES") or []
    return [str(c) for c in classes]


def _join_if_relative(root: Optional[str], value: Optional[str]) -> str:
    if not value:
        return ""
    if os.path.isabs(value) or not root:
        return value
    return os.path.normpath(os.path.join(root, value))


def load_dataset(
    config_path: Optional[str] = None,
    ann_file: Optional[str] = None,
    img_prefix: Optional[str] = None,
    dataset_type: str = "CocoDataset",
    classes: Optional[Sequence[str]] = None,
    split: str = "test",
    analysis_config: Optional[AnalysisConfig] = None,
) -> AnalysisDataset:
    """Load a detection dataset using a config or manual annotation inputs."""

    cfg = analysis_config or AnalysisConfig()
    cfg.validate()
    if config_path:
        return load_mmdet_config_dataset(config_path, split=split, analysis_config=cfg)
    if not ann_file:
        raise DatasetLoadError("Provide either --config or --ann-file.")
    if dataset_type != "CocoDataset":
        raise DatasetLoadError(
            "Manual loading currently supports CocoDataset annotations. "
            "Use --config for other MMDetection dataset types.")
    return load_coco_dataset(ann_file, img_prefix=img_prefix, classes=classes, analysis_config=cfg)


def load_mmdet_config_dataset(
    config_path: str,
    split: str = "test",
    analysis_config: Optional[AnalysisConfig] = None,
) -> AnalysisDataset:
    """Build only the dataset portion of an MMDetection config."""

    try:
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
        from mmdet.registry import DATASETS
        from mmdet.utils import register_all_modules, replace_cfg_vals, update_data_root
    except Exception as exc:
        raise DatasetLoadError(f"MMDetection/MMEngine imports failed: {exc}") from exc

    try:
        cfg = Config.fromfile(config_path)
        cfg = replace_cfg_vals(cfg)
        update_data_root(cfg)
        register_all_modules(init_default_scope=False)
        init_default_scope(cfg.get("default_scope", "mmdet"))
        dataloader_key = f"{split}_dataloader"
        if dataloader_key not in cfg:
            dataloader_key = "test_dataloader" if "test_dataloader" in cfg else "val_dataloader"
        dataset_cfg = cfg[dataloader_key]["dataset"]
        dataset = DATASETS.build(dataset_cfg)
        if hasattr(dataset, "full_init"):
            dataset.full_init()
    except Exception as exc:
        raise DatasetLoadError(f"Failed to build dataset from config {config_path}: {exc}") from exc

    return dataset_from_mmdet_dataset(
        dataset,
        analysis_config=analysis_config or AnalysisConfig(),
        metadata={"config": str(config_path), "split": split, "dataset_type": type(dataset).__name__},
    )


def dataset_from_mmdet_dataset(
    dataset: Any,
    analysis_config: Optional[AnalysisConfig] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> AnalysisDataset:
    """Extract normalized records from a built MMDetection dataset."""

    cfg = analysis_config or AnalysisConfig()
    cfg.validate()
    if hasattr(dataset, "full_init"):
        dataset.full_init()
    metainfo = getattr(dataset, "metainfo", {}) or {}
    classes = _classes_from_metainfo(metainfo)
    images: List[ImageRecord] = []
    gts: List[GroundTruth] = []

    for idx in range(len(dataset)):
        info = dataset.get_data_info(idx) if hasattr(dataset, "get_data_info") else dataset.data_list[idx]
        image_id = str(info.get("img_id", idx))
        img_path = str(info.get("img_path") or info.get("file_name") or "")
        file_name = os.path.basename(img_path)
        width = info.get("width")
        height = info.get("height")
        images.append(ImageRecord(idx, image_id, file_name, img_path, width, height))
        for ann_idx, instance in enumerate(info.get("instances", [])):
            label = int(instance.get("bbox_label", instance.get("label", 0)))
            class_name = classes[label] if 0 <= label < len(classes) else str(label)
            box = to_xyxy(instance["bbox"], fmt="xyxy")
            box_area = float(instance.get("area", area(box)))
            crowd = bool(instance.get("iscrowd", False) or instance.get("crowd", False))
            ignored = bool(instance.get("ignore", False))
            if instance.get("ignore_flag", 0):
                crowd = True
            gts.append(GroundTruth(
                gt_id=f"{image_id}:{ann_idx}",
                image_id=image_id,
                image_index=idx,
                bbox_xyxy=box,
                class_id=label,
                class_name=class_name,
                area=box_area,
                object_size=object_size(box_area, cfg.area_ranges),
                ignored=ignored,
                crowd=crowd,
                extra={k: v for k, v in instance.items() if k not in {"bbox", "bbox_label", "label"}},
            ))

    return AnalysisDataset(
        images=images,
        ground_truths=gts,
        classes=classes,
        metadata={**(metadata or {}), "metainfo": metainfo},
        ann_file=_find_ann_file(dataset),
    )


def load_coco_dataset(
    ann_file: str,
    img_prefix: Optional[str] = None,
    classes: Optional[Sequence[str]] = None,
    analysis_config: Optional[AnalysisConfig] = None,
) -> AnalysisDataset:
    """Load image and annotation records from a COCO JSON file."""

    cfg = analysis_config or AnalysisConfig()
    cfg.validate()
    with Path(ann_file).open("r", encoding="utf-8") as f:
        data = json.load(f)
    categories = sorted(data.get("categories", []), key=lambda c: c.get("id", 0))
    class_names = list(classes) if classes else [str(c.get("name", c.get("id"))) for c in categories]
    cat_id_to_label = {int(c.get("id")): i for i, c in enumerate(categories)}
    anns_by_image: Dict[int, List[Dict[str, Any]]] = {}
    for ann in data.get("annotations", []):
        anns_by_image.setdefault(int(ann.get("image_id")), []).append(ann)

    images: List[ImageRecord] = []
    gts: List[GroundTruth] = []
    for idx, img in enumerate(data.get("images", [])):
        raw_id = img.get("id", idx)
        image_id = str(raw_id)
        file_name = str(img.get("file_name", image_id))
        img_path = _join_if_relative(img_prefix, file_name)
        width = img.get("width")
        height = img.get("height")
        images.append(ImageRecord(idx, image_id, file_name, img_path, width, height))
        for ann_idx, ann in enumerate(anns_by_image.get(int(raw_id), [])):
            if ann.get("ignore", False):
                ignored = True
            else:
                ignored = False
            cat_id = int(ann.get("category_id", 0))
            label = cat_id_to_label.get(cat_id, cat_id)
            class_name = class_names[label] if 0 <= label < len(class_names) else str(cat_id)
            box = to_xyxy(ann["bbox"], fmt="xywh")
            box_area = float(ann.get("area", area(box)))
            gts.append(GroundTruth(
                gt_id=f"{image_id}:{ann.get('id', ann_idx)}",
                image_id=image_id,
                image_index=idx,
                bbox_xyxy=box,
                class_id=label,
                class_name=class_name,
                area=box_area,
                object_size=object_size(box_area, cfg.area_ranges),
                ignored=ignored,
                crowd=bool(ann.get("iscrowd", False)),
                extra={"category_id": cat_id, "annotation_id": ann.get("id")},
            ))

    return AnalysisDataset(
        images=images,
        ground_truths=gts,
        classes=class_names,
        metadata={"dataset_type": "CocoDataset", "categories": categories},
        ann_file=str(ann_file),
    )


def _find_ann_file(dataset: Any) -> Optional[str]:
    """Best-effort annotation path extraction from wrapped datasets."""

    seen = set()
    queue: List[Any] = [dataset]
    while queue:
        item = queue.pop(0)
        if id(item) in seen:
            continue
        seen.add(id(item))
        ann_file = getattr(item, "ann_file", None)
        data_root = getattr(item, "data_root", None)
        if ann_file:
            return _join_if_relative(data_root, str(ann_file))
        if hasattr(item, "dataset"):
            queue.append(item.dataset)
        if hasattr(item, "datasets"):
            queue.extend(list(item.datasets))
    return None

