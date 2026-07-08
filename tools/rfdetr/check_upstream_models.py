#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""Verify RF-DETR mirrored configs and optional official package models."""
import argparse
import hashlib
import importlib
import importlib.util
from pathlib import Path

import yaml
from mmengine.config import Config

EXPECTED_PACKAGE_MODELS = [
    'RFDETRNano',
    'RFDETRSmall',
    'RFDETRMedium',
    'RFDETRLarge',
    'RFDETRSegNano',
    'RFDETRSegSmall',
    'RFDETRSegMedium',
    'RFDETRSegLarge',
    'RFDETRSegXLarge',
    'RFDETRSeg2XLarge',
]

EXPECTED_MMDET_CONFIGS = {
    'rfdetr_nano_r50_8xb8-12e_coco.py': 384,
    'rfdetr_small_r50_8xb6-12e_coco.py': 512,
    'rfdetr_base_r50_8xb4-12e_coco.py': 560,
    'rfdetr_medium_r50_8xb4-12e_coco.py': 576,
    'rfdetr_large_r50_8xb2-12e_coco.py': 704,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Check RF-DETR model/config integration files.')
    parser.add_argument(
        '--manifest',
        default='configs/rfdetr/upstream_manifest.yml',
        help='Manifest containing expected upstream YAML hashes/classes.')
    parser.add_argument(
        '--external-yaml-dir',
        default='configs/rfdetr/external_yaml',
        help='Directory containing mirrored Roboflow RF-DETR YAML configs.')
    parser.add_argument(
        '--mmdet-config-dir',
        default='configs/rfdetr',
        help='Directory containing MMDetection RF-DETR Python configs.')
    parser.add_argument(
        '--check-installed-package',
        action='store_true',
        help='Also import the installed official rfdetr package and check model exports.')
    return parser.parse_args()


def check_external_yaml(manifest_path, yaml_dir):
    manifest = yaml.safe_load(manifest_path.read_text())
    failures = []
    for item in manifest['configs']:
        path = yaml_dir / item['file']
        if not path.is_file():
            failures.append(f'missing YAML: {path}')
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item['sha256']:
            failures.append(f'sha256 mismatch for {path.name}')
        data = yaml.safe_load(path.read_text())
        model_class = data['model']['model_config']['class_path']
        train_class = data['model']['train_config']['class_path']
        if model_class != item['class_path']:
            failures.append(f'model class mismatch for {path.name}')
        if train_class != item['train_config']:
            failures.append(f'train config mismatch for {path.name}')
        print(f'YAML OK: {path.name} -> {model_class}')
    return failures


def check_mmdet_configs(config_dir):
    failures = []
    for filename, image_size in EXPECTED_MMDET_CONFIGS.items():
        path = config_dir / filename
        if not path.is_file():
            failures.append(f'missing MMDetection config: {path}')
            continue
        cfg = Config.fromfile(path)
        if cfg.model.type != 'RFDETR':
            failures.append(f'{filename}: expected model.type RFDETR')
        if cfg.model.num_queries != 300:
            failures.append(f'{filename}: expected 300 queries')
        if cfg.image_size != image_size:
            failures.append(f'{filename}: expected image_size {image_size}')
        print(f'MMDet OK: {filename} -> image_size={cfg.image_size}')
    return failures


def check_installed_package():
    failures = []
    if importlib.util.find_spec('rfdetr') is None:
        return ['official rfdetr package is not installed']
    package = importlib.import_module('rfdetr')
    for model_name in EXPECTED_PACKAGE_MODELS:
        if not hasattr(package, model_name):
            failures.append(f'installed rfdetr missing {model_name}')
        else:
            print(f'Package OK: rfdetr.{model_name}')
    return failures


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    yaml_dir = Path(args.external_yaml_dir)
    config_dir = Path(args.mmdet_config_dir)

    failures = []
    failures.extend(check_external_yaml(manifest_path, yaml_dir))
    failures.extend(check_mmdet_configs(config_dir))
    if args.check_installed_package:
        failures.extend(check_installed_package())

    if failures:
        print('\nRF-DETR integration check failed:')
        for failure in failures:
            print(f'  - {failure}')
        raise SystemExit(1)
    print('\nRF-DETR integration check passed.')


if __name__ == '__main__':
    main()
