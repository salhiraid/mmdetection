# Copyright (c) OpenMMLab. All rights reserved.
"""Build an RF-DETR model through MMDetection only.

This script intentionally imports only MMDetection/MMEngine and the package-local
``MMDetRFDETR`` integration. It does not import the reference ``rf-detr`` package
or add ``rf-detr/src`` to ``sys.path``.
"""
import argparse

import torch
from mmengine.config import Config
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config', default='configs/rfdetr/rfdetr_nano_1xb2_coco.py')
    parser.add_argument('--device', default='cpu')
    parser.add_argument(
        '--forward', action='store_true', help='Run a dummy tensor forward.')
    return parser.parse_args()


def main():
    args = parse_args()
    register_all_modules(init_default_scope=False)
    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model)
    model.to(args.device)
    model.eval()
    print(f'Built {model.__class__.__name__} variant={model.variant}')
    print(f'num_classes={model.model_config.num_classes}')
    print(f'num_queries={model.model_config.num_queries}')
    print(f'resolution={model.model_config.resolution}')
    if args.forward:
        image = torch.randn(1, 3, model.model_config.resolution,
                            model.model_config.resolution, device=args.device)
        with torch.no_grad():
            outputs = model(image, mode='tensor')
        print('output keys:', sorted(outputs.keys()))
        for key, value in outputs.items():
            if torch.is_tensor(value):
                print(f'{key}: shape={tuple(value.shape)} dtype={value.dtype}')


if __name__ == '__main__':
    main()
