# Copyright (c) OpenMMLab. All rights reserved.
"""Native MMDetection RF-DETR integration.

The RF-DETR implementation used by this detector lives inside the MMDetection
package at :mod:`mmdet.models.detectors.rfdetr._rfdetr`. It does not import
from the reference ``rf-detr`` source tree at runtime.
"""
from typing import Any, Dict, List, Optional, Union

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from ..base import BaseDetector
from ._rfdetr.config import (RFDETRBaseConfig, RFDETRLargeConfig,
                             RFDETRMediumConfig, RFDETRNanoConfig,
                             RFDETRSegLargeConfig, RFDETRSegMediumConfig,
                             RFDETRSegNanoConfig, RFDETRSegSmallConfig,
                             RFDETRSegXLargeConfig, RFDETRSmallConfig,
                             TrainConfig)
from ._rfdetr.models import (apply_lora, build_criterion_from_config,
                             build_model_from_config, load_pretrain_weights)


_CONFIG_CLASSES = {
    'nano': RFDETRNanoConfig,
    'small': RFDETRSmallConfig,
    'base': RFDETRBaseConfig,
    'medium': RFDETRMediumConfig,
    'large': RFDETRLargeConfig,
    'seg_nano': RFDETRSegNanoConfig,
    'seg_small': RFDETRSegSmallConfig,
    'seg_medium': RFDETRSegMediumConfig,
    'seg_large': RFDETRSegLargeConfig,
    'seg_xlarge': RFDETRSegXLargeConfig,
}


@MODELS.register_module()
class MMDetRFDETR(BaseDetector):
    """Build and train RF-DETR directly through MMDetection.

    This detector contains a package-local copy of the RF-DETR architecture,
    matcher, criterion, and post-processing code. The external/reference
    ``rf-detr`` folder is only a development reference and is not added to
    ``sys.path`` or imported at runtime.
    """

    def __init__(self,
                 variant: str = 'base',
                 model_config: Optional[Dict[str, Any]] = None,
                 train_config: Optional[Dict[str, Any]] = None,
                 load_pretrain: bool = False,
                 data_preprocessor: Optional[Dict[str, Any]] = None,
                 init_cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if variant not in _CONFIG_CLASSES:
            raise ValueError(
                f'Unsupported RF-DETR variant: {variant}. Supported variants: '
                f'{sorted(_CONFIG_CLASSES)}')
        model_config = dict(model_config or {})
        train_config = dict(train_config or {})
        if not load_pretrain:
            model_config.setdefault('pretrain_weights', None)
        self.variant = variant
        self.model_config = _CONFIG_CLASSES[variant](**model_config)
        self.train_config = TrainConfig(
            dataset_dir='.', output_dir='.', **train_config)
        self.model = build_model_from_config(self.model_config, self.train_config)
        if load_pretrain and self.model_config.pretrain_weights is not None:
            load_pretrain_weights(self.model, self.model_config)
        if getattr(self.model_config, 'backbone_lora', False):
            apply_lora(self.model)
        self.criterion, self.postprocess = build_criterion_from_config(
            self.model_config, self.train_config)

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: Optional[SampleList] = None) -> Dict[str, Tensor]:
        return self.model(batch_inputs)

    def extract_feat(self, batch_inputs: Tensor) -> Dict[str, Tensor]:
        return self._forward(batch_inputs)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        targets = self._data_samples_to_targets(batch_data_samples,
                                                batch_inputs.device)
        outputs = self.model(batch_inputs, targets)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        losses = {f'rfdetr_{k}': v for k, v in loss_dict.items()}
        losses['loss'] = sum(loss_dict[k] * weight_dict[k]
                             for k in loss_dict if k in weight_dict)
        return losses

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        outputs = self.model(batch_inputs)
        target_sizes = torch.stack([
            torch.as_tensor(
                sample.metainfo.get('ori_shape', sample.metainfo['img_shape'])[:2],
                device=batch_inputs.device)
            for sample in batch_data_samples
        ])
        results = self.postprocess(outputs, target_sizes)
        for sample, result in zip(batch_data_samples, results):
            pred = InstanceData()
            pred.bboxes = result['boxes']
            pred.scores = result['scores']
            pred.labels = result['labels']
            if 'masks' in result:
                pred.masks = result['masks']
            sample.pred_instances = pred
        return batch_data_samples

    @staticmethod
    def _data_samples_to_targets(
            batch_data_samples: SampleList,
            device: Union[str, torch.device]) -> List[Dict[str, Tensor]]:
        targets = []
        for sample in batch_data_samples:
            img_h, img_w = sample.metainfo['img_shape'][:2]
            instances = sample.gt_instances
            bboxes = instances.bboxes.to(device=device, dtype=torch.float32)
            labels = instances.labels.to(device=device, dtype=torch.long)
            scale = bboxes.new_tensor([img_w, img_h, img_w,
                                       img_h]).clamp(min=1)
            boxes = bboxes / scale
            boxes = torch.stack(((boxes[:, 0] + boxes[:, 2]) * 0.5,
                                 (boxes[:, 1] + boxes[:, 3]) * 0.5,
                                 boxes[:, 2] - boxes[:, 0],
                                 boxes[:, 3] - boxes[:, 1]), dim=-1)
            target = {
                'boxes': boxes.clamp(min=0, max=1),
                'labels': labels,
                'orig_size': torch.as_tensor(
                    sample.metainfo.get('ori_shape', (img_h, img_w))[:2],
                    device=device),
                'size': torch.as_tensor((img_h, img_w), device=device),
            }
            if hasattr(instances, 'masks'):
                target['masks'] = instances.masks.to_tensor(
                    dtype=torch.float32, device=device)
            targets.append(target)
        return targets
