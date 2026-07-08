# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS

from .dino import DINO


@MODELS.register_module()
class RFDETR(DINO):
    r"""RF-DETR-style detector for MMDetection.

    RF-DETR is a real-time DETR-family detector introduced by Roboflow. This
    class intentionally reuses MMDetection's DINO implementation because
    RF-DETR keeps the same end-to-end set-prediction interface: multi-scale
    image features are encoded, decoder queries predict boxes/classes, and
    Hungarian matching trains the detector without NMS.

    The architecture-specific choices that distinguish RF-DETR variants (for
    example backbone, input resolution, query count, and optimization schedule)
    are configured in config files rather than hard-coded here. This keeps the
    model compatible with existing MMDetection DINO heads, losses, data
    preprocessors, training loops, and checkpoint utilities.
    """
