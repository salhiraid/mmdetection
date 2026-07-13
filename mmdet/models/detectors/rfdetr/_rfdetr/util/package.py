# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Deprecated: use ``rfdetr.utilities.package`` instead."""

from mmdet.models.detectors.rfdetr._rfdetr.utilities.decorators import _warn_deprecated_module

_warn_deprecated_module("mmdet.models.detectors.rfdetr._rfdetr.util.package", "mmdet.models.detectors.rfdetr._rfdetr.utilities.package", deprecated_in="1.6.0", remove_in="1.9.0")

from mmdet.models.detectors.rfdetr._rfdetr.utilities.package import get_sha, get_version  # noqa: F401, E402
