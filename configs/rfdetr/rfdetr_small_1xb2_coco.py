# Copyright (c) OpenMMLab. All rights reserved.
_base_ = './rfdetr_nano_1xb2_coco.py'

model = dict(
    variant='small',
    model_config=dict(pretrain_weights=None, resolution=512),
    data_preprocessor=dict(pad_size_divisor=16))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='PackDetInputs')
]
test_pipeline = train_pipeline
