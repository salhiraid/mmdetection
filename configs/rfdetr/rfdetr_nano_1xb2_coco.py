# Copyright (c) OpenMMLab. All rights reserved.
_base_ = '../_base_/default_runtime.py'

custom_imports = dict(
    imports=['mmdet.models.detectors.rfdetr'], allow_failed_imports=False)

model = dict(
    type='MMDetRFDETR',
    variant='nano',
    model_config=dict(num_classes=80, pretrain_weights=None),
    train_config=dict(num_select=300),
    load_pretrain=False,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=True,
        pad_size_divisor=16))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(384, 384), keep_ratio=False),
    dict(type='PackDetInputs')
]
test_pipeline = train_pipeline

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root='data/coco/',
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root='data/coco/',
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/coco/annotations/instances_val2017.json',
    metric='bbox')
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(type='MultiStepLR', by_epoch=True, milestones=[11], gamma=0.1)
]
