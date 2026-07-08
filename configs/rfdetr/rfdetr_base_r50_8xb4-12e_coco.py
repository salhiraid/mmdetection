_base_ = '../dino/dino-4scale_r50_8xb2-12e_coco.py'

# RF-DETR Base reference configuration for MMDetection.
# Official Roboflow RF-DETR Base uses a DINOv2 backbone and 560x560
# inputs. This in-repository reference keeps the same RF-DETR input scale
# and DETR-style training recipe while using MMDetection's native ResNet-50
# + DINO implementation.
model = dict(
    type='RFDETR',
    num_queries=300,
    test_cfg=dict(max_per_img=300))

image_size = 560
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoiceResize',
        scales=[504, 532, 560, 588, 616],
        keep_ratio=True),
    dict(
        type='Pad',
        size=(image_size, image_size),
        pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline))

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(image_size, image_size), keep_ratio=True),
    dict(
        type='Pad',
        size=(image_size, image_size),
        pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
