# RF-DETR

RF-DETR is a real-time DETR-family detector introduced by Roboflow. The native
Roboflow implementation uses DINOv2 vision-transformer backbones and publishes
its own checkpoints/API. This directory adds MMDetection-native RF-DETR-style
entry points that reuse the in-repository DINO transformer, head, Hungarian
assigner, losses, runners, and COCO tooling.

## MMDetection reference configs

These configs preserve the public RF-DETR variant input resolutions and starter
batch sizes while using MMDetection's native ResNet-50 + DINO implementation.
They are intended as trainable MMDetection references, not checkpoint-compatible
copies of Roboflow's DINOv2 models.

| Config | RF-DETR variant | Input | Batch/GPU | Queries |
| --- | --- | --- | --- | --- |
| `rfdetr_nano_r50_8xb8-12e_coco.py` | Nano | 384 square | 8 | 300 |
| `rfdetr_small_r50_8xb6-12e_coco.py` | Small | 512 square | 6 | 300 |
| `rfdetr_base_r50_8xb4-12e_coco.py` | Base | 560 square | 4 | 300 |
| `rfdetr_medium_r50_8xb4-12e_coco.py` | Medium | 576 square | 4 | 300 |
| `rfdetr_large_r50_8xb2-12e_coco.py` | Large | 704 square | 2 | 300 |

`rfdetr_r50_8xb2-12e_coco.py` remains as a backward-compatible alias for the
Base reference config added in the initial integration.

## Upstream Roboflow YAML examples

The `external_yaml/` directory mirrors Roboflow RF-DETR CLI example configs for
all currently published detection and segmentation variants:

- `rfdetr_nano.yaml`
- `rfdetr_small.yaml`
- `rfdetr_base.yaml`
- `rfdetr_medium.yaml`
- `rfdetr_large.yaml`
- `rfdetr_seg_nano.yaml`
- `rfdetr_seg_small.yaml`
- `rfdetr_seg_medium.yaml`
- `rfdetr_seg_large.yaml`
- `rfdetr_seg_xlarge.yaml`
- `rfdetr_seg_2xlarge.yaml`

Those YAML files are for the external `rfdetr fit --config ...` CLI. The Python
configs above are the MMDetection-native configs consumed by `tools/train.py`.
`upstream_manifest.yml` records the expected SHA-256 digest and official model
class for each mirrored YAML so the copy can be verified.

## Install the official RF-DETR package

To use the exact official Roboflow RF-DETR model code, install the optional
extra from this repository:

```bash
pip install -e .[rfdetr]
```

The MMDetection `RFDETR` class is a native compatibility entry point; the exact
Roboflow implementation remains available through the official `rfdetr` Python
package and CLI.

## Verify mirrored models/configs

Run the checker to validate every mirrored YAML, every MMDetection RF-DETR
variant config, and optionally the installed official RF-DETR package exports:

```bash
python tools/rfdetr/check_upstream_models.py
python tools/rfdetr/check_upstream_models.py --check-installed-package
```

## Create a custom COCO config

```bash
python tools/rfdetr/create_coco_config.py \
  --template configs/rfdetr/rfdetr_base_r50_8xb4-12e_coco.py \
  --output configs/rfdetr/rfdetr_r50_custom.py \
  --data-root data/my_coco/ \
  --num-classes 5
```

Then train with the standard MMDetection launcher:

```bash
python tools/train.py configs/rfdetr/rfdetr_r50_custom.py
```

## Scope

This integration intentionally avoids vendoring Roboflow's external `rfdetr`
package. It provides MMDetection model registration, RF-DETR-sized reference
configs, mirrored upstream YAML examples, and a config-generation helper so
RF-DETR-style experiments can be run with standard MMDetection workflows.
