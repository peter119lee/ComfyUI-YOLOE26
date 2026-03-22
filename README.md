# ComfyUI-YOLOE26

Open-vocabulary prompt segmentation nodes for ComfyUI powered by Ultralytics YOLOE-26.

Core segmentation logic is adapted from `prompt_segment.py` by [spawner1145](https://github.com/spawner1145).

## Features

- YOLOE-26 text-prompt segmentation inside ComfyUI
- Merged binary mask output
- Per-instance mask output
- Batch support for annotated preview + merged masks
- Local-model-only loading for predictable behavior

## Included Nodes

### 1. YOLOE-26 Load Model
Loads and validates a local YOLOE-26 model file.

**Input**
- `model_name`: local weight file name, e.g. `yoloe-26s-seg.pt`
- `device`: `auto`, `cpu`, `cuda`, `cuda:0`, `cuda:1`

**Output**
- `YOLOE_MODEL`

### 2. YOLOE-26 Prompt Segment
Runs text-prompt segmentation and returns:
- annotated preview image
- merged binary mask
- detection count

### 3. YOLOE-26 Instance Masks
Runs text-prompt segmentation and returns one mask per detected instance.

Note: this node currently supports **batch size 1 only**.

## Installation

Clone this repository into your ComfyUI `custom_nodes` folder:

```bash
git clone https://github.com/peter119lee/ComfyUI-YOLOE26 ComfyUI-YOLOE26
```

Install dependencies in the same Python environment used by ComfyUI:

```bash
pip install -r requirements.txt
```

## Model Placement

Place YOLOE-26 weights in one of these directories:

- `ComfyUI/models/ultralytics/segm/`
- `ComfyUI/models/ultralytics/bbox/`
- `ComfyUI/models/ultralytics/`
- `ComfyUI/models/yoloe/`

Example model name:

```text
yoloe-26s-seg.pt
```

This node pack currently supports **local model files only**. Automatic downloads are intentionally disabled.

## Prompt Format

Use comma-separated text classes:

```text
person
person, car, dog
red apple, green bottle
```

## Outputs

- `IMAGE`: ComfyUI image tensor with YOLOE annotations
- `MASK`: float mask tensor in ComfyUI format
- `INT`: detection count

## Credits

- [spawner1145](https://github.com/spawner1145) for the original `prompt_segment.py`
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOE / YOLO26

## Current Limitations

- `YOLOE-26 Instance Masks` supports only a single input image at a time
- The loader accepts only local `.pt` model files in supported ComfyUI model directories
- No automatic model download in this initial release

## Suggested Workflow

```text
Load Image
  -> YOLOE-26 Load Model
  -> YOLOE-26 Prompt Segment
  -> Use mask downstream for inpainting / compositing / cropping
```
