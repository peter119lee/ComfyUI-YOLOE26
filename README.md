# ComfyUI-YOLOE26

Open-vocabulary prompt segmentation nodes for ComfyUI powered by Ultralytics YOLOE-26.

This is a ComfyUI custom node pack, not a standalone application.

## Status

- Usable **beta / release candidate**
- Helper / node-level tests are passing
- Real ComfyUI smoke / integration validation is still recommended before wider public release
- **Local-first model loading**; automatic download is optional, disabled by default, and downloaded and verified against an allowlisted SHA256 digest

## What this node pack adds

This repository adds a small YOLOE-26 node set for ComfyUI so you can:

- load a local YOLOE-26 model once and reuse it in a workflow
- segment objects from text prompts like `person`, `car`, or `red apple`
- get either a merged mask, per-instance masks, or per-class merged masks
- keep structured JSON metadata for downstream routing or automation
- refine existing `MASK` batches without re-running detection
- pick the single best instance mask from an instance-mask result set

## Included nodes

| Node | Purpose | Main outputs |
| --- | --- | --- |
| `YOLOE-26 Load Model` | Load and validate a local YOLOE-26 model file | `YOLOE_MODEL` |
| `YOLOE-26 Prompt Segment` | Run prompt-based segmentation and preview results | `IMAGE`, `MASK`, `INT` |
| `YOLOE-26 Detection Metadata` | Return detections as JSON metadata | `STRING`, `INT` |
| `YOLOE-26 Instance Masks` | Return one output mask per detected instance | `MASK`, `STRING`, `INT` |
| `YOLOE-26 Class Masks` | Return one merged output mask per prompt class | `MASK`, `STRING`, `INT` |
| `YOLOE-26 Refine Mask` | Post-process an existing `MASK` batch | `MASK`, `STRING`, `INT` |
| `YOLOE-26 Select Best Instance` | Pick one best mask from instance-mask outputs | `MASK`, `STRING`, `INT` |

## Quick start

### 1. Install into ComfyUI

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
git clone https://github.com/peter119lee/ComfyUI-YOLOE26 ComfyUI-YOLOE26
```

Install dependencies in the **same Python environment used by ComfyUI**:

```bash
pip install -r requirements.txt
```

This node pack expects the host ComfyUI environment to already provide compatible `torch`, `numpy`, and `cv2` support.
Only `ultralytics` is listed in `requirements.txt` to reduce the chance of conflicting with the host environment.

### 2. Place a local YOLOE-26 model

The loader is local-first. Put your `.pt` file in one of these supported locations, or enable `auto_download` in the loader node if you want Ultralytics to try downloading a missing official weight automatically. Auto-download is currently allowlisted for official YOLOE-26 segmentation weights `yoloe-26n-seg.pt`, `yoloe-26s-seg.pt`, `yoloe-26m-seg.pt`, `yoloe-26l-seg.pt`, and `yoloe-26x-seg.pt`, and the checkpoint is first downloaded to a local path, then verified against a pinned SHA256 digest, and only then loaded.

Put your `.pt` file in one of these supported locations:

- `ComfyUI/models/ultralytics/segm/`
- `ComfyUI/models/ultralytics/bbox/`
- `ComfyUI/models/ultralytics/`
- `ComfyUI/models/yoloe/`

Example model name:

```text
yoloe-26s-seg.pt
```

### 3. Restart ComfyUI

After restart, confirm the 7 `YOLOE-26 ...` nodes appear in the node list.

### 4. Build the smallest useful workflow

```text
Load Image --------------------\
Prompt String -------------------> YOLOE-26 Prompt Segment -> use mask downstream
YOLOE-26 Load Model ----------/
```

Recommended first test:

- `model_name`: `yoloe-26s-seg.pt`
- `device`: `auto`
- `prompt`: `person`
- `conf`: `0.1`
- `iou`: `0.7`
- `max_det`: `300`
- `mask_threshold`: `0.5`
- `imgsz`: `640`

## Common workflow patterns

### Simple prompt segmentation

```text
Load Image --------------------\
Prompt String -------------------> YOLOE-26 Prompt Segment -> use merged mask for inpainting / compositing / cropping
YOLOE-26 Load Model ----------/
```

### Keep per-object masks

```text
Load Image --------------------\
Prompt String -------------------> YOLOE-26 Instance Masks -> use instance_masks + instance_metadata_json downstream
YOLOE-26 Load Model ----------/
```

### Keep one best object only

```text
YOLOE-26 Instance Masks
  ├-> instance_masks -----------\
  └-> instance_metadata_json ----> YOLOE-26 Select Best Instance -> keep one selected mask
```

### Refine an existing mask batch

```text
Any MASK-producing node
  -> YOLOE-26 Refine Mask
  -> use refined_masks downstream
```

### Build one mask per prompt class

```text
Load Image --------------------\
Prompt String -------------------> YOLOE-26 Class Masks -> use class_masks + class_metadata_json downstream
YOLOE-26 Load Model ----------/
```

## Prompt format

Use comma-separated text classes:

```text
person
person, car, dog
red apple, green bottle
```

## Node reference

### 1. YOLOE-26 Load Model

Loads and validates a local YOLOE-26 model file.
Only load `.pt` files you trust.

**Inputs**
- `model_name`: local weight file name, for example `yoloe-26s-seg.pt`
- `device`: `auto`, `cpu`, `cuda`, `cuda:N`, and `mps` when available
- `auto_download`: when `true`, download an allowlisted official model to a local path, verify its SHA256 digest, and then load it; currently allowlisted for `yoloe-26n-seg.pt`, `yoloe-26s-seg.pt`, `yoloe-26m-seg.pt`, `yoloe-26l-seg.pt`, and `yoloe-26x-seg.pt`

**Outputs**
- `YOLOE_MODEL`

### 2. YOLOE-26 Prompt Segment

Runs text-prompt segmentation and returns:
- annotated preview image
- merged binary mask
- detection count

**Required inputs**
- `model`
- `image`
- `prompt`

**Optional controls**
- `conf`
- `iou`
- `max_det`
- `mask_threshold`
- `imgsz`
- `show_boxes`
- `show_labels`
- `show_conf`
- `show_masks`

### 3. YOLOE-26 Detection Metadata

Runs text-prompt segmentation and returns:
- `metadata_json` with per-image detections, boxes, scores, mask areas, and class names
- `detection_count`

**Required inputs**
- `model`
- `image`
- `prompt`

**Optional controls**
- `conf`
- `iou`
- `max_det`
- `mask_threshold`
- `imgsz`

### 4. YOLOE-26 Instance Masks

Runs text-prompt segmentation and returns one mask per detected instance across the input batch.

**Required inputs**
- `model`
- `image`
- `prompt`

**Returns**
- `instance_masks`
- `instance_metadata_json` describing only detections that produced an output mask, including `batch_index`, class, score, box, and `output_mask_index`
- `count`

If no detections are found, the node currently returns a placeholder zero mask and reports `count = 0`.

**Optional controls**
- `conf`
- `iou`
- `max_det`
- `mask_threshold`
- `imgsz`

### 5. YOLOE-26 Class Masks

Runs text-prompt segmentation and returns one merged mask per prompt class for each input image.

**Required inputs**
- `model`
- `image`
- `prompt`

**Returns**
- `class_masks`
- `class_metadata_json` describing output mask order, source instances, and `output_mask_count`
- `output_mask_count` as the number of output class masks in the returned `MASK` batch

The node always returns one mask per prompt class for each input image. If a class is not detected, that class mask is all zeros.

**Optional controls**
- `conf`
- `iou`
- `max_det`
- `mask_threshold`
- `imgsz`

### 6. YOLOE-26 Refine Mask

Applies binary post-processing to an incoming `MASK` batch without re-running detection.

**Required inputs**
- `masks`

**Returns**
- `refined_masks`
- `refined_metadata_json` with appended `refinement` settings
- `count`

**Optional controls**
- `method`
- `kernel_size`
- `iterations`
- `min_area`
- `metadata_json`

Supported refinement methods:
- `threshold`
- `open`
- `close`
- `dilate`
- `erode`
- `largest_component`
- `fill_holes`

### 7. YOLOE-26 Select Best Instance

Selects a single best mask from `YOLOE-26 Instance Masks` outputs using instance metadata.

**Required inputs**
- `instance_masks`
- `instance_metadata_json`

**Returns**
- `best_mask`
- `best_instance_metadata_json` with `selection_mode`, `candidate_count`, and `selected_detection`
- `selected_mask_index`

**Optional controls**
- `selection_mode`

Supported selection modes:
- `highest_confidence`
- `largest_area`
- `confidence_then_area`

## Outputs and metadata

This node pack uses standard ComfyUI types:

- `IMAGE`: ComfyUI image tensor with YOLOE annotations
- `MASK`: float mask tensor in ComfyUI format
- `INT`: detection count, output mask count, or selected mask index depending on node
- `STRING`: JSON metadata for detections, instance-mask indexing, class-mask indexing, refinement settings, or best-instance selection

Metadata outputs are plain JSON strings so they stay easy to inspect, save, or parse in downstream workflows.
Treat them as untrusted data if you render them in external UIs or web views.

Notes:

- For `YOLOE-26 Class Masks`, `class_count` in metadata is the number of prompt classes, while `output_mask_count` is the number of masks returned in the `MASK` batch.
- For `YOLOE-26 Select Best Instance`, the node expects `instance_metadata_json` produced by `YOLOE-26 Instance Masks` so `output_mask_index` stays aligned with the `instance_masks` batch.

## Example workflow

A minimal ComfyUI **API-format** workflow example is included at:

- `examples/basic_api_workflow.json`

This example uses the compact API prompt style (`class_type` + `inputs`) rather than a full UI-exported workflow graph.
If you are testing through the ComfyUI API, treat it as an API prompt payload reference.
If you are testing only in the UI, rebuild the same nodes and parameters manually instead of importing it as a UI graph export.
It also demonstrates the optional `iou`, `max_det`, and `mask_threshold` controls on the segmentation nodes.
Before running it, place an input image named `example.png` in your ComfyUI input directory and make sure `yoloe-26s-seg.pt` is available in a supported model directory.

## Validation and release docs

For practical setup, smoke testing, and release checks, see:

- `SMOKE_TEST_GUIDE.md`
- `RELEASE_CHECKLIST_SHORT.md`
- `TODO_RELEASE_AND_USAGE.md`

## Current limitations

- The loader still accepts only `.pt` model names and local-first resolution from supported ComfyUI model directories
- Automatic model download depends on Ultralytics upstream model availability, network access, local write permissions, the current allowlist of approved official model names, and successful SHA256 verification of the downloaded checkpoint
- Metadata outputs are JSON strings rather than a custom ComfyUI structured type
- `YOLOE-26 Instance Masks` returns a placeholder zero mask when no detections are found and reports `count = 0`
- `YOLOE-26 Class Masks` always returns one mask per prompt class for each input image; masks are all-zero when a class is not detected
- The repo should still be validated in real ComfyUI workflows before broad public release

## Troubleshooting

- If model loading fails, verify the `.pt` file is placed in one of the supported ComfyUI model directories.
- If `auto_download` is enabled and loading still fails, verify that Ultralytics can reach its model source and that the environment can write downloaded weights to its cache directory.
- If inference fails after upgrading Ultralytics, verify that your installed version still exposes `from ultralytics import YOLOE`.
- This node pack currently targets the `ultralytics>=8.3.200` API family and may require updates if upstream YOLOE APIs change.
- If you only need the mask, you can ignore the JSON outputs.
- For batch inputs, inspect the returned JSON metadata to map each output mask back to its source image.

## Attribution status

This repository references implementation ideas associated with a `prompt_segment.py` script attributed to [spawner1145](https://github.com/spawner1145).
The exact upstream repository, file URL, and license for that script have not yet been independently verified from this repo.
Until that provenance is confirmed, this repository does **not** include a project-level `LICENSE` file.
[Ultralytics](https://github.com/ultralytics/ultralytics) remains a separate third-party dependency with its own license terms.
