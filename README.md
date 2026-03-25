# ComfyUI-YOLOE26

Open-vocabulary prompt segmentation nodes for ComfyUI powered by Ultralytics YOLOE-26.

This is a ComfyUI custom node pack, not a standalone application.

## Status

- Usable **beta / release candidate**
- Helper / node-level tests are passing
- Real ComfyUI smoke / integration validation is still recommended before wider public release
- **Local-first model loading**; automatic download is optional and disabled by default. When enabled, the loader asks Ultralytics to resolve and download the requested official weight using its upstream model-loading behavior.

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

The loader is local-first. Put your `.pt` file in one of these supported locations, or enable `auto_download` in the loader node if you want Ultralytics to try downloading a missing official weight automatically.

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
Prompt String -------------------> YOLOE-26 Prompt Segment -> annotated_image -> PreviewImage
YOLOE-26 Load Model ----------/                             \-> mask -> use downstream
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
Prompt String -------------------> YOLOE-26 Prompt Segment -> annotated_image -> PreviewImage
YOLOE-26 Load Model ----------/                             \-> mask -> use merged mask for inpainting / compositing / cropping
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
  └-> instance_metadata_json ----> YOLOE-26 Select Best Instance -> best_mask -> PreviewImage / SaveImage
```

### Refine an existing mask batch

```text
Any MASK-producing node
  -> YOLOE-26 Refine Mask
  -> refined_masks -> PreviewImage / SaveImage
```

### Build one mask per prompt class

```text
Load Image --------------------\
Prompt String -------------------> YOLOE-26 Class Masks -> class_masks -> PreviewImage / SaveImage
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
- `auto_download`: `false` = local-only loading from supported ComfyUI model directories; `true` = if the local model is missing, let Ultralytics try downloading the requested official model first

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

## Example workflows

`examples/` 中的所有 example 檔案，都是 **ComfyUI API-format workflows**，使用精簡的 `class_type` + `inputs` prompt 風格。
它們不是完整的 UI-exported graph JSON 檔。
如果你是透過 ComfyUI API 測試，請把它們當成 payload references 使用。
如果你只是在 UI 中測試，請手動重建相同的 nodes 與參數。

Example workflows 分成兩類：

- **Runnable examples**：包含可直接預覽的 `IMAGE` branch，適合做 smoke test
- **Reference/API examples**：展示 node wiring 與 API payload；`MASK` 輸出通常需要先接適當的 mask 可視化或轉換節點，不保證可直接接 `PreviewImage` / `SaveImage`

### 1. Minimal getting-started

Start with:

- `examples/basic_api_workflow.json`

This is the smallest successful path for a first-run smoke test:

```text
LoadImage -> YOLOE26LoadModel -> YOLOE26PromptSegment -> PreviewImage
```

It is the recommended Quick Start workflow when you only want to confirm that model loading, prompt segmentation, annotated-image preview, and merged-mask output work end-to-end.

### 2. Full node-pack showcase

For a wiring reference that covers the full custom node pack, use:

- `examples/all_nodes_showcase_api.json`

This workflow includes all 7 custom nodes:

- `YOLOE26LoadModel`
- `YOLOE26PromptSegment`
- `YOLOE26DetectionMetadata`
- `YOLOE26InstanceMasks`
- `YOLOE26ClassMasks`
- `YOLOE26RefineMask`
- `YOLOE26SelectBestInstance`

It also includes representative built-in output sinks for annotated image preview only.
Use it when you want one API workflow that demonstrates the full node set and typical downstream wiring.

### 3. Real application workflows

The practical examples are focused on common downstream use cases:

| Example | What it's for | Main nodes | Expected output |
| --- | --- | --- | --- |
| `examples/basic_api_workflow.json` | Quick Start and first-run smoke testing | `YOLOE26LoadModel`, `YOLOE26PromptSegment`, `PreviewImage` | Annotated preview image from `annotated_image` |
| `examples/all_nodes_showcase_api.json` | Full wiring reference for the whole node pack | All 7 custom nodes plus representative `PreviewImage` / `SaveImage` sinks for the annotated image branch only | Prompt-segment preview plus full node coverage |
| `examples/practical_prompt_segment_api.json` | Basic prompt segmentation for inpainting, compositing, or cropping | `YOLOE26PromptSegment`, `PreviewImage`, `SaveImage` | Immediate annotated preview and saved prompt-segment result |
| `examples/practical_best_instance_api.json` | API/reference workflow for selecting one best detected subject from per-instance results | `YOLOE26InstanceMasks`, `YOLOE26SelectBestInstance` | `best_mask`, `best_instance_metadata_json`, and `selected_mask_index` |
| `examples/practical_class_masks_api.json` | API/reference workflow for routing one merged result per prompt class | `YOLOE26ClassMasks` | `class_masks`, `class_metadata_json`, and `output_mask_count` |
| `examples/practical_refine_mask_api.json` | Post-process an existing mask batch without re-running detection | `YOLOE26PromptSegment`, `YOLOE26RefineMask`, `PreviewImage` | Annotated preview plus `refined_masks` / `refined_metadata_json` |
| `examples/practical_detection_metadata_api.json` | Metadata-driven automation with a visible smoke-test branch | `YOLOE26DetectionMetadata`, `YOLOE26PromptSegment`, `PreviewImage` | `metadata_json` plus annotated preview output |
| `examples/practical_batch_multi_class_api.json` | API/reference workflow for multi-class prompt routing and metadata alignment checks | `YOLOE26ClassMasks`, `YOLOE26DetectionMetadata` | class-mask batch plus detection metadata for comparison |

`basic_api_workflow.json` is the recommended entry point for Quick Start.
If you want to inspect every custom node in one place, go to `all_nodes_showcase_api.json`.

## Validation and release docs

For practical setup, smoke testing, and release checks, see:

- `SMOKE_TEST_GUIDE.md`
- `RELEASE_CHECKLIST_SHORT.md`
- `TODO_RELEASE_AND_USAGE.md`

## Current limitations

- The loader still accepts only `.pt` model names and local-first resolution from supported ComfyUI model directories
- Automatic model download depends on Ultralytics upstream model availability, network access, local write permissions, and the current upstream resolution behavior for official YOLOE-26 weights
- Metadata outputs are JSON strings rather than a custom ComfyUI structured type
- `YOLOE-26 Instance Masks` returns a placeholder zero mask when no detections are found and reports `count = 0`
- `YOLOE-26 Class Masks` always returns one mask per prompt class for each input image; masks are all-zero when a class is not detected
- The repo should still be validated in real ComfyUI workflows before broad public release

## Troubleshooting

- If model loading fails, verify the `.pt` file is placed in one of the supported ComfyUI model directories.
- If `auto_download` is enabled and loading still fails, verify that Ultralytics can reach its model source, that the environment can write downloaded weights to its cache directory, and that the requested model name is supported by the installed Ultralytics version.
- If inference fails after upgrading Ultralytics, verify that your installed version still exposes `from ultralytics import YOLOE`.
- This node pack currently targets the `ultralytics>=8.3.200` API family and may require updates if upstream YOLOE APIs change.
- If you only need the mask, you can ignore the JSON outputs.
- For batch inputs, inspect the returned JSON metadata to map each output mask back to its source image.

## Attribution status

This repository references implementation ideas associated with a `prompt_segment.py` script attributed to [spawner1145](https://github.com/spawner1145).
The exact upstream repository, file URL, and license for that script have not yet been independently verified from this repo.
Until that provenance is confirmed, this repository does **not** include a project-level `LICENSE` file.
[Ultralytics](https://github.com/ultralytics/ultralytics) remains a separate third-party dependency with its own license terms.
