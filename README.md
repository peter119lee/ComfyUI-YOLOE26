# ComfyUI-YOLOE26

[English](README.md) | [简体中文](README.zh-CN.md)

Open-vocabulary prompt segmentation nodes for ComfyUI powered by Ultralytics YOLOE-26.

This is a ComfyUI custom node pack, not a standalone application.

## Status

- Public release
- All node-level tests passing
- Smoke-tested end-to-end in a real ComfyUI environment
- **Tested baseline:** ComfyUI 0.18.1 · Python 3.12.7 · PyTorch 2.7.1+cu118 · ultralytics 8.3.207
- **Local-first model loading**; automatic download is optional and disabled by default. When enabled, only the allowlisted official YOLOE-26 segmentation weights are supported, and resolution still depends on Ultralytics' current upstream asset naming and download behavior.
- The model picker is a dropdown: labels ending in `(local)` exist only when the `.pt` file is already present in one of the supported ComfyUI model folders. Labels ending in `(downloadable)` are official segmentation presets that can be fetched only when `auto_download` is enabled. Example workflow JSONs no longer force a `(downloadable)` label, but importing a workflow does not itself prove that network download or local-model discovery was exercised.

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

The loader is local-first. Use the dropdown in `YOLOE-26 Load Model`: labels ending in `(local)` appear only when the exact `.pt` file already exists in one of the supported ComfyUI model folders below. Labels ending in `(downloadable)` are official segmentation presets that can be fetched when `auto_download` is enabled.

> **First-time auto-download note:** When you run a workflow with a `(downloadable)` model for the first time, the model is downloaded and the workflow runs successfully. However, the dropdown label will still show `(downloadable)` until you restart ComfyUI (press **F5** or restart the server). After restart, the same model will appear as `(local)` in the dropdown. You do **not** need to restart before the first run — it will work either way.

Current auto-download limitation:

- only the allowlisted official segmentation weights are supported: `yoloe-26n-seg.pt`, `yoloe-26s-seg.pt`, `yoloe-26m-seg.pt`, `yoloe-26l-seg.pt`, `yoloe-26x-seg.pt`
- non-allowlisted names, non-seg presets, or renamed files are not part of the current auto-download contract
- successful download also depends on Ultralytics continuing to resolve those official asset names the same way upstream

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

That `mask` output is intended for other MASK-aware nodes. If you connect it to an image sink, convert it first with a mask-to-image node.

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
  └-> instance_metadata_json ----> YOLOE-26 Select Best Instance -> best_mask -> MaskToImage -> PreviewImage / SaveImage
```

### Refine an existing mask batch

```text
Any MASK-producing node
  -> YOLOE-26 Refine Mask
  -> refined_masks -> MaskToImage -> PreviewImage / SaveImage
```

### Build one mask per prompt class

```text
Load Image --------------------\
Prompt String -------------------> YOLOE-26 Class Masks -> class_masks -> MaskToImage -> PreviewImage / SaveImage
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
The model dropdown shows official presets as `(downloadable)` and existing files in supported ComfyUI model folders as `(local)`. If a file that you expect to be local still appears as `(downloadable)`, check that the weight really lives in one of the supported ComfyUI model directories rather than elsewhere in the workspace.

**Inputs**
- `model_name`: local weight file name, for example `yoloe-26s-seg.pt`
- `device`: `auto`, `cpu`, `cuda`, `cuda:N`, and `mps` when available
- `auto_download`: `false` = local-only loading from supported ComfyUI model directories; `true` = if the local model is missing, let Ultralytics try downloading the requested allowlisted official segmentation model first

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

All files in `examples/` are **ComfyUI API-format workflows** using the compact `class_type` + `inputs` style.
They are not full UI-exported graph JSON files.
If you are testing through the ComfyUI API, use them as payload references.
If you are testing only in the UI, rebuild the same nodes and parameters manually.

At the current evidence level, treat the examples as follows:

- **Smoke-target examples**: JSON payloads whose graph shape matches the repo's current intended smoke path or visualization path
- **Reference examples**: JSON payloads that document node wiring and downstream usage patterns; they still need real ComfyUI execution evidence before they should be treated as release-grade proof

### 1. Minimal getting-started

Start with:

- `examples/basic_api_workflow.json`

This is the smallest documented path for a first-run smoke attempt:

```text
LoadImage -> YOLOE26LoadModel -> YOLOE26PromptSegment -> PreviewImage
```

It is the recommended Quick Start workflow when you only want to confirm that model loading, prompt segmentation, annotated-image preview, and merged-mask output are wired end-to-end.
It should still be treated as a smoke target to execute, not as already-proven public-release evidence.

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

It also includes representative built-in sinks and preview helpers, including `MaskToImage` branches for mask visualization.
Use it as a coverage-oriented wiring reference for the whole node pack, not as evidence that every branch has already been smoke-validated in a real ComfyUI session.

### 3. Practical workflows

The practical examples are focused on common downstream use cases:

| Example | What it's for | Main nodes | Current evidence level |
| --- | --- | --- | --- |
| `examples/basic_api_workflow.json` | Quick Start and first-run smoke target | `YOLOE26LoadModel`, `YOLOE26PromptSegment`, `PreviewImage` | Documented smoke target; real ComfyUI evidence still pending in release docs |
| `examples/all_nodes_showcase_api.json` | Full wiring reference for the whole node pack | All 7 custom nodes plus representative preview/save helpers and `MaskToImage` visualization branches | Wiring/reference example, not release-proof smoke evidence |
| `examples/practical_prompt_segment_api.json` | Basic prompt segmentation for inpainting, compositing, or cropping | `YOLOE26PromptSegment`, `PreviewImage`, `SaveImage` | Smoke-target example for annotated-image output |
| `examples/practical_best_instance_api.json` | Practical best-instance selection with mask visualization via `MaskToImage` | `YOLOE26InstanceMasks`, `YOLOE26SelectBestInstance`, `MaskToImage` | Reference workflow pending real ComfyUI smoke confirmation |
| `examples/practical_class_masks_api.json` | Practical class-mask routing with mask visualization via `MaskToImage` | `YOLOE26ClassMasks`, `MaskToImage` | Reference workflow pending real ComfyUI smoke confirmation |
| `examples/practical_refine_mask_api.json` | Post-process an existing mask batch without re-running detection | `YOLOE26PromptSegment`, `YOLOE26RefineMask`, `MaskToImage`, `PreviewImage` | Smoke-target/reference hybrid; still needs checked-in real ComfyUI evidence |
| `examples/practical_detection_metadata_api.json` | Metadata-driven automation with a visible annotated preview branch | `YOLOE26DetectionMetadata`, `YOLOE26PromptSegment`, `PreviewImage` | Smoke-target example for metadata plus preview |
| `examples/practical_batch_multi_class_api.json` | Multi-class prompt routing and metadata alignment checks with mask visualization via `MaskToImage` | `YOLOE26ClassMasks`, `YOLOE26DetectionMetadata`, `MaskToImage` | Reference workflow pending real ComfyUI smoke confirmation |

`basic_api_workflow.json` is still the recommended Quick Start entry point.
If you want the fastest sanity smoke, start there.
If you want one compact workflow that exercises the whole custom node pack, use `all_nodes_showcase_api.json`.

## Validation and release docs

For practical setup, smoke testing, and release checks, see:

- `SMOKE_TEST_GUIDE.md`
- `RELEASE_CHECKLIST_SHORT.md`
- `TODO_RELEASE_AND_USAGE.md`

Workflow import note:
- example workflows use ComfyUI `LoadImage` nodes, so importing them may still require you to choose or remap an input image in the ComfyUI UI.
- `basic_api_workflow.json` no longer forces the explicit `upload` flag, but a `LoadImage` node still represents a user-provided image input rather than a bundled repo asset.

## Current limitations

- The loader still accepts only `.pt` model names and local-first resolution from supported ComfyUI model directories
- Automatic model download is limited to the allowlisted official YOLOE-26 segmentation weights and depends on Ultralytics upstream model availability, network access, local write permissions, and the current upstream resolution behavior for those official assets
- Metadata outputs are JSON strings rather than a custom ComfyUI structured type
- `YOLOE-26 Instance Masks` returns a placeholder zero mask when no detections are found and reports `count = 0`
- `YOLOE-26 Class Masks` always returns one mask per prompt class for each input image; masks are all-zero when a class is not detected
- Real ComfyUI smoke evidence, compatibility matrix work, and LICENSE / provenance confirmation are still pending, so the repo should not yet be described as fully public-release-ready

## Troubleshooting

- If model loading fails, verify the `.pt` file is placed in one of the supported ComfyUI model directories.
- If `auto_download` is enabled and loading still fails, verify that Ultralytics can reach its model source, that the environment can write downloaded weights to its cache directory, and that the requested model name is supported by the installed Ultralytics version.
- If inference fails after upgrading Ultralytics, verify that your installed version still exposes `from ultralytics import YOLOE`.
- This node pack currently targets the `ultralytics>=8.3.200` API family and may require updates if upstream YOLOE APIs change.
- If you only need the mask, you can ignore the JSON outputs.
- For batch inputs, inspect the returned JSON metadata to map each output mask back to its source image.

## Attribution and licensing status

This repository references implementation ideas associated with a `prompt_segment.py` script attributed to [spawner1145](https://github.com/spawner1145).
The exact upstream repository, file URL, and license context for that script have not yet been independently verified from this repo, so provenance notes remain important.

This repository now includes a project-level `LICENSE` file for the original source code and documentation contained here.
That MIT license applies only to this repository's original code and docs; it does **not** relicense third-party dependencies, model weights, downloaded assets, or upstream projects.
[Ultralytics](https://github.com/ultralytics/ultralytics) remains a separate third-party dependency with its own license terms, and official YOLOE / Ultralytics model weights remain subject to their own upstream terms.
