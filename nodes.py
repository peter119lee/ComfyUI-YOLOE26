"""
ComfyUI nodes for YOLOE-26 open-vocabulary prompt segmentation.

Core segmentation logic adapted from spawner1145's prompt_segment.py
https://github.com/spawner1145
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

MAX_PROMPT_LENGTH = 512
MAX_PROMPT_CLASSES = 20
MAX_IMGSZ = 2048
ALLOWED_MODEL_EXTENSIONS = (".pt",)
ALLOWED_DEVICES = {"auto", "cpu", "cuda", "cuda:0", "cuda:1"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _candidate_model_dirs() -> tuple[Path, ...]:
    """Return supported local model directories inside ComfyUI."""
    comfy_base = Path(__file__).resolve().parents[2]
    return (
        comfy_base / "models" / "ultralytics" / "segm",
        comfy_base / "models" / "ultralytics" / "bbox",
        comfy_base / "models" / "ultralytics",
        comfy_base / "models" / "yoloe",
    )


def _resolve_model_path(model_name: str) -> str:
    """Resolve a local YOLOE model path from supported ComfyUI model directories.

    For first publication we only support local model files. Implicit downloads and
    arbitrary paths are intentionally disabled so repository behavior stays
    predictable and safe.
    """
    if not isinstance(model_name, str):
        raise TypeError("model_name must be a string.")

    name = model_name.strip()
    if not name:
        raise ValueError("model_name cannot be empty.")

    if Path(name).name != name:
        raise ValueError(
            "model_name must be a file name only. Put the model file inside "
            "ComfyUI/models/ultralytics/... or ComfyUI/models/yoloe/."
        )

    if not name.endswith(ALLOWED_MODEL_EXTENSIONS):
        raise ValueError(
            f"Unsupported model extension for '{name}'. Supported extensions: "
            f"{', '.join(ALLOWED_MODEL_EXTENSIONS)}."
        )

    for directory in _candidate_model_dirs():
        candidate = directory / name
        if candidate.exists() and candidate.is_file():
            return str(candidate)

    search_locations = ", ".join(str(path) for path in _candidate_model_dirs())
    raise FileNotFoundError(
        f"Model '{name}' was not found in local ComfyUI model directories. "
        f"Place the weight file in one of: {search_locations}"
    )


def _create_yoloe(model_path: str):
    from ultralytics import YOLOE

    return YOLOE(model_path)


def _validate_model_bundle(model: dict) -> tuple[object, str, str]:
    if not isinstance(model, dict):
        raise TypeError("model input must be a YOLOE model bundle returned by YOLOE-26 Load Model.")

    runtime_model = model.get("model")
    model_path = model.get("model_path")
    device = model.get("device", "auto")

    if runtime_model is None:
        raise ValueError("Invalid model bundle: missing loaded model instance.")

    if not isinstance(model_path, str) or not model_path:
        raise ValueError("Invalid model bundle: missing model_path.")

    if device not in ALLOWED_DEVICES:
        raise ValueError(f"Unsupported device '{device}'.")

    return runtime_model, model_path, device


def _validate_image_batch(image: torch.Tensor, node_name: str) -> None:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"{node_name} expected IMAGE input as torch.Tensor.")

    if image.ndim != 4:
        raise ValueError(
            f"{node_name} expected IMAGE tensor in ComfyUI format (B, H, W, C), "
            f"got shape {tuple(image.shape)}."
        )

    if image.shape[0] == 0:
        raise ValueError(f"{node_name} received an empty image batch.")

    if image.shape[-1] != 3:
        raise ValueError(
            f"{node_name} expected RGB images with 3 channels, got shape {tuple(image.shape)}."
        )


def _parse_classes(prompt: str) -> list[str]:
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string.")

    cleaned_prompt = prompt.strip()
    if not cleaned_prompt:
        raise ValueError("Prompt cannot be empty. Provide at least one class name.")

    if len(cleaned_prompt) > MAX_PROMPT_LENGTH:
        raise ValueError(
            f"Prompt is too long. Limit prompts to {MAX_PROMPT_LENGTH} characters."
        )

    classes = [item.strip() for item in cleaned_prompt.split(",") if item.strip()]
    if not classes:
        raise ValueError("Prompt cannot be empty. Provide at least one class name.")

    if len(classes) > MAX_PROMPT_CLASSES:
        raise ValueError(
            f"Too many prompt classes. Limit to {MAX_PROMPT_CLASSES} comma-separated classes."
        )

    return classes


def _build_predict_kwargs(device: str, conf: float, imgsz: int) -> dict:
    if not isinstance(conf, (int, float)):
        raise TypeError("conf must be a number between 0 and 1.")

    if not 0.0 <= float(conf) <= 1.0:
        raise ValueError("conf must be between 0 and 1.")

    if not isinstance(imgsz, int):
        raise TypeError("imgsz must be an integer.")

    if imgsz < 64 or imgsz > MAX_IMGSZ:
        raise ValueError(f"imgsz must be between 64 and {MAX_IMGSZ}.")

    predict_kwargs = {
        "conf": float(conf),
        "imgsz": imgsz,
        "verbose": False,
    }
    if device != "auto":
        predict_kwargs["device"] = device

    return predict_kwargs


def _load_runtime_model(model: dict):
    runtime_model, model_path, device = _validate_model_bundle(model)
    return runtime_model, device, model_path


def _comfy_image_to_bgr(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert a single ComfyUI IMAGE tensor (H, W, C) float32 [0,1] RGB
    to numpy uint8 BGR for OpenCV / Ultralytics."""
    if image_tensor.ndim != 3 or image_tensor.shape[-1] != 3:
        raise ValueError(
            "Expected a single image tensor with shape (H, W, 3) in RGB format, "
            f"got {tuple(image_tensor.shape)}."
        )

    img_np = (image_tensor.detach().cpu().float().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)


def _bgr_to_comfy_image(bgr: np.ndarray) -> torch.Tensor:
    """Convert numpy uint8 BGR back to ComfyUI IMAGE tensor (H, W, C)
    float32 [0,1] RGB."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.astype(np.float32) / 255.0)


def _build_binary_mask(result, height: int, width: int) -> np.ndarray:
    """Merge all instance masks into a single binary mask (float32 0/1).

    Adapted from spawner1145's prompt_segment.py.
    """
    merged = np.zeros((height, width), dtype=np.float32)

    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        return merged

    for mask_tensor in result.masks.data:
        mask = mask_tensor.detach().cpu().numpy()
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        merged = np.maximum(merged, (mask > 0.5).astype(np.float32))

    return merged


def _build_per_instance_masks(result, height: int, width: int) -> list[np.ndarray]:
    """Return a list of individual binary masks (float32 0/1), one per instance."""
    masks: list[np.ndarray] = []

    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        return masks

    for mask_tensor in result.masks.data:
        mask = mask_tensor.detach().cpu().numpy()
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        masks.append((mask > 0.5).astype(np.float32))

    return masks


# ---------------------------------------------------------------------------
# Node: Model Loader
# ---------------------------------------------------------------------------

class YOLOE26LoadModel:
    """Validate and prepare a local YOLOE-26 model for prompt segmentation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {
                    "default": "yoloe-26s-seg.pt",
                    "tooltip": (
                        "Local YOLOE-26 model file name. Place the weight file in "
                        "ComfyUI/models/ultralytics/segm/, ComfyUI/models/ultralytics/, "
                        "or ComfyUI/models/yoloe/."
                    ),
                }),
            },
            "optional": {
                "device": (["auto", "cpu", "cuda", "cuda:0", "cuda:1"], {
                    "default": "auto",
                    "tooltip": "Inference device. 'auto' lets Ultralytics choose.",
                }),
            },
        }

    RETURN_TYPES = ("YOLOE_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "YOLOE26"

    def load_model(self, model_name: str, device: str = "auto"):
        if device not in ALLOWED_DEVICES:
            raise ValueError(f"Unsupported device '{device}'.")

        resolved = _resolve_model_path(model_name)

        try:
            runtime_model = _create_yoloe(resolved)
        except Exception as exc:
            raise RuntimeError(f"Failed to validate YOLOE-26 model '{resolved}': {exc}") from exc

        return ({"model": runtime_model, "model_path": resolved, "device": device},)


# ---------------------------------------------------------------------------
# Node: Prompt Segment
# ---------------------------------------------------------------------------

class YOLOE26PromptSegment:
    """Run open-vocabulary segmentation with a text prompt on YOLOE-26.

    Returns an annotated preview image and a merged binary mask.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("YOLOE_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "person",
                    "multiline": False,
                    "tooltip": (
                        "Text prompt for open-vocabulary segmentation. "
                        "Separate multiple classes with commas, e.g. "
                        "'person, car, dog'."
                    ),
                }),
            },
            "optional": {
                "conf": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Detection confidence threshold.",
                }),
                "imgsz": ("INT", {
                    "default": 640,
                    "min": 64,
                    "max": MAX_IMGSZ,
                    "step": 32,
                    "tooltip": "Inference image size.",
                }),
                "show_boxes": ("BOOLEAN", {"default": True}),
                "show_labels": ("BOOLEAN", {"default": True}),
                "show_conf": ("BOOLEAN", {"default": True}),
                "show_masks": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("annotated_image", "mask", "detection_count")
    FUNCTION = "segment"
    CATEGORY = "YOLOE26"

    def segment(
        self,
        model: dict,
        image: torch.Tensor,
        prompt: str,
        conf: float = 0.1,
        imgsz: int = 640,
        show_boxes: bool = True,
        show_labels: bool = True,
        show_conf: bool = True,
        show_masks: bool = True,
    ):
        _validate_image_batch(image, "YOLOE-26 Prompt Segment")
        classes = _parse_classes(prompt)
        _, _, device = _validate_model_bundle(model)
        predict_kwargs = _build_predict_kwargs(device, conf, imgsz)
        yoloe, _, model_path = _load_runtime_model(model)

        try:
            yoloe.set_classes(classes)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to configure YOLOE-26 classes {classes} for model '{model_path}': {exc}"
            ) from exc

        batch_annotated: list[torch.Tensor] = []
        batch_masks: list[torch.Tensor] = []
        total_detections = 0

        for index in range(image.shape[0]):
            img_bgr = _comfy_image_to_bgr(image[index])
            height, width = img_bgr.shape[:2]

            try:
                result = yoloe.predict(img_bgr, **predict_kwargs)[0]
            except Exception as exc:
                raise RuntimeError(
                    f"YOLOE-26 inference failed for batch item {index} with prompt '{prompt}', "
                    f"imgsz={imgsz}: {exc}"
                ) from exc

            annotated_bgr = result.plot(
                conf=show_conf,
                labels=show_labels,
                boxes=show_boxes,
                masks=show_masks,
            )
            batch_annotated.append(_bgr_to_comfy_image(annotated_bgr))

            merged = _build_binary_mask(result, height, width)
            batch_masks.append(torch.from_numpy(merged))

            num = 0 if result.boxes is None else len(result.boxes)
            total_detections += num

        annotated_batch = torch.stack(batch_annotated, dim=0)
        mask_batch = torch.stack(batch_masks, dim=0)

        return (annotated_batch, mask_batch, total_detections)


# ---------------------------------------------------------------------------
# Node: Instance Masks (per-object individual masks)
# ---------------------------------------------------------------------------

class YOLOE26InstanceMasks:
    """Run YOLOE-26 prompt segmentation and output per-instance masks.

    This node currently accepts a single input image only, because per-instance
    masks from multiple images would need additional indexing metadata.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("YOLOE_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "person",
                    "multiline": False,
                    "tooltip": (
                        "Text prompt for open-vocabulary segmentation. "
                        "Separate multiple classes with commas."
                    ),
                }),
            },
            "optional": {
                "conf": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "imgsz": ("INT", {
                    "default": 640,
                    "min": 64,
                    "max": MAX_IMGSZ,
                    "step": 32,
                }),
            },
        }

    RETURN_TYPES = ("MASK", "INT")
    RETURN_NAMES = ("instance_masks", "count")
    FUNCTION = "segment_instances"
    CATEGORY = "YOLOE26"

    def segment_instances(
        self,
        model: dict,
        image: torch.Tensor,
        prompt: str,
        conf: float = 0.1,
        imgsz: int = 640,
    ):
        _validate_image_batch(image, "YOLOE-26 Instance Masks")
        if image.shape[0] != 1:
            raise ValueError(
                "YOLOE-26 Instance Masks currently supports batch size 1 only. "
                "Use a single image or split the batch before this node."
            )

        classes = _parse_classes(prompt)
        _, _, device = _validate_model_bundle(model)
        predict_kwargs = _build_predict_kwargs(device, conf, imgsz)
        yoloe, _, model_path = _load_runtime_model(model)

        try:
            yoloe.set_classes(classes)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to configure YOLOE-26 classes {classes} for model '{model_path}': {exc}"
            ) from exc

        img_bgr = _comfy_image_to_bgr(image[0])
        height, width = img_bgr.shape[:2]

        try:
            result = yoloe.predict(img_bgr, **predict_kwargs)[0]
        except Exception as exc:
            raise RuntimeError(
                f"YOLOE-26 instance-mask inference failed for prompt '{prompt}', imgsz={imgsz}: {exc}"
            ) from exc

        all_masks = [torch.from_numpy(mask) for mask in _build_per_instance_masks(result, height, width)]

        if not all_masks:
            all_masks.append(torch.zeros((height, width), dtype=torch.float32))

        mask_batch = torch.stack(all_masks, dim=0)
        count = len(all_masks) if len(all_masks) > 1 or torch.count_nonzero(mask_batch[0]) > 0 else 0

        return (mask_batch, count)


# ---------------------------------------------------------------------------
# Mappings
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "YOLOE26LoadModel": YOLOE26LoadModel,
    "YOLOE26PromptSegment": YOLOE26PromptSegment,
    "YOLOE26InstanceMasks": YOLOE26InstanceMasks,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YOLOE26LoadModel": "YOLOE-26 Load Model",
    "YOLOE26PromptSegment": "YOLOE-26 Prompt Segment",
    "YOLOE26InstanceMasks": "YOLOE-26 Instance Masks",
}
